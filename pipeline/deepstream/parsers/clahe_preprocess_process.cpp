#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLAHE pre-inference luminance conditioning
// ─────────────────────────────────────────────────────────────────────────────
// This node has NO upstream model — it operates directly on the NV12 Y plane.
// Algorithm from the PIR pseudo_code and reference:
//   1. Extract the Y plane (limited-range luma 16..235).
//   2. Subsample every 4th pixel, normalise to [0,1] via (Y-16)/219.
//   3. Compute median (p50).
//   4. Ramp: off=0.3, full=0.15 → strength s = clamp((0.3-m)/(0.3-0.15), 0, 1).
//   5. If s>0, apply CLAHE (clipLimit=2.0, tileGrid=8×8) to Y, blend with original.
//   6. Write conditioned Y back; chroma (UV) unchanged.

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[clahe_preprocess] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    (void) find_rs_meta; // suppress unused warnings for helpers we don't need
    (void) collect_tensors;
    (void) find_tensor;
    (void) nms_classaware;
    (void) place_panel;
    (void) colormap_minmax;
    (void) write_nv12;
    (void) read_nv12_bgr;

    // ── Step 1: wrap Y plane into a cv::Mat respecting pitch ──
    // We must NOT assume contiguous rows; Y_pitch >= W.
    cv::Mat Y_pitched(H, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(Y_pitched.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);

    // ── Step 2: subsample every 4th pixel, compute normalised median ──
    // sample = (y[::4, ::4].astype(float) - 16.0) / 219.0
    const int STEP = 4;
    int sH = (H + STEP - 1) / STEP;
    int sW = (W + STEP - 1) / STEP;
    int nSamples = sH * sW;
    std::vector<float> samples(nSamples);
    int idx = 0;
    for (int r = 0; r < H; r += STEP) {
        const uint8_t *row = Y_pitched.ptr(r);
        for (int c = 0; c < W; c += STEP) {
            samples[idx++] = (static_cast<float>(row[c]) - 16.0f) / 219.0f;
        }
    }
    nSamples = idx; // actual count

    // ── Step 3: median ──
    std::nth_element(samples.begin(), samples.begin() + nSamples / 2, samples.begin() + nSamples);
    float median = samples[nSamples / 2];

    // ── Step 4: ramp strength ──
    // ramp: off=0.3, full=0.15
    // s = clamp((off - m) / (off - full), 0, 1)
    const float RAMP_OFF  = 0.3f;
    const float RAMP_FULL = 0.15f;
    float s = (RAMP_OFF - median) / (RAMP_OFF - RAMP_FULL);
    s = std::min<float>(std::max<float>(s, 0.0f), 1.0f);

    if (s > 0.0f) {
        // ── Step 5: CLAHE on Y plane ──
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        cv::Mat fixed;
        clahe->apply(Y_pitched, fixed);

        // Blend: y = fixed if s >= 1 else addWeighted(y, 1-s, fixed, s, 0)
        cv::Mat blended;
        if (s >= 1.0f) {
            blended = fixed;
        } else {
            cv::addWeighted(Y_pitched, static_cast<double>(1.0f - s),
                            fixed, static_cast<double>(s), 0.0, blended);
        }

        // ── Step 6: write conditioned Y back; UV unchanged ──
        for (int r = 0; r < H; ++r)
            memcpy(in->Y_host + (size_t)r * in->Y_pitch, blended.ptr(r), (size_t)W);
    }
    // If s == 0, Y plane is untouched. UV plane is never modified.

    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);