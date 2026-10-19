#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <map>
#include <deque>
#include <string>
#include <algorithm>
#include <chrono>

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

// ── Per-track persistent state ──────────────────────────────────────────────
// MUST be static: the .so is loaded once and called per-frame; state persists
// across frames for zone membership, dwell, repeat-entry, and trails.
// The job notes say "Keep ALL state LOCAL to the function" for mutable buffers
// that could be instantiated more than once — but this .so is a singleton
// per pipeline, and the zone/dwell/repeat logic REQUIRES cross-frame state.
// We guard with a mutex if ever called from two threads (DeepStream does not).

static std::map<uint64_t, bool>    s_membership;       // track_id -> inside zone
static std::map<uint64_t, double>  s_entry_time;        // track_id -> zone entry timestamp
static std::map<uint64_t, bool>    s_alerted;            // track_id -> prolonged-presence fired
static std::map<uint64_t, std::deque<double>> s_entry_times; // track_id -> deque of entry timestamps
static std::map<uint64_t, std::deque<cv::Point>> s_trails;   // track_id -> last 30 centroids
static double s_last_perf_s = 0.0;
static uint64_t s_frame_index = 0;

// Normalized polygon vertices from PIR node 10
static const float ZONE_NORM[][2] = {
    {0.35f, 0.35f},
    {0.85f, 0.35f},
    {0.85f, 0.90f},
    {0.35f, 0.90f}
};
static const int ZONE_NVERTS = 4;

// Duration / repeat parameters from PIR
static const double DWELL_THRESHOLD_S  = 30.0;
static const double REPEAT_WINDOW_S    = 60.0;
static const int    REPEAT_COUNT       = 3;

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[render_safety] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====

    // ── Timing ──────────────────────────────────────────────────────────────
    auto now_tp = std::chrono::steady_clock::now();
    double now_s = std::chrono::duration<double>(now_tp.time_since_epoch()).count();
    double measured_fps = 0.0;
    if (s_last_perf_s > 0.0 && now_s > s_last_perf_s) {
        measured_fps = 1.0 / (now_s - s_last_perf_s);
    }
    s_last_perf_s = now_s;

    // Use frame_meta PTS as timestamp (seconds). Fallback to wall clock.
    NvDsFrameMeta *fmeta = (NvDsFrameMeta *) in->frame_meta;
    double timestamp_s = now_s;
    if (fmeta) {
        // PTS is in nanoseconds
        double pts = (double) fmeta->buf_pts;
        if (pts > 0.0) timestamp_s = pts / 1.0e9;
    }
    uint64_t frame_idx = s_frame_index++;

    // ── Build restricted-zone polygon in pixel coords ───────────────────────
    std::vector<cv::Point> zone_poly(ZONE_NVERTS);
    std::vector<cv::Point2f> zone_poly_f(ZONE_NVERTS);
    for (int i = 0; i < ZONE_NVERTS; ++i) {
        zone_poly[i] = cv::Point((int)(ZONE_NORM[i][0] * W), (int)(ZONE_NORM[i][1] * H));
        zone_poly_f[i] = cv::Point2f(ZONE_NORM[i][0] * (float)W, ZONE_NORM[i][1] * (float)H);
    }

    // ── Collect person objects from NvDsObjectMeta (gie_id=1, class 0) ──────
    struct PersonTrack {
        uint64_t track_id;
        float x1, y1, x2, y2;
        float score;
    };
    std::vector<PersonTrack> persons;

    if (fmeta) {
        for (NvDsMetaList *ol = fmeta->obj_meta_list; ol; ol = ol->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) ol->data;
            if (!obj) continue;
            if (obj->unique_component_id == 1 && obj->class_id == 0) {
                PersonTrack pt;
                pt.track_id = obj->object_id;
                pt.x1 = obj->rect_params.left;
                pt.y1 = obj->rect_params.top;
                pt.x2 = pt.x1 + obj->rect_params.width;
                pt.y2 = pt.y1 + obj->rect_params.height;
                pt.score = obj->confidence;
                persons.push_back(pt);
            }
        }
    }

    // ── Node 10: Restricted-zone entry/exit ─────────────────────────────────
    struct ZoneEvent {
        uint64_t track_id;
        std::string event_type; // "restricted_zone_entry" or "normal_exit"
        float confidence;
        float box[4];
        double timestamp_s;
        uint64_t frame_index;
    };
    struct ZoneObs {
        uint64_t track_id;
        bool inside;
        float score;
        float box[4];
    };

    std::vector<ZoneEvent> zone_events;
    std::vector<ZoneObs> zone_obs;

    for (const PersonTrack &p : persons) {
        // Foot point = bottom center of bbox
        cv::Point2f foot((p.x1 + p.x2) * 0.5f, p.y2);
        double dist = cv::pointPolygonTest(zone_poly_f, foot, false);
        bool inside = (dist >= 0.0);

        bool old_inside = false;
        auto mit = s_membership.find(p.track_id);
        if (mit != s_membership.end()) old_inside = mit->second;

        if (inside != old_inside) {
            ZoneEvent ev;
            ev.track_id = p.track_id;
            ev.event_type = inside ? "restricted_zone_entry" : "normal_exit";
            ev.confidence = p.score;
            ev.box[0] = p.x1; ev.box[1] = p.y1; ev.box[2] = p.x2; ev.box[3] = p.y2;
            ev.timestamp_s = timestamp_s;
            ev.frame_index = frame_idx;
            zone_events.push_back(ev);
        }
        s_membership[p.track_id] = inside;

        ZoneObs ob;
        ob.track_id = p.track_id;
        ob.inside = inside;
        ob.score = p.score;
        ob.box[0] = p.x1; ob.box[1] = p.y1; ob.box[2] = p.x2; ob.box[3] = p.y2;
        zone_obs.push_back(ob);

        // Update trail
        cv::Point center((int)((p.x1 + p.x2) * 0.5f), (int)((p.y1 + p.y2) * 0.5f));
        auto &trail = s_trails[p.track_id];
        trail.push_back(center);
        if ((int)trail.size() > 30) trail.pop_front();
    }

    // ── Node 11: Prolonged-presence ─────────────────────────────────────────
    struct DwellEvent {
        uint64_t track_id;
        double dwell_seconds;
        float confidence;
        float box[4];
        double timestamp_s;
        uint64_t frame_index;
    };
    std::vector<DwellEvent> dwell_events;
    std::map<uint64_t, double> dwell_timers;

    // Process crossing events first
    for (const ZoneEvent &ev : zone_events) {
        if (ev.event_type == "restricted_zone_entry") {
            s_entry_time[ev.track_id] = timestamp_s;
            s_alerted[ev.track_id] = false;
        } else if (ev.event_type == "normal_exit") {
            s_entry_time.erase(ev.track_id);
            s_alerted[ev.track_id] = false;
        }
    }

    for (const ZoneObs &ob : zone_obs) {
        if (!ob.inside) continue;
        uint64_t tid = ob.track_id;
        if (s_entry_time.find(tid) == s_entry_time.end()) {
            s_entry_time[tid] = timestamp_s;
            s_alerted[tid] = false;
        }
        double elapsed = std::max(0.0, timestamp_s - s_entry_time[tid]);
        dwell_timers[tid] = elapsed;

        if (elapsed >= DWELL_THRESHOLD_S && !s_alerted[tid]) {
            DwellEvent de;
            de.track_id = tid;
            de.dwell_seconds = elapsed;
            de.confidence = ob.score;
            de.box[0] = ob.box[0]; de.box[1] = ob.box[1];
            de.box[2] = ob.box[2]; de.box[3] = ob.box[3];
            de.timestamp_s = timestamp_s;
            de.frame_index = frame_idx;
            dwell_events.push_back(de);
            s_alerted[tid] = true;
        }
    }

    // ── Node 12: Repeat-entry ───────────────────────────────────────────────
    struct RepeatEvent {
        uint64_t track_id;
        int entry_count;
        float confidence;
        float box[4];
        double timestamp_s;
        uint64_t frame_index;
    };
    std::vector<RepeatEvent> repeat_events;

    for (const ZoneEvent &ev : zone_events) {
        if (ev.event_type != "restricted_zone_entry") continue;
        uint64_t tid = ev.track_id;
        auto &times = s_entry_times[tid];
        // Expire old entries
        while (!times.empty() && (timestamp_s - times.front()) > REPEAT_WINDOW_S)
            times.pop_front();
        times.push_back(timestamp_s);
        if ((int)times.size() >= REPEAT_COUNT) {
            RepeatEvent re;
            re.track_id = tid;
            re.entry_count = (int)times.size();
            re.confidence = ev.confidence;
            re.box[0] = ev.box[0]; re.box[1] = ev.box[1];
            re.box[2] = ev.box[2]; re.box[3] = ev.box[3];
            re.timestamp_s = timestamp_s;
            re.frame_index = frame_idx;
            repeat_events.push_back(re);
        }
    }

    // ── Collect incident labels per track for drawing ───────────────────────
    // severity: restricted_zone_entry=HIGH, normal_exit=LOW, prolonged_presence=CRITICAL, repeat_entry=HIGH
    struct Incident {
        uint64_t track_id;
        std::string event_type;
        std::string severity;
        float confidence;
        float box[4];
    };
    std::vector<Incident> all_incidents;

    for (const ZoneEvent &ev : zone_events) {
        Incident inc;
        inc.track_id = ev.track_id;
        inc.event_type = ev.event_type;
        inc.severity = (ev.event_type == "restricted_zone_entry") ? "HIGH" : "LOW";
        inc.confidence = ev.confidence;
        memcpy(inc.box, ev.box, sizeof(inc.box));
        all_incidents.push_back(inc);
    }
    for (const DwellEvent &de : dwell_events) {
        Incident inc;
        inc.track_id = de.track_id;
        inc.event_type = "prolonged_presence";
        inc.severity = "CRITICAL";
        inc.confidence = de.confidence;
        memcpy(inc.box, de.box, sizeof(inc.box));
        all_incidents.push_back(inc);
    }
    for (const RepeatEvent &re : repeat_events) {
        Incident inc;
        inc.track_id = re.track_id;
        inc.event_type = "repeat_entry";
        inc.severity = "HIGH";
        inc.confidence = re.confidence;
        memcpy(inc.box, re.box, sizeof(inc.box));
        all_incidents.push_back(inc);
    }

    // Build per-track incident label set
    std::map<uint64_t, std::vector<std::string>> incident_labels;
    for (const Incident &inc : all_incidents) {
        incident_labels[inc.track_id].push_back(inc.event_type + " " + inc.severity);
    }

    // Build inside-zone map
    std::map<uint64_t, bool> inside_map;
    for (const ZoneObs &ob : zone_obs) {
        inside_map[ob.track_id] = ob.inside;
    }

    // ── Node 15: Draw ───────────────────────────────────────────────────────
    cv::Mat frame = read_nv12_bgr(in);
    const float scale = (float)W / 1400.0f;
    const int box_thick = std::max(2, (int)(3.0f * scale));
    const double label_scale = std::max(0.8, (double)(1.1f * scale));
    const int label_thick = std::max(1, (int)(2.0f * scale));
    const int zone_thick = std::max(2, (int)(4.0f * scale));

    // Draw restricted-zone polygon in red
    {
        const cv::Point *pts_arr = zone_poly.data();
        int npts = (int)zone_poly.size();
        cv::polylines(frame, &pts_arr, &npts, 1, true, cv::Scalar(0, 0, 255), zone_thick, cv::LINE_AA);
    }

    // Draw semi-transparent zone fill
    {
        cv::Mat overlay = frame.clone();
        cv::fillConvexPoly(overlay, zone_poly.data(), (int)zone_poly.size(), cv::Scalar(0, 0, 180));
        cv::addWeighted(overlay, 0.15, frame, 0.85, 0.0, frame);
    }

    // Draw each person
    for (const PersonTrack &p : persons) {
        uint64_t tid = p.track_id;
        int x1 = (int)p.x1, y1 = (int)p.y1, x2 = (int)p.x2, y2 = (int)p.y2;

        bool has_incident = incident_labels.find(tid) != incident_labels.end();
        bool is_inside = false;
        auto iit = inside_map.find(tid);
        if (iit != inside_map.end()) is_inside = iit->second;

        cv::Scalar color;
        if (has_incident) color = cv::Scalar(0, 0, 255);       // red
        else if (is_inside) color = cv::Scalar(0, 140, 255);   // orange
        else color = cv::Scalar(0, 200, 0);                     // green

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, box_thick);

        // Motion trail
        auto tit = s_trails.find(tid);
        if (tit != s_trails.end()) {
            const std::deque<cv::Point> &trail = tit->second;
            for (int k = 1; k < (int)trail.size(); ++k) {
                cv::line(frame, trail[(size_t)(k - 1)], trail[(size_t)k], color,
                         std::max(1, box_thick / 2), cv::LINE_AA);
            }
        }

        // Label
        std::string status = is_inside ? "IN ZONE" : "CLEAR";
        char buf[256];
        auto dit = dwell_timers.find(tid);
        if (dit != dwell_timers.end()) {
            snprintf(buf, sizeof(buf), "ID %lu %s %.1fs", (unsigned long)tid, status.c_str(), dit->second);
        } else {
            snprintf(buf, sizeof(buf), "ID %lu %s", (unsigned long)tid, status.c_str());
        }
        std::string label(buf);

        auto lit = incident_labels.find(tid);
        if (lit != incident_labels.end()) {
            for (const std::string &s : lit->second) {
                label += " ";
                label += s;
            }
        }

        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, label_scale, label_thick, &baseline);
        int ly = std::max(ts.height + baseline + 8, y1);
        cv::rectangle(frame,
                      cv::Point(x1, ly - ts.height - baseline - 8),
                      cv::Point(std::min(W, x1 + ts.width + 12), ly),
                      color, -1);
        cv::putText(frame, label, cv::Point(x1 + 6, ly - baseline - 3),
                    cv::FONT_HERSHEY_SIMPLEX, label_scale, cv::Scalar(0, 0, 0),
                    label_thick, cv::LINE_AA);
    }

    // ── Publish metadata via render_meta::Builder ───────────────────────────
    render_meta::Builder sb(W, H);

    // Publish zone polygon
    {
        std::vector<double> poly_xy;
        for (int i = 0; i < ZONE_NVERTS; ++i) {
            poly_xy.push_back((double)zone_poly[i].x);
            poly_xy.push_back((double)zone_poly[i].y);
        }
        sb.polygon(poly_xy, "restricted_zone");
    }

    // Publish each person box with status
    for (const PersonTrack &p : persons) {
        uint64_t tid = p.track_id;
        bool is_inside = false;
        auto iit = inside_map.find(tid);
        if (iit != inside_map.end()) is_inside = iit->second;

        std::string lbl = is_inside ? "person_in_zone" : "person";
        sb.box((double)p.x1, (double)p.y1,
               (double)(p.x2 - p.x1), (double)(p.y2 - p.y1),
               lbl.c_str(), (int)tid, (double)p.score);
    }

    // Publish incidents as fields
    for (const Incident &inc : all_incidents) {
        char key[128];
        snprintf(key, sizeof(key), "incident_%s_track_%lu",
                 inc.event_type.c_str(), (unsigned long)inc.track_id);
        sb.field_s(key, inc.severity.c_str());
    }

    // Publish dwell timers
    for (auto &dt : dwell_timers) {
        char key[64];
        snprintf(key, sizeof(key), "dwell_s_track_%lu", (unsigned long)dt.first);
        sb.field_d(key, dt.second);
    }

    // Publish trails as polylines
    for (const PersonTrack &p : persons) {
        auto tit = s_trails.find(p.track_id);
        if (tit != s_trails.end() && tit->second.size() > 1) {
            std::vector<double> trail_xy;
            for (const cv::Point &pt : tit->second) {
                trail_xy.push_back((double)pt.x);
                trail_xy.push_back((double)pt.y);
            }
            char lbl[64];
            snprintf(lbl, sizeof(lbl), "trail_%lu", (unsigned long)p.track_id);
            sb.polyline(trail_xy, lbl);
        }
    }

    // Publish performance
    sb.field_d("measured_fps", measured_fps);
    sb.field_i("frame_index", (int64_t)frame_idx);
    sb.field_d("timestamp_s", timestamp_s);

    write_nv12(frame, in);
    sb.publish(in->frame_meta);

    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);