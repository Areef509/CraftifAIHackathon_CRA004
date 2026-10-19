#!/bin/bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── VMS mode ──────────────────────────────────────────────────────
# Generated from run.sh. Same sources, models, tracker and boxes — only
# the sink differs: instead of a window or a file this serves
#     video     tcp://0.0.0.0:${VMS_MJPEG_PORT}   multipart MJPEG
#     metadata  tcp://0.0.0.0:${VMS_META_PORT}    newline JSON
# and appends every logged frame to ${VMS_ROOT}/events/cam<N>.jsonl.
# Connect the console with:  python3 vms_console.py --host 127.0.0.1
# Start this FIRST, or just leave the console open — it reconnects.
export VMS_ROOT="${VMS_ROOT:-${DIR}/store}"
export VMS_MIN_CONF="${VMS_MIN_CONF:-0.3}"
export VMS_INTERVAL="${VMS_INTERVAL:-5}"
export VMS_SKIP_EMPTY="${VMS_SKIP_EMPTY:-1}"
export VMS_STATUS="${VMS_STATUS:-2.0}"
export VMS_META_PORT="${VMS_META_PORT:-7002}"
export VMS_MJPEG_PORT="${VMS_MJPEG_PORT:-7010}"
export VMS_JPEG_QUALITY="${VMS_JPEG_QUALITY:-80}"
export VMS_MAX_SHAPES="${VMS_MAX_SHAPES:-64}"
mkdir -p "${VMS_ROOT}/events" 2>/dev/null || true
echo "[run.vms.sh] video tcp://0.0.0.0:${VMS_MJPEG_PORT}  metadata tcp://0.0.0.0:${VMS_META_PORT}" >&2


CAM_W=${CAM_W:-1920}
CAM_H=${CAM_H:-1080}
CAM_FPS=${CAM_FPS:-30}
CAM_FMT="${CAM_FMT:-YUYV}"

# ── H.264 encoder selection (run time, not build time) ────────────────
# NVENC needs /dev/v4l2-nvenc, which dGPU workstations lack even though the
# nvv4l2h264enc ELEMENT is installed — so probe the device node, not the
# element. FORCE_SW_ENC=1 forces x264enc; BITRATE is in kbit/s for both.
BITRATE=${BITRATE:-8000}
if [ -e /dev/v4l2-nvenc ] && gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1 \
   && [ -z "${FORCE_SW_ENC:-}" ]; then
  # Hardware: NVMM in, bitrate in bits/s.
  H264ENC="nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc idrinterval=30 bitrate=$(( BITRATE * 1000 ))"
  echo "[run.sh] encoder: nvv4l2h264enc (NVENC) @ ${BITRATE}kbit/s" >&2
else
  # Software: needs SYSTEM memory, so nvvideoconvert pulls it off NVMM first;
  # bitrate is already kbit/s. key-int-max matches the hardware idrinterval so
  # a client can join an rtsp stream mid-way either way.
  H264ENC="nvvideoconvert ! video/x-raw,format=I420 ! videoconvert ! x264enc speed-preset=veryfast tune=zerolatency bitrate=${BITRATE} key-int-max=30"
  echo "[run.sh] encoder: x264enc (software) @ ${BITRATE}kbit/s — no /dev/v4l2-nvenc" >&2
fi

# ── Metadata sink config ──────────────────────────────────────────
# Read by the nvcustomprocess user libraries in build/ (the element
# has no config-file property). Override any of these at run time:
#   MQTT_HOST=10.0.0.9 ./run.sh      JSON_OUT=/tmp/d.json ./run.sh
export JSON_OUT="${JSON_OUT:-/home/areefjeevan/.pipegen/abdulwahab141978/5682b226-3a09-49f1-9998-ea7bcc204082/output/metadata.json}"
export JSON_MIN_CONF="${JSON_MIN_CONF:-0.0}"
export JSON_MASK_OUTLINE="${JSON_MASK_OUTLINE:-0}"
mkdir -p "$(dirname "${JSON_OUT}")" 2>/dev/null || true
echo "[run.sh] JSON -> ${JSON_OUT}" >&2

# ── Recording output dirs ─────────────────────────────────────────
# A missing directory here is a hard failure at pipeline start (the
# sink element does not create one for you); mkdir -p up front so a
# fresh project layout never needs a manual mkdir.
mkdir -p "$(dirname "/home/areefjeevan/.pipegen/abdulwahab141978/5682b226-3a09-49f1-9998-ea7bcc204082/output/output.mp4")" 2>/dev/null || true

if [ -f "${DIR}/parsers/Makefile" ] && command -v make >/dev/null 2>&1; then
  make -C "${DIR}/parsers" || { echo "parser/.so build failed" >&2; exit 1; }
fi

gst-launch-1.0 -e   nvstreammux name=mux batch-size=1 width=1920 height=1080 batched-push-timeout=33000 live-source=1   nvurisrcbin uri="file:///home/areefjeevan/Downloads/pipegen/bulk-to-loose-process-in-large-warehouse-distribution-centre_ciCvOEuJ.mp4" ! identity sync=true ! mux.sink_0   mux. ! queue max-size-buffers=4 ! nvvideoconvert     ! nvcustomprocess custom-lib-path="${DIR}/build/libclahe_preprocess_process.so" process-func-name=NvDsCustomPostProcess     ! queue max-size-buffers=4     ! tee name=t   t. ! queue max-size-buffers=4     ! nvinfer_backport config-file-path="${DIR}/configs/config_person_det.txt"     ! queue max-size-buffers=4 ! meta.sink_0   t. ! queue max-size-buffers=4     ! nvinfer_backport config-file-path="${DIR}/configs/config_ppe_det.txt"     ! queue max-size-buffers=4 ! meta.sink_1   nvdsmetamux name=meta config-file="${DIR}/configs/config_metamux.txt"     ! queue max-size-buffers=4     ! nvtracker tracker-width=640 tracker-height=384         ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so         ll-config-file="${DIR}/configs/config_tracker.yml"         display-tracking-id=1     ! queue max-size-buffers=4     ! nvcustomprocess custom-lib-path="${DIR}/build/librender_safety_process.so" process-func-name=NvDsCustomPostProcess     ! queue max-size-buffers=4     ! nvcustomprocess custom-lib-path="${DIR}/build/libmeta_json.so" process-func-name=MetaJson     ! queue max-size-buffers=4 ! nvcustomprocess custom-lib-path=${DIR}/build/libvms_detect.so process-func-name=VmsDetect ! queue max-size-buffers=4     ! nvvideoconvert     ! nvmultistreamtiler rows=1 columns=1 width=1920 height=1080     ! nvvideoconvert ! queue max-size-buffers=4 ! nvvideoconvert ! "video/x-raw,format=I420" ! jpegenc quality=${VMS_JPEG_QUALITY} ! multipartmux boundary=vmsframe ! tcpserversink host=0.0.0.0 port=${VMS_MJPEG_PORT} sync=false recover-policy=latest
