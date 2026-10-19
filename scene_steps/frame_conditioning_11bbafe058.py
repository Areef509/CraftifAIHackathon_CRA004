import cv2
import numpy as np

# Reference for the scene steps that run before the model (catalogue §3 Scene and
# §4 Camera mount), written to be transcribed into a per-frame C++ worker.
#
# The frame is NV12 and is changed IN PLACE:
#   y   H x W uint8, limited-range luma (16..235)
#   uv  H/2 x W/2 x 2 uint8, interleaved U,V
# In C++ both planes wrap the mapped surface directly, e.g.
#   cv::Mat Y(H, W, CV_8UC1, Y_host, Y_pitch);  cv::Mat UV(H / 2, W / 2, CV_8UC2, UV_host, UV_pitch);
# so luma-only steps need no BGR round trip.
#
# Every step is stateless: whether and how hard it works is decided from this
# frame alone (rule R), so nothing is kept between frames.
#   s = clamp((off - m) / (off - full), 0, 1)     m measured on this frame
#   out = Y * (1 - s) + fix(Y) * s                 nothing is touched when s == 0
# off > full ramps up as m falls (dark frames); off < full ramps up as m rises.
#
# params = {"ops": [{"op": "clahe_luma", "clip_limit": 2.0, "tiles": [8, 8], "ramp": {...}?}, ...],
#           "ramp": {...}?}   a top-level ramp applies to every op that has none of its own.
# A ramp belongs to its op; an op with no ramp runs on every frame. A ramp may also be a list (the
# strongest one wins), and "only_if" gates a ramp on a second measure. chroma_offset is read from UV.
# Every ramp is measured on the frame as it arrives, before any op changes it; the step does no work
# at all only when every op's strength is 0. Keys an op does not take (a label, source ids) are ignored.

STEP = 4                    # measures sample every 4th row and column
DARK_BELOW = 0.12           # dark_fraction: share of luma under this
CLIPPED_AT = 0.98           # clipped_fraction: share of luma at or above this


def luma_sample(y):
    return (y[::STEP, ::STEP].astype(np.float32) - 16.0) / 219.0


def measure(y, uv, name):
    if name == "chroma_offset":
        # How far the frame's average colour sits from grey, 0 = neutral.
        u, v = uv[::STEP, ::STEP].reshape(-1, 2).astype(np.float32).mean(axis=0) - 128.0
        return float(np.hypot(u, v) / 128.0)
    luma = luma_sample(y)
    if name == "p50":
        return float(np.median(luma))
    if name == "dark_fraction":
        return float(np.mean(luma < DARK_BELOW))
    if name == "clipped_fraction":
        return float(np.mean(luma >= CLIPPED_AT))
    if name == "rms_contrast":
        return float(luma.std())
    raise ValueError(f"unknown measure {name!r}")


def strength(y, uv, ramp):
    if not ramp:
        return 1.0
    best = 0.0
    for rule in ramp if isinstance(ramp, list) else [ramp]:
        gate = rule.get("only_if")
        if gate and measure(y, uv, gate["measure"]) < gate["min"]:
            continue
        m = measure(y, uv, rule["measure"])
        s = (rule["off"] - m) / (rule["off"] - rule["full"])
        best = max(best, min(max(s, 0.0), 1.0))
    return best


def clahe_luma(y, clip_limit=2.0, tiles=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tiles[0]), int(tiles[1])))
    return clahe.apply(np.ascontiguousarray(y))


def contrast_stretch_luma(y, low=1.0, high=99.0):
    # Maps the 1st..99th percentile of this frame onto the full limited range.
    lo, hi = np.percentile(y[::STEP, ::STEP], (low, high))
    if hi - lo < 1.0:
        return y.copy()
    lut = np.clip((np.arange(256, dtype=np.float32) - lo) * (219.0 / (hi - lo)) + 16.0, 0, 255)
    return cv2.LUT(y, np.round(lut).astype(np.uint8))


def gamma_luma(y, gamma=0.7):
    # gamma < 1 lifts the dark tones; the lookup keeps the limited range.
    x = np.clip((np.arange(256, dtype=np.float32) - 16.0) / 219.0, 0.0, 1.0)
    return cv2.LUT(y, np.round(16.0 + 219.0 * x ** float(gamma)).astype(np.uint8))


def gaussian3_luma(y):
    return cv2.GaussianBlur(y, (3, 3), 0)


LUMA_OPS = {"clahe_luma": clahe_luma, "contrast_stretch_luma": contrast_stretch_luma, "gamma_luma": gamma_luma,
            "gaussian3_luma": gaussian3_luma}
OPTIONS = {"clahe_luma": ("clip_limit", "tiles"), "contrast_stretch_luma": ("low", "high"), "gamma_luma": ("gamma",),
           "gaussian3_luma": ()}
OPS = (*LUMA_OPS, "grey_world_chroma", "flip180")


def grey_world_chroma(uv, s):
    # Grey world on NV12: move the average U and V back to 128 (neutral), luma untouched.
    shift = s * (uv[::STEP, ::STEP].reshape(-1, 2).astype(np.float32).mean(axis=0) - 128.0)
    uv[...] = np.clip(np.round(uv.astype(np.float32) - shift), 0, 255).astype(np.uint8)


def flip180(y, uv):
    y[...] = cv2.flip(y, -1)
    uv[...] = cv2.flip(uv, -1)


def process(y, uv, params):
    # Measures every op's strength on the incoming frame, then runs the ops in the given order.
    # Returns each op's strength (0 = skipped).
    ops = params.get("ops") or []
    for op in ops:
        if op.get("op") not in OPS:
            raise ValueError(f"unknown op {op.get('op')!r}")
    default = params.get("ramp")
    strengths = [strength(y, uv, op.get("ramp", default)) for op in ops]
    applied = {}
    for op, s in zip(ops, strengths):
        name = op["op"]
        applied[name] = s
        if s <= 0.0:
            continue
        if name == "flip180":
            flip180(y, uv)
        elif name == "grey_world_chroma":
            grey_world_chroma(uv, s)
        else:
            options = {k: op[k] for k in OPTIONS[name] if k in op}
            fixed = LUMA_OPS[name](y, **options)
            y[...] = fixed if s >= 1.0 else cv2.addWeighted(np.ascontiguousarray(y), 1.0 - s, fixed, s, 0.0)
    return applied
