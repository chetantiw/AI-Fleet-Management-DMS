#!/usr/bin/env python3
"""
DMS stage 1: capture -> face detection -> landmarks -> drowsiness / distraction events.

Hardware: Raspberry Pi 5, AI HAT+ 13 TOPS (Hailo-8L), HQ Camera (IMX477) on CSI.
Models:   scrfd_2.5g.hef (face), face_landmarks_lite.hef (468-point mesh).

  python3 dms.py                 # live preview window
  python3 dms.py --headless      # no window, events to console + events.jsonl
  python3 dms.py --video x.mp4   # run on a recorded clip instead of the camera
  python3 dms.py --cam 0         # USB webcam (PC, or Pi before the CSI camera arrives)

Without the AI HAT it falls back to a CPU backend (pip install mediapipe) that
produces the same 468-point mesh, so everything downstream is identical.
"""
import argparse
import collections
import json
import os
import time

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
CFG = dict(
    det_score=0.5,          # face detection confidence
    calib_seconds=5.0,      # first estimate of this driver's open-eye EAR and neutral pose
    baseline_window_s=120.0,  # then both keep adapting over this rolling window
    ear_ratio=0.60,         # eye "closed" when EAR < ratio * this driver's open-eye EAR
    closure_max_yaw=45.0,   # beyond this head turn, eye state is not trusted
    max_turn=0.6,           # face_turn() above this (about 45 deg to the camera): eyes/mouth not measured
    min_eye_px=30,          # outer eye corners closer than this: face too small to measure
    sunglass_contrast=0.38, # eye-band contrast vs cheek (frames, reflections)
    sunglass_dark=0.70,     # eye band darker than 70% of cheek brightness = sunglasses
    ear_fallback=0.20,      # used until calibration finishes
    eyes_closed_s=1.5,      # continuous closure -> DROWSY (microsleep)
    perclos_window_s=60.0,
    perclos_limit=0.20,     # >20% of the last minute with eyes closed -> FATIGUE
    mar_yawn=0.60,          # mouth aspect ratio for a yawn
    yawn_min_s=1.0,
    yawn_count=3,           # yawns ...
    yawn_window_s=300.0,    # ... within this window -> FATIGUE
    yaw_limit=30.0,         # degrees away from calibrated neutral
    pitch_limit=20.0,
    look_away_s=2.0,        # -> DISTRACTED
    no_face_s=3.0,          # -> DRIVER_ABSENT
    cooldown_s=5.0,         # min gap between repeats of the same event
    fatigue_cooldown_s=120.0,  # fatigue is a state, not a moment: remind at most every 2 min
)

# MediaPipe face-mesh indices
EYE_R = [33, 160, 158, 133, 153, 144]     # subject's right eye (image left)
EYE_L = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14]                 # corners, inner upper lip, inner lower lip
POSE_IDX = [1, 152, 33, 263, 61, 291]     # nose, chin, eye corners, mouth corners
# Generic head model (mm-ish), x right, y down, z away from camera
POSE_3D = np.array([
    [0, 0, 0], [0, 330, 65],
    [-225, -170, 135], [225, -170, 135],
    [-150, 150, 125], [150, 150, 125]], dtype=np.float64)


# ----------------------------------------------------------------------------
# Hailo inference (one shared device, several models, scheduler switches them)
# ----------------------------------------------------------------------------
class HailoDevice:
    def __init__(self):
        from hailo_platform import VDevice, HailoSchedulingAlgorithm
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)

    def load(self, hef_path):
        return HailoModel(self.vdevice, hef_path)


class HailoModel:
    def __init__(self, vdevice, hef_path):
        from hailo_platform import FormatType
        self.im = vdevice.create_infer_model(hef_path)
        self.im.set_batch_size(1)
        for name in self.im.output_names:
            self.im.output(name).set_format_type(FormatType.FLOAT32)
        self.cm = self.im.configure()
        self.h, self.w = self.im.input().shape[:2]
        self.out_shapes = {n: tuple(self.im.output(n).shape) for n in self.im.output_names}

    def run(self, rgb_u8):
        """rgb_u8: HxWx3 uint8 already at model input size. Returns {name: float32 array}."""
        bufs = {n: np.empty(s, dtype=np.float32) for n, s in self.out_shapes.items()}
        b = self.cm.create_bindings(output_buffers=bufs)
        b.input().set_buffer(np.ascontiguousarray(rgb_u8))
        self.cm.run([b], 1000)
        return {n: b.output(n).get_buffer() for n in self.out_shapes}


class HailoBackend:
    """Face box + 468 landmarks using the two HEF models on the Hailo-8L."""
    def __init__(self, model_dir):
        dev = HailoDevice()
        self.det = dev.load(os.path.join(model_dir, "scrfd_2.5g.hef"))
        self.lmk = dev.load(os.path.join(model_dir, "face_landmarks_lite.hef"))
        print("detector outputs:", self.det.out_shapes)
        print("landmark outputs:", self.lmk.out_shapes)

    def __call__(self, rgb):
        box = detect_driver(self.det, rgb)
        return box, (face_landmarks(self.lmk, rgb, box) if box is not None else None)


class CpuBackend:
    """Same 468-point mesh on plain CPU (MediaPipe). For PC / Pi without the AI HAT.
    pip install mediapipe ; the model file downloads on first use."""
    URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
           "face_landmarker/float16/1/face_landmarker.task")

    def __init__(self, model_dir):
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        path = os.path.join(model_dir, "face_landmarker.task")
        if not os.path.exists(path):
            os.makedirs(model_dir, exist_ok=True)
            print("Downloading face_landmarker.task ...")
            urllib.request.urlretrieve(self.URL, path)
        self.mp, self.ts, self.prev, self.fd = mp, 0, None, None
        self.lmk = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))

    def __call__(self, rgb):
        """Find the face (SCRFD, or last frame's mesh), then fit the mesh on a square
        close-up crop. Fitting on the whole frame misses small faces in a wide view."""
        self.ts += 33
        h, w = rgb.shape[:2]
        box = self.prev
        if box is None:
            if self.fd is None:
                import frs
                self.fd = frs.FaceDetector(frs.ensure_models()[0])
            faces = self.fd(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not faces:
                return None, None
            box = faces[0]["box"]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        side = int(max(box[2] - box[0], box[3] - box[1]) * 2.0)
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        crop = np.zeros((side, side, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
        if sx1 <= sx0 or sy1 <= sy0:
            self.prev = None
            return None, None
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
        crop = cv2.resize(crop, (256, 256))
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
        res = self.lmk.detect_for_video(img, self.ts)
        if not res.face_landmarks:
            self.prev = None
            return None, None
        lm = np.array([[p.x, p.y] for p in res.face_landmarks[0][:468]]) * side + [x0, y0]
        b = np.concatenate([lm.min(0), lm.max(0)])
        self.prev = b
        return b, lm

    def close(self):
        self.lmk.close()


# ----------------------------------------------------------------------------
# SCRFD face detector post-processing
# ----------------------------------------------------------------------------
def letterbox(img, size):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    out = np.zeros((size, size, 3), np.uint8)
    out[:nh, :nw] = cv2.resize(img, (nw, nh))
    return out, s


def scrfd_decode(outputs, in_size, score_thr, nms_thr=0.4):
    """outputs: {name: array (H,W,C)}. Layers are matched by shape, not by name:
    H gives the stride; C = 2 scores, 8 box, 20 keypoints (2 anchors per cell)."""
    by_stride = collections.defaultdict(dict)
    for arr in outputs.values():
        a = np.asarray(arr)
        a = a.reshape(a.shape[-3:])
        kind = {2: "score", 8: "box", 20: "kps"}.get(a.shape[2])
        if kind:
            by_stride[in_size // a.shape[0]][kind] = a
    boxes, scores, kpss = [], [], []
    for stride, o in by_stride.items():
        if len(o) < 3:
            continue
        n = o["score"].shape[0]
        sc = o["score"].reshape(-1)
        if sc.min() < 0 or sc.max() > 1:          # raw logits -> probabilities
            sc = 1 / (1 + np.exp(-sc))
        keep = sc > score_thr
        if not keep.any():
            continue
        ys, xs = np.mgrid[:n, :n]
        ctr = np.stack([xs, ys], -1).reshape(-1, 2).repeat(2, axis=0)[keep] * stride
        d = o["box"].reshape(-1, 4)[keep] * stride
        k = o["kps"].reshape(-1, 5, 2)[keep] * stride
        boxes.append(np.concatenate([ctr - d[:, :2], ctr + d[:, 2:]], 1))
        kpss.append(ctr[:, None, :] + k)
        scores.append(sc[keep])
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 5, 2))
    boxes, scores, kpss = np.concatenate(boxes), np.concatenate(scores), np.concatenate(kpss)
    xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), score_thr, nms_thr)
    idx = np.asarray(idx).reshape(-1)
    return boxes[idx], scores[idx], kpss[idx]


def detect_driver(model, frame_rgb):
    """Returns the largest face box (x1,y1,x2,y2) in frame pixels, or None."""
    inp, s = letterbox(frame_rgb, model.w)
    boxes, _, _ = scrfd_decode(model.run(inp), model.w, CFG["det_score"])
    if len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(np.argmax(areas))] / s


# ----------------------------------------------------------------------------
# Landmarks + geometry
# ----------------------------------------------------------------------------
def face_landmarks(model, frame_rgb, box, margin=0.25):
    """Returns 468x2 landmark array in frame pixels, or None."""
    H, W = frame_rgb.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * margin)
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    side = int(side)
    if side < 20:
        return None
    crop = np.zeros((side, side, 3), np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, W), min(y0 + side, H)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame_rgb[sy0:sy1, sx0:sx1]
    out = model.run(cv2.resize(crop, (model.w, model.h)))
    lm = next((np.asarray(v).reshape(-1) for v in out.values() if np.asarray(v).size == 1404), None)
    if lm is None:
        raise RuntimeError(f"No 1404-value landmark output; got {model.out_shapes}")
    lm = lm.reshape(468, 3)[:, :2]
    if lm.max() > 1.5:                      # model-pixel units -> 0..1
        lm = lm / model.w
    return lm * side + [x0, y0]


def ear(lm, idx):
    p = lm[idx]
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (
        2 * np.linalg.norm(p[0] - p[3]) + 1e-6)


def mar(lm):
    c0, c1, up, lo = lm[MOUTH]
    return np.linalg.norm(up - lo) / (np.linalg.norm(c0 - c1) + 1e-6)


def face_turn(lm):
    """0 = facing the camera, 1 = full profile. Uses where the nose tip sits between
    the outer eye corners, which stays meaningful when the mesh fit degrades."""
    a, b, n = lm[33], lm[263], lm[1]
    t = np.dot(n - a, b - a) / (np.dot(b - a, b - a) + 1e-6)
    return float(min(abs(t - 0.5) * 2, 1.0))


def eyes_visible(gray, lm):
    """False when the eyes are hidden (dark sunglasses): the eye band is darker than the cheeks
    below it, or dark-ish with the high contrast of frames / lens reflections. Eyes lit from
    above at night are bright and are not mistaken for sunglasses.
    Validated on 58 labelled bus crops: 15/16 sunglasses caught, 1/42 plain eyes flagged.
    Returns (visible, darkness ratio)."""
    a, b = lm[33], lm[263]
    c, w = (a + b) / 2, np.linalg.norm(b - a)
    H, W = gray.shape
    x0, x1 = int(max(c[0] - w * 0.9, 0)), int(min(c[0] + w * 0.9, W))
    eye = gray[int(max(c[1] - w * 0.35, 0)):int(min(c[1] + w * 0.35, H)), x0:x1]
    cheek = gray[int(min(c[1] + w * 0.45, H)):int(min(c[1] + w * 0.8, H)), x0:x1]
    if eye.size == 0 or cheek.size == 0:
        return True, np.nan
    ref = cheek.mean() + 1e-6
    contrast, dark = eye.std() / ref, eye.mean() / ref
    hidden = dark < CFG["sunglass_dark"] or (contrast > CFG["sunglass_contrast"] and dark < 0.9)
    return bool(not hidden), float(dark)


def measurable(lm):
    return face_turn(lm) <= CFG["max_turn"] and np.linalg.norm(lm[263] - lm[33]) >= CFG["min_eye_px"]


def head_pose(lm, frame_w, frame_h):
    """Returns (pitch, yaw, roll) in degrees, or None."""
    f = float(frame_w)                      # focal length guess; fine for relative angles
    K = np.array([[f, 0, frame_w / 2], [0, f, frame_h / 2], [0, 0, 1]], dtype=np.float64)
    pts = lm[POSE_IDX].astype(np.float64)
    # Start from "face upright, in front of the camera". Without this the solver
    # sometimes lands on the mirror solution (head behind the camera, roll 180).
    z = f * 450.0 / (np.linalg.norm(pts[3] - pts[2]) + 1e-6)
    tvec0 = np.array([[(pts[0, 0] - frame_w / 2) * z / f], [(pts[0, 1] - frame_h / 2) * z / f], [z]])
    ok, rvec, _ = cv2.solvePnP(POSE_3D, pts, K, None, np.zeros((3, 1)), tvec0,
                               useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = cv2.RQDecomp3x3(R)[0]
    return pitch, yaw, roll


# ----------------------------------------------------------------------------
# Temporal engine: per-frame signals -> events
# ----------------------------------------------------------------------------
class DmsEngine:
    def __init__(self, cfg=CFG):
        self.c = cfg
        self.t0 = None
        self.hist = collections.deque()  # (t, ear, pitch, yaw) at ~5 Hz, rolling baseline
        self.last_hist = self.last_rebase = -1e9
        self.ear_thr = cfg["ear_fallback"]
        self.neutral = (0.0, 0.0)
        self.calibrated = False
        self.closed_since = self.away_since = self.yawn_since = None
        self.noface_since = None
        self.yawn_counted = False
        self.eye_hist = collections.deque()   # (t, closed)
        self.yawns = collections.deque()
        self.last_fired = {}
        self.state = {}

    def _fire(self, name, t, events):
        gap = self.c["fatigue_cooldown_s"] if name.startswith("FATIGUE") else self.c["cooldown_s"]
        if t - self.last_fired.get(name, -1e9) >= gap:
            self.last_fired[name] = t
            events.append(name)

    def _rebase(self, t):
        """Open-eye EAR = 90th percentile, neutral pose = median, over the rolling
        window. The driver looks at the road with eyes open most of the time, so
        this self-corrects if the first seconds were unrepresentative."""
        a = np.array(self.hist)
        self.ear_thr = float(np.percentile(a[:, 1], 90) * self.c["ear_ratio"])
        self.neutral = (float(np.median(a[:, 2])), float(np.median(a[:, 3])))
        self.last_rebase = t

    def update(self, t, ear_v=None, mar_v=None, pose=None, measurable=True, eyes_ok=True):
        """Call once per frame. Pass ear_v=None when no face was found, and
        measurable=False when the face is too turned or too small to read."""
        c, events = self.c, []
        if ear_v is None:
            self.noface_since = self.noface_since or t
            self.closed_since = self.away_since = self.yawn_since = None
            if t - self.noface_since >= c["no_face_s"]:
                self._fire("DRIVER_ABSENT", t, events)
            self.state = dict(face=False)
            return events
        self.noface_since = None
        if not measurable:
            # In profile the eye and mouth ratios are meaningless, so nothing is
            # scored from them. A head turned this far is itself "looking away".
            self.closed_since = self.yawn_since = None
            self.yawn_counted = False
            if self.calibrated:
                self.away_since = self.away_since if self.away_since is not None else t
                if t - self.away_since >= c["look_away_s"]:
                    self._fire("DISTRACTED_LOOK_AWAY", t, events)
            self.state = dict(face=True, measurable=False)
            return events
        pitch, yaw = (pose[0], pose[1]) if pose else self.neutral
        if eyes_ok and t - self.last_hist >= 0.2:
            self.last_hist = t
            self.hist.append((t, ear_v, pitch, yaw))
            while t - self.hist[0][0] > c["baseline_window_s"]:
                self.hist.popleft()
        if self.calibrated and t - self.last_rebase >= 1.0:
            self._rebase(t)

        if not self.calibrated:
            self.t0 = self.t0 if self.t0 is not None else t
            if t - self.t0 >= c["calib_seconds"] and len(self.hist) >= 15:
                self._rebase(t)
                self.calibrated = True
                events.append("CALIBRATED")
            self.state = dict(face=True, calibrating=True, ear=ear_v)
            return events

        dp, dy = pitch - self.neutral[0], yaw - self.neutral[1]
        closed = eyes_ok and ear_v < self.ear_thr
        away = abs(dy) > c["yaw_limit"] or abs(dp) > c["pitch_limit"]

        # Eye closure. Not trusted at a large head turn: EAR is unreliable in profile.
        closed = closed and abs(dy) <= c["closure_max_yaw"]
        if closed:
            self.closed_since = self.closed_since or t
            if t - self.closed_since >= c["eyes_closed_s"]:
                self._fire("DROWSY_EYES_CLOSED", t, events)
        else:
            self.closed_since = None

        # PERCLOS
        self.eye_hist.append((t, closed))
        while self.eye_hist and t - self.eye_hist[0][0] > c["perclos_window_s"]:
            self.eye_hist.popleft()
        span = t - self.eye_hist[0][0]
        perclos = sum(x for _, x in self.eye_hist) / len(self.eye_hist)
        if span >= c["perclos_window_s"] * 0.5 and perclos > c["perclos_limit"]:
            self._fire("FATIGUE_PERCLOS", t, events)

        # Yawns
        if mar_v is not None and mar_v > c["mar_yawn"]:
            self.yawn_since = self.yawn_since or t
            if not self.yawn_counted and t - self.yawn_since >= c["yawn_min_s"]:
                self.yawn_counted = True
                self.yawns.append(t)
                events.append("YAWN")
        else:
            self.yawn_since, self.yawn_counted = None, False
        while self.yawns and t - self.yawns[0] > c["yawn_window_s"]:
            self.yawns.popleft()
        if len(self.yawns) >= c["yawn_count"]:
            self._fire("FATIGUE_YAWNING", t, events)

        # Looking away. Skipped while the mouth is wide open: a yawn drags the chin
        # and mouth corners and makes the head-pose estimate swing.
        if away and not (mar_v is not None and mar_v > 0.35):
            self.away_since = self.away_since or t
            if t - self.away_since >= c["look_away_s"]:
                self._fire("DISTRACTED_LOOK_AWAY", t, events)
        else:
            self.away_since = None

        self.state = dict(face=True, ear=ear_v, ear_thr=self.ear_thr, mar=mar_v,
                          dpitch=dp, dyaw=dy, perclos=perclos, closed=closed, away=away,
                          yawns=len(self.yawns))
        return events


# ----------------------------------------------------------------------------
# Frame sources
# ----------------------------------------------------------------------------
def camera_frames(width, height, fps):
    from picamera2 import Picamera2
    cam = Picamera2()
    # Picamera2 "RGB888" arrives as BGR in numpy, which is what OpenCV wants.
    cam.configure(cam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        controls={"FrameRate": fps}))
    cam.start()
    try:
        while True:
            yield cam.capture_array()
    finally:
        cam.stop()


def video_frames(path, width=None, height=None):
    """path: video file, or an int for a USB webcam index."""
    cap = cv2.VideoCapture(path)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def draw(frame, box, lm, st, banner, fps):
    if box is not None:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if lm is not None:
        for i in []:  # no landmark dots: they hide the eyes the reviewer needs to see
            cv2.circle(frame, tuple(lm[i].astype(int)), 2, (0, 255, 255), -1)
    lines = [f"{fps:4.1f} fps"]
    if st.get("measurable") is False:
        lines.append("HEAD TURNED: eyes not measurable")
    elif st.get("calibrating"):
        lines.append("CALIBRATING: look at the road, eyes open")
    elif st.get("face"):
        lines += [f"EAR {st['ear']:.2f} (thr {st['ear_thr']:.2f})  MAR {st['mar']:.2f}",
                  f"yaw {st['dyaw']:+.0f}  pitch {st['dpitch']:+.0f}",
                  f"PERCLOS {st['perclos'] * 100:.0f}%  yawns {st['yawns']}"]
    else:
        lines.append("NO FACE")
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if banner:
        cv2.rectangle(frame, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        cv2.putText(frame, banner, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.expanduser("~/dms/models"))
    ap.add_argument("--video", help="video file instead of the camera")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="events")
    ap.add_argument("--save", help="write the annotated video to this .mp4")
    ap.add_argument("--save-from", type=float, default=0.0, help="seconds into the run")
    ap.add_argument("--save-to", type=float, default=1e9)
    ap.add_argument("--cam", type=int, help="USB webcam index instead of the CSI camera")
    ap.add_argument("--backend", choices=["auto", "hailo", "cpu"], default="auto")
    args = ap.parse_args()

    if args.backend == "auto":
        try:
            import hailo_platform  # noqa: F401
            args.backend = "hailo"
        except ImportError:
            args.backend = "cpu"
    print("backend:", args.backend)
    backend = (HailoBackend if args.backend == "hailo" else CpuBackend)(args.models)

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "events.jsonl"), "a")
    engine = DmsEngine()
    if args.video:
        frames = video_frames(args.video)
    elif args.cam is not None:
        frames = video_frames(args.cam, args.width, args.height)
    else:
        frames = camera_frames(args.width, args.height, args.fps)
    banner, banner_until, fps, last, n = "", 0.0, 0.0, time.monotonic(), 0
    writer, t_start = None, None

    for frame in frames:                              # BGR
        # For recorded clips use the clip's own timeline so timers stay correct.
        t = n / args.fps if args.video else time.monotonic()
        n += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box, lm = backend(rgb)
        if lm is not None:
            e = (ear(lm, EYE_L) + ear(lm, EYE_R)) / 2
            events = engine.update(t, e, mar(lm), head_pose(lm, frame.shape[1], frame.shape[0]),
                                   measurable(lm))
        else:
            events = engine.update(t)

        now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        for ev in events:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            rec = dict(time=stamp, event=ev,
                       **{k: round(float(v), 3) for k, v in engine.state.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool)})
            print(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if ev != "CALIBRATED":
                banner, banner_until = ev.replace("_", " "), t + 3
                if ev != "YAWN":
                    cv2.imwrite(os.path.join(args.out, f"{stamp}_{ev}.jpg"), frame)
        t_start = t if t_start is None else t_start
        saving = args.save and args.save_from <= t - t_start <= args.save_to
        if not args.headless or saving:
            draw(frame, box, lm, engine.state, banner if t < banner_until else "", fps)
        if saving:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if not args.headless:
            cv2.imshow("DMS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if writer is not None:
        writer.release()
    if hasattr(backend, "close"):
        backend.close()


if __name__ == "__main__":
    main()
