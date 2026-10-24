"""
Live per-frame measurements for the alert engine (engine2.Engine2) and driver ID (identity.py).

Same measurements, same models and same crops as the recorded-video pipeline (extract.py,
hands_extract.py, belt.py, faces_extract.py), so the tuned thresholds carry over:

  every frame      face box + 5 key points (SCRFD, CPU), face mesh + eye blendshapes on a
                   close-up crop (MediaPipe, CPU), sunglasses check
  every 0.2 s      hand landmarks around the face (MediaPipe, CPU)
  every frame*     phone / bottle / cup detection - on the AI HAT (Hailo-8L) when present,
                   otherwise YOLOX-tiny on CPU every 0.5 s
  every 0.5 s      seat-belt score, face embedding for identity (ArcFace, CPU)
"""
import collections
import os
import time

import cv2
import numpy as np

import belt
import dms
import frs

KEEP_EYES = dms.EYE_L + dms.EYE_R


class Perception:
    def __init__(self, models_dir, det_size=320, hef=None, use_hailo=True, log=print):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self.mp = mp
        p = lambda n: os.path.join(models_dir, n)
        self.face_det = frs.FaceDetector(p("det_500m.onnx"), size=det_size, score=0.3)
        self.embedder = frs.FaceEmbedder(p("w600k_mbf.onnx"))
        self.mesh = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=p("face_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True,
            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=p("hand_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.3))
        self.objects, self.on_hailo = None, False
        if use_hailo and hef and os.path.exists(hef):
            try:
                import hailo_objects
                self.objects = hailo_objects.HailoObjects(hef)
                self.on_hailo = True
                log(f"phone/bottle detector: AI HAT (Hailo) using {os.path.basename(hef)}")
            except Exception as e:                      # no HAT / driver problem: fall back to CPU
                log(f"AI HAT not usable ({e}); using CPU detector")
        if self.objects is None:
            import phone
            phone.MODEL_DIR = models_dir
            self.objects = phone.YoloX("yolox_tiny")
            log("phone/bottle detector: CPU (YOLOX-tiny) - slower, lower rate")
        self.period = dict(hands=0.2, belt=0.5, embed=0.5, objects=0.0 if self.on_hailo else 0.5)
        self.last = collections.defaultdict(lambda: -1e9)
        self.hold = dict(hmouth=np.nan, hear=np.nan)
        self.eye_hist = collections.deque(maxlen=20)
        self.timing = collections.defaultdict(lambda: collections.deque(maxlen=50))

    def _due(self, name, t):
        if t - self.last[name] >= self.period[name]:
            self.last[name] = t
            return True
        return False

    def _timed(self, name, fn, *a):
        t0 = time.perf_counter()
        r = fn(*a)
        self.timing[name].append(time.perf_counter() - t0)
        return r

    def timings_ms(self):
        return {k: round(1000 * float(np.mean(v)), 1) for k, v in self.timing.items() if v}

    def reset_driver(self):
        self.eye_hist.clear()

    # ------------------------------------------------------------------
    def _mesh(self, frame, b):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        side = int(max(b[2] - b[0], b[3] - b[1]) * 2.0)
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        H, W = frame.shape[:2]
        crop = np.zeros((side, side, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H)
        if sx1 <= sx0 or sy1 <= sy0:
            return None, None
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
        crop = cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2RGB)
        res = self.mesh.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=crop))
        if not res.face_landmarks:
            return None, None
        lm = np.array([[q.x, q.y] for q in res.face_landmarks[0][:468]]) * side + [x0, y0]
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
        return lm, bs

    def _hands(self, frame, b, kps):
        fw = b[2] - b[0]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        side = int(fw * 3.2)
        x0, y0 = int(cx - side / 2), int(cy - side * 0.4)
        H, W = frame.shape[:2]
        crop = np.zeros((side, side, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H)
        if sx1 <= sx0 or sy1 <= sy0:
            return np.nan, np.nan
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
        img = cv2.cvtColor(cv2.resize(crop, (384, 384)), cv2.COLOR_BGR2RGB)
        r = self.hands.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=img))
        mouth = (kps[3] + kps[4]) / 2
        dm = de = 9.0
        for h in r.hand_landmarks:
            p = np.array([[q.x, q.y] for q in h]) * side + [x0, y0]
            dm = min(dm, np.min(np.linalg.norm(p - mouth, axis=1)) / fw)
            side_dist = np.min(np.minimum(abs(p[:, 0] - b[0]), abs(p[:, 0] - b[2]))) / fw
            in_band = np.any((p[:, 1] > b[1]) & (p[:, 1] < b[3]))
            de = min(de, side_dist if in_band else 9.0)
        return dm, de

    # ------------------------------------------------------------------
    def __call__(self, frame, t):
        """frame: BGR image. Returns (features for Engine2.update, extras for identity/overlay)."""
        faces = self._timed("face", self.face_det, frame)
        ex = dict(lm=None, embedding=None, embed_good=False, objects=[])
        nan4 = np.full(4, np.nan)
        f = dict(face=bool(faces), mesh=False, yawp=np.nan, eyespan=np.nan, blink=np.nan, down=np.nan,
                 eyes_hidden=False, hmouth=np.nan, hear=np.nan, phone_run=False,
                 phone_box=np.full(5, np.nan), face_box=nan4, fw=np.nan, frame_w=frame.shape[1],
                 belt=np.nan)
        # objects run even without a face (cheap on the HAT)
        if self._due("objects", t):
            dets = self._timed("objects", self.objects, frame, (phone_cls(),) + (39, 41)) \
                if not self.on_hailo else self._timed("objects", self.objects, frame)
            ex["objects"] = dets
            ph = [d for d in dets if d[5] == phone_cls()]
            f["phone_run"] = True
            if ph:
                f["phone_box"] = np.array(ph[0][:5], float)
        if not faces:
            self.eye_hist.clear()
            return f, ex
        fa = faces[0]
        b, kp = fa["box"], fa["kps"]
        le, re, no = kp[0], kp[1], kp[2]
        eyew = re[0] - le[0]
        f.update(face_box=b, fw=b[2] - b[0], yawp=((no[0] - le[0]) / (eyew + 1e-6) - 0.5) * 2,
                 eyespan=eyew / (b[2] - b[0] + 1e-6))
        lm, bs = self._timed("mesh", self._mesh, frame, b)
        if lm is not None:
            ex["lm"] = lm
            f.update(mesh=True, blink=(bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2,
                     down=(bs["eyeLookDownLeft"] + bs["eyeLookDownRight"]) / 2)
            vis, _ = dms.eyes_visible(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), lm)
            self.eye_hist.append(vis)
            f["eyes_hidden"] = sum(self.eye_hist) * 2 <= len(self.eye_hist)
        if self._due("hands", t):
            self.hold["hmouth"], self.hold["hear"] = self._timed("hands", self._hands, frame, b, kp)
        f["hmouth"], f["hear"] = self.hold["hmouth"], self.hold["hear"]
        if self._due("belt", t):
            f["belt"] = self._timed("belt", lambda: belt.belt_score(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), b)[0])
        if self._due("embed", t):
            ex["embedding"] = self._timed("embed", self.embedder, frame, kp)
            ex["embed_good"] = frs.face_quality(fa, frame) is None
        return f, ex

    def close(self):
        for m in (self.mesh, self.hands):
            try:
                m.close()
            except Exception:
                pass
        if self.on_hailo:
            self.objects.close()


def phone_cls():
    return 67          # COCO "cell phone"
