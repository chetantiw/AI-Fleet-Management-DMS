#!/usr/bin/env python3
"""
Phone-use detection for the DMS (CPU / ONNX Runtime; same model family runs on Hailo-8L).

Model: YOLOX (Apache-2.0 licence) trained on COCO, using its "cell phone" class.

  python3 phone.py --video clip.mp4            # preview
  python3 phone.py --cam 0
  python3 phone.py --video clip.mp4 --headless --save out.mp4

Events: PHONE_CALL (phone held at the head) and PHONE_IN_HAND (phone visible elsewhere),
raised when the phone is seen for most of a 2-second window.
"""
import argparse
import collections
import json
import os
import time
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{}.onnx"
COCO_PHONE, COCO_PERSON = 67, 0

CFG = dict(
    model="yolox_s",        # yolox_tiny is 2-3x faster but misses more small phones
    score=0.30,             # detection confidence for a phone
    every_n=3,              # run the detector on every Nth frame (10 fps at 30 fps input)
    window_s=2.0,           # phone must be seen ...
    min_fraction=0.5,       # ... in at least this share of detector runs in the window
    cooldown_s=10.0,
    min_median_score=0.45,  # and the typical confidence of those sightings must reach this
    min_size=0.36,          # phone box side / face width; smaller boxes are bottle caps, fingers, etc.
)


class YoloX:
    def __init__(self, name=None):
        import onnxruntime as ort
        name = name or CFG["model"]
        path = os.path.join(MODEL_DIR, name + ".onnx")
        if not os.path.exists(path):
            os.makedirs(MODEL_DIR, exist_ok=True)
            print(f"Downloading {name}.onnx ...")
            urllib.request.urlretrieve(MODEL_URL.format(name), path)
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.size = self.sess.get_inputs()[0].shape[2]
        grids, strides = [], []
        for s in (8, 16, 32):
            n = self.size // s
            ys, xs = np.mgrid[:n, :n]
            grids.append(np.stack([xs, ys], -1).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grid, self.stride = np.concatenate(grids), np.concatenate(strides)

    def __call__(self, bgr, classes=(COCO_PHONE,), score=None, roi=None):
        """Returns [(x1, y1, x2, y2, score, class_id)] in image pixels, best first.
        roi=(x1,y1,x2,y2) runs the detector on that crop only (bigger phone, fewer strays)."""
        score = CFG["score"] if score is None else score
        ox = oy = 0
        if roi is not None:
            x1, y1, x2, y2 = [int(v) for v in roi]
            x1, y1 = max(x1, 0), max(y1, 0)
            bgr, ox, oy = bgr[y1:y2, x1:x2], x1, y1
            if bgr.size == 0:
                return []
        h, w = bgr.shape[:2]
        r = min(self.size / w, self.size / h)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:int(round(h * r)), :int(round(w * r))] = cv2.resize(bgr, (int(round(w * r)), int(round(h * r))))
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)      # BGR, 0-255, as trained
        out = self.sess.run(None, {self.inp: blob})[0][0]
        xy = (out[:, :2] + self.grid) * self.stride
        wh = np.exp(out[:, 2:4]) * self.stride
        dets = []
        for c in classes:
            sc = out[:, 4] * out[:, 5 + c]
            keep = sc > score
            if not keep.any():
                continue
            b = np.concatenate([xy[keep] - wh[keep] / 2, wh[keep]], 1) / r
            idx = np.asarray(cv2.dnn.NMSBoxes(b.tolist(), sc[keep].tolist(), score, 0.45)).reshape(-1)
            for i in idx:
                x, y, bw, bh = b[i]
                dets.append((x + ox, y + oy, x + bw + ox, y + bh + oy, float(sc[keep][i]), c))
        return sorted(dets, key=lambda d: -d[4])


def phone_mode(phone, face_box):
    """'call' = phone beside the head, 'hand' = phone elsewhere in view,
    None = box too small relative to the face to be a phone. face_box may be None."""
    if face_box is None:
        return "hand"
    fw = max(face_box[2] - face_box[0], 1)
    if np.sqrt(max(phone[2] - phone[0], 0) * max(phone[3] - phone[1], 0)) / fw < CFG["min_size"]:
        return None
    dx = ((phone[0] + phone[2]) / 2 - (face_box[0] + face_box[2]) / 2) / fw
    dy = ((phone[1] + phone[3]) / 2 - (face_box[1] + face_box[3]) / 2) / fw
    return "call" if abs(dx) >= 0.45 and abs(dy) <= 0.6 else "hand"


class PhoneEngine:
    """Per-detector-run results -> PHONE_CALL / PHONE_IN_HAND events."""
    def __init__(self, cfg=CFG):
        self.c, self.hist, self.last = cfg, collections.deque(), {}
        self.state = dict(phone=False)

    def update(self, t, mode, score=1.0):
        """mode: None (no phone), 'call' or 'hand'. Call once per detector run."""
        c, events = self.c, []
        self.hist.append((t, mode, score))
        while t - self.hist[0][0] > c["window_s"]:
            self.hist.popleft()
        seen = [m for _, m, _ in self.hist if m]
        frac = len(seen) / len(self.hist)
        full = t - self.hist[0][0] >= c["window_s"] * 0.8
        active = (full and frac >= c["min_fraction"]
                  and np.median([s for _, m, s in self.hist if m]) >= c["min_median_score"])
        if active:
            name = "PHONE_CALL" if seen.count("call") >= len(seen) / 2 else "PHONE_IN_HAND"
            # one cooldown for both names so a call that drifts to "hand" does not re-alert
            if t - self.last.get("phone", -1e9) >= c["cooldown_s"]:
                self.last["phone"] = t
                events.append(name)
        self.state = dict(phone=bool(mode), mode=mode, fraction=frac, active=active)
        return events


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save")
    args = ap.parse_args()
    import frs
    det_path, _ = frs.ensure_models()
    face_det, yolo, eng = frs.FaceDetector(det_path), YoloX(), PhoneEngine()
    cap = cv2.VideoCapture(args.video if args.video else args.cam)
    n, writer, dets, face, banner, until = 0, None, [], None, "", 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / args.fps if args.video else time.monotonic()
        if n % CFG["every_n"] == 0:
            faces = face_det(frame)
            face = faces[0]["box"] if faces else None
            dets = yolo(frame)
            mode = phone_mode(dets[0], face) if dets else None
            if mode is None:
                dets = []
            for ev in eng.update(t, mode, dets[0][4] if dets else 0.0):
                print(json.dumps(dict(t=round(t, 1), event=ev)), flush=True)
                banner, until = ev.replace("_", " "), t + 3
        n += 1
        if args.headless and not args.save:
            continue
        for d in dets:
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 0, 255), 2)
            cv2.putText(frame, f"phone {d[4]:.2f}", (int(d[0]), int(d[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if face is not None:
            cv2.rectangle(frame, tuple(face[:2].astype(int)), tuple(face[2:].astype(int)), (0, 255, 0), 2)
        if t < until:
            cv2.rectangle(frame, (0, frame.shape[0] - 50), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            cv2.putText(frame, banner, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, frame.shape[1::-1])
            writer.write(frame)
        if not args.headless:
            cv2.imshow("Phone", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if writer is not None:
        writer.release()


if __name__ == "__main__":
    main()
