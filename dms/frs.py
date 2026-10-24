#!/usr/bin/env python3
"""
FRS stage 1: face detection + driver face recognition on plain CPU (ONNX Runtime).
Runs on a PC or on the Raspberry Pi 5 without the AI HAT.

  pip install onnxruntime opencv-python numpy

  python frs.py enroll "Ramesh Kumar" photos/ramesh/      # from a folder of photos
  python frs.py enroll "Ramesh Kumar" --cam 0             # from a webcam (SPACE = capture, q = done)
  python frs.py list
  python frs.py remove "Ramesh Kumar"
  python frs.py run --cam 0                               # live identification
  python frs.py run --video clip.mp4
  python frs.py identify photo.jpg                        # one image, prints result

Models (SCRFD-500M detector, ArcFace MobileFaceNet) download on first use.
"""
import argparse
import glob
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
MODEL_ZIP = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
GALLERY = os.path.join(HERE, "gallery.npz")

DET_SIZE = 640
DET_SCORE = 0.5
MATCH_THR = 0.35        # cosine similarity. LFW benchmark: 0.00% false accept, 2.1% false reject
                        # (0.30 -> 0.03% / 1.1%; 0.40 -> 0.00% / 4.7%)
MIN_FACE_PX = 80        # faces smaller than this are too poor to identify
VERIFY_FRAMES = 5       # consecutive agreeing frames before identity is confirmed

# ArcFace 112x112 alignment template: eyes, nose, mouth corners
ARC_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def ensure_models():
    need = [os.path.join(MODEL_DIR, n) for n in ("det_500m.onnx", "w600k_mbf.onnx")]
    if all(os.path.exists(p) for p in need):
        return need
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading models (15 MB)...")
    with urllib.request.urlopen(MODEL_ZIP) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(MODEL_DIR)
    return need


# ----------------------------------------------------------------------------
# Detector (SCRFD)
# ----------------------------------------------------------------------------
class FaceDetector:
    def __init__(self, path, size=None, score=None):
        """size: square input side (640 default; 320 is ~4x faster and fine for a close cabin camera)."""
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3          # the model declares 640x640 shapes; other sizes are fine
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.size = size or DET_SIZE
        self.score = score

    def __call__(self, bgr):
        """Returns list of dicts {box, score, kps} in image pixels, largest face first."""
        DET_SIZE = self.size
        DET_SCORE = self.score if self.score is not None else globals()["DET_SCORE"]
        h, w = bgr.shape[:2]
        s = min(DET_SIZE / w, DET_SIZE / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        canvas = np.zeros((DET_SIZE, DET_SIZE, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128.0, (DET_SIZE, DET_SIZE),
                                     (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(None, {self.inp: blob})
        # Group the 9 outputs by anchor count (stride) and channel count (type).
        by_stride = {}
        for o in outs:
            o = o.reshape(-1, o.shape[-1])
            stride = int(round(DET_SIZE / np.sqrt(o.shape[0] / 2)))
            by_stride.setdefault(stride, {})[o.shape[1]] = o
        boxes, scores, kpss = [], [], []
        for stride, o in by_stride.items():
            n = DET_SIZE // stride
            sc = o[1].reshape(-1)
            keep = sc > DET_SCORE
            if not keep.any():
                continue
            ys, xs = np.mgrid[:n, :n]
            ctr = np.stack([xs, ys], -1).reshape(-1, 2).repeat(2, axis=0)[keep] * stride
            d = o[4][keep] * stride
            boxes.append(np.concatenate([ctr - d[:, :2], ctr + d[:, 2:]], 1))
            kpss.append(ctr[:, None, :] + o[10][keep].reshape(-1, 5, 2) * stride)
            scores.append(sc[keep])
        if not boxes:
            return []
        boxes, scores, kpss = np.concatenate(boxes), np.concatenate(scores), np.concatenate(kpss)
        xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1)
        idx = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), DET_SCORE, 0.4)).reshape(-1)
        faces = [dict(box=boxes[i] / s, score=float(scores[i]), kps=kpss[i] / s) for i in idx]
        faces.sort(key=lambda f: -(f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
        return faces


# ----------------------------------------------------------------------------
# Recogniser (ArcFace)
# ----------------------------------------------------------------------------
class FaceEmbedder:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    @staticmethod
    def align(bgr, kps):
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARC_DST, method=cv2.LMEDS)
        return cv2.warpAffine(bgr, M, (112, 112), borderValue=0)

    def __call__(self, bgr, kps):
        """Returns a unit-length 512-d embedding."""
        chip = self.align(bgr, kps)
        blob = cv2.dnn.blobFromImage(chip, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        e = self.sess.run(None, {self.inp: blob})[0][0]
        return e / np.linalg.norm(e)


def face_quality(face, bgr):
    """Returns None if the face is usable, else a short reason."""
    x1, y1, x2, y2 = face["box"]
    if min(x2 - x1, y2 - y1) < MIN_FACE_PX:
        return "too small / too far"
    le, re, nose = face["kps"][0], face["kps"][1], face["kps"][2]
    eye_d = np.linalg.norm(re - le) + 1e-6
    if abs(nose[0] - (le[0] + re[0]) / 2) / eye_d > 0.35:
        return "face turned"
    crop = bgr[max(int(y1), 0):int(y2), max(int(x1), 0):int(x2)]
    if crop.size and cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 20:
        return "blurred"
    return None


# ----------------------------------------------------------------------------
# Gallery of enrolled drivers
# ----------------------------------------------------------------------------
class Gallery:
    def __init__(self, path=GALLERY):
        self.path = path
        self.names, self.embs = [], np.zeros((0, 512), np.float32)
        if os.path.exists(path):
            d = np.load(path, allow_pickle=False)
            self.names, self.embs = [str(n) for n in d["names"]], d["embs"]

    def save(self):
        np.savez(self.path, names=np.array(self.names), embs=self.embs)

    def add(self, name, embs):
        self.names += [name] * len(embs)
        self.embs = np.concatenate([self.embs, np.asarray(embs, np.float32)])
        self.save()

    def remove(self, name):
        keep = [n != name for n in self.names]
        removed = len(keep) - sum(keep)
        self.names = [n for n in self.names if n != name]
        self.embs = self.embs[np.array(keep, bool)] if keep else self.embs
        self.save()
        return removed

    def match(self, emb):
        """Returns (name or None, best similarity). Per driver, the best sample counts."""
        if not self.names:
            return None, 0.0
        sims = self.embs @ emb
        i = int(np.argmax(sims))
        return (self.names[i] if sims[i] >= MATCH_THR else None), float(sims[i])


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def open_source(args):
    if getattr(args, "video", None):
        return cv2.VideoCapture(args.video)
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def cmd_enroll(args, det, emb, gal):
    embs = []
    if args.path:
        files = sorted(f for f in glob.glob(os.path.join(args.path, "*")) if
                       f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        for f in files:
            img = cv2.imread(f)
            faces = det(img) if img is not None else []
            why = "no face" if not faces else face_quality(faces[0], img)
            if why:
                print(f"  skip {os.path.basename(f)}: {why}")
                continue
            embs.append(emb(img, faces[0]["kps"]))
            print(f"  ok   {os.path.basename(f)}")
    else:
        cap = open_source(args)
        print("SPACE = capture, q = finish. Take 5+ shots: straight, slight left/right, with/without glasses.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces = det(frame)
            why = "no face" if not faces else face_quality(faces[0], frame)
            view = frame.copy()
            if faces:
                x1, y1, x2, y2 = faces[0]["box"].astype(int)
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255) if why else (0, 255, 0), 2)
            cv2.putText(view, f"{args.name}: {len(embs)} captured  {why or 'ready'}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Enroll", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == 32 and not why:
                embs.append(emb(frame, faces[0]["kps"]))
    if not embs:
        sys.exit("No usable face captured; nothing enrolled.")
    if len(embs) > 1:       # sanity check: all samples should be the same person
        E = np.array(embs)
        low = float((E @ E.T).min())
        if low < MATCH_THR:
            print(f"  WARNING: samples disagree (min similarity {low:.2f}). Mixed people or poor photos?")
    gal.add(args.name, embs)
    print(f"Enrolled {args.name} with {len(embs)} sample(s). Gallery: {sorted(set(gal.names))}")


def cmd_identify(args, det, emb, gal):
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot read {args.image}")
    faces = det(img)
    if not faces:
        print(json.dumps(dict(result="NO_FACE")))
        return
    name, sim = gal.match(emb(img, faces[0]["kps"]))
    print(json.dumps(dict(result="MATCH" if name else "UNKNOWN", driver=name,
                          similarity=round(sim, 3), quality_issue=face_quality(faces[0], img))))


def cmd_run(args, det, emb, gal):
    cap = open_source(args)
    streak_name, streak, confirmed, fps, last = None, 0, None, 0.0, time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = det(frame)
        label, colour = "NO FACE", (128, 128, 128)
        if faces:
            f = faces[0]                      # the driver = largest face
            why = face_quality(f, frame)
            if why:
                label, colour = why, (0, 165, 255)
            else:
                name, sim = gal.match(emb(frame, f["kps"]))
                streak = streak + 1 if name == streak_name else 1
                streak_name = name
                if streak >= VERIFY_FRAMES and name != confirmed:
                    confirmed = name
                    print(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"),
                                          event="DRIVER_VERIFIED" if name else "UNKNOWN_DRIVER",
                                          driver=name, similarity=round(sim, 3))), flush=True)
                label = f"{name or 'UNKNOWN'} {sim:.2f}"
                colour = (0, 255, 0) if name else (0, 0, 255)
            x1, y1, x2, y2 = f["box"].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            for p in f["kps"].astype(int):
                cv2.circle(frame, tuple(p), 2, (0, 255, 255), -1)
        else:
            streak, streak_name = 0, None
        now = time.monotonic()
        fps, last = 0.9 * fps + 0.1 / max(now - last, 1e-6), now
        if not args.headless:
            cv2.putText(frame, f"{label}   {fps:.0f} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
            cv2.imshow("FRS", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("enroll"); p.add_argument("name"); p.add_argument("path", nargs="?"); p.add_argument("--cam", type=int, default=0)
    p = sub.add_parser("identify"); p.add_argument("image")
    p = sub.add_parser("run"); p.add_argument("--cam", type=int, default=0); p.add_argument("--video"); p.add_argument("--headless", action="store_true")
    sub.add_parser("list")
    p = sub.add_parser("remove"); p.add_argument("name")
    args = ap.parse_args()

    gal = Gallery()
    if args.cmd == "list":
        for n in sorted(set(gal.names)):
            print(f"{n}: {gal.names.count(n)} sample(s)")
        return
    if args.cmd == "remove":
        print(f"Removed {gal.remove(args.name)} sample(s) for {args.name}")
        return
    det_path, emb_path = ensure_models()
    det, emb = FaceDetector(det_path), FaceEmbedder(emb_path)
    {"enroll": cmd_enroll, "identify": cmd_identify, "run": cmd_run}[args.cmd](args, det, emb, gal)


if __name__ == "__main__":
    main()
