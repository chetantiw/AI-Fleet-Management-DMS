"""Face embeddings at 2 Hz for driver identification.

  python3 faces_extract.py video features.npz out.npz

Uses the face boxes/key points already found by extract.py; saves time, 512-d ArcFace
embedding, and whether the face passed the quality check (size, turn, blur)."""
import sys

import cv2
import numpy as np

import frs

video, xf, out = sys.argv[1:4]
d = np.load(xf)
T, box, kps, fsc = d["t"], d["box"], d["kps"], d["fscore"]
emb = frs.FaceEmbedder(frs.ensure_models()[1])
cap = cv2.VideoCapture(video)
fps = cap.get(5)
step = max(int(round(fps / 2)), 1)
ts, es, good, i = [], [], [], 0
while True:
    if i % step:
        if not cap.grab():
            break
        i += 1
        continue
    ok, f = cap.read()
    if not ok:
        break
    s = i / fps
    i += 1
    k = int(np.argmin(abs(T - s)))
    if abs(T[k] - s) > 0.06 or np.isnan(box[k, 0]):
        continue
    face = dict(box=box[k], kps=kps[k], score=fsc[k])
    ts.append(s)
    es.append(emb(f, kps[k]))
    good.append(frs.face_quality(face, f) is None)
np.savez(out, t=np.array(ts), emb=np.array(es, np.float32), good=np.array(good))
print(out, len(ts), "faces,", int(np.sum(good)), "good quality")
