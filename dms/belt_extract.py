"""Belt scores at 2 Hz: python3 belt_extract.py video features.npz out.npy"""
import sys, cv2, numpy as np, belt
video, xf, out = sys.argv[1:4]
d = np.load(xf); T, box = d["t"], d["box"]
cap = cv2.VideoCapture(video); fps = cap.get(5); step = max(int(round(fps / 2)), 1)
rows, i = [], 0
while True:
    if i % step:
        if not cap.grab(): break
        i += 1; continue
    ok, f = cap.read()
    if not ok: break
    s = i / fps; i += 1
    k = int(np.argmin(abs(T - s))); b = box[k]
    if np.isnan(b[0]): rows.append([s, np.nan, np.nan, np.nan]); continue
    sc, a, off = belt.belt_score(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), b)
    rows.append([s, sc, a, off])
np.save(out, np.array(rows)); print(out, len(rows))
