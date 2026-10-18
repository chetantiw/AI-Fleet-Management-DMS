"""Hand-to-face distances at ~5 Hz: python3 hands_extract.py video extract.npz out.npz"""
import sys, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
video, xf, out = sys.argv[1:4]
hl = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.3))
d = np.load(xf); T, box, kps = d["t"], d["box"], d["kps"]
cap = cv2.VideoCapture(video); fps = cap.get(5); step = max(int(round(fps / 5)), 1)
rows, i = [], 0
while True:
    if i % step:
        if not cap.grab(): break
        i += 1; continue
    ok, f = cap.read()
    if not ok: break
    s = i / fps; i += 1
    j = int(np.argmin(abs(T - s))); b = box[j]
    if np.isnan(b[0]):
        rows.append([s, np.nan, np.nan, np.nan]); continue
    fw = b[2] - b[0]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    side = int(fw * 3.2); x0, y0 = int(cx - side / 2), int(cy - side * 0.4)
    H, W = f.shape[:2]; crop = np.zeros((side, side, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H)
    if sx1 <= sx0 or sy1 <= sy0:
        rows.append([s, np.nan, np.nan, np.nan]); continue
    crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = f[sy0:sy1, sx0:sx1]
    r = hl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cv2.resize(crop, (384, 384)), cv2.COLOR_BGR2RGB)))
    mouth = (kps[j, 3] + kps[j, 4]) / 2; ears_x = (kps[j, 0], kps[j, 1])
    dm = de = 9.0
    for h in r.hand_landmarks:
        p = np.array([[q.x, q.y] for q in h]) * side + [x0, y0]
        dm = min(dm, np.min(np.linalg.norm(p - mouth, axis=1)) / fw)
        # "at the ear": beside the face at eye-to-mouth height
        side_dist = np.min(np.minimum(abs(p[:, 0] - b[0]), abs(p[:, 0] - b[2]))) / fw
        in_band = np.any((p[:, 1] > b[1]) & (p[:, 1] < b[3]))
        de = min(de, side_dist if in_band else 9.0)
    rows.append([s, dm, de, len(r.hand_landmarks)])
np.save(out, np.array(rows))
print(out, len(rows))
