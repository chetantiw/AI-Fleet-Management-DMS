#!/usr/bin/env python3
"""
Extract every per-frame measurement from a video once, so alert logic can be tuned offline.

  python3 extract.py video.mov out.npz [--fps 10]

Saved per processed frame: time, SCRFD face box + 5 keypoints + score, the 468-point mesh
(eyes / mouth / pose subset), MediaPipe blendshapes (eyeBlink, eyeLookDown, jawOpen...),
and every 3rd frame the best phone / bottle / cup detections.
"""
import argparse

import cv2
import numpy as np

import dms
import frs
import phone

KEEP = sorted(set(dms.EYE_L + dms.EYE_R + dms.MOUTH + dms.POSE_IDX + [10, 152, 234, 454, 168, 6]))
BS_NAMES = ["eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
            "eyeLookOutLeft", "eyeLookOutRight", "eyeLookInLeft", "eyeLookInRight",
            "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "jawOpen"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=10)
    a = ap.parse_args()
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    lmk = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
    fd = frs.FaceDetector(frs.ensure_models()[0])
    fd_thr = frs.DET_SCORE
    frs.DET_SCORE = 0.3
    yolo = phone.YoloX()
    cap = cv2.VideoCapture(a.video)
    src = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(src / a.fps)), 1)
    R = dict(t=[], box=[], kps=[], fscore=[], lm=[], bs=[], obj=[])
    i = k = 0
    frame_hw = (0, 0)
    while True:
        if i % step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        frame_hw = frame.shape[:2]
        t = i / src
        i += 1
        faces = fd(frame)
        if faces:
            f = faces[0]
            R["box"].append(f["box"]); R["kps"].append(f["kps"]); R["fscore"].append(f["score"])
            b = f["box"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            side = int(max(b[2] - b[0], b[3] - b[1]) * 2.0)
            x0, y0 = int(cx - side / 2), int(cy - side / 2)
            H, W = frame.shape[:2]
            crop = np.zeros((side, side, 3), np.uint8)
            sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H)
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
            crop = cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2RGB)
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            if res.face_landmarks:
                lm = np.array([[p.x, p.y] for p in res.face_landmarks[0][:468]]) * side + [x0, y0]
                bsd = {c.category_name: c.score for c in res.face_blendshapes[0]}
                R["lm"].append(lm[KEEP]); R["bs"].append([bsd.get(n, np.nan) for n in BS_NAMES])
            else:
                R["lm"].append(np.full((len(KEEP), 2), np.nan)); R["bs"].append([np.nan] * len(BS_NAMES))
        else:
            R["box"].append([np.nan] * 4); R["kps"].append(np.full((5, 2), np.nan)); R["fscore"].append(0)
            R["lm"].append(np.full((len(KEEP), 2), np.nan)); R["bs"].append([np.nan] * len(BS_NAMES))
        if k % 3 == 0:
            d = yolo(frame, classes=(phone.COCO_PHONE, 39, 41), score=0.05)
            best = {}
            for x in d:
                best.setdefault(x[5], x)
            R["obj"].append([list(best.get(c, (np.nan,) * 5))[:5] for c in (phone.COCO_PHONE, 39, 41)])
        else:
            R["obj"].append([[np.nan] * 5] * 3)
        k += 1
        R["t"].append(t)
    frs.DET_SCORE = fd_thr
    np.savez(a.out, **{n: np.array(v, float) for n, v in R.items()}, keep=np.array(KEEP),
             bs_names=np.array(BS_NAMES), frame_hw=np.array(frame_hw))
    print(a.out, len(R["t"]), "frames")


if __name__ == "__main__":
    main()
