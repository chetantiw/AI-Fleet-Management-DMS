#!/usr/bin/env python3
"""
Upper-body pose for the digital twin:  python3 twin_extract.py video extract.npz out.npz [--fps 10]

The camera looks down on the driver from high up, so the pose model is run on a crop centred
on the driver's face (from extract.py), 5 face-widths wide and 6 tall. That keeps passengers
and the windscreen out and makes the body big enough for the model.

Saved per processed frame (frame pixel coordinates):
  t        time (s)
  img      MediaPipe pose points 0 (nose), 11-16 (shoulders, elbows, wrists): x, y, visibility
  world    the same points in the model's 3-D body space (metres, hip-centred): x, y, z
"""
import argparse
import os

import cv2
import numpy as np

PTS = [0, 11, 12, 13, 14, 15, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("extract")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--model", default="models/pose_landmarker_heavy.task")
    a = ap.parse_args()
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.2, min_pose_presence_confidence=0.2, min_tracking_confidence=0.3))
    d = np.load(a.extract)
    T, box = d["t"], d["box"]
    cap = cv2.VideoCapture(a.video)
    src = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(src / a.fps)), 1)
    R = dict(t=[], img=[], world=[])
    last_box, i = None, 0
    nan_img, nan_w = np.full((len(PTS), 3), np.nan), np.full((len(PTS), 3), np.nan)
    while True:
        if i % step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = i / src
        i += 1
        j = int(np.argmin(abs(T - t)))
        b = box[j]
        if np.isnan(b[0]):
            b = last_box                    # face lost for a moment: keep the last driver area
        else:
            last_box = b
        R["t"].append(t)
        if b is None:
            R["img"].append(nan_img); R["world"].append(nan_w)
            continue
        fw = b[2] - b[0]
        cx = (b[0] + b[2]) / 2
        H, W = frame.shape[:2]
        x0, y0 = max(int(cx - fw * 2.5), 0), max(int(b[1] - fw * 0.6), 0)
        x1, y1 = min(int(cx + fw * 2.5), W), min(int(b[1] + fw * 5.4), H)
        crop = np.ascontiguousarray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        # VIDEO mode keeps tracking between frames; the crop moves a little, which it tolerates
        r = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop), int(t * 1000))
        if not r.pose_landmarks:
            R["img"].append(nan_img); R["world"].append(nan_w)
            continue
        p, w = r.pose_landmarks[0], r.pose_world_landmarks[0]
        ch, cw = crop.shape[:2]
        R["img"].append([[p[k].x * cw + x0, p[k].y * ch + y0, p[k].visibility] for k in PTS])
        R["world"].append([[w[k].x, w[k].y, w[k].z] for k in PTS])
    np.savez(a.out, t=np.array(R["t"]), img=np.array(R["img"], float), world=np.array(R["world"], float),
             pts=np.array(PTS))
    print(a.out, len(R["t"]), "frames")
    os._exit(0)                              # mediapipe's close() can crash at interpreter exit


if __name__ == "__main__":
    main()
