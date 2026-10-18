#!/usr/bin/env python3
"""
Score the DMS pipeline against a DMD drowsiness clip (face video + annotation JSON).

  python3 eval_dmd.py clip_rgb_face.mp4 clip_rgb_ann_drowsiness.json

Reports frame-level eye-closure accuracy, yawn detection, and the events the
temporal engine fired. Per-frame signals are cached next to the video (.npz).
"""
import json
import os
import sys

import cv2
import numpy as np

import dms


def signals(video):
    cache = video + ".signals.npz"
    if os.path.exists(cache):
        return dict(np.load(cache))
    be = dms.CpuBackend(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, lm = be(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if lm is None:
            rows.append([np.nan] * 5)
            continue
        p = dms.head_pose(lm, frame.shape[1], frame.shape[0]) or (np.nan, np.nan, 0)
        rows.append([(dms.ear(lm, dms.EYE_L) + dms.ear(lm, dms.EYE_R)) / 2, dms.mar(lm), p[0], p[1], 1])
    be.close()
    a = np.array(rows)
    out = dict(ear=a[:, 0], mar=a[:, 1], pitch=a[:, 2], yaw=a[:, 3], fps=np.array(fps))
    np.savez(cache, **out)
    return out


def intervals(ann, prefix):
    out = []
    for a in ann["actions"].values():
        if a["type"].lower().startswith(prefix):
            out += [(i["frame_start"], i["frame_end"]) for i in a["frame_intervals"]]
    return sorted(out)


def mask(iv, n):
    m = np.zeros(n, bool)
    for s, e in iv:
        m[s:e + 1] = True
    return m


def main(video, ann_path):
    ann = json.load(open(ann_path))
    ann = ann.get("openlabel") or ann.get("vcd") or ann
    s = signals(video)
    n, fps = len(s["ear"]), float(s["fps"])
    print(f"{os.path.basename(video)}: {n} frames, {n / fps:.0f} s, face found in "
          f"{100 * np.mean(~np.isnan(s['ear'])):.1f}% of frames")

    # Replay through the engine exactly as dms.py would
    eng, fired, closed_pred = dms.DmsEngine(), [], np.zeros(n, bool)
    for i in range(n):
        t = i / fps
        if np.isnan(s["ear"][i]):
            ev = eng.update(t)
        else:
            ev = eng.update(t, s["ear"][i], s["mar"][i], (s["pitch"][i], s["yaw"][i], 0))
            closed_pred[i] = bool(eng.state.get("closed"))
        fired += [(i, e) for e in ev]
    print(f"calibrated EAR threshold {eng.ear_thr:.3f}, neutral pitch/yaw {eng.neutral[0]:.0f}/{eng.neutral[1]:.0f}")

    # Frame-level eye state (only frames labelled fully open or fully closed)
    gt_open, gt_closed = mask(intervals(ann, "eyes_state/open"), n), mask(intervals(ann, "eyes_state/close"), n)
    gt_open &= ~mask(intervals(ann, "eyes_state/opening"), n)
    valid = (gt_open | gt_closed) & ~np.isnan(s["ear"])
    valid[:int(dms.CFG["calib_seconds"] * fps) + 1] = False
    tp = np.sum(closed_pred & gt_closed & valid)
    fp = np.sum(closed_pred & gt_open & valid)
    fn = np.sum(~closed_pred & gt_closed & valid)
    print(f"eye closed, per frame: GT closed {np.sum(gt_closed & valid)}, GT open {np.sum(gt_open & valid)} | "
          f"recall {tp / max(tp + fn, 1):.2f}  precision {tp / max(tp + fp, 1):.2f}  "
          f"false-closed rate on open frames {fp / max(np.sum(gt_open & valid), 1):.3f}")
    print(f"  EAR median: open {np.nanmedian(s['ear'][gt_open & valid]):.3f}  "
          f"closed {np.nanmedian(s['ear'][gt_closed & valid]) if np.any(gt_closed & valid) else float('nan'):.3f}")

    # Long closures (the ones that should raise DROWSY)
    need = dms.CFG["eyes_closed_s"] * fps
    long_gt = [(a, b) for a, b in intervals(ann, "eyes_state/close") if b - a + 1 >= need]
    drowsy = [i for i, e in fired if e == "DROWSY_EYES_CLOSED"]
    print(f"closures >= {dms.CFG['eyes_closed_s']} s in GT: {len(long_gt)}; DROWSY_EYES_CLOSED fired {len(drowsy)} "
          f"at {[round(i / fps, 1) for i in drowsy]}")

    # Yawns, event level
    gy = intervals(ann, "yawning")
    merged = []
    for a, b in gy:                         # merge touching with-hand / without-hand pieces
        if merged and a <= merged[-1][1] + fps:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    yawns = [i for i, e in fired if e == "YAWN"]
    hit = sum(any(a - fps <= y <= b + fps for y in yawns) for a, b in merged)
    false = sum(not any(a - fps <= y <= b + fps for a, b in merged) for y in yawns)
    print(f"yawns: GT {len(merged)}, detected {hit}, false {false}")
    other = sorted({e for _, e in fired} - {"YAWN", "DROWSY_EYES_CLOSED", "CALIBRATED"})
    for e in other:
        print(f"{e}: {[round(i / fps, 1) for i, x in fired if x == e]}")
    return s, ann


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
