#!/usr/bin/env python3
"""
Score phone-use detection against a DMD distraction clip (face video + annotation JSON).

  python3 eval_phone_dmd.py clip_rgb_face.mp4 clip_rgb_ann_distraction.json [model] [step]

Detections are cached next to the video so thresholds can be re-scored instantly.
"""
import json
import os
import sys

import cv2
import numpy as np

import frs
import phone


def detect(video, model, step):
    cache = f"{video}.{model}.s{step}.npz"
    if os.path.exists(cache):
        return dict(np.load(cache))
    yolo, face_det = phone.YoloX(model), frs.FaceDetector(frs.ensure_models()[0])
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    idx, boxes, faces, i = [], [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            d = yolo(frame, score=0.05)
            f = face_det(frame)
            idx.append(i)
            boxes.append(d[0][:5] if d else (0, 0, 0, 0, 0))
            faces.append(f[0]["box"] if f else (0, 0, 0, 0))
        i += 1
    out = dict(idx=np.array(idx), box=np.array(boxes, float), face=np.array(faces, float), fps=np.array(fps), n=np.array(i))
    np.savez(cache, **out)
    return out


def gt_mask(ann, prefixes, n):
    m = np.zeros(n, bool)
    for a in ann["actions"].values():
        if a["type"].lower().startswith(tuple(prefixes)):
            for i in a["frame_intervals"]:
                m[i["frame_start"]:i["frame_end"] + 1] = True
    return m


def main(video, ann_path, model="yolox_s", step=6):
    ann = json.load(open(ann_path))
    ann = ann.get("openlabel") or ann.get("vcd") or ann
    d = detect(video, model, int(step))
    n, fps, idx = int(d["n"]), float(d["fps"]), d["idx"]
    call = gt_mask(ann, ["driver_actions/phonecall"], n)[idx]
    text = gt_mask(ann, ["driver_actions/texting"], n)[idx]
    other = ~(call | text)
    score = d["box"][:, 4]
    print(f"{os.path.basename(video)} [{model}]: {n / fps:.0f} s, {len(idx)} detector runs; "
          f"GT call {call.sum()}, texting {text.sum()}, no phone {other.sum()}")
    for thr in (0.2, 0.3, 0.4, 0.5):
        hit = score >= thr
        print(f"  thr {thr}: call {100 * hit[call].mean() if call.any() else float('nan'):5.1f}%  "
              f"texting {100 * hit[text].mean() if text.any() else float('nan'):5.1f}%  "
              f"false on no-phone runs {100 * hit[other].mean():5.2f}%")
    # False alarms by what the driver was doing
    hit = score >= phone.CFG["score"]
    for a in sorted({a["type"] for a in ann["actions"].values() if a["type"].startswith("driver_actions/")}):
        m = gt_mask(ann, [a.lower()], n)[idx]
        if m.any():
            print(f"    {a[15:]:22s} runs {m.sum():5d}  phone seen {100 * hit[m].mean():5.1f}%")
    # Event level, as the live program would raise them
    eng, fired = phone.PhoneEngine(), []
    for k, i in enumerate(idx):
        mode = None
        if score[k] >= phone.CFG["score"]:
            mode = phone.phone_mode(d["box"][k], d["face"][k] if d["face"][k].any() else None)
        fired += [(round(float(i / fps), 1), e) for e in eng.update(i / fps, mode, score[k])]
    use = gt_mask(ann, ["driver_actions/phonecall", "driver_actions/texting"], n)
    for a in ann["actions"].values():
        if a["type"].lower().startswith(("driver_actions/phonecall", "driver_actions/texting")):
            for iv in a["frame_intervals"]:
                s, e = iv["frame_start"] / fps, iv["frame_end"] / fps
                ev = [(t, x) for t, x in fired if s <= t <= e + 1]
                print(f"  GT {a['type'][15:]:16s} {s:6.1f}-{e:6.1f}s -> {ev if ev else 'MISSED'}")
    false = [(t, x) for t, x in fired if not use[max(int(t * fps) - int(3 * fps), 0):int(t * fps) + 1].any()]
    print(f"  false events: {false}")


if __name__ == "__main__":
    main(*sys.argv[1:])
