#!/usr/bin/env python3
"""
Run the whole prototype (drowsiness/distraction + phone use + face embeddings) over one
recorded video, the way the in-bus unit would see it.

  python3 run_all.py video.mov --out results/ [--fps 15] [--save annotated.mp4]

Writes <out>/<name>.events.csv, <name>.signals.npz (per-frame values for later analysis),
<name>.faces.npz (one face embedding per second, for identity checks) and optionally an
annotated video at reduced size.
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

import dms
import frs
import phone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fps", type=float, default=15.0, help="processing rate")
    ap.add_argument("--phone_hz", type=float, default=3.0)
    ap.add_argument("--save")
    args = ap.parse_args()
    name = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.out, exist_ok=True)

    det_path, emb_path = frs.ensure_models()
    face_det, face_emb = frs.FaceDetector(det_path), frs.FaceEmbedder(emb_path)
    mesh, yolo = dms.CpuBackend(os.path.join(os.path.dirname(__file__), "models")), phone.YoloX()
    eng, peng = dms.DmsEngine(), phone.PhoneEngine()

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(src_fps / args.fps)), 1)
    phone_every = max(int(round(args.fps / args.phone_hz)), 1)
    rows, events, faces, writer = [], [], [], None
    i = k = 0
    last_phone, last_face_t, banner, until = [], -1e9, "", 0.0
    t0 = time.time()
    import collections
    eye_hist = collections.deque(maxlen=int(2 * args.fps))
    while True:
        if i % step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = i / src_fps
        i += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        box, lm = mesh(rgb)
        if lm is not None:
            e = (dms.ear(lm, dms.EYE_L) + dms.ear(lm, dms.EYE_R)) / 2
            p = dms.head_pose(lm, frame.shape[1], frame.shape[0]) or (np.nan, np.nan, 0)
            meas = dms.measurable(lm)
            vis, eye_ratio = dms.eyes_visible(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), lm)
            eye_hist.append(vis)
            eyes_ok = sum(eye_hist) * 2 > len(eye_hist)       # majority over ~2 s
            evs = eng.update(t, e, dms.mar(lm), p, meas, eyes_ok)
            rows.append([t, e, dms.mar(lm), p[0], p[1], meas, dms.face_turn(lm), np.linalg.norm(lm[263] - lm[33]), eye_ratio])
        else:
            evs = eng.update(t)
            rows.append([t] + [np.nan] * 8)
        st = eng.state

        if k % phone_every == 0:
            fd = face_det(frame)
            fbox = fd[0]["box"] if fd else None
            d = yolo(frame)
            mode = phone.phone_mode(d[0], fbox) if d else None
            last_phone = d if mode else []
            evs += peng.update(t, mode, d[0][4] if d else 0.0)
            if fd and t - last_face_t >= 1.0:
                last_face_t = t
                faces.append((t, face_emb(frame, fd[0]["kps"]), frs.face_quality(fd[0], frame) is None))
        k += 1

        for ev in evs:
            if ev == "CALIBRATED":
                continue
            events.append((round(t, 1), ev, {kk: round(float(v), 3) for kk, v in st.items()
                                             if isinstance(v, (int, float)) and not isinstance(v, bool)}))
            banner, until = ev.replace("_", " "), t + 3

        if args.save:
            dms.draw(frame, box, lm, st, banner if t < until else "", args.fps)
            for d in last_phone:
                cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 0, 255), 4)
            cv2.putText(frame, f"t={int(t // 60):02d}:{t % 60:04.1f}", (10, frame.shape[0] - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, small.shape[1::-1])
            writer.write(small)

    if writer is not None:
        writer.release()
    mesh.close()
    np.savez(os.path.join(args.out, name + ".signals.npz"),
             sig=np.array(rows, float), cols=np.array(["t", "ear", "mar", "pitch", "yaw", "measurable", "turn", "eye_px", "eye_ratio"]))
    if faces:
        np.savez(os.path.join(args.out, name + ".faces.npz"), t=np.array([f[0] for f in faces]),
                 emb=np.array([f[1] for f in faces]), good=np.array([f[2] for f in faces]))
    with open(os.path.join(args.out, name + ".events.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video_time", "seconds", "event", "details"])
        for t, ev, det in events:
            w.writerow([f"{int(t // 60):02d}:{t % 60:04.1f}", t, ev, det])
    print(f"{name}: {i / src_fps:.0f} s of video, {len(rows)} frames processed in {time.time() - t0:.0f} s, "
          f"{len(events)} events")


if __name__ == "__main__":
    main()
