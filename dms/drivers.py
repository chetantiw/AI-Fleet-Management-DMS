#!/usr/bin/env python3
"""
Manage enrolled drivers (stored in the gallery file named in config.json).

  python3 drivers.py add "Ramesh Kumar"              # enrol from the Pi camera (driver sits in the seat)
  python3 drivers.py add "Ramesh Kumar" --photos dir # enrol from photos (5+ clear face photos)
  python3 drivers.py list
  python3 drivers.py remove "Ramesh Kumar"
  python3 drivers.py test                            # say who the camera sees right now

Camera enrolment: the driver sits normally and looks ahead, then slowly turns the head a little
left, right, up and down. 8 good shots are taken automatically within ~30 s (blurred, turned,
or too-small faces are skipped). Photos are kept in <data_dir>/enrol/<name>/ for records.
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import app  # noqa: E402  (config loading)
import frs  # noqa: E402
import sources  # noqa: E402


def models(cfg):
    det = frs.FaceDetector(os.path.join(cfg["models_dir"], "det_500m.onnx"), size=cfg.get("face_det_size", 320))
    emb = frs.FaceEmbedder(os.path.join(cfg["models_dir"], "w600k_mbf.onnx"))
    return det, emb


def from_camera(cfg, name, n=8, timeout=40, source=None):
    det, emb = models(cfg)
    src = sources.open_source(cfg["camera"], source)
    folder = os.path.join(cfg["data_dir"], "enrol", name.replace("/", "_"))
    os.makedirs(folder, exist_ok=True)
    shots, poses, t0, last = [], [], time.time(), 0.0
    print(f"Enrolling {name}: sit normally, look ahead, then slowly turn the head a little "
          f"left, right, up and down.", flush=True)
    try:
        while len(shots) < n and time.time() - t0 < timeout:
            frame, _ = src.read()
            if frame is None:
                break
            faces = det(frame)
            if not faces:
                continue
            why = frs.face_quality(faces[0], frame)
            if why or time.time() - last < 0.6:
                continue
            kp = faces[0]["kps"]
            yaw = ((kp[2][0] - kp[0][0]) / (kp[1][0] - kp[0][0] + 1e-6) - 0.5) * 2
            pitch = (kp[2][1] - (kp[0][1] + kp[1][1]) / 2) / (np.linalg.norm(kp[1] - kp[0]) + 1e-6)
            if any(abs(yaw - y) < 0.08 and abs(pitch - p) < 0.06 for y, p in poses) and len(shots) >= 3:
                continue                                   # want some variety after the first few
            e = emb(frame, kp)
            shots.append(e)
            poses.append((yaw, pitch))
            last = time.time()
            cv2.imwrite(os.path.join(folder, f"{int(time.time())}_{len(shots)}.jpg"), frame)
            print(f"  shot {len(shots)}/{n}", flush=True)
    finally:
        src.close()
    return shots


def from_photos(cfg, folder):
    det, emb = models(cfg)
    embs = []
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        img = cv2.imread(f)
        faces = det(img) if img is not None else []
        why = "no face" if not faces else frs.face_quality(faces[0], img)
        print(f"  {'skip' if why else 'ok  '} {os.path.basename(f)}{': ' + why if why else ''}")
        if not why:
            embs.append(emb(img, faces[0]["kps"]))
    return embs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["add", "list", "remove", "test"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--photos")
    ap.add_argument("--source", help="camera override, e.g. a webcam index or video file")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    a = ap.parse_args()
    cfg = app.load_config(a.config)
    os.makedirs(os.path.dirname(cfg["gallery"]), exist_ok=True)
    gal = frs.Gallery(cfg["gallery"])
    if a.cmd == "list":
        names = sorted(set(gal.names))
        print("\n".join(f"{n}: {gal.names.count(n)} samples" for n in names) or "no drivers enrolled")
    elif a.cmd == "remove":
        print(f"removed {gal.remove(a.name)} samples of {a.name}")
    elif a.cmd == "add":
        if not a.name:
            sys.exit("give the driver's name")
        embs = from_photos(cfg, a.photos) if a.photos else from_camera(cfg, a.name, source=a.source)
        if len(embs) < 3:
            sys.exit(f"only {len(embs)} good shots - not enrolled. Check light, distance and that the face "
                     f"is not turned or blurred, then try again.")
        E = np.array(embs)
        if float((E @ E.T).min()) < frs.MATCH_THR:
            print("WARNING: the shots disagree - was more than one person in view?")
        if a.name in gal.names:
            gal.remove(a.name)
        gal.add(a.name, embs)
        print(f"enrolled {a.name} with {len(embs)} samples. Drivers: {sorted(set(gal.names))}")
    elif a.cmd == "test":
        det, emb = models(cfg)
        src = sources.open_source(cfg["camera"], a.source)
        try:
            t_end, embs = time.time() + 20, []
            while time.time() < t_end and len(embs) < 5:
                frame, _ = src.read()
                if frame is None:
                    break
                faces = det(frame)
                if faces and not frs.face_quality(faces[0], frame):
                    embs.append(emb(frame, faces[0]["kps"]))
            if not embs:
                print("no clear face seen in 20 s")
            else:
                e = np.mean(embs, 0)
                name, sim = gal.match(e / np.linalg.norm(e))
                print(f"{name or 'UNKNOWN'}  (similarity {sim:.2f} over {len(embs)} shots, match at >= {frs.MATCH_THR})")
        finally:
            src.close()


if __name__ == "__main__":
    main()
