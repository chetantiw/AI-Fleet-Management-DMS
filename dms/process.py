#!/usr/bin/env python3
"""
One command for a recorded cabin video: measure everything, identify drivers, raise alerts,
and write the annotated video + event list.

  python3 process.py VIDEO [--gallery gallery.npz] [--out results/]

Steps (each result is cached in <out>/cache, so a re-run only redoes what is missing):
  1. extract.py        face boxes, face mesh, eye/gaze scores, phone/bottle detections (10 fps)
  2. hands_extract.py  hand-to-mouth / hand-to-ear distances (5 fps)
  3. belt_extract.py   seat-belt score (2 fps)
  4. faces_extract.py  face embeddings for driver identification (2 fps)
  5. render_session.py annotated video + events CSV
Steps 1-4 run two at a time.

Enrol drivers first (photos, 5+ per driver):
  python3 frs.py enroll "Ramesh Kumar" photos/ramesh/
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, log):
    with open(log, "w") as fh:
        return subprocess.Popen([sys.executable] + cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--gallery", default=os.path.join(HERE, "gallery.npz"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--twin", action="store_true",
                    help="also write the digital-twin stream (<name>_twin.json) for the 3-D viewer")
    a = ap.parse_args()
    video = os.path.abspath(a.video)
    name = os.path.splitext(os.path.basename(video))[0]
    cache = os.path.abspath(os.path.join(a.out, "cache"))
    os.makedirs(cache, exist_ok=True)
    base = os.path.join(cache, name)
    t0 = time.time()

    if not os.path.exists(base + ".npz"):
        print("1/5 measuring face, eyes, phone ...", flush=True)
        if run(["extract.py", video, base + ".npz"], base + ".extract.log").wait():
            sys.exit(f"extract failed, see {base}.extract.log")
    jobs = []
    steps = [("hands_extract.py", ".hands.npy", "2/5 hands"),
             ("belt_extract.py", ".belt.npy", "3/5 seat belt"),
             ("faces_extract.py", ".faces.npz", "4/5 face identity")]
    if a.twin:
        steps.append(("twin_extract.py", ".twin.npz", "twin: body pose"))
    for script, suffix, label in steps:
        if not os.path.exists(base + suffix):
            print(label, "...", flush=True)
            jobs.append(run([script, video, base + ".npz", base + suffix], base + suffix + ".log"))
            if len(jobs) == 2:
                if jobs[0].wait():
                    sys.exit(f"step failed, see logs in {cache}")
                jobs.pop(0)
    for j in jobs:
        if j.wait():
            sys.exit(f"step failed, see logs in {cache}")

    print("5/5 alerts + annotated video ...", flush=True)
    out_mp4 = os.path.join(a.out, name + "_annotated.mp4")
    out_csv = os.path.join(a.out, name + "_events.csv")
    cmd = ["render_session.py", "--out", out_mp4, "--csv", out_csv, f"{video}:{cache}"]
    if a.twin:
        cmd[1:1] = ["--state", base + ".state.jsonl"]
    if os.path.exists(a.gallery):
        cmd[1:1] = ["--gallery", os.path.abspath(a.gallery)]
    else:
        print(f"(no gallery at {a.gallery}: every driver will be reported as Unknown #n)")
    if run(cmd, base + ".render.log").wait():
        sys.exit(f"render failed, see {base}.render.log")
    if a.twin:
        out_twin = os.path.join(a.out, name + "_twin.json")
        if run(["twin_stream.py", base, base + ".state.jsonl", out_twin], base + ".twin.log").wait():
            sys.exit(f"twin stream failed, see {base}.twin.log")
        print("digital twin stream ->", out_twin)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_mp4}, {out_csv}")


if __name__ == "__main__":
    main()
