#!/usr/bin/env python3
"""
Replay one or more videos as ONE continuous session (e.g. several drivers taking turns),
with driver identification + all DMS alerts, and write an annotated video and event list.

  python3 render_session.py --gallery gallery.npz --out session.mp4 --csv session.csv \
      vidA.mov:dirA  vidB.mov:dirB ...

Each `video:dir` pair names a video and the folder holding its cached measurements
(<name>.npz from extract.py, .hands.npy, .belt.npy, .faces.npz). A single video works too.
When a different driver is confirmed, the alert engine restarts so each driver gets
their own calibration (head position, normal eye openness, face size).
"""
import argparse
import csv
import os

import cv2
import numpy as np

import dms
import engine2
import frs
import identity


def segment_frames(video, base):
    """Yield (local_t, frame, measurements) for each processed frame of one video."""
    d = np.load(base + ".npz")
    T, K, LM, BS, NM = d["t"], list(d["keep"]), d["lm"], d["bs"], list(d["bs_names"])
    box, kps, obj = d["box"], d["kps"], d["obj"]
    h = np.load(base + ".hands.npy")
    hj = np.clip(np.searchsorted(h[:, 0], T), 0, len(h) - 1)
    bel = np.load(base + ".belt.npy")
    fc = np.load(base + ".faces.npz")
    cap = cv2.VideoCapture(video)
    step = max(int(round(cap.get(cv2.CAP_PROP_FPS) / 10)), 1)
    i = k = bi = fi = 0
    while k < len(T):
        if i % step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        t = T[k]
        lm = np.full((468, 2), np.nan)
        lm[K] = LM[k]
        belt = np.nan
        while bi < len(bel) and bel[bi, 0] <= t + 1e-6:
            belt = bel[bi, 1]
            bi += 1
        faces = []
        while fi < len(fc["t"]) and fc["t"][fi] <= t + 1e-6:
            faces.append((fc["emb"][fi], bool(fc["good"][fi])))
            fi += 1
        yield t, frame, dict(lm=lm, mesh=not np.isnan(LM[k, 0, 0]), box=box[k], kps=kps[k],
                             blink=(BS[k, NM.index("eyeBlinkLeft")] + BS[k, NM.index("eyeBlinkRight")]) / 2,
                             down=(BS[k, NM.index("eyeLookDownLeft")] + BS[k, NM.index("eyeLookDownRight")]) / 2,
                             hmouth=h[hj[k], 1], hear=h[hj[k], 2], phone=obj[k, 0], phone_run=k % 3 == 0,
                             belt=belt, faces=faces)
        k += 1


def run_session(segments, gallery=None):
    """Replay `segments` ("video:cache_dir") as one session. Yields, for each processed frame (10 fps):
    (t, annotated_frame, events, status) where events is a list of (event, details) raised on this
    frame and status is the current driver state (for telemetry)."""
    gal = frs.Gallery(gallery) if gallery else None
    ident = identity.DriverIdentity(gal)
    eng = engine2.Engine2()
    banner, until, offset = "", -1, 0.0
    eye_hist = []
    for seg in segments:
        video, cache = seg.rsplit(":", 1)
        base = os.path.join(cache, os.path.splitext(os.path.basename(video))[0])
        last_t = 0.0
        for lt, frame, m in segment_frames(video, base):
            t = offset + lt
            last_t = lt
            b, kp, lm = m["box"], m["kps"], m["lm"]
            face = not np.isnan(b[0])
            # ---- identity ----
            id_events = []
            if not face:
                ident.no_face(t)
            for e, good in m["faces"]:
                id_events += ident.update(t, e, good)
            confirmed = any(ev in ("DRIVER_VERIFIED", "UNKNOWN_DRIVER") for ev, _, _ in id_events)
            if confirmed and getattr(eng, "driver", None) not in (None, ident.current):
                eng = engine2.Engine2()          # a different person: fresh calibration for them
                eye_hist = []
            if not face:
                eye_hist = []
            if ident.current is not None:
                eng.driver = ident.current
            # ---- DMS ----
            hidden = False
            if m["mesh"]:
                vis, _ = dms.eyes_visible(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), lm)
                eye_hist = (eye_hist + [vis])[-20:]
                hidden = sum(eye_hist) * 2 <= len(eye_hist)
            le, re, no = kp[0], kp[1], kp[2]
            eyew = re[0] - le[0]
            f = dict(face=face, mesh=m["mesh"],
                     yawp=((no[0] - le[0]) / (eyew + 1e-6) - 0.5) * 2 if face else np.nan,
                     eyespan=eyew / (b[2] - b[0] + 1e-6) if face else np.nan,
                     blink=m["blink"], down=m["down"], eyes_hidden=hidden,
                     hmouth=m["hmouth"], hear=m["hear"], phone_run=m["phone_run"], phone_box=m["phone"],
                     face_box=b, fw=b[2] - b[0], frame_w=frame.shape[1], belt=m["belt"])
            evs = [(e, "") for e in eng.update(t, f)]
            evs += [(e, f"{lab} ({sim:.2f})" if e != "DRIVER_CHANGED" else lab) for e, lab, sim in id_events]
            st = eng.state
            for e, info in evs:
                banner = e.replace("_", " ") + (f": {info}" if info else "")
                until = t + 3
            # ---- overlay ----
            alert = t < until
            if face:
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (0, 0, 255) if alert else (0, 220, 0), 4)
            ps, _ = getattr(eng, "phone_state", (False, False))
            if ps and not np.isnan(m["phone"][0]):
                o = m["phone"]
                cv2.rectangle(frame, (int(o[0]), int(o[1])), (int(o[2]), int(o[3])), (0, 165, 255), 4)
            who = ident.current or ("identifying..." if face else "-")
            lines = [f"Driver: {who}"]
            status = dict(driver=ident.current or "", driver_status=(
                "none" if ident.current is None else "unknown" if ident.current.startswith("Unknown")
                else "verified"), face=bool(st.get("face")))
            if st.get("face"):
                head = "facing road" if st["facing"] else "HEAD TURNED"
                eyes = ("hidden (sunglasses)" if st["eyes_hidden"] else "n/a (turned)" if not st["facing"] else
                        "CLOSED" if st["closed"] else "open" if not np.isnan(st["closure"]) else "n/a")
                hands = "at mouth" if st["hand_mouth"] else "at ear" if st["hand_ear"] else "-"
                belt = "ON" if st.get("belt") else "OFF" if st.get("belt") is False else "checking"
                lines += [f"Head: {head}", f"Eyes: {eyes}", f"Hand: {hands}   Phone: {'yes' if ps else '-'}",
                          f"Tiredness (PERCLOS 60s): {st['perclos'] * 100:.0f}%", "Seat belt: " + belt]
                status.update(head="facing" if st["facing"] else "turned",
                              eyes=eyes.split(" ")[0].lower(), hand=hands.replace("at ", "") if hands != "-" else "none",
                              phone=bool(ps), perclos=round(float(st["perclos"]) * 100, 1),
                              seatbelt={"ON": "on", "OFF": "off"}.get(belt, "unknown"))
            else:
                lines += ["No driver face"]
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40 + 46 * len(lines)), (0, 0, 0), -1)
            for j, s in enumerate(lines):
                cv2.putText(frame, s, (16, 48 + 46 * j), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 255) if j == 0 else (255, 255, 255), 3)
            cv2.putText(frame, f"t={int(t // 60):02d}:{t % 60:04.1f}", (16, frame.shape[0] - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
            if alert:
                cv2.rectangle(frame, (0, frame.shape[0] - 90), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                cv2.putText(frame, banner[:46], (16, frame.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (255, 255, 255), 4)
            yield t, frame, evs, status
        offset += last_t + 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segments", nargs="+", help="video.mov:cache_dir")
    ap.add_argument("--gallery")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    writer, events, size = None, [], None
    for t, frame, evs, status in run_session(a.segments, a.gallery):
        for e, info in evs:
            events.append((t, e, info, status["driver"]))
        if size is None:
            size = (frame.shape[1] // 2, frame.shape[0] // 2)
        small = cv2.resize(frame, size)
        if writer is None:
            writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
        writer.write(small)
    writer.release()
    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session_time", "seconds", "event", "details", "current_driver"])
        for t, e, info, who in events:
            w.writerow([f"{int(t // 60):02d}:{t % 60:04.1f}", round(t, 1), e, info, who])
    print(a.out, len(events), "events")


if __name__ == "__main__":
    main()
