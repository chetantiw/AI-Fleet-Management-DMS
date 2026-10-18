#!/usr/bin/env python3
"""
Replay a video with the v2 alert engine, using measurements cached by extract.py and
hands_extract.py, and write an annotated video + event list.

  python3 render_v2.py video.mov features.npz hands.npy out.mp4 events.csv

Overlay: face box (green normal, red during an alert), a status panel (head, eyes, hands,
phone), the accepted phone box, and the alert banner. No landmark dots over the eyes.
"""
import csv
import sys

import cv2
import numpy as np

import dms
import engine2


def main(video, xf, hf, out_mp4, out_csv, bf=None):
    d = np.load(xf)
    T, K, LM, BS, NM = d["t"], list(d["keep"]), d["lm"], d["bs"], list(d["bs_names"])
    box, kps, obj = d["box"], d["kps"], d["obj"]
    h = np.load(hf)
    hj = np.clip(np.searchsorted(h[:, 0], T), 0, len(h) - 1)
    eng = engine2.Engine2()
    bel = np.load(bf) if bf else None
    bel_i = 0
    cap = cv2.VideoCapture(video)
    src = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(round(src / 10)), 1)
    writer, events, banner, until = None, [], "", -1
    eye_hist = []
    i = k = 0
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
        mesh = not np.isnan(LM[k, 0, 0])
        b, kp = box[k], kps[k]
        face = not np.isnan(b[0])
        hidden = False
        if mesh:
            vis, _ = dms.eyes_visible(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), lm)
            eye_hist = (eye_hist + [vis])[-20:]
            hidden = sum(eye_hist) * 2 <= len(eye_hist)
        le, re, no = kp[0], kp[1], kp[2]
        eyew = re[0] - le[0]
        f = dict(face=face, mesh=mesh,
                 yawp=((no[0] - le[0]) / (eyew + 1e-6) - 0.5) * 2 if face else np.nan,
                 eyespan=eyew / (b[2] - b[0] + 1e-6) if face else np.nan,
                 blink=(BS[k, NM.index("eyeBlinkLeft")] + BS[k, NM.index("eyeBlinkRight")]) / 2,
                 down=(BS[k, NM.index("eyeLookDownLeft")] + BS[k, NM.index("eyeLookDownRight")]) / 2,
                 eyes_hidden=hidden, hmouth=h[hj[k], 1], hear=h[hj[k], 2],
                 phone_run=k % 3 == 0, phone_box=obj[k, 0], face_box=b, fw=b[2] - b[0],
                 frame_w=frame.shape[1])
        f["belt"] = np.nan
        if bel is not None:
            while bel_i < len(bel) and bel[bel_i, 0] <= t + 1e-6:
                f["belt"] = bel[bel_i, 1]
                bel_i += 1
        evs = eng.update(t, f)
        st = eng.state
        for e in evs:
            events.append((t, e))
            banner, until = e.replace("_", " "), t + 3
        # ---- overlay ----
        alert = t < until
        if face:
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255) if alert else (0, 220, 0), 4)
        ps, _ = getattr(eng, "phone_state", (False, False))
        if ps and not np.isnan(obj[k, 0, 0]):
            o = obj[k, 0]
            cv2.rectangle(frame, (int(o[0]), int(o[1])), (int(o[2]), int(o[3])), (0, 165, 255), 4)
        if st.get("face"):
            head = "facing road" if st["facing"] else "HEAD TURNED"
            eyes = ("hidden (sunglasses)" if st["eyes_hidden"] else
                    "n/a (turned)" if not st["facing"] else
                    "CLOSED" if st["closed"] else "open" if not np.isnan(st["closure"]) else "n/a")
            hands = "at mouth" if st["hand_mouth"] else "at ear" if st["hand_ear"] else "-"
            lines = [f"Head: {head}", f"Eyes: {eyes}", f"Hand: {hands}   Phone: {'yes' if ps else '-'}",
                     f"Tiredness (PERCLOS 60s): {st['perclos'] * 100:.0f}%",
                     "Seat belt: " + ("ON" if st.get("belt") else "OFF" if st.get("belt") is False else "checking")]
        else:
            lines = ["No driver face"]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40 + 46 * len(lines)), (0, 0, 0), -1)
        for j, s in enumerate(lines):
            cv2.putText(frame, s, (16, 48 + 46 * j), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, f"t={int(t // 60):02d}:{t % 60:04.1f}", (16, frame.shape[0] - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
        if alert:
            cv2.rectangle(frame, (0, frame.shape[0] - 90), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            cv2.putText(frame, banner, (16, frame.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 4)
        small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        if writer is None:
            writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 10, small.shape[1::-1])
        writer.write(small)
        k += 1
    writer.release()
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video_time", "seconds", "alert"])
        for t, e in events:
            w.writerow([f"{int(t // 60):02d}:{t % 60:04.1f}", round(t, 1), e])
    print(out_mp4, len(events), "alerts")


if __name__ == "__main__":
    main(*sys.argv[1:7])
