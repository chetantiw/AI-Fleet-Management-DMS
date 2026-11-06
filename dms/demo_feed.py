#!/usr/bin/env python3
"""
Demo vehicle for the fleet platform, from already-processed recordings, until real hardware runs.

Plays the annotated DMS videos (e.g. v3_IMG_0107_annotated.mp4) in real time, one after another, in
a loop, and feeds them to the fleet agent exactly like the live Pi app does:
  - each alert from the review CSV is sent at its moment in the video (with a snapshot of that frame)
    -> ThingsBoard alarms and the Events table;
  - driver state (driver, seat belt, phone, eyes, tiredness) follows the alerts;
  - position from the demo route; live video on the dashboard button shows the annotated video.

  python3 demo_feed.py --config demo.device.json --csv v3_alerts_for_review.csv --videos /path/to/videos

CSV columns used (the "alerts for review" export): video, driver, video_time (mm:ss.s), alert.
The vehicle is marked as a demo in ThingsBoard (client attribute demo_mode = true).
--local makes MQTT and the video push go to 127.0.0.1 (when it runs on the ThingsBoard server itself).
"""
import argparse
import csv
import glob
import logging
import os
import re
import time

import cv2

import fleet_agent
from tb_link import load_config

log = logging.getLogger("demo_feed")

PHONE = ("PHONE_CALL", "PHONE_IN_HAND")


def mmss(s):
    m, sec = s.strip().split(":")
    return int(m) * 60 + float(sec)


def load_playlist(csv_path, video_dir):
    clips = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            vid = (row.get("video") or "").strip()
            alert = (row.get("alert") or "").strip().replace("\\_", "_")
            if not vid or not alert:
                continue
            c = clips.setdefault(vid, {"driver": (row.get("driver") or vid).split(",")[0].strip(), "events": []})
            c["events"].append((mmss(row["video_time"]), alert))
    out = []
    for vid, c in clips.items():
        files = sorted(glob.glob(os.path.join(video_dir, f"*{vid}*annotated*.mp4"))) or \
            sorted(glob.glob(os.path.join(video_dir, f"*{vid}*.mp4")))
        if not files:
            log.warning("no video for %s in %s, skipped", vid, video_dir)
            continue
        out.append(dict(name=vid, path=files[0], driver=c["driver"], events=sorted(c["events"])))
    if not out:
        raise SystemExit("no clips: check --csv and --videos")
    return out


def near(events, t, names, before, after):
    return any(n in names and e - before <= t <= e + after for e, n in events)


def status_at(clip, t):
    ev = clip["events"]
    absent = near(ev, t, ("DRIVER_ABSENT",), 3, 3)
    s = dict(driver=clip["driver"], driver_status="verified", face=not absent)
    if absent:
        return s
    calling = near(ev, t, ("PHONE_CALL",), 4, 4)
    drinking = near(ev, t, ("DRINKING",), 3, 2)
    perclos = 6.0
    for e, n in ev:                 # tiredness rises at fatigue / eyes-closed alerts, then decays
        if n in ("FATIGUE_PERCLOS", "DROWSY_EYES_CLOSED") and e - 8 <= t:
            perclos = max(perclos, 6 + 24 * max(0.0, 1 - max(t - e, 0) / 60) * min(1, (t - e + 8) / 8))
    s.update(head="turned" if near(ev, t, ("DISTRACTED_LOOK_AWAY",), 2, 1) else "facing",
             eyes="closed" if near(ev, t, ("DROWSY_EYES_CLOSED",), 2, 2) else "open",
             hand="ear" if calling else "mouth" if drinking else "none",
             phone=near(ev, t, PHONE, 4, 4),
             perclos=round(perclos, 1),
             # the DMS raises NO_SEATBELT after 10 s unbuckled, then reminds every 60 s
             seatbelt="off" if near(ev, t, ("NO_SEATBELT",), 10, 62) else "unknown" if t < 10 else "on")
    return s


def localize(cfg):
    cfg["thingsboard"]["host"] = "127.0.0.1"
    st = cfg.get("stream", {})
    if st.get("publish_url"):
        st["publish_url"] = re.sub(r"@[^:/]+", "@127.0.0.1", st["publish_url"], count=1)


def keep_streaming(agent):
    st = agent.stream
    if st is not None and st.status()["seconds_left"] < 120:
        st.start(60)
        agent.link.attributes({"stream_active": True})


def play(agent, clip, prev_driver, always=False):
    cap = cv2.VideoCapture(clip["path"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    step = max(int(round(fps / 10)), 1)
    pending = list(clip["events"])
    t0, i = time.time(), 0
    log.info("playing %s (%s, %d alerts)", os.path.basename(clip["path"]), clip["driver"], len(pending))
    first = True
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        t = i / fps
        wait = t0 + t - time.time()
        if wait > 0:
            time.sleep(wait)
        evs = []
        if first and t >= 1.0:
            first = False
            evs.append(("DRIVER_VERIFIED", f"{clip['driver']} (demo)"))
            if prev_driver and prev_driver != clip["driver"]:
                evs.append(("DRIVER_CHANGED", clip["driver"]))
        while pending and pending[0][0] <= t:
            evs.append((pending.pop(0)[1], "recorded demo"))
        if always and i % 100 == 0:
            keep_streaming(agent)
        agent.frame(t0 + t, frame, evs, status_at(clip, t))
    cap.release()
    return clip["driver"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="device.json of the demo vehicle")
    ap.add_argument("--csv", required=True, help="alerts CSV (video, driver, video_time, alert)")
    ap.add_argument("--videos", required=True, help="folder with the annotated videos")
    ap.add_argument("--local", action="store_true", help="ThingsBoard and media server on this machine")
    ap.add_argument("--once", action="store_true", help="play the list once instead of looping")
    ap.add_argument("--outbox", help="where to keep the offline buffer (default: from the config)")
    ap.add_argument("--stream-always", action="store_true",
                    help="keep the live video on all the time (demo on the server only: no SIM data)")
    ap.add_argument("--no-gps", action="store_true",
                    help="send no position (when something else already sends this bus's location)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    clips = load_playlist(a.csv, a.videos)
    cfg = load_config(a.config)
    if a.local:
        localize(cfg)
    if a.outbox:
        cfg["outbox"] = a.outbox
    if a.no_gps:
        cfg["gnss"] = {"type": "none"}
    agent = fleet_agent.Agent(cfg)
    agent.start()
    agent.link.attributes({"demo_mode": True,
                           "demo_note": "Replay of recorded bus videos (DMS processed offline)"})
    driver = None
    try:
        while True:
            for clip in clips:
                driver = play(agent, clip, driver, a.stream_always)
            if a.once:
                break
    except KeyboardInterrupt:
        pass
    finally:
        agent.close()


if __name__ == "__main__":
    main()
