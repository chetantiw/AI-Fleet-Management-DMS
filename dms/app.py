#!/usr/bin/env python3
"""
Live Driver Monitoring System for Raspberry Pi 5 + AI HAT+ (Hailo-8L) + Pi camera.

  python3 app.py                       # uses config.json next to this file
  python3 app.py --source video.mov    # test on a recorded video instead of the camera
  python3 app.py --source 0            # USB webcam
  python3 app.py --show                # also show a window on the Pi's own screen

Open http://<pi-address>:8080 on a phone/laptop on the same network for the live view.
Events go to <data_dir>/events/YYYY-MM-DD.jsonl, an annotated photo of each alert to
<data_dir>/snapshots/, optional annotated recordings to <data_dir>/recordings/.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import signal
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import engine2  # noqa: E402
import frs  # noqa: E402
import identity  # noqa: E402
import overlay  # noqa: E402
import perception  # noqa: E402
import sources  # noqa: E402

ALERTS_BEEP = {"DROWSY_EYES_CLOSED": (0.6, 3), "FATIGUE_PERCLOS": (0.6, 2), "DISTRACTED_LOOK_AWAY": (0.2, 2),
               "PHONE_CALL": (0.2, 3), "PHONE_IN_HAND": (0.2, 3), "DRINKING": (0.2, 1),
               "NO_SEATBELT": (0.3, 2), "UNKNOWN_DRIVER": (0.1, 1)}


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load_config(path):
    """config.json (shipped defaults) + config.local.json next to it (this vehicle's own settings,
    never overwritten by `git pull`)."""
    with open(path) as fh:
        cfg = json.load(fh)
    local = os.path.join(os.path.dirname(os.path.abspath(path)), "config.local.json")
    if os.path.exists(local):
        with open(local) as fh:
            _merge(cfg, json.load(fh))
    base = os.path.dirname(os.path.abspath(path))
    for k in ("models_dir", "data_dir", "gallery"):
        if k in cfg and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.normpath(os.path.join(base, cfg[k]))
    dc = cfg.get("fleet", {}).get("device_config")
    if dc and not os.path.isabs(dc):
        cfg["fleet"]["device_config"] = os.path.normpath(os.path.join(base, dc))
    return cfg


class Buzzer:
    def __init__(self, pin):
        self.b = None
        if pin is None:
            return
        try:
            from gpiozero import Buzzer as B
            self.b = B(int(pin))
        except Exception as e:
            print(f"buzzer on GPIO {pin} not available: {e}", flush=True)

    def beep(self, on_s, n):
        if self.b is None:
            return
        def run():
            for _ in range(n):
                self.b.on(); time.sleep(on_s); self.b.off(); time.sleep(0.15)
        threading.Thread(target=run, daemon=True).start()


def start_fleet_agent(path, log):
    """ThingsBoard link + on-demand live video (fleet_agent.py, docs/thingsboard_integration.md).
    Off when device.json is missing."""
    if not path or not os.path.exists(path):
        log("fleet platform: off (no device.json; see docs/thingsboard_integration.md)")
        return None
    try:
        import logging
        import fleet_agent
        from tb_link import load_config as tb_config
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
        cfg = tb_config(path)
        if not os.path.isabs(cfg.get("outbox", "tb_outbox.db")):
            cfg["outbox"] = os.path.join(os.path.dirname(os.path.abspath(path)), cfg.get("outbox", "tb_outbox.db"))
        agent = fleet_agent.Agent(cfg)
        agent.start()
        log(f"fleet platform: ThingsBoard {cfg['thingsboard'].get('host')} as {cfg.get('vehicle_id', '?')}")
        return agent
    except Exception as e:
        log(f"fleet platform not started: {type(e).__name__}: {e}")
        return None


def driver_status(st, ident, phone_seen):
    """Driver state for telemetry (same keys as render_session.run_session)."""
    cur = ident.current
    s = dict(driver=cur or "", driver_status="none" if cur is None else
             "unknown" if cur.startswith("Unknown") else "verified", face=bool(st.get("face")))
    if st.get("face"):
        s.update(head="facing" if st["facing"] else "turned",
                 eyes="hidden" if st["eyes_hidden"] else "n/a" if not st["facing"] else
                 "closed" if st["closed"] else "open",
                 hand="mouth" if st["hand_mouth"] else "ear" if st["hand_ear"] else "none",
                 phone=bool(phone_seen), perclos=round(float(st["perclos"]) * 100, 1),
                 seatbelt={True: "on", False: "off"}.get(st.get("belt"), "unknown"))
    return s


class Recorder:
    """Annotated video in segments, oldest data deleted when the data folder passes max_gb."""
    def __init__(self, folder, fps, segment_min, max_gb, data_dir):
        self.folder, self.fps, self.seg, self.max_gb, self.data_dir = folder, fps, segment_min * 60, max_gb, data_dir
        os.makedirs(folder, exist_ok=True)
        self.w, self.started = None, 0

    def write(self, frame):
        now = time.time()
        if self.w is None or now - self.started > self.seg:
            if self.w is not None:
                self.w.release()
            self._prune()
            name = os.path.join(self.folder, dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4")
            self.w = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, frame.shape[1::-1])
            self.started = now
        self.w.write(frame)

    def _prune(self):
        files = []
        for root, _, fs in os.walk(self.data_dir):
            for f in fs:
                if f.endswith((".mp4", ".jpg")):
                    p = os.path.join(root, f)
                    files.append((os.path.getmtime(p), os.path.getsize(p), p))
        total = sum(s for _, s, _ in files)
        for _, s, p in sorted(files):
            if total <= self.max_gb * 1e9:
                break
            os.remove(p)
            total -= s

    def close(self):
        if self.w is not None:
            self.w.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--source", help="'picamera', a webcam index, or a video file (overrides config)")
    ap.add_argument("--show", action="store_true", help="show a window on the Pi's screen")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--record", action="store_true", help="save annotated video (overrides config)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this long (testing)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    data = cfg["data_dir"]
    for sub in ("events", "snapshots"):
        os.makedirs(os.path.join(data, sub), exist_ok=True)
    log = lambda *m: print(dt.datetime.now().strftime("%H:%M:%S"), *m, flush=True)

    # engine settings from config (anything in engine2.CFG can be overridden)
    for k, v in cfg.get("engine", {}).items():
        if k in engine2.CFG:
            engine2.CFG[k] = v

    src = sources.open_source(cfg["camera"], a.source)
    log("camera:", "Pi camera" if isinstance(src, sources.PiCamera) else a.source or cfg["camera"].get("source"))
    perc = perception.Perception(cfg["models_dir"], cfg.get("face_det_size", 320), cfg.get("hef"),
                                 cfg.get("use_hailo", True), log)
    gal = frs.Gallery(cfg["gallery"]) if os.path.exists(cfg["gallery"]) else None
    log("enrolled drivers:", sorted(set(gal.names)) if gal else "none (everyone will be 'Unknown #n')")
    ident = identity.DriverIdentity(gal)
    eng = engine2.Engine2()
    web = None
    if cfg.get("web", {}).get("enabled", True) and not a.no_web:
        import webview
        port = int(cfg.get("web", {}).get("port", 8080))
        web = webview.LiveView(port)
        log(f"live view on http://<this-pi>:{port}/")
    buzzer = Buzzer(cfg.get("buzzer_gpio"))
    agent = start_fleet_agent(cfg.get("fleet", {}).get("device_config"), log)
    rec_cfg = cfg.get("recording", {})
    rec = Recorder(os.path.join(data, "recordings"), rec_cfg.get("fps", 10), rec_cfg.get("segment_minutes", 5),
                   rec_cfg.get("max_gb", 20), data) if (a.record or rec_cfg.get("enabled")) else None

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *x: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *x: stop.update(now=True))

    banner, until, fps, last_tick, n, t_start = "", -1.0, 0.0, time.monotonic(), 0, None
    last_report, web_every = time.monotonic(), 0.0
    try:
        while not stop["now"]:
            frame, t = src.read()
            if frame is None:
                log("end of video")
                break
            t_start = t if t_start is None else t_start
            if a.max_seconds and t - t_start > a.max_seconds:
                break
            f, ex = perc(frame, t)

            # ---- who is driving ----
            id_events = []
            if not f["face"]:
                ident.no_face(t)
            elif ex["embedding"] is not None:
                id_events = ident.update(t, ex["embedding"], ex["embed_good"])
            confirmed = any(e in ("DRIVER_VERIFIED", "UNKNOWN_DRIVER") for e, _, _ in id_events)
            if confirmed and getattr(eng, "driver", None) not in (None, ident.current):
                eng = engine2.Engine2()                  # different person: fresh calibration
                perc.reset_driver()
            if ident.current is not None:
                eng.driver = ident.current

            # ---- alerts ----
            evs = [(e, "") for e in eng.update(t, f)]
            evs += [(e, f"{lab} ({sim:.2f})" if e != "DRIVER_CHANGED" else lab) for e, lab, sim in id_events]
            st = eng.state
            phone_seen, _ = getattr(eng, "phone_state", (False, False))
            for e, info in evs:
                banner, until = e.replace("_", " ") + (f": {info}" if info else ""), t + 3
            alert = t < until
            overlay.draw(frame, f, st, ident.current or ("identifying..." if f["face"] else "-"),
                         banner, alert, phone_seen, fps)
            now = dt.datetime.now()
            for e, info in evs:
                ev = dict(time=now.isoformat(timespec="seconds"), t=round(t, 1), event=e,
                          driver=ident.current or "", details=info)
                with open(os.path.join(data, "events", now.strftime("%Y-%m-%d") + ".jsonl"), "a") as fh:
                    fh.write(json.dumps(ev) + "\n")
                snap = os.path.join(data, "snapshots", now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{e}.jpg")
                cv2.imwrite(snap, cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                log(e, info, ident.current or "")
                if web:
                    web.add_event(ev)
                if e in ALERTS_BEEP:
                    buzzer.beep(*ALERTS_BEEP[e])

            # ---- outputs ----
            status = driver_status(st, ident, phone_seen)
            n += 1
            tick = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(tick - last_tick, 1e-3) if n > 1 else 0.0
            last_tick = tick
            small = None
            if rec is not None or web is not None or agent is not None:
                small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            if agent is not None:
                agent.frame(time.time(), small, evs, status)
            if rec is not None:
                rec.write(small)
            if web is not None and tick - web_every >= 0.1:
                web_every = tick
                ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                web.publish(jpg.tobytes(), dict(
                    driver=ident.current or "-", fps=round(fps, 1),
                    fleet_link=("connected" if agent.link.connected else f"offline, {agent.link.backlog} buffered")
                    if agent is not None else "off",
                    head=("facing road" if st.get("facing") else "turned") if st.get("face") else "no face",
                    eyes=("hidden" if st.get("eyes_hidden") else "closed" if st.get("closed") else "open")
                    if st.get("face") else "-",
                    seat_belt={True: "on", False: "OFF"}.get(st.get("belt"), "checking"),
                    phone="yes" if phone_seen else "-",
                    tiredness_pct=round(100 * st.get("perclos", 0)),
                    detector="AI HAT" if perc.on_hailo else "CPU",
                    stage_ms=perc.timings_ms()))
            if a.show:
                cv2.imshow("DMS", cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if tick - last_report >= 30:
                last_report = tick
                log(f"{fps:.1f} fps | driver {ident.current or '-'} | ms per stage {perc.timings_ms()}")
    finally:
        src.close()
        perc.close()
        if rec:
            rec.close()
        if web:
            web.close()
        if agent is not None:
            agent.close()
        log(f"stopped after {n} frames")


if __name__ == "__main__":
    main()
