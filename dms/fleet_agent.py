#!/usr/bin/env python3
"""
In-vehicle agent: connects the DMS to the fleet platform (ThingsBoard CE) and the media server.

What it sends to ThingsBoard (MQTT, see tb_link.py and docs/thingsboard_integration.md):
  every `telemetry_interval_s` (default 5 s): position + speed + driver state
  every 60 s:                               device health (CPU, memory, disk, temperature)
  at once:                                  every DMS / identity event, with position, speed,
                                            driver and a small snapshot picture
What the dashboard can ask it (RPC):
  stream_start {"minutes": 5}   live video to the media server (stops by itself)
  stream_stop
  snapshot                       current picture (JPEG, as a data URL)
  get_status                     driver state, position, stream and buffer status

Until the camera hardware is fitted the agent replays recorded, already-processed videos in
real time, exactly as the live pipeline will feed it:

  python3 fleet_agent.py --config device.json --replay bus.mov:results/cache [--loop]

or, to test only the link and the video (no driver monitoring), play any video as the camera:

  python3 fleet_agent.py --config device.json --video results/bus_annotated.mp4 --loop

On the device, the live DMS loop calls Agent.frame(t, frame, events, status) for each frame.
"""
import argparse
import base64
import json
import logging
import time

import cv2

import gnss
import health
from streamer import LiveStream
from tb_link import TbLink, load_config

log = logging.getLogger("fleet_agent")

FW_VERSION = "0.4.0"
DEFAULTS = dict(telemetry_interval_s=5.0, health_interval_s=60.0, stream_max_minutes=10,
                stream_width=640, stream_fps=10, stream_bitrate_k=400, event_snapshot=True,
                overspeed_kmh=60)


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.s = dict(DEFAULTS, **cfg.get("settings", {}))
        tb = cfg["thingsboard"]
        self.link = TbLink(tb["host"], tb["token"], port=tb.get("port"), tls=tb.get("tls", False),
                           ca_file=tb.get("ca_file"), outbox=cfg.get("outbox", "tb_outbox.db"),
                           shared_keys=list(DEFAULTS))
        self.link.on_shared = self._settings
        st = cfg.get("stream", {})
        self.stream = LiveStream(st["publish_url"], width=self.s["stream_width"], fps=self.s["stream_fps"],
                                 bitrate_k=self.s["stream_bitrate_k"],
                                 max_minutes=self.s["stream_max_minutes"]) if st.get("publish_url") else None
        g = cfg.get("gnss", {"type": "demo"})
        self.gps = gnss.NmeaGnss(g["port"], g.get("baud", 9600)) if g.get("type") == "nmea" else \
            gnss.RouteReplay(speed_kmh=g.get("speed_kmh", 32))
        self.last_tel = self.last_health = 0.0
        self.last_status = {}
        self.last_frame = None
        self.speeding = False
        for name, fn in (("stream_start", self._rpc_stream_start), ("stream_stop", self._rpc_stream_stop),
                         ("snapshot", self._rpc_snapshot), ("get_status", self._rpc_status)):
            self.link.on_rpc(name, fn)

    def start(self):
        self.link.start()
        st = self.cfg.get("stream", {})
        self.link.attributes({
            "vehicle_id": self.cfg.get("vehicle_id", ""), "fw_version": FW_VERSION,
            "dms_models": "scrfd500m+arcface-mbf+mediapipe+yolox", "stream_path": st.get("path", ""),
            "stream_view_url": st.get("view_url", ""), "stream_hls_url": st.get("hls_url", ""),
            "stream_active": False})
        health.snapshot()

    # ---------------------------------------------------------------- per frame
    def frame(self, t_wall, frame, events, status):
        """t_wall: epoch seconds of the frame. events: [(name, details)]. status: driver state dict."""
        self.last_frame = frame
        if self.stream is not None:
            was = self.stream.active
            self.stream.push(frame)
            if was and not self.stream.active and self.stream.until <= time.time():
                self.link.attributes({"stream_active": False})
        pos = self.gps.fix()
        self._speed_events(t_wall, pos, status)
        for name, details in events:
            extra = {k: pos.get(k) for k in ("latitude", "longitude", "speed")}
            if self.s["event_snapshot"]:
                extra["event_image"] = self._jpeg(frame, 320, 60, 24_000)
            self.link.event(name, ts=t_wall, driver=status.get("driver", ""), details=details, **extra)
        changed = any(status.get(k) != self.last_status.get(k) for k in ("driver", "seatbelt", "face"))
        if changed or t_wall - self.last_tel >= self.s["telemetry_interval_s"]:
            self.last_tel = t_wall
            self.last_status = status
            v = dict(pos, **{k: status.get(k) for k in ("driver", "driver_status", "face", "head", "eyes",
                                                          "hand", "phone", "perclos", "seatbelt")
                             if status.get(k) is not None})
            v["buffered"] = self.link.backlog
            self.link.telemetry(v, ts=t_wall)
        if t_wall - self.last_health >= self.s["health_interval_s"]:
            self.last_health = t_wall
            h = health.snapshot()
            h["streaming"] = bool(self.stream and self.stream.active)
            self.link.telemetry(h, ts=t_wall)

    def _speed_events(self, t, pos, status):
        fast = pos.get("speed", 0) > self.s["overspeed_kmh"]
        if fast and not self.speeding:
            self.link.event("OVERSPEED", ts=t, driver=status.get("driver", ""),
                            details=f"{pos['speed']:.0f} km/h", latitude=pos.get("latitude"),
                            longitude=pos.get("longitude"), speed=pos.get("speed"))
        self.speeding = fast

    # ---------------------------------------------------------------- RPC + settings
    def _rpc_stream_start(self, params):
        if self.stream is None:
            raise RuntimeError("no media server configured on this device")
        minutes = (params or {}).get("minutes", self.s["stream_max_minutes"]) if isinstance(params, dict) else \
            self.s["stream_max_minutes"]
        r = self.stream.start(minutes)
        self.link.attributes({"stream_active": True})
        r["view_url"] = self.cfg.get("stream", {}).get("view_url", "")
        return r

    def _rpc_stream_stop(self, params):
        if self.stream is None:
            return {"streaming": False}
        self.link.attributes({"stream_active": False})
        return self.stream.stop()

    def _rpc_snapshot(self, params):
        if self.last_frame is None:
            raise RuntimeError("no camera frame yet")
        return {"image": self._jpeg(self.last_frame, 480, 70, 50_000)}

    def _rpc_status(self, params):
        return dict(self.last_status, **self.gps.fix(), buffered=self.link.backlog,
                    stream=self.stream.status() if self.stream else None, fw_version=FW_VERSION)

    def _settings(self, d):
        for k, v in d.items():
            if k in DEFAULTS and v is not None:
                self.s[k] = type(DEFAULTS[k])(v)
        if self.stream is not None:
            self.stream.max_minutes = self.s["stream_max_minutes"]
            self.stream.bitrate_k = self.s["stream_bitrate_k"]
            self.stream.width = self.s["stream_width"]
        log.info("settings from server: %s", d)

    @staticmethod
    def _jpeg(frame, width, quality, max_chars):
        """Small JPEG as a data URL, shrunk until it fits in max_chars (MQTT message limit)."""
        h, w = frame.shape[:2]
        while True:
            small = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
            url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            if len(url) <= max_chars or width <= 160:
                return url
            width, quality = int(width * 0.8), max(quality - 10, 35)

    def close(self):
        if self.stream is not None:
            self.stream.stop()
        self.link.stop()


def replay(agent, segments, gallery, loop):
    """Feed processed recordings to the agent at real-time speed (10 fps)."""
    import render_session
    while True:
        t0 = time.time()
        n = 0
        for t, frame, evs, status in render_session.run_session(segments, gallery):
            wait = t0 + t - time.time()
            if wait > 0:
                time.sleep(wait)
            agent.frame(t0 + t, frame, evs, status)
            n += 1
            for e, info in evs:
                log.info("event %s %s", e, info)
        log.info("replay finished (%d frames)", n)
        if not loop:
            return


def play_video(agent, path, loop):
    """Link test without DMS processing: plays any video (e.g. an annotated output) as the camera,
    sending position, health and the live stream, but no driver state or events."""
    while True:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        step = max(int(round(fps / 10)), 1)
        t0, i = time.time(), 0
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
            agent.frame(t0 + t, frame, [], {})
        cap.release()
        if not loop:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="device.json (see device.example.json)")
    ap.add_argument("--replay", nargs="+", metavar="VIDEO:CACHE", help="processed videos to replay")
    ap.add_argument("--video", help="link test: play this video as the camera (no DMS processing)")
    ap.add_argument("--gallery", help="enrolled drivers (gallery.npz)")
    ap.add_argument("--loop", action="store_true", help="replay forever (bench demo)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    agent = Agent(load_config(a.config))
    agent.start()
    try:
        if a.replay:
            replay(agent, a.replay, a.gallery, a.loop)
        elif a.video:
            play_video(agent, a.video, a.loop)
        else:
            raise SystemExit("live camera mode runs inside the on-device DMS loop; use --replay or --video")
    except KeyboardInterrupt:
        pass
    finally:
        agent.close()


if __name__ == "__main__":
    main()
