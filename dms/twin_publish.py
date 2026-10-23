#!/usr/bin/env python3
"""
Replay a digital-twin stream into ThingsBoard, frame by frame in real time, exactly as the bus unit
will send it. Use it to try the "Cabin driver twin" widget before the hardware exists.

  python3 twin_publish.py results/v0112_twin.json --host tb.example.com --token DEVICE_TOKEN
  python3 twin_publish.py results/v0112_twin.json --host 192.168.1.20:8080 --token ... --http --start 160 --speed 1

Default transport is MQTT (port 1883, needs `pip install paho-mqtt`); --http uses the device HTTP API
(no extra packages). Each frame goes out as telemetry {"twin": {...}} plus a few flat keys
(dms_alert, dms_belt, dms_phone, dms_eyes_closed, dms_perclos, driver) that ordinary ThingsBoard
widgets, rule chains and alarms can use directly.

ThingsBoard notes
- 10 messages/s per device. If the tenant profile has transport rate limits, allow at least
  20 messages per second per device (or use --rate 5: the widget smooths between frames).
- Timestamps are "now", so the widget's realtime window shows them live.
"""
import argparse
import http.client
import json
import sys
import time


def flat(f):
    return {
        "twin": {k: v for k, v in f.items() if k != "t"},
        "dms_alert": f.get("alert", ""),
        "dms_belt": f.get("belt", -1),
        "dms_phone": f.get("phone", 0),
        "dms_eyes_closed": f.get("closed", 0),
        "dms_perclos": f.get("perclos", 0),
        "driver": f.get("driver", ""),
    }


class HttpSender:
    def __init__(self, host, token, tls):
        h, _, p = host.partition(":")
        self.conn_args = (h, int(p) if p else (443 if tls else 80))
        self.cls = http.client.HTTPSConnection if tls else http.client.HTTPConnection
        self.path = f"/api/v1/{token}/telemetry"
        self.c = None

    def send(self, payload):
        for attempt in (1, 2):
            try:
                if self.c is None:
                    self.c = self.cls(*self.conn_args, timeout=5)
                self.c.request("POST", self.path, body=json.dumps(payload, separators=(",", ":")),
                               headers={"Content-Type": "application/json"})
                r = self.c.getresponse()
                r.read()
                if r.status >= 300:
                    sys.exit(f"ThingsBoard answered HTTP {r.status}: check the host and the device access token")
                return
            except (OSError, http.client.HTTPException):
                self.c = None
                if attempt == 2:
                    raise

    def close(self):
        if self.c:
            self.c.close()


class MqttSender:
    def __init__(self, host, token, tls):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            sys.exit("MQTT needs paho-mqtt (pip install paho-mqtt), or run with --http")
        h, _, p = host.partition(":")
        try:
            self.cl = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:                       # paho-mqtt 1.x
            self.cl = mqtt.Client()
        self.cl.username_pw_set(token)
        if tls:
            self.cl.tls_set()
        import threading
        self.ready, self.rc = threading.Event(), None

        def on_connect(cl, userdata, flags, rc, props=None):
            self.rc = rc
            self.ready.set()
        self.cl.on_connect = on_connect
        self.cl.connect(h, int(p) if p else (8883 if tls else 1883), keepalive=30)
        self.cl.loop_start()
        if not self.ready.wait(10):
            sys.exit("no answer from the MQTT broker: check host / port 1883 is open")
        if self.rc != 0:
            sys.exit(f"ThingsBoard refused the connection ({self.rc}): check the device access token")

    def send(self, payload):
        r = self.cl.publish("v1/devices/me/telemetry", json.dumps(payload, separators=(",", ":")), qos=1)
        if r.rc != 0:
            print("  publish failed, reconnecting ...", flush=True)

    def close(self):
        time.sleep(0.5)                              # let the last messages go out
        self.cl.loop_stop()
        self.cl.disconnect()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream", help="*_twin.json from process.py --twin")
    ap.add_argument("--host", required=True, help="ThingsBoard host[:port]")
    ap.add_argument("--token", required=True, help="device access token (Device -> Copy access token)")
    ap.add_argument("--http", action="store_true", help="use the HTTP device API instead of MQTT")
    ap.add_argument("--tls", action="store_true", help="HTTPS / MQTTS")
    ap.add_argument("--start", type=float, default=0, help="start at this second of the recording")
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=10, help="frames per second to send (10 = every frame)")
    ap.add_argument("--loop", action="store_true", help="start again at the end")
    a = ap.parse_args()

    frames = [f for f in json.load(open(a.stream))["frames"] if a.start <= f["t"] <= a.end]
    if not frames:
        sys.exit("no frames in that time range")
    step = max(1, round(10 / a.rate))
    frames = frames[::step]
    tx = HttpSender(a.host, a.token, a.tls) if a.http else MqttSender(a.host, a.token, a.tls)
    print(f"sending {len(frames)} frames ({frames[0]['t']:.1f}-{frames[-1]['t']:.1f} s) "
          f"to {a.host} over {'HTTP' if a.http else 'MQTT'}; Ctrl+C to stop", flush=True)
    try:
        while True:
            t0, w0 = frames[0]["t"], time.monotonic()
            last_alert = None
            for f in frames:
                wait = w0 + (f["t"] - t0) / a.speed - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                tx.send(flat(f))
                if f.get("alert") and f["alert"] != last_alert:
                    print(f"  {f['t']:7.1f} s  {f['alert']}", flush=True)
                last_alert = f.get("alert")
            if not a.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        tx.close()
    print("done")


if __name__ == "__main__":
    main()
