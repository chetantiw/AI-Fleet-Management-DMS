"""
Link from the in-vehicle device to ThingsBoard CE over MQTT (device API).

  link = TbLink(host="iot.example.com", token="<device access token>")
  link.on_rpc("stream_start", handler)          # dashboard button -> device
  link.start()
  link.telemetry({"latitude": 22.71, "longitude": 75.85, "speed": 42})
  link.event("PHONE_CALL", driver="Ramesh Kumar", details="...", lat=..., lon=...)
  link.attributes({"fw_version": "0.3"})

Design points
- Every message goes into a small SQLite outbox first and is deleted only when ThingsBoard
  acknowledges it (MQTT QoS 1 PUBACK). With no 4G coverage the device keeps working and the
  outbox fills; after reconnection it is sent oldest-first, each record carrying its own original
  timestamp ("ts"), so the history on the server is correct.
- Telemetry topics:  v1/devices/me/telemetry   (time series)
                     v1/devices/me/attributes  (client attributes: versions, stream URL, ...)
  RPC from the dashboard: v1/devices/me/rpc/request/<id>  -> reply on .../response/<id>
  Shared attributes (settings pushed from the server): v1/devices/me/attributes
- Thread-safe: call from the video loop; the network runs on paho's own thread.
"""
import json
import logging
import os
import sqlite3
import ssl
import threading
import time

import paho.mqtt.client as mqtt

log = logging.getLogger("tb_link")

TELEMETRY = "v1/devices/me/telemetry"
ATTRIBUTES = "v1/devices/me/attributes"
ATTR_REQUEST = "v1/devices/me/attributes/request/"
ATTR_RESPONSE = "v1/devices/me/attributes/response/+"
RPC_REQUEST = "v1/devices/me/rpc/request/+"
RPC_RESPONSE = "v1/devices/me/rpc/response/"
# ThingsBoard's MQTT transport drops the connection on messages over 64 KB (MQTT_MAX_PAYLOAD_SIZE);
# anything bigger is refused here instead, so one oversized message can never block the link.
MAX_PAYLOAD = 60_000

# Severity used for the alarm each DMS event raises on the server (rule chain reads "severity").
SEVERITY = {
    "DROWSY_EYES_CLOSED": "CRITICAL",
    "FATIGUE_PERCLOS": "MAJOR",
    "DISTRACTED_LOOK_AWAY": "MINOR",
    "PHONE_CALL": "MAJOR",
    "PHONE_IN_HAND": "MAJOR",
    "DRINKING": "MINOR",
    "NO_SEATBELT": "MAJOR",
    "DRIVER_ABSENT": "WARNING",
    "UNKNOWN_DRIVER": "MAJOR",
    "DRIVER_VERIFIED": None,      # information only, no alarm
    "DRIVER_CHANGED": None,
    "ALCOHOL_FAIL": "CRITICAL",
    "ALCOHOL_TEST_INVALID": "MAJOR",
    "ALCOHOL_PASS": None,
    "CAMERA_FAULT": "MAJOR",
    "OVERSPEED": "MAJOR",
}


class Outbox:
    """Persistent FIFO of (topic, payload) waiting for ThingsBoard's acknowledgement."""

    def __init__(self, path, max_rows=200_000):
        self.max_rows = max_rows
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS q (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "topic TEXT, payload TEXT, prio INTEGER DEFAULT 0)")
        self.lock = threading.Lock()

    def put(self, topic, payload, prio=0):
        if len(payload) > MAX_PAYLOAD:
            log.error("dropping %d-byte message on %s (limit %d)", len(payload), topic, MAX_PAYLOAD)
            return
        with self.lock:
            self.db.execute("INSERT INTO q (topic, payload, prio) VALUES (?,?,?)", (topic, payload, prio))
            n = self.db.execute("SELECT COUNT(*) FROM q").fetchone()[0]
            if n > self.max_rows:   # storage full: drop the oldest routine telemetry, never events
                self.db.execute("DELETE FROM q WHERE id IN (SELECT id FROM q WHERE prio=0 ORDER BY id LIMIT ?)",
                                (n - self.max_rows,))

    def head(self, after_id, limit):
        with self.lock:
            return self.db.execute("SELECT id, topic, payload FROM q WHERE id > ? ORDER BY id LIMIT ?",
                                   (after_id, limit)).fetchall()

    def delete(self, row_id):
        with self.lock:
            self.db.execute("DELETE FROM q WHERE id = ?", (row_id,))

    def __len__(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM q").fetchone()[0]


class TbLink:
    def __init__(self, host, token, port=None, tls=False, ca_file=None, outbox="tb_outbox.db",
                 client_id=None, shared_keys=(), in_flight=20):
        self.host, self.token, self.tls = host, token, tls
        self.port = port or (8883 if tls else 1883)
        self.box = Outbox(outbox)
        self.rpc = {}
        self.shared = {}
        self.shared_keys = list(shared_keys)
        self.on_shared = None                     # callback(dict of changed shared attributes)
        self.connected = False
        self.in_flight = in_flight
        self.pending = {}                         # mid -> outbox row id
        self.sent_upto = 0                        # highest outbox id handed to paho
        self.plock = threading.Lock()
        self.wake = threading.Event()
        self.stop_ev = threading.Event()
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id or "",
                             protocol=mqtt.MQTTv311)
        self.c.username_pw_set(token)
        if tls:
            self.c.tls_set(ca_certs=ca_file, cert_reqs=ssl.CERT_REQUIRED if ca_file else ssl.CERT_NONE)
            if not ca_file:
                self.c.tls_insecure_set(True)
        self.c.reconnect_delay_set(1, 60)
        self.c.on_connect = self._on_connect
        self.c.on_disconnect = self._on_disconnect
        self.c.on_publish = self._on_publish
        self.c.on_message = self._on_message
        self.sender = threading.Thread(target=self._send_loop, daemon=True)

    # ---------------------------------------------------------------- public API
    def start(self):
        self.c.connect_async(self.host, self.port, keepalive=30)
        self.c.loop_start()
        self.sender.start()

    def stop(self, flush_s=5.0):
        end = time.time() + flush_s
        while len(self.box) and self.connected and time.time() < end:
            time.sleep(0.1)
        self.stop_ev.set()
        self.wake.set()
        self.c.disconnect()
        self.c.loop_stop()

    def on_rpc(self, method, fn):
        """fn(params) -> JSON-able result (returned to the dashboard), or raise to report an error."""
        self.rpc[method] = fn

    def telemetry(self, values, ts=None, prio=0):
        """values: dict of key -> number/string/bool. ts in seconds (default now)."""
        rec = {"ts": int((ts or time.time()) * 1000), "values": values}
        self.box.put(TELEMETRY, json.dumps(rec, separators=(",", ":")), prio)
        self.wake.set()

    def event(self, name, ts=None, driver="", details="", severity="auto", **extra):
        """One DMS / alcohol / vehicle event. The server's rule chain turns events with a severity
        into alarms (type = event name) and keeps all of them as time series for history."""
        sev = SEVERITY.get(name) if severity == "auto" else severity
        v = {"event": name, "event_driver": driver or "", "event_details": details or ""}
        if sev:
            v["severity"] = sev
        v.update({k: x for k, x in extra.items() if x is not None})
        self.telemetry(v, ts, prio=1)

    def attributes(self, values):
        self.box.put(ATTRIBUTES, json.dumps(values, separators=(",", ":")), 1)
        self.wake.set()

    @property
    def backlog(self):
        return len(self.box)

    # ---------------------------------------------------------------- MQTT callbacks
    def _on_connect(self, c, ud, flags, rc, props=None):
        if rc != 0:
            log.warning("ThingsBoard refused connection: %s (check the access token)", rc)
            return
        log.info("connected to ThingsBoard %s:%s", self.host, self.port)
        c.subscribe([(RPC_REQUEST, 1), (ATTRIBUTES, 1), (ATTR_RESPONSE, 1)])
        if self.shared_keys:      # fetch current settings once on every (re)connect
            c.publish(ATTR_REQUEST + "1", json.dumps({"sharedKeys": ",".join(self.shared_keys)}), 1)
        with self.plock:          # anything un-acknowledged from the last connection is resent
            self.pending.clear()
            self.sent_upto = 0
        self.connected = True
        self.wake.set()

    def _on_disconnect(self, c, ud, flags, rc, props=None):
        if self.connected:
            log.warning("disconnected from ThingsBoard (%s); buffering locally", rc)
        self.connected = False

    def _on_publish(self, c, ud, mid, rc=None, props=None):
        with self.plock:
            row = self.pending.pop(mid, None)
        if row is not None:
            self.box.delete(row)
        self.wake.set()

    def _on_message(self, c, ud, msg):
        try:
            data = json.loads(msg.payload or b"{}")
        except ValueError:
            return
        if msg.topic.startswith("v1/devices/me/rpc/request/"):
            threading.Thread(target=self._handle_rpc, args=(msg.topic.rsplit("/", 1)[1], data),
                             daemon=True).start()
        elif msg.topic.startswith("v1/devices/me/attributes/response/"):
            self._shared_update(data.get("shared", {}))
        elif msg.topic == ATTRIBUTES:
            self._shared_update(data)

    def _shared_update(self, d):
        if not d:
            return
        self.shared.update(d)
        if self.on_shared:
            try:
                self.on_shared(d)
            except Exception:
                log.exception("shared attribute handler failed")

    def _handle_rpc(self, req_id, data):
        method, params = data.get("method"), data.get("params")
        fn = self.rpc.get(method)
        try:
            res = fn(params) if fn else {"error": f"unknown method {method}"}
        except Exception as e:  # report the failure to the dashboard instead of timing out
            log.exception("rpc %s failed", method)
            res = {"error": str(e)}
        out = json.dumps(res if res is not None else {"ok": True})
        if len(out) > MAX_PAYLOAD:
            out = json.dumps({"error": f"reply too large ({len(out)} bytes)"})
        self.c.publish(RPC_RESPONSE + req_id, out, 1)

    # ---------------------------------------------------------------- sender
    def _send_loop(self):
        while not self.stop_ev.is_set():
            self.wake.wait(1.0)
            self.wake.clear()
            if not self.connected:
                continue
            with self.plock:
                room = self.in_flight - len(self.pending)
                after = self.sent_upto
            if room <= 0:
                continue
            for row_id, topic, payload in self.box.head(after, room):
                info = self.c.publish(topic, payload, qos=1)
                if info.rc != mqtt.MQTT_ERR_SUCCESS:
                    break
                with self.plock:
                    self.pending[info.mid] = row_id
                    self.sent_upto = row_id
                # a fast ack can arrive before pending[] is filled; check for that race
                if info.is_published():
                    self._on_publish(None, None, info.mid)


def load_config(path):
    """Device config (JSON). Environment variables TB_HOST / TB_TOKEN override the file."""
    with open(path) as fh:
        cfg = json.load(fh)
    cfg.setdefault("thingsboard", {})
    if os.environ.get("TB_HOST"):
        cfg["thingsboard"]["host"] = os.environ["TB_HOST"]
    if os.environ.get("TB_TOKEN"):
        cfg["thingsboard"]["token"] = os.environ["TB_TOKEN"]
    return cfg
