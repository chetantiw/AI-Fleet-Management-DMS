#!/usr/bin/env python3
"""
Login check for the media server (MediaMTX "authMethod: http"), so that

- each vehicle can only PUBLISH its own stream (path = vehicle id, own password), and
- only people logged in to ThingsBoard can WATCH: the dashboard adds the viewer's ThingsBoard
  login token to the player URL (?token=...), and this service asks ThingsBoard whether that
  token is valid (and, if the user is a customer user, whether the vehicle belongs to them).

No browser password prompt, nothing to share with operators, and a stream link copied out of the
dashboard stops working when the ThingsBoard session ends.

  python3 stream_auth.py --tb http://127.0.0.1:8080 --vehicles vehicles.json --port 9100

vehicles.json:  {"veh-102": "publish-password", "veh-103": "..."}
Standard library only. Run it on the same server as MediaMTX (it listens on 127.0.0.1).
"""
import argparse
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("stream_auth")
CACHE_S = 60


class Auth:
    def __init__(self, tb_url, vehicles_file):
        self.tb = tb_url.rstrip("/")
        self.vehicles_file = vehicles_file
        self.cache = {}                  # (token, path) -> (allowed, expiry)
        self.lock = threading.Lock()

    def vehicles(self):
        with open(self.vehicles_file) as fh:   # re-read every time: add vehicles without a restart
            return json.load(fh)

    def tb_get(self, path, token):
        req = urllib.request.Request(self.tb + path, headers={"X-Authorization": "Bearer " + token})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)

    def viewer_ok(self, token, path):
        key = (token, path)
        with self.lock:
            hit = self.cache.get(key)
            if hit and hit[1] > time.time():
                return hit[0]
        ok = False
        try:
            user = self.tb_get("/api/auth/user", token)
            if user.get("authority") in ("TENANT_ADMIN", "SYS_ADMIN"):
                ok = True
            elif user.get("authority") == "CUSTOMER_USER":
                # the vehicle (device named like the path, case-insensitive) must be assigned to them
                cid = user["customerId"]["id"]
                q = urllib.parse.urlencode({"pageSize": 50, "page": 0, "textSearch": path})
                devs = self.tb_get(f"/api/customer/{cid}/devices?{q}", token)["data"]
                ok = any(d["name"].lower() == path.lower() for d in devs)
        except (urllib.error.URLError, KeyError, ValueError) as e:
            log.info("viewer token rejected: %s", e)
        with self.lock:
            self.cache[key] = (ok, time.time() + CACHE_S)
            if len(self.cache) > 5000:
                self.cache.clear()
        return ok

    def check(self, req):
        action, path = req.get("action"), (req.get("path") or "").strip("/")
        if action == "publish":
            pw = self.vehicles().get(path)
            return pw is not None and req.get("user") == path and req.get("password") == pw
        if action in ("read", "playback"):
            q = urllib.parse.parse_qs(req.get("query") or "")
            token = (q.get("token") or [""])[0] or req.get("token") or ""
            return bool(token) and path.startswith("veh-") and self.viewer_ok(token, path)
        return False


def make_handler(auth):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                ok = auth.check(req)
            except Exception:
                log.exception("auth check failed")
                ok, req = False, {}
            log.info("%s %s %s from %s -> %s", req.get("action"), req.get("path"), req.get("protocol"),
                     req.get("ip"), "allow" if ok else "deny")
            self.send_response(200 if ok else 401)
            self.end_headers()

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", default="http://127.0.0.1:8080", help="ThingsBoard URL as seen from this server")
    ap.add_argument("--vehicles", default="vehicles.json")
    ap.add_argument("--port", type=int, default=9100)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(Auth(a.tb, a.vehicles)))
    log.info("stream auth on 127.0.0.1:%d, ThingsBoard %s", a.port, a.tb)
    srv.serve_forever()


if __name__ == "__main__":
    main()
