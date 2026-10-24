"""
Tiny built-in web page for watching the DMS live from a phone or laptop on the same network.

  http://<pi-address>:8080/          live view + current state + recent events
  http://<pi-address>:8080/stream    MJPEG stream
  http://<pi-address>:8080/status    JSON: driver, state, fps, timings
  http://<pi-address>:8080/events    JSON: last 100 events
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DMS live</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#111;color:#eee}
.wrap{display:flex;flex-wrap:wrap;gap:12px;padding:12px}
img{max-width:100%;height:auto;border-radius:6px;background:#000}
.col{flex:1 1 320px;min-width:0}
h1{font-size:18px;margin:0 0 8px}
table{width:100%;border-collapse:collapse;font-size:14px}
td{padding:4px 6px;border-bottom:1px solid #333;vertical-align:top}
.k{color:#9aa}
.alert{color:#ff6b6b;font-weight:600}
</style></head><body><div class="wrap">
<div class="col"><img src="/stream" alt="live camera"></div>
<div class="col"><h1>Driver monitor</h1><table id="st"></table>
<h1 style="margin-top:14px">Recent events</h1><table id="ev"></table></div></div>
<script>
async function tick(){
 try{
  const s=await (await fetch('/status')).json();
  document.getElementById('st').innerHTML=Object.entries(s).map(([k,v])=>
   `<tr><td class="k">${k}</td><td>${typeof v==='object'?JSON.stringify(v):v}</td></tr>`).join('');
  const e=await (await fetch('/events')).json();
  document.getElementById('ev').innerHTML=e.slice(-30).reverse().map(x=>
   `<tr><td class="k">${x.time.slice(11,19)}</td><td class="alert">${x.event}</td><td>${x.driver||''} ${x.details||''}</td></tr>`).join('');
 }catch(err){}
}
setInterval(tick,1000);tick();
</script></body></html>"""


class LiveView:
    def __init__(self, port=8080):
        self.jpeg = None
        self.status = {}
        self.events = []
        self.cond = threading.Condition()
        view = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8", PAGE.encode())
                elif self.path == "/status":
                    self._send(200, "application/json", json.dumps(view.status, default=str).encode())
                elif self.path == "/events":
                    self._send(200, "application/json", json.dumps(view.events[-100:], default=str).encode())
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        while True:
                            with view.cond:
                                view.cond.wait(timeout=2.0)
                                jpg = view.jpeg
                            if jpg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self._send(404, "text/plain", b"not found")

        self.server = ThreadingHTTPServer(("0.0.0.0", port), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, jpeg_bytes, status):
        with self.cond:
            self.jpeg = jpeg_bytes
            self.status = status
            self.cond.notify_all()

    def add_event(self, ev):
        self.events.append(ev)
        if len(self.events) > 500:
            self.events = self.events[-300:]

    def close(self):
        self.server.shutdown()
