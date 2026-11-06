# Task for Claude on the ThingsBoard server: install the Fleet DMS server side

You are running on the Linux server that hosts **ThingsBoard CE** at **https://iot.mutechautomation.com**.
On behalf of Chetan (owner), install the fleet DMS server side: the ThingsBoard rule chain, device
profile, vehicle devices and dashboard, plus the MediaMTX media server for live video. Everything
needed is in this repository; full background is in `docs/thingsboard_integration.md`.

Vehicles to create: **VEH-102** (add more only if Chetan names them).

## Rules
- This is a production server with live ThingsBoard data. Do not restart, upgrade or reconfigure
  ThingsBoard itself, and do not delete or change existing devices, rule chains or dashboards.
  The setup script only adds a "DMS Events" rule chain, a "DMS Vehicle" profile, the vehicle devices
  and a "Fleet DMS" dashboard.
- Before editing any system file (nginx site, firewall), copy it to `<file>.bak-<date>`, show Chetan
  the change, and validate it (`nginx -t`) before reloading.
- Ask Chetan for the ThingsBoard tenant admin login when you reach that step. Pass it only as the
  environment variables `TB_USER` / `TB_PASS` for that one command; never write it to a file, a
  commit or a log.
- `server/mediamtx/vehicles.json` and `server/vehicles/*.device.json` hold passwords and tokens.
  Keep them `chmod 600`, never commit them (they are git-ignored), and do not print them in full.
- If a step fails, stop and report the exact error instead of improvising workarounds.

## Steps

1. **Look around first (read only) and report back before changing anything:**
   - ThingsBoard: how it runs (Docker / systemd / package) and its local HTTP port
     (`ss -ltnp | grep -E '8080|9090'`, check the nginx site). Expected: `http://127.0.0.1:8080`.
   - `curl -s http://127.0.0.1:<port>/api/system/info` → version (needs 4.0 or later).
   - nginx: which site file serves `iot.mutechautomation.com` on 443.
   - Docker and `docker compose` present? Python 3 and pip present? `ffmpeg` not needed here.
   - Firewall in use: ufw, firewalld, iptables, and/or a cloud security group (ask Chetan).
   - Ports 8554, 8889, 8189, 9100, 9997 free (`ss -ltnup`).
   - MQTT port 1883 already listening (ThingsBoard MQTT transport)?

2. **Get the code:**
   `git clone https://github.com/chetantiw/AI-Fleet-Management-DMS /opt/fleet-dms && cd /opt/fleet-dms`
   (or `git pull` if it is already there).

3. **Run the installer** (ThingsBoard port from step 1 in `TB_URL`; if 9100 is taken set
   `AUTH_PORT`; if vehicles must use another MQTT port than 1883 set `MQTT_PORT`, and `MQTT_TLS=true`
   if that port is TLS):
   ```bash
   cd /opt/fleet-dms
   sudo TB_URL=http://127.0.0.1:8080 AUTH_PORT=9100 MQTT_PORT=1883 \
        TB_USER='<from Chetan>' TB_PASS='<from Chetan>' \
        ./server/install_server.sh iot.mutechautomation.com VEH-102
   ```
   Per-server settings are written to `server/mediamtx/.env` (git-ignored). Never edit tracked files
   (`docker-compose.yml`, `mediamtx.yml`) for local ports; that breaks `git pull`.
   It starts MediaMTX + the viewer login check (`stream_auth.py`) with Docker, generates a video
   password per vehicle, creates the ThingsBoard items and writes
   `server/vehicles/VEH-102.device.json`. It prints an nginx block and a firewall line; it does not
   apply them.
   If Docker is missing, install it from the official Docker docs only after Chetan agrees.

4. **nginx:** add the printed `location /live/ { ... }` block (from `server/mediamtx/nginx-live.conf`)
   inside the existing 443 `server { }` block for `iot.mutechautomation.com`. Back up first, then
   `sudo nginx -t && sudo systemctl reload nginx`.

5. **Firewall:** open `8554/tcp` (video from vehicles), `8189/udp` and `8189/tcp` (WebRTC media), and
   `1883/tcp` if MQTT is not already reachable from outside. If there is a cloud security group,
   tell Chetan exactly which rules to add there.

6. **Verify and report:**
   - `docker compose -f server/mediamtx/docker-compose.yml ps` → both services up;
     `docker logs` show no errors.
   - `curl -s -o /dev/null -w '%{http_code}' https://iot.mutechautomation.com/live/veh-102/` → `401`
     (refused without a ThingsBoard login, which is correct).
   - Wrong-password publish is refused:
     `ffmpeg -re -f lavfi -i testsrc=size=320x240:rate=10 -t 3 -c:v libx264 -f rtsp -rtsp_transport tcp rtsp://veh-102:wrong@127.0.0.1:8554/veh-102`
     (only if ffmpeg is installed) → fails; `stream_auth` logs `deny`.
   - In ThingsBoard: device VEH-102 (profile "DMS Vehicle"), rule chain "DMS Events", dashboard
     "Fleet DMS" exist.
   - Tell Chetan where `server/vehicles/VEH-102.device.json` is. It goes on the vehicle's
     Raspberry Pi as `dms/device.json` in the repo checkout (the live app picks it up on restart).
     Its `gnss` entry is `{"type": "demo"}` (fake route); on a real vehicle set it to
     `{"type": "nmea", "port": "/dev/ttyUSB1", "baud": 9600}` for the GPS receiver, or `"none"`.

## Updating later / adding vehicles
```bash
cd /opt/fleet-dms && git pull
cd server/mediamtx && docker compose up -d        # picks up changes; settings stay in .env
```
Add a vehicle: re-run the installer with the same environment variables and the new id
(`... install_server.sh iot.mutechautomation.com VEH-103`); existing passwords are kept.
On srv1300859 use `TB_URL=http://127.0.0.1:8081 AUTH_PORT=9100 MQTT_PORT=1884 MQTT_TLS=false`.

## Demo vehicle VEH-DEMO (recorded bus videos, until the real hardware runs)

Adds a vehicle "VEH-DEMO" that replays the three processed bus videos in a loop on this server: its
alerts become ThingsBoard alarms at the moment they happen in the video (with snapshots), the
driver panel follows them, it drives a demo route around Indore, and "Start live video" shows the
annotated video. It is marked `demo_mode = true` in its attributes and alerts say "recorded demo".

1. `cd /opt/fleet-dms && git pull`
2. Create the device (same settings as before; ask Chetan for the ThingsBoard login again):
   ```bash
   sudo TB_URL=http://127.0.0.1:8081 AUTH_PORT=9100 MQTT_PORT=1884 MQTT_TLS=false \
        TB_USER='<from Chetan>' TB_PASS='<from Chetan>' \
        ./server/install_server.sh iot.mutechautomation.com VEH-DEMO
   ```
   It must not change VEH-102 (passwords are kept). Skip the nginx/firewall output; that is done.
3. Videos: put `v3_IMG_0107_annotated.mp4`, `v3_IMG_0110_annotated.mp4` and
   `v3_IMG_0112_annotated.mp4` into `/opt/fleet-dms/server/demo/videos/`. They are in Chetan's
   Google Drive folder (processed videos). Ask him to either copy them to the server (`scp`) or share
   the folder "anyone with the link" temporarily, then
   `pip install gdown && gdown --folder '<folder link>' -O /tmp/dmsvids` and move the three mp4 files.
   They show drivers' faces: keep them only in that folder (git-ignored), and remind Chetan to turn
   link sharing off again.
4. `cd /opt/fleet-dms/server/demo && docker compose up -d --build`
5. Verify: `docker compose logs --tail 20` shows "connected to ThingsBoard 127.0.0.1:1884" and
   "playing ..."; on the Fleet DMS dashboard VEH-DEMO is online, events and alarms appear within a
   minute; "Start live video" on its page plays (Chetan checks this in his browser).
   Stop the demo any time with `docker compose down` in server/demo.

The alert list it replays is `demo/v3_alerts_for_review.csv` (in git; alert names and times only).

## Demo on an existing bus (GARIMA-BUS-01) and the "Bus Device Details" dashboard

Chetan wants the demo on one of the existing Garima buses, shown on his own dashboards ("Garima
School Transport Management System" and "Bus Device Details", sections "Live Streaming" and
"Driver Monitoring"), instead of a separate VEH-DEMO. Use **GARIMA-BUS-01** unless he names another.

Rules: export a backup of every dashboard, rule chain and device profile before you change it
(`GET /api/dashboard/<id>` etc. → `/opt/fleet-dms/backups/<name>-<date>.json`), and show Chetan the
change before saving. Do not change the other buses.

1. **Look first and report:**
   - GARIMA-BUS-01: its device profile, the rule chain that profile uses, its latest telemetry and
     attributes, and whether something already sends its position/telemetry (simulator, gateway,
     rule chain generator). If something sends latitude/longitude, the demo must run with `--no-gps`.
   - "Bus Device Details": export it and list, for the **Live Streaming** and **Driver Monitoring**
     widgets, which attributes/telemetry keys they read and how (widget settings and JS code).
     The Live Streaming widget shows "No camera URL configured for this bus", so note the attribute
     it expects.
2. **Create the stream login and device file for the bus** (keeps its own profile; the script no
   longer moves existing devices):
   ```bash
   sudo TB_URL=http://127.0.0.1:8081 AUTH_PORT=9100 MQTT_PORT=1884 MQTT_TLS=false \
        TB_USER='<from Chetan>' TB_PASS='<from Chetan>' \
        ./server/install_server.sh iot.mutechautomation.com GARIMA-BUS-01
   ```
   This adds `garima-bus-01` to `server/mediamtx/vehicles.json`, sets server attribute
   `stream_view_url = https://iot.mutechautomation.com/live/garima-bus-01/` on the bus, adds the DMS
   settings as shared attributes, and writes `server/vehicles/GARIMA-BUS-01.device.json` (with the
   bus's existing access token). It does not change the bus's profile. Restart the auth service so
   it sees the new path: `cd server/mediamtx && docker compose restart stream-auth`.
3. **Live Streaming widget:** make it play `stream_view_url` + `?token=<the viewer's ThingsBoard
   token>` in an `<iframe>` (the media server refuses viewers without a valid ThingsBoard token;
   in widget JS: `localStorage.getItem('jwt_token')`). Simplest is to point the widget's camera-URL
   attribute at that value or edit its HTML. A plain `<video src>` will not play WebRTC. Keep the
   widget's look; only change where the video comes from.
4. **Driver Monitoring widget:** make it read the DMS keys the demo (and later the real Pi) sends:
   | Widget field | DMS telemetry key |
   |---|---|
   | Drowsiness | `drowsy` (true/false) |
   | Distraction | `distracted` (true/false) |
   | Phone use | `phone_use` (true/false) |
   | Face detected | `face_detected` (true/false) |
   | Camera online | `camera_ok` (true) and data younger than ~30 s |
   | Driver attention % | `attention` (0–100, rough score) |
   | DMS events | `event` (latest alert name), `event_details`, `event_image` (snapshot) |
   | also available | `driver`, `driver_status`, `seatbelt` (on/off/unknown), `perclos`, `eyes`, `head` |
   Prefer changing the widget's key names to these (keep a backup) rather than adding new keys.
5. **Alarms (optional, ask Chetan):** DMS alerts become alarms only through the "DMS Events" rule
   chain. If the bus's profile uses the Root Rule Chain, setting that profile's default rule chain
   to "DMS Events" is additive (it raises alarms for messages with an `event` key and forwards
   everything to Root unchanged), but it applies to all buses with that profile. If the profile
   uses another chain, report that instead of changing it.
6. **Run the demo on the bus** (videos in `server/demo/videos/` as in the VEH-DEMO section):
   ```bash
   cd /opt/fleet-dms/server/demo
   printf 'DEMO_DEVICE=GARIMA-BUS-01\nDEMO_ARGS=--stream-always\n' > .env    # add --no-gps if step 1 found another position source
   docker compose up -d --build
   ```
   `--stream-always` keeps the video on (no start button needed; it runs on the server, so no SIM
   data). If VEH-DEMO was already running, this replaces it.
7. **Verify with Chetan:** Bus Device Details for GARIMA-BUS-01 plays the video and Driver Monitoring
   changes as the alerts happen; the bus still appears on the Garima map.

## Report to Chetan at the end
A short summary: what was installed, ThingsBoard version, the nginx and firewall changes made
(with backup file names), test results from step 6, anything you could not do, and the location
of the device.json file(s). Remind him to change the ThingsBoard password if it was shared in chat.
