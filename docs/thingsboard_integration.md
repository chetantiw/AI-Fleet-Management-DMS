# ThingsBoard CE integration: telemetry + live video

Tested end to end on ThingsBoard CE 4.2.1 and MediaMTX 1.21.2 (24 Oct 2026); the setup script reads
widget defaults from your own server, so ThingsBoard 4.0 or later works.

```
 Vehicle (Pi 5 / AX8850)                         Linux server
 ┌──────────────────────────┐   MQTT 1883      ┌──────────────────────────────┐
 │ DMS + face ID ─┐         │ ───────────────▶ │ ThingsBoard CE               │
 │ GNSS ──────────┼─ fleet_ │  telemetry,      │  rule chain "DMS Events"     │
 │ health ────────┘  agent  │  events, RPC     │  → alarms, dashboard         │
 │       ▲ outbox (SQLite)  │ ◀─────────────── │  "Start live video" button   │
 │       │                  │                  │                              │
 │ camera frames ─▶ ffmpeg  │   RTSP 8554      │ MediaMTX ◀─ stream_auth.py   │
 │   (H.264, on demand)     │ ───────────────▶ │  WebRTC 8889 ─▶ browser      │
 └──────────────────────────┘  (vehicle pushes)└──────────────────────────────┘
```

The vehicle always connects **out** to the server (MQTT and RTSP). A 4G SIM sits behind the
operator's NAT with no public address, so the server can never connect in to the vehicle.

## 1. Server setup (once)

On the ThingsBoard server:

```bash
git clone https://github.com/chetantiw/AI-Fleet-Management-DMS && cd AI-Fleet-Management-DMS

# a) ThingsBoard: rule chain, device profile, vehicle device, dashboard
pip install requests
python3 server/thingsboard/tb_setup.py --url http://localhost:8080 \
    --user <tenant admin e-mail> --password <password> \
    --vehicle VEH-102 --stream-view http://<server>:8889/veh-102/
#   -> prints the device access token for the vehicle's device.json
#   Run again with --vehicle VEH-103 ... for each further vehicle (safe to re-run).

# b) Media server + its login check
cd server/mediamtx
cp vehicles.example.json vehicles.json      # "veh-102": "<publish password>", one line per vehicle
docker compose up -d
```

Firewall: open **1883/tcp** (MQTT, probably already open), **8554/tcp** (RTSP from vehicles),
**8889/tcp** (video player) and **8189/udp + 8189/tcp** (WebRTC media). In `mediamtx.yml` put the
server's public name in `webrtcAdditionalHosts` if browsers reach it from outside the LAN.

If ThingsBoard is served over **HTTPS**, add `nginx-live.conf` to its nginx site and use
`--stream-view https://<server>/live/veh-102/` instead (a browser refuses an http:// video frame
inside an https:// page).

Without Docker: download the `mediamtx` binary from its GitHub releases and run
`./mediamtx mediamtx.yml` and `python3 stream_auth.py --vehicles vehicles.json` as two systemd services.

## 2. Vehicle setup

```bash
pip install -r requirements.txt && sudo apt install ffmpeg
cp dms/device.example.json dms/device.json    # fill host, token, stream password
```

| `device.json` field | Meaning |
|---|---|
| `thingsboard.host / port / token` | server, 1883 (8883 with `"tls": true`), access token from `tb_setup.py` |
| `stream.publish_url` | `rtsp://veh-102:<password>@<server>:8554/veh-102` (password from `vehicles.json`) |
| `stream.view_url` | player address shown in the dashboard |
| `gnss` | `{"type": "nmea", "port": "/dev/ttyUSB1", "baud": 9600}` for a real receiver; `{"type": "demo"}` drives a route around Indore |

Until the camera hardware is fitted, the agent replays processed recordings in real time:

```bash
cd dms
python3 fleet_agent.py --config device.json --replay bus.mov:results/cache --gallery gallery.npz
python3 fleet_agent.py --config device.json --video results/bus_annotated.mp4 --loop   # link + video only
```

On the device the live DMS loop will call `Agent.frame(t, frame, events, status)` for every frame.

## 3. What is sent

**Every 5 s** (setting `telemetry_interval_s`), and at once when the driver, seat belt or face
presence changes — one telemetry record:

| Key | Example | |
|---|---|---|
| `latitude`, `longitude`, `speed`, `heading`, `altitude`, `satellites`, `hdop`, `gps_fix` | 22.7186, 75.8553, 32.0 | km/h, degrees |
| `driver`, `driver_status` | "Ramesh Kumar", `verified` / `unknown` / `none` | |
| `face`, `head`, `eyes`, `hand`, `phone` | true, `facing`, `open`/`closed`/`hidden`, `mouth`/`ear`/`none`, false | |
| `perclos`, `seatbelt` | 12.5 (%), `on` / `off` / `unknown` | |
| `buffered` | 0 | messages waiting on the device (no network) |

**Every 60 s:** `cpu`, `mem`, `disk` (%), `cpu_temp` (°C), `uptime_h`, `load`, `streaming`.

**Each event, immediately:** `event` (name), `event_driver`, `event_details`, `severity`,
`latitude`, `longitude`, `speed`, `event_image` (320-px JPEG snapshot, ≤ 24 KB, data URL).

| Event | Alarm severity |
|---|---|
| `DROWSY_EYES_CLOSED`, `ALCOHOL_FAIL` | Critical |
| `FATIGUE_PERCLOS`, `PHONE_CALL`, `PHONE_IN_HAND`, `NO_SEATBELT`, `UNKNOWN_DRIVER`, `OVERSPEED`, `ALCOHOL_TEST_INVALID`, `CAMERA_FAULT` | Major |
| `DISTRACTED_LOOK_AWAY`, `DRINKING` | Minor |
| `DRIVER_ABSENT` | Warning |
| `DRIVER_VERIFIED`, `DRIVER_CHANGED`, `ALCOHOL_PASS` | none (history only) |

The rule chain turns each event with a severity into an alarm of that type (details: driver, place,
speed, count). A repeat of the same event updates the open alarm and increases its `count` until an
operator clears it. Every event also stays in the time series (Events table, with snapshot).

**Client attributes:** `vehicle_id`, `fw_version`, `dms_models`, `stream_path`, `stream_active`.

**Settings from the server** (shared attributes, change them on the device page in ThingsBoard;
the vehicle applies them at once): `telemetry_interval_s` 5, `stream_max_minutes` 10,
`stream_bitrate_k` 400, `stream_width` 640, `event_snapshot` true, `overspeed_kmh` 60.

**Commands (RPC)** — used by the dashboard buttons, or any two-way RPC call:
`stream_start {"minutes": 5}`, `stream_stop`, `snapshot` (current picture), `get_status`.

## 4. No network coverage

Every message goes to a SQLite outbox on the device first and is deleted only after ThingsBoard
acknowledges it (MQTT QoS 1). With no coverage the vehicle keeps working and buffering; after
reconnection the backlog is sent oldest first with each record's original time, so history, routes
and alarms are correct. Messages over 60 KB are refused on the device (ThingsBoard drops the whole
connection on anything over 64 KB). If storage fills (200 000 messages), the oldest routine telemetry
is dropped first, never events.

## 5. Live video

Operator presses **Start live video** on the vehicle page → RPC to the vehicle → the vehicle starts
pushing H.264 (RTSP over TCP) to MediaMTX → the dashboard shows it with WebRTC (~0.5 s delay).
It stops by itself after `stream_max_minutes` (default 10, button asks for 5), so a forgotten browser
tab cannot use up the SIM's data.

- Data: 640-px wide, 10 fps, 400 kbit/s ≈ **180 MB per hour of viewing**; nothing is sent when nobody watches.
- The stream shows the annotated DMS view (driver, eye state, alerts drawn on the picture).
- Raspberry Pi 5 has no hardware H.264 encoder: software x264 at this size costs about half a CPU
  core while streaming. The AX8850 has a hardware encoder — switch the codec when porting.
- **Who can watch:** `stream_auth.py` lets a browser in only with a valid ThingsBoard login token
  (the dashboard adds it to the player address). Customer users can watch only vehicles assigned to
  their customer. Each vehicle can publish only to its own path, with its own password.
- Recording on the server for incident review: set `record: true` in `mediamtx.yml` (kept 7 days).

## 6. Dashboard "Fleet DMS"

- **Fleet:** map of all vehicles, vehicles table (online, driver, speed, seat belt, tiredness, last
  event; click a row to open the vehicle), active DMS alarms (acknowledge / clear / assign).
- **Vehicle:** live video with start/stop, driver now, device health, route of the last hour (with
  replay), events of the last 24 h with snapshots, speed + tiredness chart, alarms of 7 days.

## 7. Test results (local ThingsBoard CE 4.2.1 + MediaMTX 1.21.2)

| Test | Result |
|---|---|
| Telemetry, attributes, settings pushed from the server | received; settings applied on connect |
| Events → alarms | 5 event types → 5 alarms with correct severity, driver, place; repeat raised `count` to 2 |
| Network cut 60 s (container unplugged) while 3 events happened | all 3 events + 12 positions arrived 2 s after reconnection, original timestamps exact |
| Backlog of 1 000 messages | sent in 0.7 s, stored by ThingsBoard in 1.4 s |
| Dashboard "Start live video" button | vehicle started streaming; video played in the dashboard (WebRTC, 540×960) |
| Viewer login | valid ThingsBoard token → plays; wrong / no token → refused |
| Wrong publish password, or another vehicle's account | refused |
| Customer user | sees the video of their own vehicle; another customer's user is refused |
| Snapshot command | 44 KB JPEG returned |

Known limitation: after a long outage, a burst of the same event type can occasionally open two
alarms instead of one (ThingsBoard processes the burst in parallel). Clearing either is enough.
