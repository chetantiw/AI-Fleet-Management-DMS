# Digital twin on ThingsBoard — handoff for Claude on the fleet server

This file is the brief for a Claude Code session running on the fleet server. Read it fully before
changing anything.

## What already exists (branch `digital-twin`)

| Path | What it is |
|---|---|
| `dms/twin_extract.py`, `dms/twin_stream.py` | Turn a cabin video into a per-frame twin stream (`*_twin.json`) |
| `dms/twin_publish.py` | Replays a `*_twin.json` into a ThingsBoard device at 10 frames/s (MQTT or HTTP) |
| `viewer/thingsboard/cabin_twin_widget.json` | ThingsBoard widget type "Cabin driver twin" (fqn `cabin_driver_twin`), latest-values widget, reads key `twin` |
| `viewer/thingsboard/widget.{html,css,js}` | The same widget as separate files |
| `viewer/build_tb_widget.py` | Rebuilds the widget from `viewer/cabin_twin_template.html` (edit the template, not widget.js) |

Telemetry sent per frame (device topic `v1/devices/me/telemetry`):

- `twin`: JSON object with `yaw, pitch, roll` (deg vs the driver's normal), `eyeL, eyeR, jaw` (0..1),
  `lean, lE, lW, rE, rW` (arm points in shoulder widths), `armOk, face, facing, closed, hidden, phone,
  mouth, ear, belt (1/0/-1), perclos, hand ("L"/"R"), alert, ev[], driver`
- flat keys for normal widgets / rule chains: `dms_alert, dms_belt, dms_phone, dms_eyes_closed, dms_perclos, driver`

Twin streams of the 13 Oct bus videos (`v0112_twin.json`, `v0110_twin.json`, `v0107_twin.json`) are
NOT in git; Chetan copies them to the server (e.g. `~/dms-twin/`).

## The job

Add the driver twin to the **existing fleet-management dashboard** without disturbing it.

1. **Find the setup.** ThingsBoard URL (often `http://localhost:8080`), tenant-admin login from the
   environment variables `TB_URL`, `TB_USER`, `TB_PASS` — ask Chetan to export them; never write
   credentials into files or git.
2. **Back up first.** `GET /api/dashboard/{id}` for the fleet dashboard → save to
   `~/dms-twin/backup/<dashboard>_<date>.json`. Same for any widget bundle you touch.
3. **Import the widget type.** `POST /api/widgetType` with the body of `cabin_twin_widget.json`
   (skip if a widget with fqn `cabin_driver_twin` already exists; update it instead).
4. **Test device.** Create device `DMS-Twin-Test` (or use the vehicle's real DMS device if Chetan says
   so); read its access token via `GET /api/device/{id}/credentials`.
5. **Add to the dashboard.** In the backed-up JSON, add one widget instance of `cabin_driver_twin`:
   - a new entity alias for the DMS device (or reuse the dashboard's vehicle alias if each vehicle
     device carries the `twin` key — ask which),
   - datasource key `twin` (type timeseries), realtime window 1 min,
   - place it in its own new state/page "Driver twin" or a free area of the main layout — do not move or
     resize existing widgets. Show Chetan the planned change before `POST /api/dashboard`.
   - Optional: a small "Latest values" card next to it with `dms_alert`, `dms_belt`, `dms_phone`,
     `dms_perclos`, `driver`.
6. **Run a replay** and check it on the dashboard:
   `python3 dms/twin_publish.py ~/dms-twin/v0112_twin.json --host localhost --token <token> --start 160`
   (MQTT 1883 needs `pip install paho-mqtt`; or `--http --host localhost:8080`).
7. **Rate limits.** If the tenant profile has transport rate limits below 20 msg/s per device, tell
   Chetan rather than changing the profile yourself; `--rate 5` works meanwhile.
8. **Report** what changed (dashboard, widget type, device) and where the backup is.

## Rules

- Never delete or overwrite existing dashboards, widgets, devices, rule chains or tenant settings.
  Every change is additive and has a backup.
- The widget loads three.js from cdn.jsdelivr.net. If dashboard users have no internet access, host the
  two files under ThingsBoard's resources (`POST /api/resource`, type JS_MODULE) or the server's web
  root and update the two `resources` URLs in the widget type.
- No videos or face images on the server dashboard; the twin exists so that only numbers leave the bus.
