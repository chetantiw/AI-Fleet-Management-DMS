# Fleet DMS + FRS — Driver Monitoring and Face Recognition

Prototype software for an in-cabin driver monitoring system (DMS) with driver face
recognition (FRS), for the MuTech / Sensoverse AI fleet management system.

Target hardware: Raspberry Pi 5 + AI HAT+ (Hailo-8L, 13 TOPS) + camera (later Axera AX8850).
**Current state: runs on CPU on recorded cabin videos.** The live, on-device version waits for
the hardware; every model is kept in ONNX / TFLite form so it can be compiled for Hailo later.

## What it detects

| Alert | How |
|---|---|
| `DRIVER_VERIFIED` / `UNKNOWN_DRIVER` / `DRIVER_CHANGED` | ArcFace face embeddings matched to enrolled drivers; strangers numbered "Unknown #n" for the session |
| `DISTRACTED_LOOK_AWAY` | Head turn from face key points, relative to this driver's own normal, for 2 s |
| `DROWSY_EYES_CLOSED` | Eye-blink + look-down scores vs this driver's normal, 1.5 s, confirmed 1 s later (no phone / hand at face) |
| `FATIGUE_PERCLOS` | Eyes closed > 25 % of the last 60 s while facing the road (max once per 2 min) |
| `PHONE_CALL` / `PHONE_IN_HAND` | Phone detector (YOLOX) + hand at ear; fixed objects (wheel hub) and on-face boxes filtered out |
| `DRINKING` | Hand at the mouth with no phone in view |
| `NO_SEATBELT` | Diagonal strap across the chest not seen for 10 s (no trained model needed) |
| `DRIVER_ABSENT` | No driver-sized face for 3 s |

Sunglasses are detected and eye-based alerts pause while the eyes are hidden. Each new driver gets
a fresh calibration (head position, normal eye openness, face size).

## Quick start (recorded video)

```bash
pip install -r requirements.txt
./download_models.sh
cd dms

# enrol drivers (5+ photos each, taken as in the data-collection guide)
python3 frs.py enroll "Ramesh Kumar" photos/ramesh/
python3 frs.py list

# process a cabin video: annotated video + event CSV in results/
python3 process.py /path/to/cabin_video.mov --out results/
```

`process.py` runs the steps below (results cached, so re-runs are quick):

1. `extract.py` — face boxes, face mesh, eye/gaze scores, phone/bottle detections (10 fps)
2. `hands_extract.py` — hand-to-mouth / hand-to-ear distances (5 fps)
3. `belt_extract.py` — seat-belt score (2 fps)
4. `faces_extract.py` — face embeddings for identification (2 fps)
5. `render_session.py` — identity + alert engine (`engine2.py`, `identity.py`), annotated video, events CSV

Several videos can be replayed as one session (driver changes across clips):
`python3 render_session.py --gallery gallery.npz --out s.mp4 --csv s.csv a.mov:cache b.mov:cache`

## Fleet platform link (ThingsBoard CE) and live video

`dms/fleet_agent.py` sends position, driver state, device health and every alert (with a snapshot)
to ThingsBoard over MQTT, buffering on the device when there is no network, and streams live video
to a MediaMTX media server on demand from the dashboard. `server/` sets up ThingsBoard (rule chain →
alarms, device profile, "Fleet DMS" dashboard) and the media server.
Setup, message formats and test results: **[docs/thingsboard_integration.md](docs/thingsboard_integration.md)**.

```bash
python3 server/thingsboard/tb_setup.py --user <tenant admin> --password <pw> --vehicle VEH-102 \
    --stream-view http://<server>:8889/veh-102/          # on the server, prints the device token
cd server/mediamtx && cp vehicles.example.json vehicles.json && docker compose up -d
cd dms && python3 fleet_agent.py --config device.json --replay bus.mov:results/cache   # vehicle side
```

## Digital twin (3-D driver replay)

`process.py VIDEO --twin` also writes `results/<name>_twin.json`: one small record per frame (10 fps,
about 300 bytes) with the driver's head angles, eyelids, mouth, shoulder/elbow/wrist positions and the
DMS state (phone, hand at mouth/ear, seat belt, PERCLOS, current alert). These are the same keys the
device will publish to ThingsBoard over MQTT, so the fleet dashboard can show a 3-D driver instead of
camera video.

- `twin_extract.py` — upper-body pose (MediaPipe Pose, heavy model, Apache-2.0) on a crop around the driver's face
- `twin_stream.py` — merges head pose, eye/mouth scores, arm points and the per-frame DMS state
  (`render_session.py --state`) into the telemetry stream, relative to this driver's own normal
- `viewer/cabin_twin_template.html` + `build_viewer.py` — three.js viewer: generic driver in a bus cabin, alert timeline,
  optional side-by-side local video

Extra model: `models/pose_landmarker_heavy.task` (fetched by `download_models.sh`).

## Files

| File | Purpose |
|---|---|
| `dms/engine2.py` | Alert engine (all thresholds in `CFG`, with notes on how each was tuned) |
| `dms/identity.py` | Driver identification and driver-change logic |
| `dms/frs.py` | Face detection, alignment, ArcFace embedding, gallery; CLI for enrol / identify / webcam |
| `dms/belt.py` | Seat-belt check |
| `dms/phone.py` | YOLOX wrapper (phone, bottle, cup) |
| `dms/dms.py` | Shared face-geometry helpers, sunglasses check; v1 live pipeline with Hailo / CPU backends |
| `dms/process.py` | One-command processing of a recorded video |
| `dms/labels_bus.py`, `eval_bus.py`, `score_csv.py` | Ground-truth labels of the Oct-13 bus videos and the scoring used for tuning |
| `dms/eval_dmd.py`, `eval_phone_dmd.py` | Scoring against the public DMD driver dataset |
| `dms/run_all.py`, `render_v2.py` | Earlier pipelines (v1, v2), kept for reference |
| `dms/setup_pi_hailo.sh` | Raspberry Pi 5 + Hailo-8L setup (for when the hardware arrives) |
| `dms/fleet_agent.py` | In-vehicle agent: telemetry, events, RPC, live video; replays recordings until the camera is fitted |
| `dms/tb_link.py` | ThingsBoard MQTT link with offline outbox (SQLite) |
| `dms/streamer.py` | On-demand H.264 push to the media server (ffmpeg) |
| `dms/gnss.py`, `dms/health.py` | GNSS (NMEA) reader + demo route; device health |
| `server/thingsboard/tb_setup.py` | Creates rule chain, device profile, devices and the Fleet DMS dashboard |
| `server/mediamtx/` | Media server config, ThingsBoard-login check for viewers, docker compose, nginx snippet |
| `docs/results.md` | Test results so far |

## Data and licences

- **No videos, photos or face galleries are stored in this repository.** They contain drivers'
  faces (biometric data); keep them on company storage with the drivers' written consent.
- InsightFace face models (`det_500m`, `w600k_mbf`): **non-commercial research use only** —
  must be replaced before commercial deployment.
- MediaPipe face / hand landmarkers and YOLOX: Apache-2.0.
- The public DMD dataset used for early testing is research-licensed; not used for training.

## Known limitations

- Thresholds were tuned on three bus videos (three drivers); expect retuning on new footage.
- Phone/bottle detection uses an off-the-shelf COCO model; a detector trained on cabin footage is
  the main planned improvement.
- No liveness (anti-spoofing) check yet: a photo could pass face recognition.
- Night needs the IR camera + IR illumination; ordinary phone video at night is too dark.
- Camera should sit at chin-to-eye height in front of the driver; a high camera makes looking
  down resemble closed eyes.
