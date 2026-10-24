#!/usr/bin/env bash
# DMS installer for Raspberry Pi 5 + AI HAT+ (Hailo-8L) + Raspberry Pi camera.
# Run from the unpacked bundle:   bash pi/install.sh
# Safe to run again (it only adds what is missing).
set -euo pipefail
APP="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/dms-venv"
say() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARNING: %s\033[0m\n' "$*"; }

say "1/6 System packages (AI HAT driver, camera, Python tools) - this takes a while"
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y dkms
sudo apt install -y hailo-all
sudo apt install -y hailo-models || warn "hailo-models package not found; AI HAT phone detector may fall back to CPU"
sudo apt install -y python3-picamera2 python3-venv python3-gpiozero libportaudio2 ffmpeg

say "2/6 Python environment ($VENV)"
python3 -m venv --system-site-packages "$VENV"
SYSNP=$(python3 -c 'import numpy; print(numpy.__version__)')
EXTRA=""
case "$SYSNP" in 1.*) EXTRA="opencv-contrib-python<4.11";; esac
"$VENV/bin/pip" install --upgrade pip
# keep numpy at the system version so the system camera / AI HAT libraries keep working
"$VENV/bin/pip" install "numpy==$SYSNP" "mediapipe==1.1.1" "onnxruntime>=1.17" "paho-mqtt>=2.0" $EXTRA

say "3/6 Model files"
bash "$APP/download_models.sh"

say "4/6 Data folder"
mkdir -p "$APP/data"

say "5/6 Auto-start service (installed but NOT enabled; see the guide, step 8)"
sed -e "s#__USER__#$USER#g" -e "s#__APP__#$APP#g" -e "s#__VENV__#$VENV#g" \
    "$APP/pi/dms.service" | sudo tee /etc/systemd/system/dms.service >/dev/null
sudo systemctl daemon-reload

sudo ln -sf "$APP/pi/dms" /usr/local/bin/dms

say "6/6 Quick checks"
if [ -e /dev/hailo0 ]; then hailortcli fw-control identify | grep -E "Device Architecture|Firmware Version" || true
else warn "AI HAT not visible yet (/dev/hailo0 missing). Reboot, then run: hailortcli fw-control identify"; fi
rpicam-hello --list-cameras 2>&1 | head -5 || warn "camera check failed - check the cable, then reboot"
ls /usr/share/hailo-models/yolov8s_h8l.hef >/dev/null 2>&1 && echo "AI HAT model: yolov8s_h8l.hef found" \
  || warn "yolov8s_h8l.hef not found - phone detection will run on the CPU"
"$VENV/bin/python" -c "import mediapipe, onnxruntime, cv2, numpy; print('python libs OK:', mediapipe.__version__, onnxruntime.__version__, cv2.__version__, numpy.__version__)"

printf '\nDone. Reboot once now:  sudo reboot\nThen follow the guide from step 4.\n'
