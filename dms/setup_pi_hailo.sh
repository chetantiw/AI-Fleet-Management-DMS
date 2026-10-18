#!/usr/bin/env bash
# DMS prototype setup: Raspberry Pi 5 + AI HAT+ 13 TOPS (Hailo-8L) + HQ Camera (IMX477, CSI)
# Run on the Pi (Raspberry Pi OS 64-bit):  bash setup.sh   then reboot.
set -euo pipefail

sudo apt update
sudo apt full-upgrade -y
sudo apt install -y hailo-all python3-picamera2 python3-opencv python3-numpy wget

# HQ Camera: auto-detect normally works. If `rpicam-hello --list-cameras` shows
# nothing after reboot, add  dtoverlay=imx477  to /boot/firmware/config.txt.

MODELS="$HOME/dms/models"
mkdir -p "$MODELS"
ZOO="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.14.0/hailo8l"

# Face detector: use the copy shipped with hailo-all if present, else download.
if [ -f /usr/share/hailo-models/scrfd_2.5g_h8l.hef ]; then
  cp /usr/share/hailo-models/scrfd_2.5g_h8l.hef "$MODELS/scrfd_2.5g.hef"
else
  wget -nc -O "$MODELS/scrfd_2.5g.hef" "$ZOO/scrfd_2.5g.hef"
fi
# Face landmarks (MediaPipe face mesh, 468 points, 192x192 input)
wget -nc -O "$MODELS/face_landmarks_lite.hef" "$ZOO/face_landmarks_lite.hef"

echo
echo "Done. Reboot, then check:"
echo "  hailortcli fw-control identify      # should report HAILO8L"
echo "  rpicam-hello --list-cameras         # should list imx477"
echo "  hailortcli parse-hef $MODELS/scrfd_2.5g.hef"
echo "  python3 dms.py"
