#!/usr/bin/env bash
# Downloads the model files the prototype needs into dms/models/ (~80 MB).
set -euo pipefail
cd "$(dirname "$0")/dms"
mkdir -p models && cd models
# Face detection (SCRFD-500M) + face recognition (ArcFace MobileFaceNet), InsightFace "buffalo_sc".
# LICENCE: InsightFace pretrained weights are for NON-COMMERCIAL research use only.
#          Replace with a licensed / self-trained model before any commercial deployment.
if [ ! -f det_500m.onnx ]; then
  curl -fL -o buffalo_sc.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
  unzip -o -q buffalo_sc.zip && rm buffalo_sc.zip
fi
# Face mesh + eye/gaze blendshapes, and hand landmarks (MediaPipe, Apache-2.0)
curl -fL -z face_landmarker.task -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -fL -z hand_landmarker.task -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# Phone / bottle / cup detection (YOLOX, Apache-2.0, COCO classes)
for m in yolox_s yolox_tiny; do
  [ -f $m.onnx ] || curl -fL -o $m.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/$m.onnx
done
ls -la
