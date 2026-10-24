"""
Phone / bottle / cup detection on the Raspberry Pi AI HAT+ (Hailo-8L).

Uses the YOLOv8s model that Raspberry Pi ships with the Hailo packages
(/usr/share/hailo-models/yolov8s_h8l.hef). NMS runs on the accelerator, so the output is,
per COCO class, a list of [y0, x0, y1, x1, score] in 0..1 of the model input.
Returns the same format as phone.YoloX: [(x1, y1, x2, y2, score, class_id)], best first,
in frame pixels.
"""
import cv2
import numpy as np

WANT = (67, 39, 41)          # cell phone, bottle, cup


def parse(output, sx, sy, pad_x, pad_y, score, classes=WANT):
    """output: list indexed by class id of arrays of [y0, x0, y1, x1, score] (0..1, letterboxed).
    sx, sy: model-input pixels -> frame pixels scale; pad: letterbox offsets in model pixels."""
    dets = []
    for c in classes:
        if c >= len(output):
            continue
        for d in output[c]:
            s = float(d[4])
            if s < score:
                continue
            y0, x0, y1, x1 = [float(v) for v in d[:4]]
            dets.append(((x0 * sx[0] - pad_x) * sx[1], (y0 * sy[0] - pad_y) * sy[1],
                         (x1 * sx[0] - pad_x) * sx[1], (y1 * sy[0] - pad_y) * sy[1], s, c))
    return sorted(dets, key=lambda d: -d[4])


class HailoObjects:
    def __init__(self, hef, score=0.05):
        from picamera2.devices import Hailo
        self.h = Hailo(hef)
        self.mh, self.mw, _ = self.h.get_input_shape()
        self.score = score

    def __call__(self, bgr):
        H, W = bgr.shape[:2]
        r = min(self.mw / W, self.mh / H)
        nw, nh = int(round(W * r)), int(round(H * r))
        canvas = np.full((self.mh, self.mw, 3), 114, np.uint8)
        px, py = (self.mw - nw) // 2, (self.mh - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(bgr, (nw, nh))
        out = self.h.run(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        if isinstance(out, dict):                        # some HailoRT versions return {name: list}
            out = next(iter(out.values()))
        return parse(out, (self.mw, 1 / r), (self.mh, 1 / r), px, py, self.score)

    def close(self):
        try:
            self.h.close()
        except Exception:
            pass
