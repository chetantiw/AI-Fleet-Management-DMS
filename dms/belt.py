"""
Seat-belt check without a trained model.

A worn belt is a straight strap crossing the chest diagonally, from the driver's right shoulder
(image left, window side on a right-hand-drive bus) down to the left hip. We rotate the chest
area so each candidate direction becomes vertical and look for a straight band - darker OR
lighter than the shirt on both sides - that runs the full height of the chest and passes near
its middle. An unbuckled belt hanging beside the shoulder is off-centre and is ignored.

belt_score(gray, face_box) -> (score, angle_deg, offset_in_face_widths)
"""
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

CFG = dict(
    angles=range(30, 76, 5),   # belt direction, degrees from horizontal, falling to the image right
    max_offset=0.55,           # band must pass within this many face widths of the chest centre
    on_score=0.15,             # score above this = belt seen in this frame (bus videos: off ~0.02-0.06, on ~0.23-0.39)
)


def chest_roi(gray, fb):
    fw = fb[2] - fb[0]
    cx = (fb[0] + fb[2]) / 2
    H, W = gray.shape
    x0, x1 = int(cx - 1.4 * fw), int(cx + 1.4 * fw)
    y0, y1 = int(fb[3] + 0.2 * fw), int(fb[3] + 2.4 * fw)
    if x0 < 0 or x1 > W or y1 > H or y0 >= y1:
        # clip, but give up if most of the chest is off-frame
        x0c, x1c, y1c = max(x0, 0), min(x1, W), min(y1, H)
        if (x1c - x0c) < 0.7 * (x1 - x0) or (y1c - y0) < 0.6 * (y1 - y0):
            return None, fw
        x0, x1, y1 = x0c, x1c, y1c
    roi = gray[y0:y1, x0:x1]
    s = 160.0 / fw                       # normalise: face width = 160 px
    return cv2.resize(roi, (max(int(roi.shape[1] * s), 8), max(int(roi.shape[0] * s), 8))), fw


def belt_score(gray, fb):
    roi, fw = chest_roi(gray, fb)
    if roi is None:
        return np.nan, np.nan, np.nan
    roi = cv2.GaussianBlur(roi.astype(np.float32), (5, 5), 0)
    h, w = roi.shape
    c = (w / 2, h / 2)
    best = (0.0, np.nan, np.nan)
    band = 26                            # belt width ~ 0.16 face widths at 160 px/face
    for a in CFG["angles"]:
        # rotate so a strap falling at angle a (to the right) becomes vertical
        M = cv2.getRotationMatrix2D(c, -(90 - a), 1.0)
        r = cv2.warpAffine(roi, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=np.nan)
        valid = ~np.isnan(r)
        rows = np.where(valid.mean(1) > 0.6)[0]
        if len(rows) < h * 0.5:
            continue
        segs = np.array_split(rows, 3)
        profs = []
        for sg in segs:
            blk = r[sg]
            with np.errstate(all="ignore"):
                profs.append(np.nanmean(blk, 0))
        profs = np.array(profs)                         # 3 x w
        maxoff = int(CFG["max_offset"] * 160)
        for x in range(max(int(c[0]) - maxoff, band + 8), min(int(c[0]) + maxoff, w - band - 8), 3):
            inside = profs[:, x - band // 2:x + band // 2].mean(1)
            left = profs[:, x - band // 2 - 18:x - band // 2 - 4].mean(1)
            right = profs[:, x + band // 2 + 4:x + band // 2 + 18].mean(1)
            if np.any(np.isnan(inside)) or np.any(np.isnan(left)) or np.any(np.isnan(right)):
                continue
            ref = (left + right) / 2 + 1e-6
            d1, d2 = (inside - left) / ref, (inside - right) / ref
            # same sign on both sides (band darker, or lighter, than both neighbours) in all 3 segments
            same = np.sign(d1) == np.sign(d2)
            if not same.all() or len(set(np.sign(d1))) != 1:
                continue
            sc = float(np.min(np.minimum(abs(d1), abs(d2))))
            if sc > best[0]:
                best = (sc, a, (x - c[0]) / 160)
    return best
