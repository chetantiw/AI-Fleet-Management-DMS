"""Status panel, boxes and alert banner drawn on the frame (no dots over the eyes)."""
import cv2
import numpy as np


def draw(frame, f, st, driver, banner, alert, phone_seen, fps=None, scale=None):
    H, W = frame.shape[:2]
    s = scale or max(W, H) / 1920 * 1.6          # text size relative to the frame
    th = max(int(round(2 * s)), 1)
    b = f["face_box"]
    if f["face"] and not np.isnan(b[0]):
        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (0, 0, 255) if alert else (0, 220, 0), max(th, 2))
    if phone_seen and not np.isnan(f["phone_box"][0]):
        o = f["phone_box"]
        cv2.rectangle(frame, (int(o[0]), int(o[1])), (int(o[2]), int(o[3])), (0, 165, 255), max(th, 2))
    lines = [f"Driver: {driver}"]
    if st.get("face"):
        head = "facing road" if st["facing"] else "HEAD TURNED"
        eyes = ("hidden (sunglasses)" if st["eyes_hidden"] else "n/a (turned)" if not st["facing"] else
                "CLOSED" if st["closed"] else "open" if not np.isnan(st["closure"]) else "n/a")
        hands = "at mouth" if st["hand_mouth"] else "at ear" if st["hand_ear"] else "-"
        belt = "ON" if st.get("belt") else "OFF" if st.get("belt") is False else "checking"
        lines += [f"Head: {head}   Eyes: {eyes}",
                  f"Hand: {hands}   Phone: {'yes' if phone_seen else '-'}   Seat belt: {belt}",
                  f"Tiredness (60 s): {st['perclos'] * 100:.0f}%"]
    else:
        lines += ["No driver face"]
    lh = int(30 * s)
    cv2.rectangle(frame, (0, 0), (W, int(lh * (len(lines) + 0.6))), (0, 0, 0), -1)
    for j, txt in enumerate(lines):
        cv2.putText(frame, txt, (int(10 * s), int(lh * (j + 1))), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                    (0, 255, 255) if j == 0 else (255, 255, 255), th)
    if fps:
        cv2.putText(frame, f"{fps:.1f} fps", (W - int(130 * s), H - int(12 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * s, (255, 255, 255), th)
    if alert:
        bh = int(48 * s)
        cv2.rectangle(frame, (0, H - bh - int(30 * s)), (W, H - int(30 * s)), (0, 0, 255), -1)
        cv2.putText(frame, banner[:48], (int(10 * s), H - int(42 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s,
                    (255, 255, 255), th + 1)
    return frame
