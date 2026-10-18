"""Replay extracted features through Engine2 and score alerts against labels_bus.py, episode by episode."""
import sys, numpy as np, feat, engine2
from labels_bus import L
S = sys.argv[1]; vids = sys.argv[2:] or ["IMG_0110", "IMG_0112", "IMG_0107"]
WANT = {"DISTRACTED_LOOK_AWAY": ("side", 2), "DROWSY_EYES_CLOSED": ("closed", 2), "DRINKING": ("drink", 1),
        "PHONE_CALL": ("call", 2), "DRIVER_ABSENT": ("absent", 3)}
def episodes(iv, gap=1):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + gap: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
PHONE_THR = 0.35
def phone_flags(F):
    """Phone detections (run every 3rd frame) held until the next run; 'at head' = beside the face."""
    o = F["obj"][:, 0]; seen = np.zeros(len(o), bool); head = np.zeros(len(o), bool); cur = (False, False)
    for i in range(len(o)):
        if not np.isnan(o[i, 4]) or not np.isnan(o[i, 0]):
            s = o[i, 4] if not np.isnan(o[i, 4]) else 0
            ok = s >= PHONE_THR and F["face"][i]
            hd = False
            if ok:
                px, py = (o[i, 0] + o[i, 2]) / 2, (o[i, 1] + o[i, 3]) / 2
                fw = F["fw"][i]; b = F["face_box"][i]
                hd = abs(py - (b[1] + b[3]) / 2) < 0.8 * fw and abs(px - (b[0] + b[2]) / 2) < 1.2 * fw
            cur = (ok, hd)
        seen[i], head[i] = cur
    return seen, head
WANT["PHONE_IN_HAND"] = ("hand", 2)
tot = {}
def run(v, cfg=None, verbose=True):
    F, G = feat.load(v, S)
    e = engine2.Engine2(cfg or engine2.CFG); fired = []
    hidden = G["sunglasses"]
    ph_seen, ph_head = phone_flags(F)   # stand-in for the live sunglasses check, so eye logic is judged on its own
    for i, t in enumerate(F["t"]):
        ph = F["obj"][i, 0]
        f = dict(face=bool(F["face"][i]), mesh=bool(F["mesh"][i]), yawp=F["yawp"][i], eyespan=F["eyespan"][i],
                 blink=F["blink"][i], down=F["down"][i], eyes_hidden=bool(hidden[i]),
                 hmouth=F["hmouth"][i], hear=F["hear"][i], phone_run=not np.isnan(F["obj"][i, 0, 4]) or not np.isnan(F["obj"][i, 0, 0]) or i % 3 == 0,
                 phone_box=F["obj"][i, 0], face_box=F["face_box"][i], fw=F["fw"][i], frame_w=1080)
        fired += [(t, x) for x in e.update(t, f)]
    res = {}
    for ev, (lab, minlen) in WANT.items():
        gt = [ep for ep in episodes(L[v][lab]) if ep[1] - ep[0] + 1 >= minlen]
        al = [t for t, x in fired if x == ev]
        hit = sum(any(a - 2 <= t <= b + 3 for t in al) for a, b in gt)
        allg = episodes(L[v][lab])
        false = [round(t) for t in al if not any(a - 2 <= t <= b + 3 for a, b in allg)]
        res[ev] = (hit, len(gt), false)
        if verbose:
            print(f"  {ev:22s} caught {hit}/{len(gt)} episodes   false alerts {len(false)} {false[:12]}")
    fat = [round(t) for t, x in fired if x == "FATIGUE_PERCLOS"]
    if verbose: print(f"  FATIGUE_PERCLOS at {fat}")
    return res, fired
if __name__ == "__main__":
    for v in vids:
        print("==", v); run(v)
