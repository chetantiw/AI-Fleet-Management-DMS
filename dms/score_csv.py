import csv, sys
from labels_bus import L
from eval_bus import episodes
WANT = {"DISTRACTED_LOOK_AWAY": ("side", 2), "DROWSY_EYES_CLOSED": ("closed", 2), "DRINKING": ("drink", 1),
        "PHONE_CALL": ("call", 2), "PHONE_IN_HAND": ("hand", 2), "DRIVER_ABSENT": ("absent", 3)}
def score(path, v, col):
    rows = list(csv.DictReader(open(path)))
    al = [(float(r["seconds"]), r[col]) for r in rows]
    out = {}
    for ev, (lab, mn) in WANT.items():
        gt = [ep for ep in episodes(L[v][lab]) if ep[1] - ep[0] + 1 >= mn]
        ts = [t for t, e in al if e == ev]
        hit = sum(any(a - 2 <= t <= b + 3 for t in ts) for a, b in gt)
        allg = episodes(L[v][lab])
        fa = sum(not any(a - 2 <= t <= b + 3 for a, b in allg) for t in ts)
        out[ev] = (hit, len(gt), fa)
    out["FATIGUE_PERCLOS"] = (None, None, sum(1 for t, e in al if e == "FATIGUE_PERCLOS"))
    return out
