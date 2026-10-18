import numpy as np, dms
from labels_bus import L
def load(v, S):
    d = np.load(f"{S}/bus2/x/{v}.npz"); t = d["t"]; K = list(d["keep"]); lm = d["lm"]; bs = d["bs"]; nm = list(d["bs_names"])
    P = lambda i: lm[:, K.index(i)]
    n = lambda a, b: np.linalg.norm(a - b, axis=1)
    def ear(idx):
        p = [P(i) for i in idx]; return (n(p[1], p[5]) + n(p[2], p[4])) / (2 * n(p[0], p[3]) + 1e-6)
    kps, box = d["kps"], d["box"]
    le, re, no, ml, mr = [kps[:, i] for i in range(5)]
    eyew = re[:, 0] - le[:, 0]
    F = dict(t=t, face=~np.isnan(box[:, 0]), mesh=~np.isnan(lm[:, 0, 0]),
             ear=(ear(dms.EYE_L) + ear(dms.EYE_R)) / 2,
             blink=(bs[:, nm.index("eyeBlinkLeft")] + bs[:, nm.index("eyeBlinkRight")]) / 2,
             down=(bs[:, nm.index("eyeLookDownLeft")] + bs[:, nm.index("eyeLookDownRight")]) / 2,
             jaw=bs[:, nm.index("jawOpen")],
             yawp=((no[:, 0] - le[:, 0]) / (eyew + 1e-6) - 0.5) * 2,
             eyespan=eyew / (box[:, 2] - box[:, 0] + 1e-6),
             pitchp=(no[:, 1] - (le[:, 1] + re[:, 1]) / 2) / ((ml[:, 1] + mr[:, 1]) / 2 - (le[:, 1] + re[:, 1]) / 2 + 1e-6),
             fw=box[:, 2] - box[:, 0], obj=d["obj"], face_box=box)
    try:
        h = np.load(f"{S}/bus2/x/{v}.hands.npy")
        j = np.clip(np.searchsorted(h[:, 0], t), 0, len(h) - 1)
        F["hmouth"], F["hear"] = h[j, 1], h[j, 2]
    except FileNotFoundError:
        F["hmouth"] = F["hear"] = np.full(len(t), np.nan)
    G = {}
    for k, iv in L[v].items():
        m = np.zeros(len(t), bool)
        for a, b in iv: m |= (t >= a - 0.5) & (t <= b + 0.5)
        G[k] = m
    return F, G
