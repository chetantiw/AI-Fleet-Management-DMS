#!/usr/bin/env python3
"""
Build the digital-twin telemetry stream for one video:

  python3 twin_stream.py CACHE_BASE STATE.jsonl OUT.json

CACHE_BASE is <out>/cache/<video name> (uses .npz from extract.py and .twin.npz from twin_extract.py);
STATE.jsonl is the per-frame DMS state from render_session.py --state.

Each frame becomes one small record with the same keys the device will publish to ThingsBoard
(MQTT topic v1/devices/me/telemetry). Everything is already expressed relative to this driver's
own normal, so the 3-D widget only has to apply it:

  yaw, pitch, roll   head turn in degrees vs the driver's normal (yaw +: driver turns to own left,
                     pitch +: looks down, roll +: tilts to own left)
  eyeL, eyeR         eye openness 0 (shut) .. 1 (normal open)
  jaw                mouth open 0 .. 1
  lean               [sideways, forward] torso shift vs normal, in shoulder widths
  lE, lW, rE, rW     left/right elbow and wrist relative to the shoulder centre, in shoulder widths:
                     x to the driver's left, y up, z towards the camera / windscreen
  armOk              1 if the arm points came from the camera this frame, 0 if held / defaulted
  face, facing, closed, hidden (sunglasses), phone, mouth (hand at mouth), ear (hand at ear),
  belt (1 on / 0 off / -1 checking), perclos, hand ("L"/"R": which hand is up), alert, ev, driver
"""
import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
import dms  # noqa: E402

BS = ["eyeBlinkLeft", "eyeBlinkRight", "jawOpen"]
HOLD_S = 1.0          # keep the last good arm pose this long when the camera loses the arms
ALPHA = 0.45          # smoothing of arm points (0 = frozen, 1 = raw)
HEAD_ALPHA = 0.6


def r2(v):
    return [round(float(x), 2) for x in v]


def main():
    base, state_path, out = sys.argv[1:4]
    d = np.load(base + ".npz")
    tw = np.load(base + ".twin.npz")
    st = [json.loads(line) for line in open(state_path)]
    T, LM, BSV, keep, box = d["t"], d["lm"], d["bs"], list(d["keep"]), d["box"]
    names = [str(x) for x in d["bs_names"]]
    H, W = d["frame_hw"].astype(int)
    pos = {k: i for i, k in enumerate(keep)}

    # ---- head angles per extract frame, via the repo's own solver -------------------------
    head = np.full((len(T), 3), np.nan)
    for k in range(len(T)):
        if np.isnan(LM[k, 0, 0]):
            continue
        lm = np.zeros((468, 2))
        for idx, p in pos.items():
            lm[idx] = LM[k, p]
        hp = dms.head_pose(lm, W, H)
        if hp is not None:
            head[k] = hp
    # The PnP decomposition can wrap pitch near +-180; bring it back to a small angle range
    head[:, 0] = (head[:, 0] + 180) % 360 - 180
    head[np.abs(head[:, 0]) > 90, 0] -= np.sign(head[np.abs(head[:, 0]) > 90, 0]) * 180
    blinkL = BSV[:, names.index(BS[0])]
    blinkR = BSV[:, names.index(BS[1])]
    jaw = BSV[:, names.index(BS[2])]
    # driver's normal = median over frames where the DMS says "facing the road"
    ST0 = np.array([x["t"] for x in st])
    facing = np.array([st[int(np.argmin(abs(ST0 - t)))]["facing"] for t in T])
    ok = facing & ~np.isnan(head[:, 0])
    h0 = np.nanmedian(head[ok], axis=0) if ok.sum() > 20 else np.nanmedian(head, axis=0)
    b0 = np.nanmedian((blinkL + blinkR)[ok] / 2) if ok.sum() > 20 else 0.3

    # ---- arm / shoulder points --------------------------------------------------------------
    PT = list(tw["pts"])
    IMG, TT = tw["img"], tw["t"]
    i_ls, i_rs, i_le, i_re, i_lw, i_rw, i_n = (PT.index(k) for k in (11, 12, 13, 14, 15, 16, 0))
    WRL = tw["world"]
    sw_img = np.linalg.norm(IMG[:, i_ls, :2] - IMG[:, i_rs, :2], axis=1)
    good_sh = (IMG[:, i_ls, 2] > 0.6) & (IMG[:, i_rs, 2] > 0.6)
    sh_mid = (IMG[:, i_ls, :2] + IMG[:, i_rs, :2]) / 2
    sw0 = np.nanmedian(sw_img[good_sh]) if good_sh.any() else 100.0
    mid0 = np.nanmedian(sh_mid[good_sh], axis=0) if good_sh.any() else np.array([W / 2, H / 3])

    def rel(k, j):
        """image point j of frame k -> shoulder frame (x driver-left, y up, z to camera)."""
        p = IMG[k, j, :2]
        c = sh_mid[k] if good_sh[k] else mid0
        s = sw_img[k] if good_sh[k] and sw_img[k] > 0.5 * sw0 else sw0
        x = (p[0] - c[0]) / s            # camera faces the driver: image right = driver's left
        y = -(p[1] - c[1]) / s
        # depth from the 3-D model, in shoulder widths (z < 0 means nearer the camera)
        wsw = np.linalg.norm(WRL[k, i_ls] - WRL[k, i_rs]) + 1e-6
        z = -(WRL[k, j, 2] - (WRL[k, i_ls, 2] + WRL[k, i_rs, 2]) / 2) / wsw
        return np.array([x, y, np.clip(z, -1.5, 1.5)])

    DEFAULT = dict(lE=np.array([0.75, -0.95, 0.45]), lW=np.array([0.45, -1.15, 1.1]),
                   rE=np.array([-0.75, -0.95, 0.45]), rW=np.array([-0.45, -1.15, 1.1]))
    arm = {k: v.copy() for k, v in DEFAULT.items()}
    last_good = -9.0
    ST = np.array([x["t"] for x in st])
    OBJ, KPS = d["obj"], d["kps"]
    votes = []
    frames = []
    hs = np.array([np.nan] * 3)
    for k in range(len(TT)):
        t = TT[k]
        j = int(np.argmin(abs(T - t)))
        s = st[int(np.argmin(abs(ST - t)))]
        # head
        if not np.isnan(head[j, 0]):
            hv = head[j] - h0
            hs = hv if np.isnan(hs[0]) else HEAD_ALPHA * hv + (1 - HEAD_ALPHA) * hs
        elif s["face"] is False:
            hs = np.array([0.0, 0.0, 0.0])
        hv = np.nan_to_num(hs)
        # eyes / mouth
        def openness(bv):
            if np.isnan(bv):
                return 1.0
            return float(np.clip((0.8 - bv) / max(0.8 - b0, 0.15), 0, 1))
        eL, eR = openness(blinkL[j]), openness(blinkR[j])
        if s["closed"]:
            eL = eR = 0.0
        # arms
        arm_ok = 0
        vis_arm = IMG[k, [i_le, i_re, i_lw, i_rw], 2]
        if good_sh[k] and np.nanmin(vis_arm) > 0.5:
            new = dict(lE=rel(k, i_le), lW=rel(k, i_lw), rE=rel(k, i_re), rW=rel(k, i_rw))
            for n in arm:
                arm[n] = ALPHA * new[n] + (1 - ALPHA) * arm[n]
            last_good, arm_ok = t, 1
        elif t - last_good > HOLD_S:
            for n in arm:
                arm[n] = 0.15 * DEFAULT[n] + 0.85 * arm[n]      # drift back to hands-on-wheel
        lean = [0.0, 0.0]
        if good_sh[k]:
            lean = [-(sh_mid[k, 0] - mid0[0]) / sw0, (sh_mid[k, 1] - mid0[1]) / sw0]
        # which hand is raised (for phone / drinking): vote per frame, settled per episode below
        vote = 0
        if s["mouth"] or s["ear"] or s["phone"]:
            ob = OBJ[j, 0]
            nose_x = KPS[j, 2, 0]
            if not np.isnan(ob[0]) and not np.isnan(nose_x):
                vote = 2 if (ob[0] + ob[2]) / 2 > nose_x else -2      # phone box: image right = driver's left
            else:
                n = IMG[k, i_n, :2]
                dl = np.linalg.norm(IMG[k, i_lw, :2] - n) if IMG[k, i_lw, 2] > 0.3 else 1e9
                dr = np.linalg.norm(IMG[k, i_rw, :2] - n) if IMG[k, i_rw, 2] > 0.3 else 1e9
                vote = 1 if dl < dr else -1 if dr < dl else 0
        votes.append(vote)
        hand = ""
        frames.append(dict(
            t=round(float(t), 1), yaw=round(-float(hv[1]), 1), pitch=round(float(hv[0]), 1),
            roll=round(-float(hv[2]), 1), eyeL=round(eL, 2), eyeR=round(eR, 2),
            jaw=round(float(np.clip(np.nan_to_num(jaw[j]) * 1.6, 0, 1)), 2),
            lean=r2(np.clip(lean, -1, 1)), lE=r2(arm["lE"]), lW=r2(arm["lW"]), rE=r2(arm["rE"]), rW=r2(arm["rW"]),
            armOk=arm_ok, face=int(s["face"]), facing=int(s["facing"]), closed=int(s["closed"]),
            hidden=int(s["hidden"]), phone=int(s["phone"]), mouth=int(s["mouth"]), ear=int(s["ear"]),
            belt={True: 1, False: 0, None: -1}[s["belt"]], perclos=s["perclos"], hand=hand,
            alert=s["alert"], ev=s["ev"], driver=s["driver"]))
    # one hand per episode (a run of phone / hand-at-face frames, gaps under 1 s bridged)
    active = [bool(f["phone"] or f["mouth"] or f["ear"]) for f in frames]
    k = 0
    while k < len(frames):
        if not active[k]:
            k += 1
            continue
        e = k
        while e + 1 < len(frames) and (active[e + 1] or any(active[e + 1:e + 11])):
            e += 1
        side = "L" if sum(votes[k:e + 1]) > 0 else "R"
        for q in range(k, e + 1):
            frames[q]["hand"] = side
        k = e + 1
    meta = dict(fps=10, frames=len(frames), driver_normal=dict(head_deg=r2(h0), blink=round(float(b0), 3)),
                bytes_per_frame=int(np.mean([len(json.dumps(f, separators=(",", ":"))) for f in frames])))
    with open(out, "w") as fh:
        json.dump(dict(meta=meta, frames=frames), fh, separators=(",", ":"))
    print(out, meta)


if __name__ == "__main__":
    main()
