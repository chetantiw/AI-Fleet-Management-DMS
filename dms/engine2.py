"""
DMS alert engine v2, retuned on the Oct-13 bus videos (frame-by-frame labels in labels_bus.py).

Changes from v1 (dms.DmsEngine):
- Head turn from the face detector's 5 key points (nose position between the eyes, and how
  much the eye span shrinks), relative to this driver's own normal. Works even when the
  detailed face mesh is lost, which is exactly what happens when the head turns.
- Eye closure from the mesh's eye-blink and eye-look-down scores (MediaPipe blendshapes),
  smoothed over 1 s and compared with this driver's own rolling normal. Not judged while the
  head is turned, while sunglasses hide the eyes, or while a hand/bottle is at the face.
- DRINKING: a hand (or bottle) at the mouth for >= 1 s.
- PHONE_CALL: a hand at the ear, or a phone beside the head, for >= 2 s.
- Fatigue (PERCLOS) only counts eye closure while looking ahead, and reminds at most every 2 min.
"""
import collections

import numpy as np

CFG = dict(
    baseline_s=120.0,      # rolling window for "this driver's normal"
    warmup_s=5.0,
    turn_on=0.6,           # head-turn score above this = looking away (score ~0 when facing ahead)
    look_away_s=2.0,
    closed_on=0.32,        # smoothed (blink + look-down) above the driver's normal by this = eyes closed
    closed_s=1.5,
    closed_confirm_s=1.0,
    perclos_window_s=60.0,
    perclos_limit=0.25,
    hand_mouth=0.45,       # hand landmark within this many face-widths of the mouth
    hand_ear=0.35,         # hand beside the face (within this many face-widths of the face edge)
    drink_s=2.0,           # hand at mouth in >= drink_frac of the last 2 s
    drink_frac=0.4,        # (hand detector sees a hand in only ~1/3 to 1/2 of frames, so count, do not require)
    call_s=3.0,
    call_frac=0.2,
    call_frac_no_phone=1.01, # hand at ear with NO phone seen never counts as a call (it was hair, glasses, earphones)
    hand_frac=0.3,           # phone seen in >= 30% of detector runs over 3 s = phone in hand
    away_frac=0.6,         # turned in >= 60% of the last look_away_s (tuned: 16/22 caught on 2 videos)
    phone_score=0.35,
    no_face_s=3.0,
    belt_on=0.15,          # per-frame belt score above this = belt seen
    belt_window_s=10.0,    # belt counts as worn if seen in >= belt_frac of the last 10 s
    belt_frac=0.2,
    no_belt_s=10.0,        # NO_SEATBELT after 10 s without a belt (driver seated)
    belt_cooldown_s=60.0,  # then remind every minute
    min_face_frac=0.08,    # driver's face is >= 8% of frame width; passengers behind are ~5%
    cooldown_s=8.0,
    fatigue_cooldown_s=120.0,
)


class Rolling:
    """Median of samples taken every `every` seconds over the last `span` seconds."""
    def __init__(self, span, every=0.5):
        self.span, self.every, self.q, self.last = span, every, collections.deque(), -1e9

    def add(self, t, x):
        if x is None or np.isnan(x) or t - self.last < self.every:
            return
        self.last = t
        self.q.append((t, x))
        while t - self.q[0][0] > self.span:
            self.q.popleft()

    def median(self, default=np.nan):
        return float(np.median([x for _, x in self.q])) if len(self.q) >= 5 else default


class Engine2:
    def __init__(self, cfg=CFG):
        self.c = cfg
        self.base_yaw, self.base_span = Rolling(cfg["baseline_s"]), Rolling(cfg["baseline_s"])
        self.base_close = Rolling(cfg["baseline_s"])
        self.smooth = collections.deque()           # (t, closure score) over 1 s
        self.since = {}                             # condition -> start time
        self.last_fired = {}
        self.perclos = collections.deque()
        self.t0 = None
        self.win = {}
        self.pending_closed = None
        self.belt_q = collections.deque()
        self.belt_worn = None
        self.phone_hist = collections.deque()
        self.base_fw = Rolling(cfg["baseline_s"])
        self.state = {}

    def _phone(self, t, pb, fb):
        """Filter one phone detection. Returns (seen, at_head).
        Rejected: low score, a box on the face itself (sunglasses), and fixed objects - a box that
        keeps reappearing at the same spot in several separate 20-s blocks of the last 3 min
        (steering-wheel hub, a phone lying in a holder). A phone held in the hand moves around."""
        c = self.c
        if pb is None or fb is None or np.isnan(pb[4]) or pb[4] < c["phone_score"] or np.isnan(fb[0]):
            return False, False
        fw = fb[2] - fb[0]
        px, py = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
        fx, fy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
        if fb[0] < px < fb[2] and fb[1] < py < fb[3] and abs(px - fx) < 0.3 * fw:
            return False, False
        # positions relative to the face, so a moving head does not move "fixed" objects
        rx, ry = (px - fx) / fw, (py - fy) / fw
        hist = self.phone_hist
        hist.append((t, rx, ry))
        while t - hist[0][0] > 180:
            hist.popleft()
        blocks = {int(tt // 20) for tt, x, y in hist if abs(x - rx) < 0.35 and abs(y - ry) < 0.35}
        if len(blocks) >= 4:
            return False, False
        at_head = abs(ry) < 0.8 and 0.3 < abs(rx) < 1.3
        return True, at_head

    def _frac(self, name, on, t, span):
        """Share of the last `span` seconds in which `on` was true (window must be mostly full)."""
        q = self.win.setdefault(name, collections.deque())
        q.append((t, bool(on)))
        while t - q[0][0] > span:
            q.popleft()
        if t - q[0][0] < span * 0.8:
            return 0.0
        return sum(x for _, x in q) / len(q)

    def _hold(self, name, on, t, need):
        if on:
            self.since.setdefault(name, t)
            return t - self.since[name] >= need
        self.since.pop(name, None)
        return False

    def _fire(self, name, t, ev):
        gap = (self.c["fatigue_cooldown_s"] if name.startswith("FATIGUE") else
               self.c["belt_cooldown_s"] if name == "NO_SEATBELT" else self.c["cooldown_s"])
        if t - self.last_fired.get(name, -1e9) >= gap:
            self.last_fired[name] = t
            ev.append(name)

    def update(self, t, f):
        """f: dict with face, mesh (bools), yawp, eyespan, blink, down, eyes_hidden,
        hmouth, hear (face-widths, nan if no hand), phone (score, nan if not run), phone_at_head (bool)."""
        c, ev = self.c, []
        self.t0 = t if self.t0 is None else self.t0
        # a face much smaller than this driver's usual face is a passenger further back, not the driver
        fw = f.get("fw", np.nan)
        usual = self.base_fw.median(np.nan)
        too_small = (not np.isnan(fw) and f.get("frame_w") and fw < c["min_face_frac"] * f["frame_w"])
        if f["face"] and (too_small or (not np.isnan(fw) and not np.isnan(usual) and fw < 0.6 * usual)):
            f = dict(f, face=False)
        if not f["face"]:
            if self._hold("absent", True, t, c["no_face_s"]):
                self._fire("DRIVER_ABSENT", t, ev)
            for k in ("away", "closed", "drink", "call"):
                self.since.pop(k, None)
            self.state = dict(face=False)
            return ev
        self.since.pop("absent", None)

        # seat belt (scored at ~2 Hz; NaN on frames where it was not run)
        bsc = f.get("belt", np.nan)
        if bsc is not None and not np.isnan(bsc):
            self.belt_q.append((t, bsc > c["belt_on"]))
        while self.belt_q and t - self.belt_q[0][0] > c["belt_window_s"]:
            self.belt_q.popleft()
        if len(self.belt_q) >= 4:
            self.belt_worn = bool(np.mean([x for _, x in self.belt_q]) >= c["belt_frac"])
        if t - self.t0 >= c["no_belt_s"] and self._hold("nobelt", self.belt_worn is False, t, c["no_belt_s"]):
            self._fire("NO_SEATBELT", t, ev)

        # head turn relative to this driver's normal
        by, bs = self.base_yaw.median(0.0), self.base_span.median(f["eyespan"])
        turn = max(abs(f["yawp"] - by) / 1.0, (1 - f["eyespan"] / bs) / 0.4) if not np.isnan(f["yawp"]) else 0.0
        facing = turn < c["turn_on"]

        hand_mouth = (not np.isnan(f["hmouth"])) and f["hmouth"] < c["hand_mouth"]
        hand_ear = (not np.isnan(f["hear"])) and f["hear"] < c["hand_ear"]
        if f.get("phone_run"):
            self.phone_state = self._phone(t, f.get("phone_box"), f.get("face_box"))
        phone_seen, phone_head = getattr(self, "phone_state", (False, False))
        if phone_seen:
            self.last_phone = t
        phone_recent = t - getattr(self, "last_phone", -1e9) <= 3.0

        # eye closure, only when it can be judged
        judge = facing and f["mesh"] and not f["eyes_hidden"] and not hand_mouth and not phone_recent
        score = np.nan
        if judge:
            raw = f["blink"] + f["down"]
            self.smooth.append((t, raw))
            while t - self.smooth[0][0] > 1.0:
                self.smooth.popleft()
            score = float(np.median([x for _, x in self.smooth])) - self.base_close.median(raw)
        closed = judge and not np.isnan(score) and score > c["closed_on"]

        # update baselines with "normal" frames only
        if facing and f["mesh"] and not closed:
            self.base_fw.add(t, fw)
            self.base_yaw.add(t, f["yawp"])
            self.base_span.add(t, f["eyespan"])
        if judge and not closed:
            self.base_close.add(t, f["blink"] + f["down"])

        warm = t - self.t0 >= c["warmup_s"]
        if warm and self._frac("away", not facing and not (hand_mouth or hand_ear), t, c["look_away_s"]) >= c["away_frac"]:
            self._fire("DISTRACTED_LOOK_AWAY", t, ev)
        # Confirm eye closure 1 s later: if a phone or a hand at the face shows up in that second,
        # the driver was looking down at the phone / drinking, not asleep.
        if warm and self._hold("closed", closed, t, c["closed_s"]) and self.pending_closed is None:
            self.pending_closed = t
        if self.pending_closed is not None and t - self.pending_closed >= c["closed_confirm_s"]:
            if not (phone_recent or hand_mouth or hand_ear):
                self._fire("DROWSY_EYES_CLOSED", t, ev)
            self.pending_closed = None
        drink_frac = self._frac("drink", hand_mouth, t, c["drink_s"])
        call_frac = self._frac("call", hand_ear or phone_head, t, c["call_s"])
        phone_frac = self._frac("phone", phone_seen, t, c["call_s"])
        # A phone in view decides between the two hand-at-face cases:
        #   phone seen  -> PHONE_CALL (hand at ear) / PHONE_IN_HAND
        #   no phone    -> DRINKING (hand at mouth); hand at the ear alone only counts if it stays a long time
        if phone_frac > 0:
            if call_frac >= c["call_frac"]:
                self._fire("PHONE_CALL", t, ev)
            elif phone_frac >= c["hand_frac"]:
                self._fire("PHONE_IN_HAND", t, ev)
        else:
            if drink_frac >= c["drink_frac"]:
                self._fire("DRINKING", t, ev)
            elif call_frac >= c["call_frac_no_phone"]:
                self._fire("PHONE_CALL", t, ev)
        if judge:
            self.perclos.append((t, closed))
        while self.perclos and t - self.perclos[0][0] > c["perclos_window_s"]:
            self.perclos.popleft()
        pc = np.mean([x for _, x in self.perclos]) if self.perclos else 0.0
        if warm and t - self.perclos[0][0] >= c["perclos_window_s"] * 0.5 if self.perclos else False:
            if pc > c["perclos_limit"]:
                self._fire("FATIGUE_PERCLOS", t, ev)
        self.state = dict(face=True, belt=self.belt_worn, turn=turn, facing=facing, closure=score, closed=closed,
                          perclos=pc, hand_mouth=hand_mouth, hand_ear=hand_ear, eyes_hidden=f["eyes_hidden"])
        return ev
