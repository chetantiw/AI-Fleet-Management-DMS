"""
Who is driving, and when the driver changes.

Fed one face embedding at a time (about 2 per second, only from good-quality face crops).
- Identifies the driver against the enrolled gallery (frs.Gallery); faces that match nobody
  become "Unknown #1", "Unknown #2"... and are remembered for the session, so the same
  stranger coming back keeps the same number.
- A driver is confirmed after `confirm_n` agreeing samples  -> DRIVER_VERIFIED / UNKNOWN_DRIVER.
- A different confirmed person in the seat                 -> DRIVER_CHANGED (+ the above).
- The seat being empty for `empty_reset_s` forgets the current driver, so whoever sits
  down next is identified afresh.
"""
import collections

import numpy as np

import frs

CFG = dict(
    match=frs.MATCH_THR,     # same person if cosine similarity >= this (0.35; LFW: 0% false accept)
    confirm_n=4,             # agreeing samples needed to confirm an identity (~2 s)
    change_n=4,              # samples of someone else in a row before a change is accepted
    empty_reset_s=3.0,
)


class DriverIdentity:
    def __init__(self, gallery=None, cfg=CFG):
        self.c = cfg
        self.gal = gallery
        self.unknowns = []                 # list of [label, [embeddings]]
        self.current = None                # confirmed label
        self.current_sim = 0.0
        self.template = None               # mean embedding of the confirmed driver this session
        self.votes = collections.deque(maxlen=max(cfg["confirm_n"], cfg["change_n"]))
        self.last_face = -1e9

    def _who(self, e):
        """Best label for an embedding: enrolled name, known unknown, or None."""
        name, sim = (self.gal.match(e) if self.gal is not None and self.gal.names else (None, 0.0))
        if name:
            return name, sim
        best, bs = None, 0.0
        for lab, embs in self.unknowns:
            s = float(np.max(np.array(embs) @ e))
            if s > bs:
                best, bs = lab, s
        if best is not None and bs >= self.c["match"]:
            return best, bs
        return None, max(sim, bs)

    def no_face(self, t):
        if self.current is not None and t - self.last_face >= self.c["empty_reset_s"]:
            self.current, self.template = None, None
            self.votes.clear()

    def update(self, t, e, good=True):
        """e: unit-length embedding. Returns list of (event, label, similarity)."""
        self.last_face = t
        if not good:
            return []
        ev = []
        # still the current driver?
        if self.template is not None and float(self.template @ e) >= self.c["match"]:
            self.votes.append(self.current)
            self.template = self.template * 0.9 + e * 0.1
            self.template /= np.linalg.norm(self.template)
            return ev
        label, sim = self._who(e)
        if label is None:
            label = f"cand-{len(self.unknowns) + 1}"          # numbered only once confirmed
            self.unknowns.append([label, [e]])
        else:
            for lab, embs in self.unknowns:
                if lab == label and len(embs) < 20:
                    embs.append(e)
        self.votes.append(label)
        need = self.c["confirm_n"] if self.current is None else self.c["change_n"]
        recent = list(self.votes)[-need:]
        if len(recent) == need and all(v == label for v in recent) and label != self.current:
            if label.startswith("cand-"):
                n = sum(1 for u in self.unknowns if u[0].startswith("Unknown")) + 1
                for u in self.unknowns:
                    if u[0] == label:
                        u[0] = f"Unknown #{n}"
                self.votes = collections.deque([f"Unknown #{n}" if v == label else v for v in self.votes],
                                               maxlen=self.votes.maxlen)
                label = f"Unknown #{n}"
            prev = self.current
            self.current, self.current_sim, self.template = label, sim, e.copy()
            if prev is not None:
                ev.append(("DRIVER_CHANGED", f"{prev} -> {label}", sim))
            ev.append(("UNKNOWN_DRIVER" if label.startswith("Unknown") else "DRIVER_VERIFIED", label, sim))
        return ev
