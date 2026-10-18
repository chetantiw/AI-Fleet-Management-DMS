# Test results (as of 18 Oct 2026)

All results are from CPU runs on recorded video. Labels for the bus videos were made by
reviewing one frame per second (`dms/labels_bus.py`), so short events may be missed by the labels.

## Bus videos, 13 Oct 2026 (3 drivers, ~14 min, camera mounted high on the windscreen)

Episode-level, all three videos together. "False" = alert with no matching labelled event.

| Alert | v1 (11 Oct) | Current |
|---|---|---|
| Look away | 2 / 33 caught, 4 false | 23 / 33 caught, 6 false |
| Eyes closed | 1 / 3, 7 false | 3 / 3, 2 false |
| Drinking | not built | 5 / 9, 2 false |
| Phone call | 1 / 2, 0 false | 2 / 2, 6 false (bottle / sunglasses near face read as phone) |
| Phone in hand | 1 / 2, 3 false | 2 / 2, 2 false |
| Driver absent | 1 / 3, 1 false | 2 / 3, 1 false |
| Fatigue (PERCLOS) | 38 alerts, all false | 3 alerts |
| No seat belt | not built | alerts at 20 s / 80 s / 146 s (IMG_0107) and 20 s (IMG_0112) while unbuckled; 0 false while worn |

Tuning used IMG_0110 and IMG_0112; IMG_0107 is the closest to a held-out test
(look away 8 / 11, drinking 2 / 4).

Seat belt, per-frame score: belt off ~0.02–0.06, belt on ~0.23–0.39 (threshold 0.15). With 10 s
smoothing the state was right 100 % of the time while worn and 91–100 % while unbuckled.

Sunglasses check (58 labelled eye crops): 15 / 16 sunglasses caught, 1 / 42 plain eyes flagged.

## Driver identification

- Same driver across frames: 94–97 % of sampled frames matched (similarity ≥ 0.35).
- Different drivers: highest cross-driver similarity 0.28 (threshold 0.35), no false matches.
- 4-change session (A → unenrolled C → empty seat → B → C): every driver identified ~2 s after a
  clear face; A stayed recognised through 80 s of sunglasses; C recognised as the same
  "Unknown #1" on return (optimistic: identical footage replayed).
- LFW benchmark (6,000 pairs): 99.7 % accuracy; at threshold 0.35, 0.00 % false accept,
  2.1 % false reject.

## Public DMD driver dataset (6 drivers, ~3 min each, v1 pipeline)

- Closed-eye frames detected: 95–100 %; open eyes misread as closed: 0–1.7 %.
- Eye closures ≥ 1.5 s: 34 labelled, 39 alerts.
- Phone calls: 4 / 4 episodes; texting with the phone below the camera view: 0 / 4.

## Open items

1. Liveness check before identity is tied to the alcohol test.
2. Train a cabin-specific detector (phone, bottle, hand at face) on reviewed footage.
3. Port to Hailo-8L when the hardware is ready; re-test on the same clips.
4. Send alerts, identity and alcohol readings to the fleet backend.
