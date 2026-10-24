#!/usr/bin/env python3
"""
Hardware self-test: checks each part of the DMS on the Pi and says what is wrong.

  dms selftest            (or: python3 selftest.py)

1. camera opens and delivers frames (saves data/selftest_camera.jpg so you can check focus/framing)
2. AI HAT is present and the phone detector runs on it
3. all model files load
4. one full processing pass on live frames, with the speed of each stage
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import app  # noqa: E402

OK, BAD = "\033[1;32mOK\033[0m", "\033[1;31mFAIL\033[0m"


def main():
    cfg = app.load_config(os.path.join(HERE, "config.json"))
    src_override = sys.argv[1] if len(sys.argv) > 1 else None
    failed = 0

    print("1. Camera ...", flush=True)
    import sources
    try:
        src = sources.open_source(cfg["camera"], src_override)
        frames, t0 = [], time.time()
        while len(frames) < 30 and time.time() - t0 < 10:
            f, _ = src.read()
            if f is not None:
                frames.append(f)
        cam_fps = len(frames) / (time.time() - t0)
        os.makedirs(cfg["data_dir"], exist_ok=True)
        snap = os.path.join(cfg["data_dir"], "selftest_camera.jpg")
        cv2.imwrite(snap, frames[-1])
        sharp = cv2.Laplacian(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        print(f"   {OK} {frames[-1].shape[1]}x{frames[-1].shape[0]}, {cam_fps:.1f} fps, sharpness {sharp:.0f} "
              f"(focus by turning the lens until this is highest). Photo: {snap}")
    except Exception as e:
        print(f"   {BAD} {e}\n   -> check the cable direction at both ends, then: rpicam-hello --list-cameras")
        return 1

    print("2. AI HAT ...", flush=True)
    if not os.path.exists("/dev/hailo0"):
        print(f"   {BAD} /dev/hailo0 not found -> is the HAT seated on the PCIe connector? sudo reboot; "
              f"hailortcli fw-control identify")
        failed += 1
    elif not os.path.exists(cfg.get("hef", "")):
        print(f"   {BAD} model {cfg.get('hef')} missing -> sudo apt install hailo-models")
        failed += 1
    else:
        try:
            import hailo_objects
            h = hailo_objects.HailoObjects(cfg["hef"])
            t0 = time.time()
            for f in frames[:10]:
                dets = h(f)
            ms = (time.time() - t0) / 10 * 1000
            h.close()
            print(f"   {OK} detector runs on the AI HAT, {ms:.0f} ms per frame, {len(dets)} objects in last frame")
        except Exception as e:
            print(f"   {BAD} {type(e).__name__}: {e}")
            failed += 1

    print("3+4. Models and one processing pass ...", flush=True)
    try:
        src.close()
        import perception
        perc = perception.Perception(cfg["models_dir"], cfg.get("face_det_size", 320), cfg.get("hef"),
                                     cfg.get("use_hailo", True), log=lambda *m: print("  ", *m))
        src = sources.open_source(cfg["camera"], src_override)
        n, t0, faces = 0, time.time(), 0
        while time.time() - t0 < 8:
            f, t = src.read()
            if f is None:
                break
            feat, ex = perc(f, t)
            faces += feat["face"]
            n += 1
        dt_ = time.time() - t0
        src.close()
        perc.close()
        print(f"   {OK} {n / dt_:.1f} frames/s overall, face seen in {faces}/{n} frames; "
              f"ms per stage {perc.timings_ms()}")
        if faces == 0:
            print("   (no face seen: sit in front of the camera for this test)")
    except Exception as e:
        print(f"   {BAD} {type(e).__name__}: {e}")
        failed += 1

    print("\nAll checks passed." if not failed else f"\n{failed} check(s) failed - see above.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
