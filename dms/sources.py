"""
Frame sources: the Pi camera (Picamera2 / libcamera), a USB webcam, or a video file.
All return BGR frames (what OpenCV expects) and the frame's time in seconds.
"""
import time

import cv2


class PiCamera:
    def __init__(self, width=1280, height=960, fps=15, rotate=0, hflip=False, vflip=False):
        from picamera2 import Picamera2
        from libcamera import Transform
        self.cam = Picamera2()
        # libcamera can flip; 90/270 rotation is done in software
        tr = Transform(hflip=bool(hflip) ^ (rotate == 180), vflip=bool(vflip) ^ (rotate == 180))
        cfg = self.cam.create_video_configuration(
            main={"size": (int(width), int(height)), "format": "RGB888"},   # RGB888 = BGR bytes
            controls={"FrameRate": float(fps)}, transform=tr, buffer_count=4)
        self.cam.configure(cfg)
        self.cam.start()
        self.rot = {90: cv2.ROTATE_90_CLOCKWISE, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(int(rotate))
        self.t0 = time.monotonic()
        self.live = True

    def read(self):
        f = self.cam.capture_array("main")
        if f.shape[2] == 4:
            f = f[:, :, :3]
        if self.rot is not None:
            f = cv2.rotate(f, self.rot)
        return f, time.monotonic() - self.t0

    def close(self):
        self.cam.stop()
        self.cam.close()


class OpenCVSource:
    """USB webcam (index) or video file (path). Files are read as fast as the app can process
    them, with time taken from the file (so alert timings are right even if processing is slow),
    optionally skipping frames to match `fps`."""
    def __init__(self, src, width=None, height=None, fps=15, rotate=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {src}")
        self.file = not isinstance(src, int)
        if not self.file and width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.step = max(int(round(self.src_fps / fps)), 1) if self.file else 1
        self.i = 0
        self.rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(int(rotate))
        self.t0 = time.monotonic()
        self.live = not self.file

    def read(self):
        while self.file and self.i % self.step:
            if not self.cap.grab():
                return None, None
            self.i += 1
        ok, f = self.cap.read()
        if not ok:
            return None, None
        t = self.i / self.src_fps if self.file else time.monotonic() - self.t0
        self.i += 1
        if self.rot is not None:
            f = cv2.rotate(f, self.rot)
        return f, t

    def close(self):
        self.cap.release()


def open_source(cam_cfg, override=None):
    """override: None (use config), 'picamera', a webcam index, or a video file path."""
    src = override if override is not None else cam_cfg.get("source", "picamera")
    if src == "picamera":
        return PiCamera(cam_cfg.get("width", 1280), cam_cfg.get("height", 960), cam_cfg.get("fps", 15),
                        cam_cfg.get("rotate", 0), cam_cfg.get("hflip", False), cam_cfg.get("vflip", False))
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    return OpenCVSource(src, cam_cfg.get("width"), cam_cfg.get("height"), cam_cfg.get("fps", 15),
                        cam_cfg.get("rotate", 0))
