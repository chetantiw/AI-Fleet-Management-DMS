"""
Live video from the vehicle to the media server (MediaMTX), on demand.

The device PUSHES the stream out (RTSP over TCP, or SRT) to the server. Vehicles on 4G sit behind
the operator's NAT and have no public address, so the server can never pull from them; an
outgoing connection from the device always works.

  s = LiveStream("rtsp://veh-102:secret@iot.example.com:8554/veh-102", width=640, fps=10)
  s.start(minutes=5)          # usually from the dashboard's "Live video" button (RPC)
  s.push(frame_bgr)           # every frame of the DMS loop (annotated or raw); cheap when stopped
  s.stop()

Encoding: H.264 with ffmpeg/libx264 (ultrafast, zerolatency), keyframe every 2 s, bitrate capped
(default 400 kbit/s ~ 180 MB per hour of viewing). The Raspberry Pi 5 has no hardware H.264
encoder, so this costs about half a CPU core at 640x360/10 fps; the AX8850 has a hardware encoder
(swap the codec arguments when porting). Streaming stops by itself after `minutes` so a forgotten
browser tab cannot burn the SIM's data.

Frames are handed over through a 1-frame slot: if the encoder or the network is slow, frames are
dropped instead of slowing the DMS loop.
"""
import logging
import shutil
import subprocess
import threading
import time

import cv2

log = logging.getLogger("streamer")


class LiveStream:
    def __init__(self, url, width=640, fps=10, bitrate_k=400, max_minutes=10, ffmpeg="ffmpeg"):
        self.url, self.width, self.fps = url, width, fps
        self.bitrate_k, self.max_minutes = bitrate_k, max_minutes
        self.ffmpeg = shutil.which(ffmpeg) or ffmpeg
        self.proc = None
        self.until = 0.0
        self.size = None
        self.slot = None
        self.cv = threading.Condition()
        self.lock = threading.Lock()
        self.started_at = None
        self.frames = 0

    # ---------------------------------------------------------------- control
    @property
    def active(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, minutes=None):
        """Start (or extend) streaming. Returns status dict for the dashboard."""
        minutes = min(float(minutes or self.max_minutes), 60.0)
        with self.lock:
            self.until = time.time() + minutes * 60
            if not self.active:
                self.proc = None            # the encoder opens on the next frame (needs its size)
                self.started_at = time.time()
                self.frames = 0
        log.info("live stream on for %.0f min -> %s", minutes, self._safe_url())
        return self.status()

    def stop(self):
        with self.lock:
            self.until = 0.0
            self._close()
        return self.status()

    def status(self):
        on = self.until > time.time()
        return {"streaming": on, "seconds_left": int(max(self.until - time.time(), 0)) if on else 0,
                "path": self.url.rsplit("/", 1)[-1], "frames": self.frames}

    # ---------------------------------------------------------------- frames
    def push(self, frame):
        if self.until <= time.time():
            if self.proc is not None:
                with self.lock:
                    self._close()
                log.info("live stream stopped (time limit)")
            return
        with self.lock:
            if not self.active:
                if self.proc is not None:
                    log.warning("encoder exited (%s); reconnecting", self.proc.returncode)
                self._open(frame)
        with self.cv:
            self.slot = frame            # newest frame wins; older one is dropped if unsent
            self.cv.notify()

    def _open(self, frame):
        h, w = frame.shape[:2]
        ow = min(self.width, w) // 2 * 2
        oh = int(round(h * ow / w)) // 2 * 2
        self.size = (ow, oh)
        proto = ["-rtsp_transport", "tcp", "-f", "rtsp"] if self.url.startswith("rtsp") else \
                ["-f", "mpegts"]
        cmd = [self.ffmpeg, "-loglevel", "error", "-nostdin",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", str(self.fps), "-i", "-",
               "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
               "-profile:v", "baseline", "-pix_fmt", "yuv420p",
               "-b:v", f"{self.bitrate_k}k", "-maxrate", f"{self.bitrate_k}k", "-bufsize", f"{self.bitrate_k * 2}k",
               "-g", str(self.fps * 2), "-keyint_min", str(self.fps * 2),
               *proto, self.url]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._writer, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._stderr, args=(self.proc,), daemon=True).start()

    def _writer(self, proc):
        while proc.poll() is None:
            with self.cv:
                self.cv.wait_for(lambda: self.slot is not None or proc.poll() is not None, timeout=1.0)
                frame, self.slot = self.slot, None
            if frame is None:
                continue
            if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            try:
                proc.stdin.write(frame.tobytes())
                self.frames += 1
            except (BrokenPipeError, ValueError, OSError):
                break

    def _stderr(self, proc):
        for line in proc.stderr:
            log.warning("ffmpeg: %s", line.decode(errors="replace").strip())

    def _close(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.stdin.close()
        except OSError:
            pass
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()

    def _safe_url(self):
        if "@" in self.url:
            scheme, rest = self.url.split("://", 1)
            return f"{scheme}://***@{rest.split('@', 1)[1]}"
        return self.url
