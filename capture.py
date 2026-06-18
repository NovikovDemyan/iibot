import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame_bgr: np.ndarray
    timestamp: float
    fps: float


class ScreenCapture:
    def __init__(self, region=None, prefer_dxcam=False):
        self.region = region
        self.prefer_dxcam = prefer_dxcam
        self.backend = None
        self.camera = None
        self.sct = None
        self.monitor = None
        self.last_t = None
        self.fps = 0.0

    def start(self):
        if self.prefer_dxcam:
            try:
                import dxcam
                self.camera = dxcam.create(output_color="BGR")
                self.backend = "dxcam"
                return
            except Exception as e:
                print(f"dxcam init failed, falling back to mss: {e}")
                self.camera = None

        self._start_mss()

    def _start_mss(self):
        import mss

        self.sct = mss.mss()
        if self.region is None:
            mon = self.sct.monitors[1]
            self.monitor = {
                "left": mon["left"],
                "top": mon["top"],
                "width": mon["width"],
                "height": mon["height"],
            }
        else:
            self.monitor = self.region

        self.backend = "mss"

    def read(self) -> FramePacket:
        if self.backend is None:
            raise RuntimeError("Call ScreenCapture.start() before read().")

        if self.backend == "dxcam":
            frame = self.camera.grab(region=self._dxcam_region())
            if frame is None:
                print("dxcam returned empty frame, falling back to mss.")
                self.stop()
                self.camera = None
                self.sct = None
                self._start_mss()
                return self.read()
            frame_bgr = frame

        elif self.backend == "mss":
            raw = np.array(self.sct.grab(self.monitor))
            frame_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

        now = time.perf_counter()
        if self.last_t is not None:
            dt = now - self.last_t
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = 0.9 * self.fps + 0.1 * inst_fps if self.fps else inst_fps

        self.last_t = now
        return FramePacket(frame_bgr=frame_bgr, timestamp=now, fps=self.fps)

    def _dxcam_region(self):
        if self.region is None:
            return None

        left = self.region["left"]
        top = self.region["top"]
        right = left + self.region["width"]
        bottom = top + self.region["height"]
        return left, top, right, bottom

    def stop(self):
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass

        if self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass
