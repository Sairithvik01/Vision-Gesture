"""Threaded webcam capture — freshest-frame producer with cross-platform backend support."""

import os
import platform
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import Config


@dataclass(frozen=True)
class FramePacket:
    timestamp_ms: int
    frame: np.ndarray


def _open_camera_cross_platform(cam_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Intelligently select the best camera backend for Windows, macOS, or Linux,

    with automatic fallback and device index scanning.
    """
    system_name = platform.system().lower()

    if "windows" in system_name:
        preferred_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif "darwin" in system_name:
        preferred_backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif "linux" in system_name:
        preferred_backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        preferred_backends = [cv2.CAP_ANY]

    cap = None
    # 1. Try preferred backends on requested cam_index
    for backend in preferred_backends:
        try:
            cap = cv2.VideoCapture(cam_index, backend)
            if cap.isOpened():
                break
            cap.release()
        except Exception:
            pass

    # 2. If requested cam_index failed, scan indices 0, 1, 2
    if cap is None or not cap.isOpened():
        for fallback_idx in [0, 1, 2]:
            if fallback_idx == cam_index:
                continue
            for backend in preferred_backends:
                try:
                    cap = cv2.VideoCapture(fallback_idx, backend)
                    if cap.isOpened():
                        print(f"[Camera] Device index {cam_index} unavailable; auto-switched to index {fallback_idx}")
                        break
                    cap.release()
                except Exception:
                    pass
            if cap is not None and cap.isOpened():
                break

    if cap is None or not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {cam_index} or any fallback indices (0, 1, 2).\n"
            f"Please verify webcam connection and OS camera privacy permissions.\n"
            f"(macOS: System Settings -> Privacy & Security -> Camera;\n"
            f" Windows: Settings -> Privacy -> Camera;\n"
            f" Linux: verify user is in 'video' group via 'sudo usermod -aG video $USER')."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class WebcamStream:
    """Continuously capture only the newest camera frame.

    A bounded one-frame buffer intentionally drops stale frames. For an
    interactive controller, latency matters more than processing every frame.
    """

    def __init__(self, cfg: Config):
        self.cap = _open_camera_cross_platform(
            cam_index=cfg.cam_index,
            width=cfg.cam_width,
            height=cfg.cam_height,
            fps=cfg.cam_fps,
        )

        self._condition = threading.Condition()
        self._latest: Optional[FramePacket] = None
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "WebcamStream":
        self._thread = threading.Thread(
            target=self._update,
            name="webcam-capture",
            daemon=True,
        )
        self._thread.start()
        return self

    def _update(self) -> None:
        while not self._stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.01)
                continue

            timestamp_ms = time.monotonic_ns() // 1_000_000
            packet = FramePacket(timestamp_ms=timestamp_ms, frame=frame)

            with self._condition:
                self._latest = packet
                self._condition.notify()

    def read_latest(self, last_timestamp_ms: int = -1) -> Optional[FramePacket]:
        """Return newest frame newer than last_timestamp_ms, if available."""
        with self._condition:
            packet = self._latest
            if packet is None or packet.timestamp_ms <= last_timestamp_ms:
                return None
            return FramePacket(packet.timestamp_ms, packet.frame.copy())

    def stop(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()
