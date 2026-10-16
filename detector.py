"""Async MediaPipe hand-landmark detector with auto-download and cross-platform verification."""

import os
import threading
from dataclasses import dataclass
from typing import Optional

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

import numpy as np

from config import Config


@dataclass(frozen=True)
class DetectionPacket:
    timestamp_ms: int
    result: object


class HandDetector:
    """MediaPipe live-stream wrapper with timestamped latest-result storage."""

    def __init__(self, cfg: Config):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "mediapipe is not installed in the current Python environment.\n"
                "Please install dependencies via: pip install -r requirements.txt"
            )

        model_path = cfg.model_path
        if not os.path.exists(model_path):
            try:
                import urllib.request
                url = (
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                    "hand_landmarker/float16/1/hand_landmarker.task"
                )
                print(f"[HandDetector] Model not found at '{model_path}'. Attempting auto-download from Google Cloud CDN...")
                os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
                urllib.request.urlretrieve(url, model_path)
                print(f"[HandDetector] Successfully downloaded hand_landmarker.task to '{model_path}'")
            except Exception as e:
                raise RuntimeError(
                    f"MediaPipe hand landmarker model not found at:\n  '{model_path}'\n\n"
                    f"Please download 'hand_landmarker.task' from:\n"
                    f"  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
                    f"and place it in your project folder, or set the VISION_GESTURE_MODEL_PATH environment variable."
                ) from e

        self._condition = threading.Condition()
        self._latest: Optional[DetectionPacket] = None
        self._closed = False

        def result_callback(result, _output_image, timestamp_ms):
            packet = DetectionPacket(timestamp_ms=timestamp_ms, result=result)
            with self._condition:
                self._latest = packet
                self._condition.notify()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=cfg.min_hand_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            result_callback=result_callback,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect_async(self, rgb_frame: np.ndarray, timestamp_ms: int) -> None:
        if self._closed:
            return
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.detector.detect_async(mp_image, timestamp_ms)

    def get_latest_result(self) -> Optional[DetectionPacket]:
        with self._condition:
            return self._latest

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.detector.close()
