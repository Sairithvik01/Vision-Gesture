"""Vision-Gesture — Real-time HCI Hand Gesture Control & Capstone Architecture.

Usage
-----
  python main.py          # launch real-time hand-gesture controller
  python main.py --train  # train ML gesture classifier on landmark dataset
"""

import argparse
import os
import time
from typing import Optional

import cv2
import numpy as np

try:
    import pyautogui
except ImportError:
    class _MockPyAutoGUI:
        FAILSAFE = True
        PAUSE = 0
        @staticmethod
        def size(): return (1920, 1080)
    pyautogui = _MockPyAutoGUI

from actions import ActionWorker
from camera import WebcamStream
from config import Config
from controller import HandGestureController
from cursor import CursorWorker
from detector import HandDetector
from overlay import draw_overlay, resize_for_detection

try:
    from train import train_model
except ImportError:
    train_model = None


def main(custom_cfg: Optional[Config] = None) -> None:
    cfg = custom_cfg or Config()

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
    screen_w, screen_h = pyautogui.size()

    stream: Optional[WebcamStream] = None
    detector: Optional[HandDetector] = None
    cursor_worker: Optional[CursorWorker] = None
    action_worker: Optional[ActionWorker] = None

    # Monotonic timestamps for latency tracking and frame deduplication
    last_submitted_ts = -1
    last_rendered_ts = -1
    last_processed_det_ts = -1
    last_info: Optional[dict] = None

    fps_value = 0.0
    fps_last = time.perf_counter()
    fps_frames = 0

    # ── Optional Machine Learning Classifier ─────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clf = None
    le = None
    try:
        import joblib
        model_path = os.path.join(script_dir, "gesture_model.pkl")
        encoder_path = os.path.join(script_dir, "label_encoder.pkl")
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            clf = joblib.load(model_path)
            le = joblib.load(encoder_path)
            print(f"✓ Gesture classifier loaded ({', '.join(le.classes_)})")
        else:
            print("  [Info] No gesture_model.pkl found — run with --train to enable live classification.")
    except ImportError:
        print("  [Info] joblib not available — live classification disabled.")

    try:
        cv2.setUseOptimized(True)

        stream = WebcamStream(cfg).start()
        detector = HandDetector(cfg)
        cursor_worker = CursorWorker()
        action_worker = ActionWorker()
        controller = HandGestureController(
            cfg, screen_w, screen_h, cursor_worker, action_worker,
        )

        print("\n" + "=" * 64)
        print(" Vision-Gesture Active | Tier-1 HCI Capstone Controller")
        print(" - 1:1 Middle-Finger Kinematic Tracking")
        print(" - 2-Hand Mode: Right Hand = Cursor | Left Hand = Actions Only")
        print(" Press 'q' in video window to exit.")
        print("=" * 64 + "\n")

        while True:
            packet = stream.read_latest(last_rendered_ts)
            if packet is None:
                time.sleep(0.001)
                continue

            last_rendered_ts = packet.timestamp_ms
            frame = cv2.flip(packet.frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # Submit only newest unsubmitted frame for lowest end-to-end latency
            if packet.timestamp_ms > last_submitted_ts:
                detect_frame = resize_for_detection(frame, cfg.detector_width)
                rgb_frame = np.ascontiguousarray(cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB))
                detector.detect_async(rgb_frame, packet.timestamp_ms)
                last_submitted_ts = packet.timestamp_ms

            detection_packet = detector.get_latest_result()
            info = None

            if detection_packet is not None:
                result_age = packet.timestamp_ms - detection_packet.timestamp_ms

                # Freshness guarantee: discard results older than max_result_age_ms
                if 0 <= result_age <= cfg.max_result_age_ms:
                    if detection_packet.result.hand_landmarks:
                        if detection_packet.timestamp_ms != last_processed_det_ts:
                            landmarks = detection_packet.result.hand_landmarks[0]
                            last_info = controller.update(
                                landmarks,
                                frame_w,
                                frame_h,
                                detection_packet.timestamp_ms,
                                classifier=clf,
                                label_encoder=le,
                                all_landmarks=detection_packet.result.hand_landmarks,
                                handedness=detection_packet.result.handedness,
                            )
                            last_processed_det_ts = detection_packet.timestamp_ms
                        info = last_info
                    else:
                        if last_processed_det_ts != -1:
                            controller.reset_tracking()
                            last_processed_det_ts = -1
                            last_info = None
                elif result_age > cfg.max_result_age_ms:
                    if last_processed_det_ts != -1:
                        controller.reset_tracking()
                        last_processed_det_ts = -1
                        last_info = None
            else:
                if last_processed_det_ts != -1:
                    controller.reset_tracking()
                    last_processed_det_ts = -1
                    last_info = None

            fps_frames += 1
            now = time.perf_counter()
            elapsed = now - fps_last
            if elapsed >= 0.5:
                instant_fps = fps_frames / elapsed
                fps_value = instant_fps if fps_value == 0 else fps_value * 0.65 + instant_fps * 0.35
                fps_frames = 0
                fps_last = now

            if cfg.display:
                draw_overlay(frame, cfg, info, fps_value)
                cv2.imshow("Vision-Gesture: Capstone HCI Controller", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        if action_worker is not None:
            action_worker.stop()
        if cursor_worker is not None:
            cursor_worker.stop()
        if stream is not None:
            stream.stop()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vision-Gesture: Low-Latency Capstone Hand-Gesture Controller"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the ensemble gesture classification model and save weights",
    )
    args = parser.parse_args()

    if args.train:
        if train_model is not None:
            train_model()
        else:
            print("ERROR: Training module not found or scikit-learn is missing.")
    else:
        main()