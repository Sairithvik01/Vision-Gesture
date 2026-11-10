"""Real-Time Multi-Subject Hand Landmark Dataset Recorder for Vision-Gesture.

Allows B.Tech capstone teams to record real multi-subject hand gesture landmarks
from a live webcam into `data/raw_landmarks/recorded_gestures.csv`, which `train.py`
automatically ingests for Stratified 5-Fold Cross-Validation.

Usage
-----
  python collect_dataset.py
"""

import csv
import os
import time

import cv2
import numpy as np

from camera import WebcamStream
from config import Config
from detector import HandDetector
from overlay import HAND_CONNECTIONS, resize_for_detection

GESTURE_KEY_MAP = {
    ord("1"): "Pointer",
    ord("2"): "Pinch",
    ord("3"): "Peace_Scroll",
    ord("4"): "Open_Palm",
    ord("5"): "Fist_Clutch",
    ord("6"): "Thumbs_Up",
}


def collect_landmarks() -> None:
    cfg = Config()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "data", "raw_landmarks")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "recorded_gestures.csv")

    file_exists = os.path.exists(csv_path)
    header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]

    stream = WebcamStream(cfg).start()
    detector = HandDetector(cfg)
    last_ts = -1
    counts = {name: 0 for name in GESTURE_KEY_MAP.values()}

    print("=" * 68)
    print(" Vision-Gesture: Live Multi-Subject Landmark Dataset Collector")
    print("=" * 68)
    print(" Hold a gesture in front of the webcam and press keys 1-6 to record:")
    for k, name in GESTURE_KEY_MAP.items():
        print(f"   [{chr(k)}] -> {name}")
    print(" Press [q] to finish and save dataset.")
    print("=" * 68)

    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)

            while True:
                pkt = stream.read_latest(last_ts)
                if pkt is None:
                    time.sleep(0.002)
                    continue

                last_ts = pkt.timestamp_ms
                frame = cv2.flip(pkt.frame, 1)
                h, w = frame.shape[:2]

                det_frame = resize_for_detection(frame, cfg.detector_width)
                rgb = np.ascontiguousarray(cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB))
                detector.detect_async(rgb, pkt.timestamp_ms)

                det_pkt = detector.get_latest_result()
                active_lms = None
                if det_pkt is not None and det_pkt.result.hand_landmarks:
                    active_lms = det_pkt.result.hand_landmarks[0]
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in active_lms]
                    for s, e in HAND_CONNECTIONS:
                        cv2.line(frame, pts[s], pts[e], (0, 220, 120), 2, cv2.LINE_AA)
                    for pt in pts:
                        cv2.circle(frame, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

                # HUD Overlay
                cv2.rectangle(frame, (10, 10), (340, 195), (25, 25, 25), -1)
                cv2.putText(
                    frame, "Press 1-6 to Record Frame | 'q' Quit", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA,
                )
                y_off = 60
                for k, name in GESTURE_KEY_MAP.items():
                    cv2.putText(
                        frame, f"[{chr(k)}] {name:<14}: {counts[name]} samples",
                        (25, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA,
                    )
                    y_off += 22

                cv2.imshow("Vision-Gesture Dataset Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in GESTURE_KEY_MAP and active_lms is not None:
                    label = GESTURE_KEY_MAP[key]
                    row = []
                    for lm in active_lms:
                        row.extend([f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}"])
                    row.append(label)
                    writer.writerow(row)
                    f.flush()
                    counts[label] += 1
                    print(f"  + Recorded '{label}' (total: {counts[label]})")

    finally:
        stream.stop()
        detector.close()
        cv2.destroyAllWindows()
        print(f"\n✓ Dataset saved to: {csv_path}")


if __name__ == "__main__":
    collect_landmarks()
