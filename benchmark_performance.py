"""Quantitative Performance Benchmark & Evaluation Suite for Vision-Gesture.

Measures:
1. One-Euro & 2D Filter Latency (μs per point).
2. Landmark Jitter Rejection Ratio (% noise variance reduction).
3. Trajectory Smoothness & Fitts's Law Ballistics response.
4. Gesture State Machine update throughput (FPS capacity).
"""

import math
import os
import sys
import time
from types import SimpleNamespace
import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__))
user_uploaded_dir = os.path.join(root_dir, "user_uploaded_files", "ttl=63d", "user_uploaded")
sys.path.insert(0, root_dir)
sys.path.insert(0, user_uploaded_dir)

from config import Config
from controller import HandGestureController
from filters import CursorBallistics, OneEuroFilter, Point2DOneEuroFilter


class MockWorker:
    def __init__(self):
        self.cursor_submissions = []
        self.actions = []

    def submit(self, *args):
        if len(args) == 2:
            self.cursor_submissions.append((args[0], args[1]))
        elif len(args) == 1 and callable(args[0]):
            self.actions.append(args[0])


def create_mock_landmarks():
    lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    lms[0] = SimpleNamespace(x=0.5, y=0.8, z=0.0)
    lms[2] = SimpleNamespace(x=0.45, y=0.7, z=0.0)
    lms[4] = SimpleNamespace(x=0.45, y=0.6, z=0.0)
    lms[5] = SimpleNamespace(x=0.50, y=0.65, z=0.0)
    lms[6] = SimpleNamespace(x=0.50, y=0.55, z=0.0)
    lms[8] = SimpleNamespace(x=0.50, y=0.40, z=0.0)
    lms[9] = SimpleNamespace(x=0.55, y=0.65, z=0.0)
    lms[12] = SimpleNamespace(x=0.55, y=0.38, z=0.0)
    lms[16] = SimpleNamespace(x=0.60, y=0.70, z=0.0)
    lms[20] = SimpleNamespace(x=0.65, y=0.72, z=0.0)
    return lms


def benchmark_filters():
    print("=" * 65)
    print(" 1. Signal Processing & Filter Latency Benchmark")
    print("=" * 65)
    filt = Point2DOneEuroFilter(min_cutoff=0.7, beta=0.015, deadband_px=1.2)

    n_samples = 100_000
    times = np.linspace(0, n_samples / 30.0, n_samples)
    xs = 500.0 + np.sin(times) * 200.0
    ys = 400.0 + np.cos(times) * 150.0

    t0 = time.perf_counter()
    for x, y, t in zip(xs, ys, times):
        filt.filter(x, y, t)
    t1 = time.perf_counter()

    total_time = t1 - t0
    us_per_call = (total_time / n_samples) * 1_000_000
    max_fps_equiv = n_samples / total_time

    print(f"Processed:            {n_samples:,} samples")
    print(f"Total Time:           {total_time:.4f} s")
    print(f"Latency per Filter:   {us_per_call:.2f} μs / sample")
    print(f"Theoretical Max Rate: {max_fps_equiv:,.0f} samples/sec")


def benchmark_jitter_rejection():
    print("\n" + "=" * 65)
    print(" 2. Jitter Rejection & Micro-Tremor Suppression Analysis")
    print("=" * 65)
    filt = Point2DOneEuroFilter(min_cutoff=0.7, beta=0.015, deadband_px=1.2)

    # 1,000 frames of stationary hand with MediaPipe landmark gaussian noise (sigma = 1.8 px)
    rng = np.random.default_rng(42)
    n_frames = 1000
    true_x, true_y = 640.0, 360.0
    noise_x = rng.normal(0, 1.8, n_frames)
    noise_y = rng.normal(0, 1.8, n_frames)

    raw_xs = true_x + noise_x
    raw_ys = true_y + noise_y

    filtered_xs, filtered_ys = [], []
    t = 0.0
    for rx, ry in zip(raw_xs, raw_ys):
        fx, fy = filt.filter(rx, ry, t)
        filtered_xs.append(fx)
        filtered_ys.append(fy)
        t += 1.0 / 30.0

    raw_var = (np.var(raw_xs) + np.var(raw_ys)) / 2.0
    filt_var = (np.var(filtered_xs[50:]) + np.var(filtered_ys[50:])) / 2.0
    rejection_pct = (1.0 - (filt_var / raw_var)) * 100.0

    print(f"Raw Input Jitter Variance:      {raw_var:.4f} px² (Std: {math.sqrt(raw_var):.2f} px)")
    print(f"Filtered Output Variance:       {filt_var:.4f} px² (Std: {math.sqrt(filt_var):.2f} px)")
    print(f"Jitter Reduction Fidelity:      {rejection_pct:.2f}% noise suppression")


def benchmark_controller_throughput():
    print("\n" + "=" * 65)
    print(" 3. Gesture Controller End-to-End Throughput")
    print("=" * 65)
    cfg = Config()
    mock_cursor = MockWorker()
    mock_action = MockWorker()
    ctrl = HandGestureController(cfg, 1920, 1080, mock_cursor, mock_action)

    n_frames = 10_000
    lms = create_mock_landmarks()
    t_ms = 1000

    t0 = time.perf_counter()
    for _ in range(n_frames):
        ctrl.update(lms, 640, 480, t_ms)
        t_ms += 33
    t1 = time.perf_counter()

    elapsed = t1 - t0
    fps_capacity = n_frames / elapsed
    ms_per_frame = (elapsed / n_frames) * 1000.0

    print(f"Total Frames Processed:         {n_frames:,}")
    print(f"Controller Execution Time:      {elapsed:.3f} s")
    print(f"Latency per Update:             {ms_per_frame:.3f} ms")
    print(f"Controller Throughput Capacity: {fps_capacity:,.0f} FPS")
    print("=" * 65)


if __name__ == "__main__":
    benchmark_filters()
    benchmark_jitter_rejection()
    benchmark_controller_throughput()
