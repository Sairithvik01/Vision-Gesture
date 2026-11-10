"""Unit tests for signal processing filters, One-Euro, Kalman, EMA, and Fitts' Law metrics."""

import math
import os
import sys
import unittest

import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from filters import (
    CursorBallistics,
    LowPassFilter,
    OneEuroFilter,
    Point2DEMAFilter,
    Point2DKalmanFilter,
    Point2DOneEuroFilter,
)
from fitts_law_evaluator import (
    compute_fitts_throughput,
    compute_shannon_id,
    generate_iso9241_ring_targets,
)


class TestFilters(unittest.TestCase):

    def test_low_pass_filter(self):
        lpf = LowPassFilter(alpha=0.5)
        self.assertIsNone(lpf.last_value)
        v1 = lpf.apply(10.0)
        self.assertEqual(v1, 10.0)
        v2 = lpf.apply(20.0)
        self.assertAlmostEqual(v2, 15.0)
        lpf.reset()
        self.assertIsNone(lpf.last_value)

    def test_one_euro_filter_stillness(self):
        """Stationary signal should have low cutoff and heavily filter noise."""
        filt = OneEuroFilter(min_cutoff=0.7, beta=0.015, d_cutoff=1.0)
        t = 0.0
        outputs = []
        rng = np.random.default_rng(123)
        for _ in range(60):
            noise = rng.normal(0, 0.5)
            filtered = filt.filter(100.0 + noise, t)
            outputs.append(filtered)
            t += 1.0 / 30.0

        steady_out = outputs[15:]
        self.assertLess(np.std(steady_out), 0.25)

    def test_one_euro_filter_responsiveness(self):
        """Fast movement should adapt cutoff and track with minimal lag."""
        filt = OneEuroFilter(min_cutoff=0.7, beta=0.02, d_cutoff=1.0)
        t = 0.0
        for _ in range(20):
            filt.filter(0.0, t)
            t += 1.0 / 30.0
        for i in range(10):
            val = i * 50.0
            out = filt.filter(val, t)
            t += 1.0 / 30.0
        self.assertGreater(out, 300.0)

    def test_point2d_one_euro_filter_deadband(self):
        """Sub-pixel movements within deadband should be completely stabilized."""
        filt = Point2DOneEuroFilter(min_cutoff=0.7, beta=0.015, deadband_px=1.5)
        fx, fy = filt.filter(200.0, 200.0, 0.0)
        self.assertEqual((fx, fy), (200.0, 200.0))

        # Micro-tremor perturbation (0.4 px < deadband)
        fx2, fy2 = filt.filter(200.3, 200.2, 0.033)
        self.assertEqual((fx2, fy2), (200.0, 200.0))

        # Larger motion (15 px > deadband)
        fx3, fy3 = filt.filter(215.0, 215.0, 0.066)
        self.assertGreater(fx3, 200.0)

    def test_point2d_kalman_filter_tracking_and_noise_reduction(self):
        """2D Constant-Velocity Kalman Filter should suppress stationary noise and track linear ramps."""
        kf = Point2DKalmanFilter(process_noise=120.0, measurement_noise=2.5)
        rng = np.random.default_rng(77)
        t = 0.0
        xs = []
        for _ in range(50):
            ox, _ = kf.filter(300.0 + rng.normal(0, 2.0), 400.0 + rng.normal(0, 2.0), t)
            xs.append(ox)
            t += 1.0 / 30.0
        self.assertLess(np.std(xs[15:]), 1.2)

        # Linear ramp tracking
        kf.reset()
        t = 0.0
        for i in range(20):
            ox, oy = kf.filter(i * 25.0, i * 15.0, t)
            t += 1.0 / 30.0
        self.assertGreater(ox, 350.0)
        self.assertGreater(oy, 200.0)

    def test_point2d_ema_filter(self):
        """2D EMA filter should exponentially smooth coordinates."""
        ema = Point2DEMAFilter(alpha=0.25)
        x0, y0 = ema.filter(100.0, 200.0, 0.0)
        self.assertEqual((x0, y0), (100.0, 200.0))
        x1, y1 = ema.filter(200.0, 300.0, 0.033)
        self.assertAlmostEqual(x1, 125.0)
        self.assertAlmostEqual(y1, 225.0)

    def test_fitts_law_shannon_metrics(self):
        """Verify ISO 9241-9 Shannon formulation ID = log2(D/W + 1) and Throughput calculation."""
        # D = 300, W = 100 -> ID = log2(3 + 1) = 2.0 bits
        id_bits = compute_shannon_id(300.0, 100.0)
        self.assertAlmostEqual(id_bits, 2.0)
        tp = compute_fitts_throughput(id_bits, 0.5)
        self.assertAlmostEqual(tp, 4.0)

        ring = generate_iso9241_ring_targets(500.0, 500.0, 200.0, num_targets=11)
        self.assertEqual(len(ring), 11)

    def test_cursor_ballistics(self):
        ballistics = CursorBallistics(precision_gain=0.85, ballistic_exponent=1.35)
        low_gain = ballistics.compute_gain(20.0)
        self.assertAlmostEqual(low_gain, 0.85)

        high_gain = ballistics.compute_gain(600.0)
        self.assertGreater(high_gain, 2.0)

    def test_edge_easing(self):
        v_min = CursorBallistics.ease_edge(0.0, 0.0, 100.0, 0.0, 1920.0)
        v_mid = CursorBallistics.ease_edge(50.0, 0.0, 100.0, 0.0, 1920.0)
        v_max = CursorBallistics.ease_edge(100.0, 0.0, 100.0, 0.0, 1920.0)
        self.assertAlmostEqual(v_min, 0.0)
        self.assertAlmostEqual(v_mid, 960.0)
        self.assertAlmostEqual(v_max, 1920.0)


if __name__ == "__main__":
    unittest.main()
