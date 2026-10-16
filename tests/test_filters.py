"""Unit tests for signal processing filters, One-Euro adaptive filter, and cursor ballistics."""

import math
import os
import sys
import unittest
import numpy as np

# Ensure module import works across both dev workspace and user_uploaded directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
user_uploaded_dir = os.path.join(root_dir, "user_uploaded_files", "ttl=63d", "user_uploaded")
sys.path.insert(0, root_dir)
sys.path.insert(0, user_uploaded_dir)

from filters import CursorBallistics, LowPassFilter, OneEuroFilter, Point2DOneEuroFilter


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
