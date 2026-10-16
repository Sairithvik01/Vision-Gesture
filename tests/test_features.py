"""Unit tests for landmark normalization and feature extraction invariance."""

import math
import os
import sys
import unittest
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
user_uploaded_dir = os.path.join(root_dir, "user_uploaded_files", "ttl=63d", "user_uploaded")
sys.path.insert(0, root_dir)
sys.path.insert(0, user_uploaded_dir)

from features import normalize_landmarks


class TestFeatures(unittest.TestCase):

    def setUp(self):
        # Canonical hand landmarks (21, 3)
        self.base_lms = np.array([
            [100.0, 300.0, 0.0],   # 0: wrist
            [80.0, 270.0, 0.0],    # 1: thumb CMC
            [70.0, 230.0, 0.0],    # 2: thumb MCP
            [60.0, 190.0, 0.0],    # 3: thumb IP
            [50.0, 160.0, 0.0],    # 4: thumb tip
            [85.0, 190.0, 0.0],    # 5: index MCP
            [80.0, 150.0, 0.0],    # 6: index PIP
            [75.0, 120.0, 0.0],    # 7: index DIP
            [70.0, 90.0, 0.0],     # 8: index tip
            [100.0, 180.0, 0.0],   # 9: middle MCP
            [100.0, 140.0, 0.0],   # 10: middle PIP
            [100.0, 100.0, 0.0],   # 11: middle DIP
            [100.0, 60.0, 0.0],    # 12: middle tip
            [115.0, 190.0, 0.0],   # 13: ring MCP
            [118.0, 150.0, 0.0],   # 14: ring PIP
            [120.0, 120.0, 0.0],   # 15: ring DIP
            [122.0, 90.0, 0.0],    # 16: ring tip
            [130.0, 205.0, 0.0],   # 17: pinky MCP
            [135.0, 175.0, 0.0],   # 18: pinky PIP
            [138.0, 150.0, 0.0],   # 19: pinky DIP
            [140.0, 125.0, 0.0],   # 20: pinky tip
        ], dtype=np.float64)

    def test_output_shape(self):
        feat = normalize_landmarks(self.base_lms)
        self.assertEqual(feat.shape, (63,))

    def test_translation_invariance(self):
        """Translating hand anywhere in frame should yield identical normalized features."""
        feat_base = normalize_landmarks(self.base_lms)
        translated = self.base_lms + np.array([250.0, -180.0, 50.0])
        feat_trans = normalize_landmarks(translated)
        np.testing.assert_allclose(feat_base, feat_trans, atol=1e-5)

    def test_scale_invariance(self):
        """Scaling hand (near or far from camera) should yield identical features."""
        feat_base = normalize_landmarks(self.base_lms)
        scaled = self.base_lms * 2.5
        feat_scaled = normalize_landmarks(scaled)
        np.testing.assert_allclose(feat_base, feat_scaled, atol=1e-5)

    def test_rotation_invariance(self):
        """Rotating hand in 2D image plane should align canonical axis."""
        feat_base = normalize_landmarks(self.base_lms)
        # Rotate by 45 degrees
        angle = math.radians(45.0)
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        rotated = (self.base_lms - self.base_lms[0]) @ R.T + self.base_lms[0]
        feat_rot = normalize_landmarks(rotated)
        # Middle MCP landmark (index 9) must align to (0, -1) in both
        np.testing.assert_allclose(feat_base, feat_rot, atol=1e-4)

    def test_degenerate_zero_distance(self):
        """Guard against divide-by-zero on degenerate inputs."""
        zeros = np.zeros((21, 3))
        feat = normalize_landmarks(zeros)
        self.assertEqual(feat.shape, (63,))
        self.assertFalse(np.isnan(feat).any())


if __name__ == "__main__":
    unittest.main()
