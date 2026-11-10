"""Unit tests for gesture state machine, middle-finger 1:1 sync, drag & drop, scrolling, and hybrid ML gating."""

import os
import sys
import unittest
from types import SimpleNamespace
from typing import List

import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from config import Config
from controller import ControlMode, HandGestureController, PinchState, _HandTracker
from gestures.kinematics import compute_3d_palm_scale, determine_middle_finger_point, extract_pts_px


class MockWorker:
    """Mock for CursorWorker and ActionWorker to record submitted calls."""

    def __init__(self):
        self.cursor_submissions: List[tuple] = []
        self.actions: List[str] = []

    def submit(self, *args):
        if len(args) == 2 and isinstance(args[0], (int, float)):
            self.cursor_submissions.append((args[0], args[1]))
        elif len(args) == 1 and callable(args[0]):
            action_name = getattr(args[0], "__name__", str(args[0]))
            self.actions.append(action_name)


class MockProbClassifier:
    """Mock probabilistic ML classifier returning configurable class probabilities."""

    def __init__(self, target_idx: int, confidence: float = 0.92, num_classes: int = 6):
        self.target_idx = target_idx
        self.confidence = confidence
        self.num_classes = num_classes

    def predict_proba(self, X):
        probs = np.full((len(X), self.num_classes), (1.0 - self.confidence) / (self.num_classes - 1))
        probs[:, self.target_idx] = self.confidence
        return probs


class MockLabelEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def inverse_transform(self, indices):
        return [self.classes_[i] for i in indices]


def create_mock_landmarks(
    wrist=(0.5, 0.8),
    thumb_tip=(0.45, 0.6),
    index_tip=(0.50, 0.4),
    index_pip=(0.50, 0.55),
    index_mcp=(0.50, 0.65),
    middle_tip=(0.55, 0.38),
    middle_dip=None,
    middle_pip=None,
    middle_mcp=(0.55, 0.65),
    ring_tip=(0.60, 0.70),
    pinky_tip=(0.65, 0.72),
) -> List[SimpleNamespace]:
    """Build 21 normalized landmarks matching MediaPipe Landmark structure."""
    lms = [SimpleNamespace(x=wrist[0], y=wrist[1], z=0.0) for _ in range(21)]
    lms[0] = SimpleNamespace(x=wrist[0], y=wrist[1], z=0.0)
    lms[2] = SimpleNamespace(x=wrist[0] - 0.05, y=wrist[1] - 0.10, z=0.0)
    lms[4] = SimpleNamespace(x=thumb_tip[0], y=thumb_tip[1], z=0.0)
    lms[5] = SimpleNamespace(x=index_mcp[0], y=index_mcp[1], z=0.0)
    lms[6] = SimpleNamespace(x=index_pip[0], y=index_pip[1], z=0.0)
    lms[8] = SimpleNamespace(x=index_tip[0], y=index_tip[1], z=0.0)
    lms[9] = SimpleNamespace(x=middle_mcp[0], y=middle_mcp[1], z=0.0)

    if middle_pip is None:
        middle_pip = (
            middle_mcp[0] + (middle_tip[0] - middle_mcp[0]) * 0.35,
            middle_mcp[1] + (middle_tip[1] - middle_mcp[1]) * 0.35,
        )
    if middle_dip is None:
        middle_dip = (
            middle_mcp[0] + (middle_tip[0] - middle_mcp[0]) * 0.70,
            middle_mcp[1] + (middle_tip[1] - middle_mcp[1]) * 0.70,
        )
    lms[10] = SimpleNamespace(x=middle_pip[0], y=middle_pip[1], z=0.0)
    lms[11] = SimpleNamespace(x=middle_dip[0], y=middle_dip[1], z=0.0)
    lms[12] = SimpleNamespace(x=middle_tip[0], y=middle_tip[1], z=0.0)

    lms[13] = SimpleNamespace(x=ring_tip[0], y=ring_tip[1] + 0.10, z=0.0)
    lms[14] = SimpleNamespace(x=ring_tip[0], y=ring_tip[1] + 0.05, z=0.0)
    lms[16] = SimpleNamespace(x=ring_tip[0], y=ring_tip[1], z=0.0)
    lms[17] = SimpleNamespace(x=pinky_tip[0], y=pinky_tip[1] + 0.10, z=0.0)
    lms[18] = SimpleNamespace(x=pinky_tip[0], y=pinky_tip[1] + 0.05, z=0.0)
    lms[20] = SimpleNamespace(x=pinky_tip[0], y=pinky_tip[1], z=0.0)
    return lms


class TestController(unittest.TestCase):

    def setUp(self):
        self.cfg = Config(
            frame_margin=50,
            drag_hold_ms=200,
            double_click_window_ms=400,
            click_cooldown_ms=150,
            right_pinch_confirm_ms=40,
        )
        self.mock_cursor = MockWorker()
        self.mock_action = MockWorker()
        self.tracker = _HandTracker(
            self.cfg,
            screen_w=1920,
            screen_h=1080,
            cursor_worker=self.mock_cursor,
            action_worker=self.mock_action,
            drives_cursor=True,
        )

    def test_finger_extension(self):
        lms = create_mock_landmarks(
            index_tip=(0.50, 0.30),
            middle_tip=(0.55, 0.30),
            ring_tip=(0.60, 0.75),
            pinky_tip=(0.65, 0.77),
        )
        extended = _HandTracker._get_extended_fingers(lms, 640, 480)
        self.assertTrue(extended["index"])
        self.assertTrue(extended["middle"])
        self.assertFalse(extended["ring"])
        self.assertFalse(extended["pinky"])

    def test_modular_kinematics_and_3d_palm_scale(self):
        """Verify extracted gestures.kinematics functions compute accurate palm scale and tip blend."""
        lms = create_mock_landmarks(middle_tip=(0.55, 0.38))
        pts = extract_pts_px(lms, 640, 480)
        scale = compute_3d_palm_scale(pts)
        self.assertGreater(scale, 50.0)
        px, py = determine_middle_finger_point(pts, use_kinematic_blend=True)
        self.assertAlmostEqual(px, 0.55 * 640, delta=3.5)
        self.assertAlmostEqual(py, 0.38 * 480, delta=3.5)

    def test_middle_finger_1to1_sync(self):
        """Cursor position must track the middle finger 1:1 across the workspace without drift."""
        lms_center = create_mock_landmarks(middle_tip=(0.50, 0.50))
        self.tracker.update(lms_center, 640, 480, timestamp_ms=100)
        cx, cy = self.mock_cursor.cursor_submissions[-1]
        self.assertAlmostEqual(cx, 1919.0 * 0.5, delta=2.0)
        self.assertAlmostEqual(cy, 1079.0 * 0.5, delta=2.0)

        t = 133
        for x_norm in [0.60, 0.70, 0.75, 0.65, 0.50, 0.50, 0.50]:
            lms = create_mock_landmarks(middle_tip=(x_norm, 0.50))
            self.tracker.update(lms, 640, 480, timestamp_ms=t)
            t += 33

        final_x, final_y = self.mock_cursor.cursor_submissions[-1]
        self.assertAlmostEqual(final_x, 1919.0 * 0.5, delta=3.0)
        self.assertAlmostEqual(final_y, 1079.0 * 0.5, delta=3.0)

    def test_click_anchor_lock(self):
        """Cursor moves freely during relaxed hand motion and locks only on actual pinch contact."""
        lms_open = create_mock_landmarks(
            thumb_tip=(0.35, 0.50),
            index_tip=(0.50, 0.50),
        )
        info1 = self.tracker.update(lms_open, 640, 480, timestamp_ms=100)
        self.assertGreater(info1["pinch_ratio"], self.cfg.pinch_lock_ratio)
        self.assertIsNone(self.tracker._locked_target_x)

        lms_closing = create_mock_landmarks(
            thumb_tip=(0.493, 0.50),
            index_tip=(0.50, 0.50),
        )
        info2 = self.tracker.update(lms_closing, 640, 480, timestamp_ms=133)
        self.assertLessEqual(info2["pinch_ratio"], self.cfg.pinch_lock_ratio)
        self.assertIsNotNone(self.tracker._locked_target_x)
        locked_x = self.tracker._locked_target_x

        lms_touch = create_mock_landmarks(
            thumb_tip=(0.50, 0.50),
            index_tip=(0.50, 0.50),
        )
        self.tracker.update(lms_touch, 640, 480, timestamp_ms=166)
        self.assertEqual(self.tracker._locked_target_x, locked_x)

    def test_tap_click_and_double_click(self):
        """Quick pinch-release should register click, and rapid second tap triggers double-click."""
        lms_open = create_mock_landmarks(thumb_tip=(0.35, 0.5), index_tip=(0.5, 0.5))
        self.tracker.update(lms_open, 640, 480, timestamp_ms=100)

        lms_pinch = create_mock_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.5, 0.5))
        info_down = self.tracker.update(lms_pinch, 640, 480, timestamp_ms=130)
        self.assertTrue(info_down["pinched"])

        info_up = self.tracker.update(lms_open, 640, 480, timestamp_ms=180)
        self.assertTrue(info_up["clicked"])
        self.assertIn("click", self.mock_action.actions)

        self.tracker.update(lms_pinch, 640, 480, timestamp_ms=280)

        info_dbl = self.tracker.update(lms_open, 640, 480, timestamp_ms=320)
        self.assertTrue(info_dbl["double_clicked"])
        self.assertIn("doubleClick", self.mock_action.actions)

    def test_drag_and_drop(self):
        """Holding pinch past drag_hold_ms should engage mouseDown, allow moving, and mouseUp on release."""
        lms_open = create_mock_landmarks(thumb_tip=(0.35, 0.5), index_tip=(0.5, 0.5))
        lms_pinch = create_mock_landmarks(thumb_tip=(0.5, 0.5), index_tip=(0.5, 0.5))

        self.tracker.update(lms_pinch, 640, 480, timestamp_ms=100)

        info_drag = self.tracker.update(lms_pinch, 640, 480, timestamp_ms=320)
        self.assertTrue(info_drag["dragging"])
        self.assertIn("mouseDown", self.mock_action.actions)

        info_drop = self.tracker.update(lms_open, 640, 480, timestamp_ms=450)
        self.assertTrue(info_drop["drag_released"])
        self.assertIn("mouseUp", self.mock_action.actions)

    def test_right_click(self):
        """Thumb + Middle pinch held for confirm duration should trigger right-click."""
        lms_rpinch = create_mock_landmarks(
            thumb_tip=(0.5, 0.5),
            middle_tip=(0.5, 0.5),
            index_tip=(0.4, 0.3),
        )
        self.tracker.update(lms_rpinch, 640, 480, timestamp_ms=100)
        info = self.tracker.update(lms_rpinch, 640, 480, timestamp_ms=160)
        self.assertTrue(info["right_clicked"])
        self.assertIn("rightClick", self.mock_action.actions)

    def test_proportional_smooth_scroll(self):
        """Index + Middle extended triggers scroll mode, displacement up/down scrolls."""
        lms_scroll = create_mock_landmarks(
            index_tip=(0.5, 0.3),
            middle_tip=(0.55, 0.3),
            ring_tip=(0.6, 0.75),
            pinky_tip=(0.65, 0.77),
        )
        self.tracker.update(lms_scroll, 640, 480, timestamp_ms=100)
        info1 = self.tracker.update(lms_scroll, 640, 480, timestamp_ms=230)
        self.assertEqual(info1["control_mode"], "SCROLL")

        lms_scroll_up = create_mock_landmarks(
            index_tip=(0.5, 0.15),
            middle_tip=(0.55, 0.15),
            ring_tip=(0.6, 0.75),
            pinky_tip=(0.65, 0.77),
        )
        info_scroll = self.tracker.update(lms_scroll_up, 640, 480, timestamp_ms=270)
        self.assertEqual(info_scroll["control_mode"], "SCROLL")
        self.assertGreater(info_scroll["scroll_velocity"], 0.0)

    def test_midas_clutch_standby(self):
        """Closed fist should enter STANDBY (clutch) and pause tracking."""
        lms_fist = create_mock_landmarks(
            thumb_tip=(0.48, 0.75),
            index_tip=(0.50, 0.75),
            middle_tip=(0.55, 0.75),
            ring_tip=(0.60, 0.75),
            pinky_tip=(0.65, 0.75),
        )
        self.tracker.update(lms_fist, 640, 480, timestamp_ms=100)
        info_clutch = self.tracker.update(lms_fist, 640, 480, timestamp_ms=250)
        self.assertEqual(info_clutch["control_mode"], "STANDBY")

    def test_hybrid_ml_confidence_gating(self):
        """High-confidence ML prediction accelerates mode confirmation and reports confidence."""
        ctrl = HandGestureController(
            self.cfg,
            screen_w=1920,
            screen_h=1080,
            cursor_worker=self.mock_cursor,
            action_worker=self.mock_action,
        )
        classes = ["Fist_Clutch", "Open_Palm", "Peace_Scroll", "Pinch", "Pointer", "Thumbs_Up"]
        le = MockLabelEncoder(classes)
        # Mock classifier predicting 'Fist_Clutch' (idx 0) with 94% confidence
        clf = MockProbClassifier(target_idx=0, confidence=0.94)

        # Fist with slightly noisy thumb (1 finger extended), which pure geometry would reject
        # for STANDBY, but hybrid ML gating recognizes as Fist_Clutch and confirms in half the time (60ms)
        lms_noisy_fist = create_mock_landmarks(
            thumb_tip=(0.35, 0.60),  # slightly extended thumb
            index_tip=(0.50, 0.75),
            middle_tip=(0.55, 0.75),
            ring_tip=(0.60, 0.75),
            pinky_tip=(0.65, 0.75),
        )
        ctrl.update(lms_noisy_fist, 640, 480, timestamp_ms=100, classifier=clf, label_encoder=le)
        info = ctrl.update(lms_noisy_fist, 640, 480, timestamp_ms=170, classifier=clf, label_encoder=le)
        self.assertEqual(info["gesture_label"], "Fist_Clutch")
        self.assertAlmostEqual(info["gesture_confidence"], 0.94, places=2)
        self.assertEqual(info["control_mode"], "STANDBY")


if __name__ == "__main__":
    unittest.main()
