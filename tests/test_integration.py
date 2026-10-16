"""End-to-end integration and concurrency tests for Vision-Gesture."""

import os
import sys
import time
import unittest
from types import SimpleNamespace
from typing import List

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from actions import ActionWorker
from config import Config
from controller import HandGestureController
from cursor import CursorWorker


class RecordingCursorWorker:
    """Records every cursor (x, y) submission for deterministic assertion."""

    def __init__(self):
        self.submissions: List[tuple] = []

    def submit(self, x: float, y: float) -> None:
        self.submissions.append((x, y))

    def stop(self) -> None:
        pass


class RecordingActionWorker:
    """Records every action callable dispatched by the controller."""

    def __init__(self):
        self.actions: List[str] = []

    def submit(self, action) -> None:
        name = getattr(action, "__name__", str(action))
        self.actions.append(name)

    def stop(self) -> None:
        pass


def create_hand(
    center_x: float,
    center_y: float,
    middle_offset_x: float = 0.0,
    middle_offset_y: float = -0.05,
    pinch_left: bool = False,
):
    """Build mock 21 hand landmarks around a given center point."""
    lms = []
    for i in range(21):
        lms.append(SimpleNamespace(x=center_x + (i - 10) * 0.005, y=center_y + (i - 10) * 0.005, z=0.0))
    lms[0] = SimpleNamespace(x=center_x, y=center_y + 0.15, z=0.0)        # wrist
    lms[2] = SimpleNamespace(x=center_x - 0.04, y=center_y + 0.08, z=0.0)
    if pinch_left:
        lms[4] = SimpleNamespace(x=center_x, y=center_y - 0.05, z=0.0)    # thumb touching index
    else:
        lms[4] = SimpleNamespace(x=center_x - 0.06, y=center_y, z=0.0)    # thumb open
    lms[5] = SimpleNamespace(x=center_x, y=center_y + 0.05, z=0.0)
    lms[6] = SimpleNamespace(x=center_x, y=center_y + 0.02, z=0.0)
    lms[8] = SimpleNamespace(x=center_x, y=center_y - 0.05, z=0.0)         # index tip
    lms[9] = SimpleNamespace(x=center_x + 0.03, y=center_y + 0.05, z=0.0)  # middle mcp
    lms[10] = SimpleNamespace(x=center_x + 0.03, y=center_y + 0.01, z=0.0) # middle pip
    lms[11] = SimpleNamespace(x=center_x + 0.03, y=center_y - 0.02, z=0.0) # middle dip
    lms[12] = SimpleNamespace(
        x=center_x + 0.03 + middle_offset_x,
        y=center_y + middle_offset_y,
        z=0.0,
    )  # middle tip
    lms[13] = SimpleNamespace(x=center_x + 0.05, y=center_y + 0.05, z=0.0)
    lms[16] = SimpleNamespace(x=center_x + 0.06, y=center_y + 0.10, z=0.0) # ring curled
    lms[17] = SimpleNamespace(x=center_x + 0.07, y=center_y + 0.06, z=0.0)
    lms[20] = SimpleNamespace(x=center_x + 0.08, y=center_y + 0.12, z=0.0) # pinky curled
    return lms


class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.cfg = Config(multi_hand=True, num_hands=2, right_hand_gone_timeout_ms=450)
        self.cursor_worker = RecordingCursorWorker()
        self.action_worker = ActionWorker()
        self.controller = HandGestureController(
            self.cfg,
            screen_w=1920,
            screen_h=1080,
            cursor_worker=self.cursor_worker,
            action_worker=self.action_worker,
        )

    def tearDown(self):
        self.cursor_worker.stop()
        self.action_worker.stop()

    def test_multi_hand_right_cursor_left_actions_only(self):
        """When 2 hands are detected, Right hand middle finger controls cursor; Left hand never moves cursor."""
        t_ms = 1000

        # Phase 1: Right hand stays stationary at x=0.70, while Left hand at x=0.30 moves its middle finger wildly
        for f in range(15):
            right_hand = create_hand(0.70, 0.45, middle_offset_x=0.0, middle_offset_y=-0.05)
            left_hand = create_hand(0.30, 0.50, middle_offset_x=0.02 * f, middle_offset_y=-0.05 - 0.01 * f)

            all_lms = [left_hand, right_hand]
            handedness = [
                [SimpleNamespace(category_name="Right")],  # MediaPipe raw on flipped frame for physical Left
                [SimpleNamespace(category_name="Left")],   # MediaPipe raw on flipped frame for physical Right
            ]

            info = self.controller.update(
                landmarks=left_hand,
                frame_w=640,
                frame_h=480,
                timestamp_ms=t_ms,
                all_landmarks=all_lms,
                handedness=handedness,
            )

            self.assertEqual(info["hand_role"], "Right")
            self.assertTrue(info["drives_cursor"])
            self.assertEqual(len(info["extra_hands"]), 1)
            self.assertEqual(info["extra_hands"][0]["hand_role"], "Left")
            self.assertFalse(info["extra_hands"][0]["drives_cursor"])
            t_ms += 33

        # Because Right hand middle finger was completely stationary, cursor must only have moved on frame 0
        self.assertEqual(len(self.cursor_worker.submissions), 1)
        stationary_cursor_pos = self.cursor_worker.submissions[-1]

        # Phase 2: Right hand disappears briefly (< 450ms timeout), leaving only Left hand moving
        for f in range(5):
            left_hand = create_hand(0.30 + 0.005 * f, 0.50, middle_offset_x=0.03 * f)
            info = self.controller.update(
                landmarks=left_hand,
                frame_w=640,
                frame_h=480,
                timestamp_ms=t_ms,
                all_landmarks=[left_hand],
                handedness=[[SimpleNamespace(category_name="Right")]],
            )
            # Left hand must remain actions-only and NOT move the cursor while waiting for Right hand
            self.assertEqual(info["hand_role"], "Left")
            self.assertFalse(info["drives_cursor"])
            t_ms += 33

        self.assertEqual(self.cursor_worker.submissions[-1], stationary_cursor_pos)

        # Phase 3: Right hand has genuinely gone away (> 450ms timeout) -> single hand takes over cursor
        t_ms += 500
        single_hand = create_hand(0.35, 0.45)
        info_after = self.controller.update(
            landmarks=single_hand,
            frame_w=640,
            frame_h=480,
            timestamp_ms=t_ms,
            all_landmarks=[single_hand],
            handedness=[[SimpleNamespace(category_name="Right")]],
        )
        self.assertEqual(info_after["hand_role"], "Right")
        self.assertTrue(info_after["drives_cursor"])
        self.assertGreater(len(self.cursor_worker.submissions), 1)

    def test_two_hand_simultaneous_cursor_and_left_actions(self):
        """Right hand middle finger moves cursor while Left hand executes click and scroll actions."""
        rec_cursor = RecordingCursorWorker()
        rec_action = RecordingActionWorker()
        ctrl = HandGestureController(self.cfg, 1920, 1080, rec_cursor, rec_action)

        # Frame 1: Both hands open, Right hand at x=0.65, Left hand at x=0.25
        r1 = create_hand(0.65, 0.50)
        l1 = create_hand(0.25, 0.50, pinch_left=False)
        ctrl.update(r1, 640, 480, 1000, all_landmarks=[l1, r1], handedness=[[SimpleNamespace(category_name="Right")], [SimpleNamespace(category_name="Left")]])

        # Frame 2: Right hand moves middle finger right; Left hand pinches thumb+index
        r2 = create_hand(0.72, 0.50)
        l2 = create_hand(0.25, 0.50, pinch_left=True)
        ctrl.update(r2, 640, 480, 1033, all_landmarks=[l2, r2], handedness=[[SimpleNamespace(category_name="Right")], [SimpleNamespace(category_name="Left")]])

        # Frame 3: Right hand moves middle finger further right; Left hand releases pinch -> CLICK!
        r3 = create_hand(0.78, 0.50)
        l3 = create_hand(0.25, 0.50, pinch_left=False)
        info3 = ctrl.update(r3, 640, 480, 1066, all_landmarks=[l3, r3], handedness=[[SimpleNamespace(category_name="Right")], [SimpleNamespace(category_name="Left")]])

        self.assertTrue(info3["clicked"])
        self.assertIn("click", rec_action.actions)
        # Cursor moved smoothly across all 3 frames following the Right hand's middle finger
        self.assertEqual(len(rec_cursor.submissions), 3)
        self.assertGreater(rec_cursor.submissions[2][0], rec_cursor.submissions[0][0])

    def test_action_worker_concurrency(self):
        """ActionWorker executes queued items in FIFO order without blocking."""
        results = []
        for i in range(5):
            self.action_worker.submit(lambda val=i: results.append(val))

        time.sleep(0.1)
        self.assertEqual(results, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
