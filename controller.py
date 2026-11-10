"""Orchestrator for Multi-Hand Role Isolation and Hybrid ML-FSM Gesture Control.

Decomposed Architecture:
* `gestures/kinematics.py`: 4-point middle-finger kinematic chain & 3D palm scale invariance.
* `gestures/state_machine.py`: Hybrid Deterministic-Probabilistic FSM, click-slip anchor lock,
  drag-and-drop, proportional smooth scroll, and Midas-Touch clutch.
* `gestures/multi_hand.py`: Spatial centroid sorting and two-hand role isolation.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from actions import ActionWorker
from config import Config
from cursor import CursorWorker
from features import normalize_landmarks
from gestures.multi_hand import assign_two_hand_roles, compute_palm_centroid
from gestures.state_machine import ControlMode, PinchState, _HandTracker

__all__ = [
    "ControlMode",
    "HandGestureController",
    "PinchState",
    "_HandTracker",
]


class HandGestureController:
    """Production Multi-Hand Controller with Strict Right-Hand Cursor Isolation
    and Hybrid Deterministic-Probabilistic ML Gating.

    Role assignment when 2 hands are detected:
      - Right Hand ('Right'): Middle finger exclusively controls the cursor (+ clicks/drags).
      - Left Hand ('Left'): Strictly actions-only (scroll, clicks, drag-hold); its middle
        finger NEVER moves the cursor while the Right hand is present or within the
        right-hand absence grace window (`cfg.right_hand_gone_timeout_ms`).
    """

    def __init__(
        self,
        cfg: Config,
        screen_w: int,
        screen_h: int,
        cursor_worker: CursorWorker,
        action_worker: Optional[ActionWorker] = None,
    ):
        self.cfg = cfg
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cursor_worker = cursor_worker
        self.action_worker = action_worker

        self._gesture_infer_counter = 0
        self._last_gesture_label: Optional[str] = None
        self._last_gesture_confidence: float = 0.0

        # Unified trackers:
        # "Right" tracker always has drives_cursor=True.
        # "Left" tracker always has drives_cursor=False (actions only).
        self._multi_trackers: Dict[str, _HandTracker] = {
            "Right": _HandTracker(
                cfg, screen_w, screen_h, cursor_worker, action_worker, drives_cursor=True
            ),
            "Left": _HandTracker(
                cfg, screen_w, screen_h, cursor_worker, action_worker, drives_cursor=False
            ),
        }
        self._single_tracker = self._multi_trackers["Right"]

        # Temporal state for multi-hand disambiguation and Right-hand presence persistence
        self._last_right_palm: Optional[Tuple[float, float]] = None
        self._last_left_palm: Optional[Tuple[float, float]] = None
        self._last_right_seen_ms: int = -(10 ** 9)
        self._two_hand_session_active: bool = False
        self._last_processed_ts: int = -(10 ** 9)
        self._last_combined_info: Optional[dict] = None

    def reset_tracking(self) -> None:
        for t in self._multi_trackers.values():
            t.reset()
        self._last_gesture_label = None
        self._last_gesture_confidence = 0.0
        self._last_right_palm = None
        self._last_left_palm = None
        self._last_right_seen_ms = -(10 ** 9)
        self._two_hand_session_active = False
        self._last_processed_ts = -(10 ** 9)
        self._last_combined_info = None

    @staticmethod
    def _palm_centroid(hand_lms) -> Tuple[float, float]:
        return compute_palm_centroid(hand_lms)

    def _assign_two_hand_roles(
        self, all_landmarks: List, handedness: Optional[List]
    ) -> List[Tuple[object, str, Tuple[float, float]]]:
        return assign_two_hand_roles(
            all_landmarks=all_landmarks,
            handedness=handedness,
            spatial_hand_sorting=self.cfg.spatial_hand_sorting,
            last_right_palm=self._last_right_palm,
            last_left_palm=self._last_left_palm,
        )

    def _infer_gesture_with_confidence(
        self, landmarks, frame_w: int, frame_h: int, classifier, label_encoder
    ) -> Tuple[Optional[str], float]:
        """Run pose-invariant ML inference and return (gesture_label, confidence)."""
        gesture_label = self._last_gesture_label
        confidence = self._last_gesture_confidence
        should_infer = self._gesture_infer_counter % self.cfg.gesture_infer_every == 0
        self._gesture_infer_counter += 1

        if should_infer and classifier is not None and label_encoder is not None:
            try:
                lm_arr = np.array(
                    [[lm.x * frame_w, lm.y * frame_h, lm.z * frame_w] for lm in landmarks],
                    dtype=np.float64,
                )
                features = normalize_landmarks(lm_arr).reshape(1, -1)
                if hasattr(classifier, "predict_proba"):
                    probs = classifier.predict_proba(features)[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx])
                else:
                    pred_idx = classifier.predict(features)[0]
                    confidence = 1.0
                gesture_label = str(label_encoder.inverse_transform([pred_idx])[0])
                self._last_gesture_label = gesture_label
                self._last_gesture_confidence = confidence
            except Exception:
                pass

        return gesture_label, confidence

    def _infer_gesture(
        self, landmarks, frame_w: int, frame_h: int, classifier, label_encoder
    ) -> Optional[str]:
        label, _ = self._infer_gesture_with_confidence(
            landmarks, frame_w, frame_h, classifier, label_encoder
        )
        return label

    def update(
        self,
        landmarks,
        frame_w: int,
        frame_h: int,
        timestamp_ms: int,
        classifier=None,
        label_encoder=None,
        all_landmarks: Optional[List] = None,
        handedness: Optional[List] = None,
    ) -> dict:
        """Update controller state with strict 2-hand role separation and hybrid ML-FSM gating."""
        if timestamp_ms == self._last_processed_ts and self._last_combined_info is not None:
            return self._last_combined_info
        self._last_processed_ts = timestamp_ms

        # ── Case 1: Two (or more) hands detected ─────────────────────────────
        if (
            self.cfg.multi_hand
            and all_landmarks is not None
            and len(all_landmarks) >= 2
        ):
            assigned = self._assign_two_hand_roles(all_landmarks, handedness)
            assigned.sort(key=lambda item: item[1] != "Right")

            right_lms, _, right_palm = assigned[0]
            left_lms, _, left_palm = assigned[1]

            self._last_right_palm = right_palm
            self._last_left_palm = left_palm
            self._last_right_seen_ms = timestamp_ms
            self._two_hand_session_active = True

            ml_label, ml_conf = self._infer_gesture_with_confidence(
                right_lms, frame_w, frame_h, classifier, label_encoder
            )

            # In 2-hand mode, the Right hand middle finger is dedicated to cursor tracking
            # (allow_scroll=False so two-finger pointing never freezes the cursor into scroll mode),
            # while the Left hand is dedicated to actions (scroll, clicks, drag).
            right_info = self._multi_trackers["Right"].update(
                right_lms,
                frame_w,
                frame_h,
                timestamp_ms,
                allow_scroll=False,
                ml_label=ml_label,
                ml_confidence=ml_conf,
            )
            right_info["hand_role"] = "Right"

            left_info = self._multi_trackers["Left"].update(
                left_lms, frame_w, frame_h, timestamp_ms, allow_scroll=True
            )
            left_info["hand_role"] = "Left"

            combined_info = dict(right_info)
            combined_info["extra_hands"] = [left_info]

            # Surface Left-hand actions into top-level telemetry while keeping
            # Right-hand middle finger exclusively in charge of cursor coordinates.
            combined_info["clicked"] = right_info["clicked"] or left_info["clicked"]
            combined_info["double_clicked"] = right_info["double_clicked"] or left_info["double_clicked"]
            combined_info["right_clicked"] = right_info["right_clicked"] or left_info["right_clicked"]
            combined_info["dragging"] = right_info["dragging"] or left_info["dragging"]
            combined_info["drag_released"] = right_info["drag_released"] or left_info["drag_released"]
            if left_info["scroll_action"] is not None and right_info["scroll_action"] is None:
                combined_info["scroll_action"] = left_info["scroll_action"]
                combined_info["scroll_velocity"] = left_info["scroll_velocity"]

            combined_info["gesture_label"] = ml_label
            combined_info["gesture_confidence"] = ml_conf
            self._last_combined_info = combined_info
            return combined_info

        # ── Case 2: Single hand detected ─────────────────────────────────────
        single_lms = all_landmarks[0] if (all_landmarks and len(all_landmarks) == 1) else landmarks
        palm = self._palm_centroid(single_lms)
        ml_label, ml_conf = self._infer_gesture_with_confidence(
            single_lms, frame_w, frame_h, classifier, label_encoder
        )

        # Check if we are within a 2-hand session where the Right hand just disappeared:
        # As long as the Right hand has NOT gone away beyond right_hand_gone_timeout_ms,
        # if the remaining hand is the Left hand, keep it strictly on the Left (actions-only)
        # tracker so its middle finger NEVER hijacks the cursor!
        is_left_hand_waiting_for_right = False
        if (
            self.cfg.multi_hand
            and self._two_hand_session_active
            and (timestamp_ms - self._last_right_seen_ms) < self.cfg.right_hand_gone_timeout_ms
            and self._last_left_palm is not None
            and self._last_right_palm is not None
        ):
            dist_to_left = math.hypot(
                palm[0] - self._last_left_palm[0], palm[1] - self._last_left_palm[1]
            )
            dist_to_right = math.hypot(
                palm[0] - self._last_right_palm[0], palm[1] - self._last_right_palm[1]
            )
            if dist_to_left < dist_to_right * 0.85:
                is_left_hand_waiting_for_right = True

        if is_left_hand_waiting_for_right:
            self._last_left_palm = palm
            self._multi_trackers["Right"].reset()
            info = self._multi_trackers["Left"].update(
                single_lms,
                frame_w,
                frame_h,
                timestamp_ms,
                allow_scroll=True,
                ml_label=ml_label,
                ml_confidence=ml_conf,
            )
            info["hand_role"] = "Left"
            info["extra_hands"] = []
            info["gesture_label"] = ml_label
            info["gesture_confidence"] = ml_conf
            self._last_combined_info = info
            return info

        # Otherwise, the visible hand is the active cursor hand (either the Right hand
        # remained in view, or the Right hand has completely gone away after timeout).
        if (timestamp_ms - self._last_right_seen_ms) >= self.cfg.right_hand_gone_timeout_ms:
            self._two_hand_session_active = False
            self._last_left_palm = None

        self._last_right_palm = palm
        self._last_right_seen_ms = timestamp_ms
        self._multi_trackers["Left"].reset()

        info = self._multi_trackers["Right"].update(
            single_lms,
            frame_w,
            frame_h,
            timestamp_ms,
            allow_scroll=True,
            ml_label=ml_label,
            ml_confidence=ml_conf,
        )
        info["hand_role"] = "Right"
        info["extra_hands"] = []
        info["gesture_label"] = ml_label
        info["gesture_confidence"] = ml_conf
        self._last_combined_info = info
        return info
