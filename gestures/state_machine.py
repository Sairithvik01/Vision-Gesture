"""Hybrid Deterministic-Probabilistic Gesture Finite State Machine & Hand Tracker."""

import math
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pyautogui
except ImportError:
    class _MockPyAutoGUI:
        FAILSAFE = True
        PAUSE = 0
        @staticmethod
        def click(*args, **kwargs): pass
        @staticmethod
        def doubleClick(*args, **kwargs): pass
        @staticmethod
        def rightClick(*args, **kwargs): pass
        @staticmethod
        def mouseDown(*args, **kwargs): pass
        @staticmethod
        def mouseUp(*args, **kwargs): pass
        @staticmethod
        def scroll(*args, **kwargs): pass
        @staticmethod
        def moveTo(*args, **kwargs): pass
        @staticmethod
        def size(): return (1920, 1080)
    pyautogui = _MockPyAutoGUI

from actions import ActionWorker
from config import Config
from cursor import CursorWorker
from filters import (
    CursorBallistics,
    OneEuroFilter,
    Point2DEMAFilter,
    Point2DKalmanFilter,
    Point2DOneEuroFilter,
)
from gestures.kinematics import (
    compute_3d_palm_scale,
    determine_middle_finger_point,
    extract_pts_px,
    get_extended_fingers,
    get_extended_fingers_from_pts,
    remap_and_clamp,
)


class PinchState(Enum):
    OPEN = auto()
    PINCH_CANDIDATE = auto()
    PINCHED = auto()
    DRAGGING = auto()


class ControlMode(Enum):
    """Top-level operating mode."""
    CURSOR = auto()
    SCROLL = auto()
    STANDBY = auto()


class _HandTracker:
    """Per-hand kinematic tracking, multi-point middle finger estimator, and hybrid gesture FSM."""

    def __init__(
        self,
        cfg: Config,
        screen_w: int,
        screen_h: int,
        cursor_worker: CursorWorker,
        action_worker: Optional[ActionWorker] = None,
        drives_cursor: bool = True,
    ):
        self.cfg = cfg
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cursor_worker = cursor_worker
        self.action_worker = action_worker
        self.drives_cursor = drives_cursor

        # Configurable 2-D Filter (One-Euro default, Kalman or EMA selectable for study)
        filter_type = getattr(cfg, "filter_type", "one_euro").lower()
        if filter_type == "kalman":
            self._cursor_filter = Point2DKalmanFilter(
                process_noise=getattr(cfg, "kalman_process_noise", 1e-2),
                measurement_noise=getattr(cfg, "kalman_measurement_noise", 2.5),
            )
        elif filter_type == "ema":
            self._cursor_filter = Point2DEMAFilter(alpha=0.25)
        else:
            self._cursor_filter = Point2DOneEuroFilter(
                min_cutoff=cfg.cursor_min_cutoff,
                beta=cfg.cursor_beta,
                d_cutoff=cfg.cursor_d_cutoff,
                deadband_px=cfg.deadband_px,
            )

        self._ballistics = CursorBallistics(
            precision_gain=cfg.ballistics_precision_gain,
            ballistic_exponent=cfg.ballistics_exponent,
        )

        self._prev_screen_x: Optional[float] = None
        self._prev_screen_y: Optional[float] = None
        self._prev_filtered_x: Optional[float] = None
        self._prev_filtered_y: Optional[float] = None
        self._prev_time_s: Optional[float] = None

        # Anti Click-Slip anchor lock
        self._locked_target_x: Optional[float] = None
        self._locked_target_y: Optional[float] = None

        # Left-click & Drag-and-drop state
        self._lclick_state = PinchState.OPEN
        self._lclick_candidate_since_ms: Optional[int] = None
        self._pinch_start_ms: Optional[int] = None
        self._last_lclick_ms = -(10 ** 9)
        self._prev_lclick_ms = -(10 ** 9)

        # Right-click state
        self._rclick_state = PinchState.OPEN
        self._rclick_candidate_since_ms: Optional[int] = None
        self._last_rclick_ms = -(10 ** 9)

        # Mode and Scrolling
        self._control_mode = ControlMode.CURSOR
        self._mode_candidate_since_ms: Optional[int] = None
        self._mode_break_since_ms: Optional[int] = None
        self._scroll_anchor_y: Optional[float] = None
        self._scroll_subtick_accum: float = 0.0
        self._last_scroll_tick_ms = -(10 ** 9)
        self._scroll_y_filter = OneEuroFilter(
            min_cutoff=cfg.scroll_smooth_min_cutoff,
            beta=cfg.scroll_smooth_beta,
            d_cutoff=cfg.cursor_d_cutoff,
        )

        # Clutch / Standby state
        self._clutch_candidate_since_ms: Optional[int] = None

    def _dispatch(self, action) -> None:
        if self.action_worker is not None:
            self.action_worker.submit(action)
        else:
            action()

    @staticmethod
    def _remap_and_clamp(
        val: float, in_min: float, in_max: float, out_min: float, out_max: float
    ) -> float:
        return remap_and_clamp(val, in_min, in_max, out_min, out_max)

    @staticmethod
    def _extract_pts_px(landmarks, frame_w: int, frame_h: int) -> np.ndarray:
        return extract_pts_px(landmarks, frame_w, frame_h)

    @staticmethod
    def _determine_middle_finger_point(
        pts: np.ndarray, use_kinematic_blend: bool = True
    ) -> Tuple[float, float]:
        return determine_middle_finger_point(pts, use_kinematic_blend=use_kinematic_blend)

    @staticmethod
    def _get_extended_fingers_from_pts(pts: np.ndarray) -> Dict[str, bool]:
        return get_extended_fingers_from_pts(pts)

    @staticmethod
    def _get_extended_fingers(landmarks, frame_w: int, frame_h: int) -> Dict[str, bool]:
        return get_extended_fingers(landmarks, frame_w, frame_h)

    def reset_cursor_state(self) -> None:
        """Reset only cursor filtering state without dropping active click/scroll states."""
        self._cursor_filter.reset()
        self._prev_screen_x = None
        self._prev_screen_y = None
        self._prev_filtered_x = None
        self._prev_filtered_y = None
        self._prev_time_s = None
        self._locked_target_x = None
        self._locked_target_y = None

    def reset(self) -> None:
        if self._lclick_state == PinchState.DRAGGING:
            self._dispatch(pyautogui.mouseUp)
        self._lclick_state = PinchState.OPEN
        self._lclick_candidate_since_ms = None
        self._pinch_start_ms = None
        self._rclick_state = PinchState.OPEN
        self._rclick_candidate_since_ms = None
        self.reset_cursor_state()
        self._control_mode = ControlMode.CURSOR
        self._mode_candidate_since_ms = None
        self._mode_break_since_ms = None
        self._scroll_anchor_y = None
        self._scroll_subtick_accum = 0.0
        self._scroll_y_filter.reset()
        self._clutch_candidate_since_ms = None

    # ------------------------------------------------------------------
    # Hybrid Deterministic-Probabilistic Mode Transition State Machine
    # ------------------------------------------------------------------

    def _detect_target_mode(
        self,
        extended: Dict[str, bool],
        allow_scroll: bool = True,
        ml_label: Optional[str] = None,
        ml_confidence: float = 0.0,
    ) -> ControlMode:
        num_extended = sum(extended.values())
        use_ml = (
            getattr(self.cfg, "hybrid_ml_gating", True)
            and ml_label is not None
            and ml_confidence >= getattr(self.cfg, "ml_confidence_threshold", 0.65)
        )

        # Deterministic closed-fist check OR high-confidence ML Fist_Clutch when fingers mostly curled
        if self.cfg.clutch_enabled:
            if num_extended == 0 or (use_ml and ml_label == "Fist_Clutch" and num_extended <= 1):
                return ControlMode.STANDBY

        # Deterministic two-finger scroll check OR high-confidence ML Peace_Scroll
        if allow_scroll:
            is_geom_scroll = (
                extended["index"]
                and extended["middle"]
                and not extended["ring"]
                and not extended["pinky"]
            )
            is_ml_scroll = (
                use_ml
                and ml_label == "Peace_Scroll"
                and extended["index"]
                and extended["middle"]
                and not extended["pinky"]
            )
            if is_geom_scroll or is_ml_scroll:
                return ControlMode.SCROLL

        return ControlMode.CURSOR

    def _update_control_mode(
        self,
        extended: Dict[str, bool],
        scroll_y: float,
        timestamp_ms: int,
        allow_scroll: bool = True,
        ml_label: Optional[str] = None,
        ml_confidence: float = 0.0,
    ) -> None:
        target = self._detect_target_mode(
            extended,
            allow_scroll=allow_scroll,
            ml_label=ml_label,
            ml_confidence=ml_confidence,
        )

        # When ML classifier confirms the gesture with high probability, accelerate confirmation window
        use_ml_boost = (
            getattr(self.cfg, "hybrid_ml_gating", True)
            and ml_label is not None
            and ml_confidence >= getattr(self.cfg, "ml_confidence_threshold", 0.65)
        )
        fast_ratio = getattr(self.cfg, "ml_fast_confirm_ratio", 0.50)

        scroll_confirm_ms = (
            int(self.cfg.mode_confirm_ms * fast_ratio)
            if (use_ml_boost and ml_label == "Peace_Scroll")
            else self.cfg.mode_confirm_ms
        )
        clutch_confirm_ms = (
            int(self.cfg.clutch_confirm_ms * fast_ratio)
            if (use_ml_boost and ml_label == "Fist_Clutch")
            else self.cfg.clutch_confirm_ms
        )

        if self._control_mode == ControlMode.CURSOR:
            if target == ControlMode.SCROLL:
                if self._mode_candidate_since_ms is None:
                    self._mode_candidate_since_ms = timestamp_ms
                elif timestamp_ms - self._mode_candidate_since_ms >= scroll_confirm_ms:
                    self._control_mode = ControlMode.SCROLL
                    self._mode_candidate_since_ms = None
                    self._scroll_anchor_y = scroll_y
                    self._scroll_subtick_accum = 0.0
                    self._scroll_y_filter.reset()
            elif target == ControlMode.STANDBY and self.cfg.clutch_enabled:
                if self._clutch_candidate_since_ms is None:
                    self._clutch_candidate_since_ms = timestamp_ms
                elif timestamp_ms - self._clutch_candidate_since_ms >= clutch_confirm_ms:
                    self._control_mode = ControlMode.STANDBY
                    self._clutch_candidate_since_ms = None
            else:
                self._mode_candidate_since_ms = None
                self._clutch_candidate_since_ms = None

        elif self._control_mode == ControlMode.SCROLL:
            if target != ControlMode.SCROLL:
                if self._mode_break_since_ms is None:
                    self._mode_break_since_ms = timestamp_ms
                elif timestamp_ms - self._mode_break_since_ms >= self.cfg.mode_exit_confirm_ms:
                    self._control_mode = ControlMode.CURSOR
                    self._scroll_anchor_y = None
                    self._scroll_subtick_accum = 0.0
                    self._mode_candidate_since_ms = None
                    self._mode_break_since_ms = None
                    self._scroll_y_filter.reset()
            else:
                self._mode_break_since_ms = None

        elif self._control_mode == ControlMode.STANDBY:
            if target != ControlMode.STANDBY:
                self._control_mode = ControlMode.CURSOR
                self._clutch_candidate_since_ms = None

    # ------------------------------------------------------------------
    # Continuous Proportional Scroll
    # ------------------------------------------------------------------

    def _update_scroll(
        self,
        mid_y: float,
        frame_h: int,
        timestamp_ms: int,
    ) -> Tuple[Optional[str], float]:
        ts = timestamp_ms / 1000.0
        smooth_y = self._scroll_y_filter.filter(mid_y, ts)

        if self._scroll_anchor_y is None:
            self._scroll_anchor_y = smooth_y
            return None, 0.0

        elapsed_ms = timestamp_ms - self._last_scroll_tick_ms
        if elapsed_ms < self.cfg.scroll_tick_interval_ms:
            return None, 0.0

        dy = smooth_y - self._scroll_anchor_y
        deadzone = self.cfg.scroll_deadzone_ratio * frame_h
        action: Optional[str] = None
        velocity = 0.0

        if abs(dy) > deadzone:
            excess = dy - (deadzone if dy > 0 else -deadzone)
            norm_excess = excess / max(frame_h * 0.25, 1.0)
            sign = 1.0 if excess > 0 else -1.0
            speed = min(
                abs(norm_excess) ** 1.15 * self.cfg.scroll_max_speed,
                float(self.cfg.scroll_max_speed),
            )
            velocity = -sign * speed

            dt_factor = min(elapsed_ms / 33.3, 3.0)
            self._scroll_subtick_accum += velocity * dt_factor
            ticks_to_fire = int(self._scroll_subtick_accum)

            if ticks_to_fire != 0:
                self._dispatch(lambda t=ticks_to_fire: pyautogui.scroll(t))
                self._scroll_subtick_accum -= ticks_to_fire
                self._last_scroll_tick_ms = timestamp_ms
                action = "scroll_up" if ticks_to_fire > 0 else "scroll_down"
        else:
            self._scroll_subtick_accum *= 0.5
            self._last_scroll_tick_ms = timestamp_ms

        return action, velocity

    # ------------------------------------------------------------------
    # Left-Click, Double-Click & Drag-and-Drop
    # ------------------------------------------------------------------

    def _update_left_pinch(
        self, pinch_ratio: float, timestamp_ms: int
    ) -> Tuple[bool, bool, bool, bool, bool]:
        """Return (pinched, clicked, double_clicked, dragging, drag_released)."""
        clicked = False
        double_clicked = False
        dragging = False
        drag_released = False

        if self._lclick_state == PinchState.OPEN:
            if pinch_ratio <= self.cfg.pinch_on_ratio:
                if self.cfg.pinch_confirm_ms <= 0:
                    self._lclick_state = PinchState.PINCHED
                    self._pinch_start_ms = timestamp_ms
                else:
                    self._lclick_state = PinchState.PINCH_CANDIDATE
                    self._lclick_candidate_since_ms = timestamp_ms

        elif self._lclick_state == PinchState.PINCH_CANDIDATE:
            if pinch_ratio >= self.cfg.pinch_off_ratio:
                self._lclick_state = PinchState.OPEN
                self._lclick_candidate_since_ms = None
            elif (
                self._lclick_candidate_since_ms is not None
                and timestamp_ms - self._lclick_candidate_since_ms >= self.cfg.pinch_confirm_ms
            ):
                self._lclick_state = PinchState.PINCHED
                self._pinch_start_ms = timestamp_ms
                self._lclick_candidate_since_ms = None

        elif self._lclick_state == PinchState.PINCHED:
            held_duration = (
                timestamp_ms - self._pinch_start_ms
                if self._pinch_start_ms is not None
                else 0
            )

            if pinch_ratio >= self.cfg.pinch_off_ratio:
                # Released before drag threshold: Discrete Click / Double-Click!
                self._lclick_state = PinchState.OPEN
                time_since_prev = timestamp_ms - self._prev_lclick_ms
                if (
                    self.cfg.double_click_min_interval_ms
                    <= time_since_prev
                    <= self.cfg.double_click_window_ms
                ):
                    self._dispatch(pyautogui.doubleClick)
                    double_clicked = True
                    self._prev_lclick_ms = -(10 ** 9)
                    self._last_lclick_ms = timestamp_ms
                elif timestamp_ms - self._last_lclick_ms >= self.cfg.click_cooldown_ms:
                    self._dispatch(pyautogui.click)
                    clicked = True
                    self._prev_lclick_ms = timestamp_ms
                    self._last_lclick_ms = timestamp_ms
                self._pinch_start_ms = None

            elif held_duration >= self.cfg.drag_hold_ms:
                # Sustained pinch: Engage native drag!
                self._lclick_state = PinchState.DRAGGING
                self._dispatch(pyautogui.mouseDown)
                dragging = True

        elif self._lclick_state == PinchState.DRAGGING:
            dragging = True
            if pinch_ratio >= self.cfg.pinch_off_ratio:
                # Released drag: drop item!
                self._lclick_state = PinchState.OPEN
                self._dispatch(pyautogui.mouseUp)
                drag_released = True
                self._pinch_start_ms = None
                self._last_lclick_ms = timestamp_ms

        pinched = self._lclick_state in (PinchState.PINCHED, PinchState.DRAGGING)
        return pinched, clicked, double_clicked, dragging, drag_released

    # ------------------------------------------------------------------
    # Right-Click Hysteresis
    # ------------------------------------------------------------------

    def _update_right_pinch(
        self, right_pinch_ratio: float, timestamp_ms: int
    ) -> Tuple[bool, bool]:
        """Return (right_pinched, right_clicked)."""
        right_clicked = False

        if self._rclick_state == PinchState.OPEN:
            if right_pinch_ratio <= self.cfg.right_pinch_on_ratio:
                self._rclick_state = PinchState.PINCH_CANDIDATE
                self._rclick_candidate_since_ms = timestamp_ms

        elif self._rclick_state == PinchState.PINCH_CANDIDATE:
            if right_pinch_ratio >= self.cfg.right_pinch_off_ratio:
                self._rclick_state = PinchState.OPEN
                self._rclick_candidate_since_ms = None
            elif (
                self._rclick_candidate_since_ms is not None
                and timestamp_ms - self._rclick_candidate_since_ms >= self.cfg.right_pinch_confirm_ms
            ):
                self._rclick_state = PinchState.PINCHED
                self._rclick_candidate_since_ms = None
                if timestamp_ms - self._last_rclick_ms >= self.cfg.right_click_cooldown_ms:
                    self._dispatch(pyautogui.rightClick)
                    self._last_rclick_ms = timestamp_ms
                    right_clicked = True

        elif self._rclick_state == PinchState.PINCHED:
            if right_pinch_ratio >= self.cfg.right_pinch_off_ratio:
                self._rclick_state = PinchState.OPEN

        return self._rclick_state == PinchState.PINCHED, right_clicked

    # ------------------------------------------------------------------
    # Per-frame Update
    # ------------------------------------------------------------------

    def update(
        self,
        landmarks,
        frame_w: int,
        frame_h: int,
        timestamp_ms: int,
        allow_scroll: bool = True,
        ml_label: Optional[str] = None,
        ml_confidence: float = 0.0,
    ) -> dict:
        pts = extract_pts_px(landmarks, frame_w, frame_h)

        tx, ty = float(pts[4, 0]), float(pts[4, 1])
        ix, iy = float(pts[8, 0]), float(pts[8, 1])
        mx, my = float(pts[12, 0]), float(pts[12, 1])

        # Determine middle-finger position using multi-point kinematic chain (9-10-11-12)
        point_x, point_y = determine_middle_finger_point(
            pts, use_kinematic_blend=self.cfg.middle_finger_kinematic_blend
        )

        mid_x = (ix + mx) * 0.5
        mid_y = (iy + my) * 0.5

        # 3D-invariant multi-point palm scale
        hand_scale = compute_3d_palm_scale(pts)

        # Pinch metrics
        lclick_dist = math.hypot(tx - ix, ty - iy)
        pinch_ratio = lclick_dist / hand_scale

        rclick_dist = math.hypot(tx - mx, ty - my)
        right_pinch_ratio = rclick_dist / hand_scale

        extended = get_extended_fingers_from_pts(pts)
        self._update_control_mode(
            extended=extended,
            scroll_y=mid_y,
            timestamp_ms=timestamp_ms,
            allow_scroll=allow_scroll,
            ml_label=ml_label,
            ml_confidence=ml_confidence,
        )

        scroll_action: Optional[str] = None
        scroll_velocity: float = 0.0
        pinched = False
        left_clicked = False
        double_clicked = False
        dragging = False
        drag_released = False
        right_pinched = False
        right_clicked = False

        if self._control_mode == ControlMode.STANDBY:
            pass

        elif self._control_mode == ControlMode.SCROLL:
            scroll_action, scroll_velocity = self._update_scroll(
                mid_y=mid_y, frame_h=frame_h, timestamp_ms=timestamp_ms
            )

        else:
            (
                pinched,
                left_clicked,
                double_clicked,
                dragging,
                drag_released,
            ) = self._update_left_pinch(pinch_ratio, timestamp_ms)

            right_pinched, right_clicked = self._update_right_pinch(
                right_pinch_ratio, timestamp_ms
            )

            # ONLY move the OS cursor when this tracker is designated as the cursor driver
            # (Right hand when 2 hands are present; never Left hand while Right hand is active).
            if self.drives_cursor:
                margin_x = min(self.cfg.frame_margin, max(frame_w // 4, 0))
                margin_y = min(self.cfg.frame_margin, max(frame_h // 4, 0))

                out_max_x = float(max(self.screen_w - 1, 0))
                out_max_y = float(max(self.screen_h - 1, 0))

                if self.cfg.edge_easing:
                    raw_screen_x = CursorBallistics.ease_edge(
                        point_x, margin_x, frame_w - margin_x, 0.0, out_max_x
                    )
                    raw_screen_y = CursorBallistics.ease_edge(
                        point_y, margin_y, frame_h - margin_y, 0.0, out_max_y
                    )
                else:
                    raw_screen_x = remap_and_clamp(
                        point_x, margin_x, frame_w - margin_x, 0.0, out_max_x
                    )
                    raw_screen_y = remap_and_clamp(
                        point_y, margin_y, frame_h - margin_y, 0.0, out_max_y
                    )

                ts = timestamp_ms / 1000.0

                is_pre_pinch = (
                    self.cfg.anchor_lock_on_pinch
                    and (
                        pinch_ratio <= self.cfg.pinch_lock_ratio
                        or right_pinch_ratio <= self.cfg.right_pinch_on_ratio * 1.05
                    )
                    and not dragging
                )

                if is_pre_pinch:
                    if self._locked_target_x is None or self._locked_target_y is None:
                        self._locked_target_x = raw_screen_x
                        self._locked_target_y = raw_screen_y
                    target_x = self._locked_target_x
                    target_y = self._locked_target_y
                else:
                    self._locked_target_x = None
                    self._locked_target_y = None
                    target_x = raw_screen_x
                    target_y = raw_screen_y

                filtered_x, filtered_y = self._cursor_filter.filter(target_x, target_y, ts)

                # Anchored precision damping when use_ballistics=True, or direct 1:1 sync
                # when use_ballistics=False. Never accumulates unbounded offset error.
                if (
                    self.cfg.use_ballistics
                    and self._prev_screen_x is not None
                    and self._prev_time_s is not None
                    and not is_pre_pinch
                    and not dragging
                ):
                    dt = max(ts - self._prev_time_s, 1e-4)
                    prev_fx = self._prev_filtered_x if self._prev_filtered_x is not None else self._prev_screen_x
                    prev_fy = self._prev_filtered_y if self._prev_filtered_y is not None else self._prev_screen_y
                    speed_px_s = math.hypot(filtered_x - prev_fx, filtered_y - prev_fy) / dt
                    gain = self._ballistics.compute_gain(speed_px_s)
                    if gain < 1.0:
                        curr_x = self._prev_screen_x + (filtered_x - self._prev_screen_x) * gain
                        curr_y = self._prev_screen_y + (filtered_y - self._prev_screen_y) * gain
                    else:
                        curr_x = filtered_x
                        curr_y = filtered_y
                else:
                    curr_x = filtered_x
                    curr_y = filtered_y

                if self._prev_screen_x is None or self._prev_screen_y is None:
                    should_move = True
                else:
                    disp = math.hypot(curr_x - self._prev_screen_x, curr_y - self._prev_screen_y)
                    should_move = disp >= self.cfg.min_move_px

                if should_move and (not is_pre_pinch or dragging):
                    self.cursor_worker.submit(curr_x, curr_y)
                    self._prev_screen_x = curr_x
                    self._prev_screen_y = curr_y
                elif is_pre_pinch and not dragging:
                    self._prev_screen_x = curr_x
                    self._prev_screen_y = curr_y

                self._prev_time_s = ts
                self._prev_filtered_x = filtered_x
                self._prev_filtered_y = filtered_y

        landmarks_px = [(int(pts[i, 0]), int(pts[i, 1])) for i in range(21)]

        return {
            "thumb_px":           (int(tx), int(ty)),
            "index_px":           (int(ix), int(iy)),
            "middle_px":          (int(mx), int(my)),
            "pointer_px":         (int(point_x), int(point_y)),
            "clicked":            left_clicked,
            "double_clicked":     double_clicked,
            "pinched":            pinched,
            "dragging":           dragging,
            "drag_released":      drag_released,
            "right_clicked":      right_clicked,
            "right_pinched":      right_pinched,
            "pinch_ratio":        pinch_ratio,
            "right_pinch_ratio":  right_pinch_ratio,
            "landmarks_px":       landmarks_px,
            "midpoint_px":        (int(mid_x), int(mid_y)),
            "control_mode":       self._control_mode.name,
            "scroll_action":      scroll_action,
            "scroll_velocity":    scroll_velocity,
            "extended":           extended,
            "drives_cursor":      self.drives_cursor,
        }
