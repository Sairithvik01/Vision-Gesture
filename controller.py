"""Gesture state machine, middle-finger kinematic tracking, strict 2-hand role control, and smooth scroll.

Key Engineering Capabilities:
* Multi-Point Middle Finger Tracking: Determines the middle finger position using the 4-joint
  middle-finger kinematic chain (landmarks 9, 10, 11, 12) for zero-offset, low-jitter 1:1 cursor sync.
* 3D-Invariant Multi-Point Palm Scale: Uses composite palm length and knuckle width across
  landmarks (0, 5, 9, 13, 17) so pinch thresholds remain stable across 3D hand tilt.
* Strict 2-Hand Role Isolation: When 2 hands are detected, the Right hand's middle finger
  exclusively drives the cursor (and never accidentally locks into scroll mode), while the
  Left hand is strictly locked to actions only (scroll, clicks, drag-hold) and never moves
  the cursor as long as the Right hand is present (plus a configurable dropout grace period).
* Drag-and-Drop, Double-Click, Right-Click, Proportional Smooth Scroll, and Midas Clutch.
"""

import math
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

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
from features import normalize_landmarks
from filters import CursorBallistics, OneEuroFilter, Point2DOneEuroFilter


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
    """Per-hand kinematic tracking, multi-point middle finger estimator, and gesture state machine."""

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

        # 2-D Isotropic One-Euro Filter & Ballistics
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
        if in_max <= in_min:
            return (out_min + out_max) * 0.5
        if val <= in_min:
            return out_min
        if val >= in_max:
            return out_max
        return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)

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
    # Vectorized Multi-Point Geometry & Middle-Finger Determination
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_pts_px(landmarks, frame_w: int, frame_h: int) -> np.ndarray:
        """Extract all 21 landmarks into a contiguous (21, 2) float64 pixel array."""
        pts = np.empty((21, 2), dtype=np.float64)
        fw = float(frame_w)
        fh = float(frame_h)
        for i, lm in enumerate(landmarks):
            pts[i, 0] = lm.x * fw
            pts[i, 1] = lm.y * fh
        return pts

    @staticmethod
    def _determine_middle_finger_point(
        pts: np.ndarray, use_kinematic_blend: bool = True
    ) -> Tuple[float, float]:
        """Determine the exact middle finger position using its 4-point kinematic chain.

        Landmarks:
          9  = MIDDLE_FINGER_MCP
          10 = MIDDLE_FINGER_PIP
          11 = MIDDLE_FINGER_DIP
          12 = MIDDLE_FINGER_TIP

        Uses landmark 12 as the primary middle-finger tip coordinate and stabilizes
        high-frequency single-point detector noise via bone-vector projection along
        10 -> 11 -> 12 when the middle finger joints are anatomically collinear.
        """
        mx, my = float(pts[12, 0]), float(pts[12, 1])
        if not use_kinematic_blend:
            return mx, my

        dip_x, dip_y = float(pts[11, 0]), float(pts[11, 1])
        pip_x, pip_y = float(pts[10, 0]), float(pts[10, 1])
        mcp_x, mcp_y = float(pts[9, 0]), float(pts[9, 1])

        d_dip_tip = math.hypot(mx - dip_x, my - dip_y)
        d_pip_dip = math.hypot(dip_x - pip_x, dip_y - pip_y)
        d_mcp_tip = math.hypot(mx - mcp_x, my - mcp_y)

        # Verify anatomical validity of intermediate joints (guards synthetic mocks
        # where landmark 11 may be left at the wrist default).
        if (
            d_mcp_tip > 4.0
            and 0.5 < d_dip_tip < d_mcp_tip * 0.65
            and 0.5 < d_pip_dip < d_mcp_tip * 0.65
        ):
            bone_ratio = min(max(d_dip_tip / d_pip_dip, 0.4), 1.2)
            proj_x = dip_x + (dip_x - pip_x) * bone_ratio
            proj_y = dip_y + (dip_y - pip_y) * bone_ratio

            # Clamp kinematic correction to a tight 3.5 px radius so the tracking point
            # never deviates visually from the middle fingertip (landmark 12), while
            # smoothing out single-frame MediaPipe tip jitter.
            diff_x = proj_x - mx
            diff_y = proj_y - my
            diff_mag = math.hypot(diff_x, diff_y)
            if diff_mag <= 3.5:
                return (0.80 * mx + 0.20 * proj_x, 0.80 * my + 0.20 * proj_y)

        return mx, my

    @staticmethod
    def _get_extended_fingers_from_pts(pts: np.ndarray) -> Dict[str, bool]:
        """Vectorized rotation-invariant finger extension analysis across all 21 points."""
        wx, wy = pts[0, 0], pts[0, 1]

        dx = pts[:, 0] - wx
        dy = pts[:, 1] - wy
        dists = np.hypot(dx, dy)

        extended: Dict[str, bool] = {
            "index":  bool(dists[8] > dists[5] * 1.15 and dists[8] > dists[6]),
            "middle": bool(dists[12] > dists[9] * 1.15 and dists[12] > dists[10]),
            "ring":   bool(dists[16] > dists[13] * 1.15 and dists[16] > dists[14]),
            "pinky":  bool(dists[20] > dists[17] * 1.15 and dists[20] > dists[18]),
            "thumb":  bool(dists[4] > dists[2] * 1.15),
        }
        return extended

    @staticmethod
    def _get_extended_fingers(landmarks, frame_w: int, frame_h: int) -> Dict[str, bool]:
        """Rotation-invariant finger extension analysis using joint chain vectors."""
        pts = _HandTracker._extract_pts_px(landmarks, frame_w, frame_h)
        return _HandTracker._get_extended_fingers_from_pts(pts)

    # ------------------------------------------------------------------
    # Mode Transition State Machine
    # ------------------------------------------------------------------

    def _detect_target_mode(
        self, extended: Dict[str, bool], allow_scroll: bool = True
    ) -> ControlMode:
        num_extended = sum(extended.values())
        if self.cfg.clutch_enabled and num_extended == 0:
            return ControlMode.STANDBY

        if (
            allow_scroll
            and extended["index"]
            and extended["middle"]
            and not extended["ring"]
            and not extended["pinky"]
        ):
            return ControlMode.SCROLL

        return ControlMode.CURSOR

    def _update_control_mode(
        self,
        extended: Dict[str, bool],
        scroll_y: float,
        timestamp_ms: int,
        allow_scroll: bool = True,
    ) -> None:
        target = self._detect_target_mode(extended, allow_scroll=allow_scroll)

        if self._control_mode == ControlMode.CURSOR:
            if target == ControlMode.SCROLL:
                if self._mode_candidate_since_ms is None:
                    self._mode_candidate_since_ms = timestamp_ms
                elif timestamp_ms - self._mode_candidate_since_ms >= self.cfg.mode_confirm_ms:
                    self._control_mode = ControlMode.SCROLL
                    self._mode_candidate_since_ms = None
                    self._scroll_anchor_y = scroll_y
                    self._scroll_subtick_accum = 0.0
                    self._scroll_y_filter.reset()
            elif target == ControlMode.STANDBY and self.cfg.clutch_enabled:
                if self._clutch_candidate_since_ms is None:
                    self._clutch_candidate_since_ms = timestamp_ms
                elif timestamp_ms - self._clutch_candidate_since_ms >= self.cfg.clutch_confirm_ms:
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
    ) -> dict:
        pts = self._extract_pts_px(landmarks, frame_w, frame_h)

        tx, ty = float(pts[4, 0]), float(pts[4, 1])
        ix, iy = float(pts[8, 0]), float(pts[8, 1])
        mx, my = float(pts[12, 0]), float(pts[12, 1])
        mmx, mmy = float(pts[9, 0]), float(pts[9, 1])
        wx, wy = float(pts[0, 0]), float(pts[0, 1])

        # Determine middle-finger position using multi-point kinematic chain (9-10-11-12)
        point_x, point_y = self._determine_middle_finger_point(
            pts, use_kinematic_blend=self.cfg.middle_finger_kinematic_blend
        )

        mid_x = (ix + mx) * 0.5
        mid_y = (iy + my) * 0.5

        # 3D-invariant multi-point palm scale:
        # Combines wrist->middle_MCP length with index_MCP->pinky_MCP knuckle width
        # so forward/backward hand tilt does not distort pinch ratios.
        palm_len = math.hypot(mmx - wx, mmy - wy)
        knuckle_span = math.hypot(pts[5, 0] - pts[17, 0], pts[5, 1] - pts[17, 1])
        hand_scale = max(palm_len, knuckle_span * 0.90, 1.0)

        # Pinch metrics
        lclick_dist = math.hypot(tx - ix, ty - iy)
        pinch_ratio = lclick_dist / hand_scale

        rclick_dist = math.hypot(tx - mx, ty - my)
        right_pinch_ratio = rclick_dist / hand_scale

        extended = self._get_extended_fingers_from_pts(pts)
        self._update_control_mode(
            extended=extended,
            scroll_y=mid_y,
            timestamp_ms=timestamp_ms,
            allow_scroll=allow_scroll,
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
                    raw_screen_x = self._remap_and_clamp(
                        point_x, margin_x, frame_w - margin_x, 0.0, out_max_x
                    )
                    raw_screen_y = self._remap_and_clamp(
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


# ---------------------------------------------------------------------------
# Public Controller (Multi-Hand Role Isolation & ML Inference Orchestration)
# ---------------------------------------------------------------------------

class HandGestureController:
    """Production Multi-Hand Controller with Strict Right-Hand Cursor Isolation.

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
        self._last_right_palm = None
        self._last_left_palm = None
        self._last_right_seen_ms = -(10 ** 9)
        self._two_hand_session_active = False
        self._last_processed_ts = -(10 ** 9)
        self._last_combined_info = None

    @staticmethod
    def _palm_centroid(hand_lms) -> Tuple[float, float]:
        """Compute normalized (x, y) palm centroid using rigid palm landmarks (0, 5, 9, 13, 17)."""
        indices = (0, 5, 9, 13, 17)
        cx = sum(hand_lms[i].x for i in indices) * 0.2
        cy = sum(hand_lms[i].y for i in indices) * 0.2
        return (cx, cy)

    def _assign_two_hand_roles(
        self, all_landmarks: List, handedness: Optional[List]
    ) -> List[Tuple[object, str, Tuple[float, float]]]:
        """Assign 'Right' (cursor) and 'Left' (actions-only) roles to 2 detected hands."""
        lms_0, lms_1 = all_landmarks[0], all_landmarks[1]
        p0 = self._palm_centroid(lms_0)
        p1 = self._palm_centroid(lms_1)

        # 1. If both hands have temporal history and are very close horizontally,
        # use minimum-distance centroid matching to prevent cross-over swaps.
        if (
            self._last_right_palm is not None
            and self._last_left_palm is not None
            and abs(p0[0] - p1[0]) < 0.08
        ):
            d0_r = math.hypot(p0[0] - self._last_right_palm[0], p0[1] - self._last_right_palm[1])
            d0_l = math.hypot(p0[0] - self._last_left_palm[0], p0[1] - self._last_left_palm[1])
            d1_r = math.hypot(p1[0] - self._last_right_palm[0], p1[1] - self._last_right_palm[1])
            d1_l = math.hypot(p1[0] - self._last_left_palm[0], p1[1] - self._last_left_palm[1])
            if d0_r + d1_l <= d0_l + d1_r:
                return [(lms_0, "Right", p0), (lms_1, "Left", p1)]
            return [(lms_1, "Right", p1), (lms_0, "Left", p0)]

        # 2. Spatial sorting in the mirrored webcam frame:
        # In a horizontally flipped (mirror) frame, the user's physical Right hand sits
        # on the right side of the image (larger X), and the Left hand sits on the left
        # side (smaller X).
        if self.cfg.spatial_hand_sorting and abs(p0[0] - p1[0]) >= 0.04:
            if p0[0] >= p1[0]:
                return [(lms_0, "Right", p0), (lms_1, "Left", p1)]
            else:
                return [(lms_1, "Right", p1), (lms_0, "Left", p0)]

        # 3. Handedness classification fallback (inverted for mirrored webcam frame)
        if handedness and len(handedness) >= 2 and handedness[0] and handedness[1]:
            raw_0 = handedness[0][0].category_name
            raw_1 = handedness[1][0].category_name
            role_0 = "Left" if raw_0 == "Right" else "Right"
            role_1 = "Left" if raw_1 == "Right" else "Right"
            if role_0 != role_1:
                if role_0 == "Right":
                    return [(lms_0, "Right", p0), (lms_1, "Left", p1)]
                return [(lms_1, "Right", p1), (lms_0, "Left", p0)]

        if p0[0] >= p1[0]:
            return [(lms_0, "Right", p0), (lms_1, "Left", p1)]
        return [(lms_1, "Right", p1), (lms_0, "Left", p0)]

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
        """Update controller state with strict 2-hand role separation and middle-finger tracking."""
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

            # In 2-hand mode, the Right hand middle finger is dedicated to cursor tracking
            # (allow_scroll=False so two-finger pointing never freezes the cursor into scroll mode),
            # while the Left hand is dedicated to actions (scroll, clicks, drag).
            right_info = self._multi_trackers["Right"].update(
                right_lms, frame_w, frame_h, timestamp_ms, allow_scroll=False
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

            combined_info["gesture_label"] = self._infer_gesture(
                right_lms, frame_w, frame_h, classifier, label_encoder
            )
            self._last_combined_info = combined_info
            return combined_info

        # ── Case 2: Single hand detected ─────────────────────────────────────
        single_lms = all_landmarks[0] if (all_landmarks and len(all_landmarks) == 1) else landmarks
        palm = self._palm_centroid(single_lms)

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
                single_lms, frame_w, frame_h, timestamp_ms, allow_scroll=True
            )
            info["hand_role"] = "Left"
            info["extra_hands"] = []
            info["gesture_label"] = self._infer_gesture(
                single_lms, frame_w, frame_h, classifier, label_encoder
            )
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
            single_lms, frame_w, frame_h, timestamp_ms, allow_scroll=True
        )
        info["hand_role"] = "Right"
        info["extra_hands"] = []
        info["gesture_label"] = self._infer_gesture(
            single_lms, frame_w, frame_h, classifier, label_encoder
        )
        self._last_combined_info = info
        return info

    def _infer_gesture(
        self, landmarks, frame_w, frame_h, classifier, label_encoder
    ) -> Optional[str]:
        gesture_label = self._last_gesture_label
        should_infer = self._gesture_infer_counter % self.cfg.gesture_infer_every == 0
        self._gesture_infer_counter += 1

        if should_infer and classifier is not None and label_encoder is not None:
            try:
                lm_arr = np.array(
                    [[lm.x * frame_w, lm.y * frame_h, lm.z * frame_w] for lm in landmarks],
                    dtype=np.float64,
                )
                features = normalize_landmarks(lm_arr).reshape(1, -1)
                pred_idx = classifier.predict(features)[0]
                gesture_label = label_encoder.inverse_transform([pred_idx])[0]
                self._last_gesture_label = gesture_label
            except Exception:
                pass

        return gesture_label
