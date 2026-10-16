"""Tier-1 Capstone Computer Vision HUD and Visual Feedback Overlay.

Features:
- Zero-allocation in-place glassmorphic telemetry cards and HUD status badges.
- Distinct visual separation between Right Hand (Middle-Finger Cursor Driver)
  and Left Hand (Actions Only — no cursor reticle on Left middle finger).
- Dynamic pinch meters (Left & Right) with threshold progress indicators.
- Proportional scroll velocity meter with directional indicators.
- Motion anchor reticle and active workspace boundary with corner brackets.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from config import Config


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _draw_glass_panel(
    frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.60
) -> None:
    """Draw a semi-transparent dark glass panel in-place without temporary allocations."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2]
    cv2.multiply(sub, 1.0 - alpha, dst=sub)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1, cv2.LINE_AA)


def _draw_corner_brackets(
    frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, length: int = 18, color=(255, 0, 200)
) -> None:
    """Draw stylish corner brackets for active workspace boundary."""
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, 2, cv2.LINE_AA)


def _draw_radial_meter(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    ratio: float,
    threshold: float,
    label: str,
    color_active=(0, 255, 120),
    color_idle=(80, 80, 80),
) -> None:
    """Draw circular gauge showing current pinch ratio vs trigger threshold."""
    cx, cy = center
    cv2.circle(frame, center, radius, color_idle, 2, cv2.LINE_AA)

    fill_fraction = max(0.0, min(1.0, 1.0 - (ratio / (threshold * 2.5))))
    angle_deg = int(fill_fraction * 360)

    color = color_active if ratio <= threshold else (0, 180, 255)
    cv2.ellipse(
        frame,
        center,
        (radius, radius),
        -90,
        0,
        angle_deg,
        color,
        3,
        cv2.LINE_AA,
    )
    thresh_angle_deg = int((1.0 - threshold / (threshold * 2.5)) * 360)
    thresh_rad = math.radians(-90 + thresh_angle_deg)
    tx = int(cx + (radius + 5) * math.cos(thresh_rad))
    ty = int(cy + (radius + 5) * math.sin(thresh_rad))
    cv2.circle(frame, (tx, ty), 3, (0, 60, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 3, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, cfg: Config, info: Optional[dict], fps: float) -> None:
    h, w = frame.shape[:2]

    margin_x = min(cfg.frame_margin, max(w // 4, 0))
    margin_y = min(cfg.frame_margin, max(h // 4, 0))

    # Workspace Bounding Box with styled corner brackets
    x1, y1 = margin_x, margin_y
    x2, y2 = w - margin_x - 1, h - margin_y - 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 80), 1, cv2.LINE_AA)
    _draw_corner_brackets(frame, x1, y1, x2, y2, length=20, color=(180, 50, 220))

    if info:
        mode = info.get("control_mode", "CURSOR")
        role = info.get("hand_role", "Right")
        drives_cursor = info.get("drives_cursor", role == "Right")

        # Draw Primary Hand Skeleton and Landmarks
        landmarks_px = info.get("landmarks_px", [])
        skel_color = (60, 180, 100) if drives_cursor else (160, 120, 60)
        for start, end in HAND_CONNECTIONS:
            if start < len(landmarks_px) and end < len(landmarks_px):
                cv2.line(frame, landmarks_px[start], landmarks_px[end], skel_color, 2, cv2.LINE_AA)

        for idx, pt in enumerate(landmarks_px):
            if idx == 12 and drives_cursor:
                # Middle fingertip — active cursor control point (Right hand only)
                cv2.circle(frame, pt, 9, (0, 255, 80), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 11, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(frame, pt, (255, 255, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
            elif idx in (4, 8, 12, 16, 20):
                cv2.circle(frame, pt, 6, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
            elif idx in (5, 6, 9, 10, 11):
                cv2.circle(frame, pt, 4, (255, 120, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, (120, 220, 150), -1, cv2.LINE_AA)

        if landmarks_px:
            wx0, wy0 = landmarks_px[0]
            role_tag = f"[{role}: CURSOR]" if drives_cursor else f"[{role}: ACTIONS ONLY]"
            cv2.putText(
                frame,
                role_tag,
                (max(10, wx0 - 45), min(h - 10, wy0 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 140) if drives_cursor else (255, 200, 50),
                1,
                cv2.LINE_AA,
            )

        # Draw Extra Hands (Left hand in 2-hand mode — strictly Actions Only)
        for extra in info.get("extra_hands", []):
            extra_lms = extra.get("landmarks_px", [])
            for start, end in HAND_CONNECTIONS:
                if start < len(extra_lms) and end < len(extra_lms):
                    cv2.line(frame, extra_lms[start], extra_lms[end], (160, 120, 60), 2, cv2.LINE_AA)
            for idx2, pt in enumerate(extra_lms):
                r2 = 6 if idx2 in (4, 8, 12, 16, 20) else 4
                cv2.circle(frame, pt, r2, (255, 160, 40), -1, cv2.LINE_AA)

            # Show pinch lines on the action hand
            if extra.get("thumb_px") and extra.get("index_px"):
                e_col = (0, 255, 0) if extra.get("pinched") else (180, 100, 180)
                cv2.line(frame, extra["thumb_px"], extra["index_px"], e_col, 2, cv2.LINE_AA)
            if extra.get("right_pinched") and extra.get("thumb_px") and extra.get("middle_px"):
                cv2.line(frame, extra["thumb_px"], extra["middle_px"], (255, 255, 0), 2, cv2.LINE_AA)

            extra_role = extra.get("hand_role", "Left")
            extra_mode = extra.get("control_mode", "CURSOR")
            mode_label = "ACTIONS" if extra_mode == "CURSOR" else extra_mode
            if extra_lms:
                wx2, wy2 = extra_lms[0]
                cv2.putText(
                    frame,
                    f"[{extra_role}: {mode_label} ONLY]",
                    (max(10, wx2 - 50), min(h - 10, wy2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 200, 50),
                    1,
                    cv2.LINE_AA,
                )

        # ── Mode-Specific Overlays ───────────────────────────────────────────
        if mode == "STANDBY":
            _draw_glass_panel(frame, w // 2 - 130, 15, w // 2 + 130, 65, alpha=0.7)
            cv2.putText(
                frame, "STANDBY (CLUTCH)", (w // 2 - 110, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 200, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, "Tracking Paused - Extend fingers to resume", (w // 2 - 160, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1, cv2.LINE_AA,
            )

        elif mode == "SCROLL":
            _draw_glass_panel(frame, w // 2 - 120, 15, w // 2 + 120, 65, alpha=0.7)
            cv2.putText(
                frame, "SMOOTH SCROLL", (w // 2 - 100, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 255), 2, cv2.LINE_AA,
            )

            mid_px = info.get("midpoint_px")
            vel = info.get("scroll_velocity", 0.0)
            if mid_px:
                mx, my = mid_px
                cv2.circle(frame, mid_px, 12, (0, 200, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, mid_px, 14, (255, 255, 255), 2, cv2.LINE_AA)

                vec_len = int(vel * 3.5)
                if abs(vec_len) > 2:
                    end_y = my - vec_len
                    color = (0, 255, 120) if vel > 0 else (0, 130, 255)
                    cv2.arrowedLine(frame, (mx, my), (mx, end_y), color, 3, tipLength=0.35)

            scroll_action = info.get("scroll_action")
            if scroll_action == "scroll_up":
                cv2.putText(
                    frame, "SCROLL UP", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 120), 3, cv2.LINE_AA,
                )
            elif scroll_action == "scroll_down":
                cv2.putText(
                    frame, "SCROLL DOWN", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 130, 255), 3, cv2.LINE_AA,
                )

        else:
            # ── Normal Cursor Mode ───────────────────────────────────────────
            pointer_px = info.get("pointer_px", info.get("middle_px"))

            if info.get("dragging"):
                reticle_color = (0, 160, 255)
            elif info.get("pinched"):
                reticle_color = (0, 0, 255)
            elif info.get("right_pinched"):
                reticle_color = (255, 180, 0)
            else:
                reticle_color = (255, 80, 0)

            # Draw cursor targeting reticle ONLY when this hand drives the cursor
            if pointer_px and drives_cursor:
                cv2.circle(frame, pointer_px, 10, reticle_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pointer_px, 14, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(frame, pointer_px, (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

            # Left pinch link (Thumb <-> Index)
            t_px, i_px = info["thumb_px"], info["index_px"]
            pinch_col = (0, 255, 0) if info["pinched"] else (220, 0, 220)
            cv2.line(frame, t_px, i_px, pinch_col, 2, cv2.LINE_AA)

            # Right pinch link (Thumb <-> Middle)
            if info.get("right_pinched"):
                m_px = info["middle_px"]
                cv2.line(frame, t_px, m_px, (255, 255, 0), 2, cv2.LINE_AA)

            # Flash Action Banners (including actions triggered by the Left hand in 2-hand mode)
            scroll_action = info.get("scroll_action")
            if info.get("double_clicked"):
                _draw_glass_panel(frame, w // 2 - 140, h // 2 - 35, w // 2 + 140, h // 2 + 25, alpha=0.7)
                cv2.putText(
                    frame, "DOUBLE CLICK", (w // 2 - 120, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 255), 3, cv2.LINE_AA,
                )
            elif info.get("clicked"):
                _draw_glass_panel(frame, w // 2 - 90, h // 2 - 35, w // 2 + 90, h // 2 + 25, alpha=0.7)
                cv2.putText(
                    frame, "CLICK", (w // 2 - 60, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3, cv2.LINE_AA,
                )
            elif info.get("dragging"):
                cv2.putText(
                    frame, "DRAG & DROP", (w // 2 - 110, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 160, 255), 3, cv2.LINE_AA,
                )
            elif info.get("drag_released"):
                cv2.putText(
                    frame, "DROPPED", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 180), 3, cv2.LINE_AA,
                )
            elif info.get("right_clicked"):
                _draw_glass_panel(frame, w // 2 - 130, h // 2 - 35, w // 2 + 130, h // 2 + 25, alpha=0.7)
                cv2.putText(
                    frame, "RIGHT CLICK", (w // 2 - 110, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 180, 0), 3, cv2.LINE_AA,
                )
            elif scroll_action == "scroll_up":
                cv2.putText(
                    frame, "SCROLL UP", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 120), 3, cv2.LINE_AA,
                )
            elif scroll_action == "scroll_down":
                cv2.putText(
                    frame, "SCROLL DOWN", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 130, 255), 3, cv2.LINE_AA,
                )

        # ── Bottom Telemetry Panel ───────────────────────────────────────────
        _draw_glass_panel(frame, 10, h - 80, 300, h - 10, alpha=0.65)

        l_ratio = info.get("pinch_ratio", 1.0)
        r_ratio = info.get("right_pinch_ratio", 1.0)
        _draw_radial_meter(
            frame, center=(35, h - 45), radius=18, ratio=l_ratio,
            threshold=cfg.pinch_on_ratio, label="L",
            color_active=(0, 255, 100),
        )
        cv2.putText(
            frame, f"L:{l_ratio:.2f}", (60, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

        _draw_radial_meter(
            frame, center=(150, h - 45), radius=18, ratio=r_ratio,
            threshold=cfg.right_pinch_on_ratio, label="R",
            color_active=(255, 200, 0),
        )
        cv2.putText(
            frame, f"R:{r_ratio:.2f}", (175, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

        extra_count = len(info.get("extra_hands", []))
        hands_status = f"2 Hands (R:Cursor, L:Actions)" if extra_count > 0 else f"Mode: {mode} ({role})"
        cv2.putText(
            frame, hands_status, (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 220, 255), 1, cv2.LINE_AA,
        )

        # ML Gesture Classification Badge
        label = info.get("gesture_label")
        if label:
            _draw_glass_panel(frame, w - 180, h - 45, w - 10, h - 10, alpha=0.65)
            cv2.putText(
                frame,
                f"ML: {label}",
                (w - 170, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 160),
                1,
                cv2.LINE_AA,
            )

    # ── Top Telemetry Panel (FPS & System Status) ────────────────────────────
    if cfg.show_fps:
        _draw_glass_panel(frame, 10, 10, 130, 42, alpha=0.65)
        cv2.putText(
            frame,
            f"FPS {fps:5.1f}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )


def resize_for_detection(frame: np.ndarray, detector_width: int) -> np.ndarray:
    """Downscale frame with aspect-ratio preservation for ultra-low latency inference."""
    h, w = frame.shape[:2]
    if detector_width >= w:
        return frame
    detector_height = max(1, int(round(h * detector_width / w)))
    return cv2.resize(frame, (detector_width, detector_height), interpolation=cv2.INTER_AREA)
