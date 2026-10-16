"""Centralised cross-platform configuration for Vision-Gesture."""

import os
from dataclasses import dataclass, field


def _resolve_default_model_path() -> str:
    """Intelligently locate hand_landmarker.task across Windows, macOS, and Linux."""
    # 1. Environment variable override
    env_path = os.environ.get("VISION_GESTURE_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. Candidate locations relative to project root or user home
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "hand_landmarker.task"),
        os.path.join(script_dir, "models", "hand_landmarker.task"),
        os.path.join(os.getcwd(), "hand_landmarker.task"),
        os.path.join(os.getcwd(), "models", "hand_landmarker.task"),
        os.path.expanduser("~/Desktop/Python/Vision-Gesture/hand_landmarker.task"),
        os.path.expanduser("~/Documents/Python/Vision-Gesture/hand_landmarker.task"),
        "C:/Users/bathi/OneDrive/Documents/Desktop/Python/Vision-Gesture/hand_landmarker.task",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Default to local path in script directory
    return os.path.join(script_dir, "hand_landmarker.task")


@dataclass(frozen=True)
class Config:
    cam_index: int = 0
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30

    # Auto-resolved model path (works on Windows, macOS, Linux)
    model_path: str = field(default_factory=_resolve_default_model_path)

    # Inference resolution: 512x384 provides an ideal speed/precision balance.
    detector_width: int = 512

    # Workspace margin (px) inside camera frame mapped to full screen.
    frame_margin: int = 80

    # ── Cursor Smoothing & 1:1 Middle-Finger Sync (One-Euro Filter) ──────────
    # Tuned for tight, low-latency 1:1 synchronization with the middle finger:
    #   cursor_min_cutoff: 1.8 Hz eliminates stationary micro-tremor without viscous lag.
    #   cursor_beta:       0.045 adapts immediately to finger motion for zero-lag tracking.
    #   cursor_d_cutoff:   1.2 Hz smooths velocity estimation.
    #   deadband_px:       0.5 px suppresses only sub-pixel sensor noise.
    cursor_min_cutoff: float = 1.8
    cursor_beta: float = 0.045
    cursor_d_cutoff: float = 1.2
    deadband_px: float = 0.5
    min_move_px: float = 0.1

    # Multi-point kinematic middle-finger position determination (uses landmarks 9, 10, 11, 12)
    middle_finger_kinematic_blend: bool = True

    # ── 1:1 Workspace Mapping vs Ballistics ──────────────────────────────────
    # Default False for both so the OS cursor stays 100% locked in 1:1 spatial
    # synchronization with the middle finger position inside the workspace box.
    use_ballistics: bool = False
    ballistics_precision_gain: float = 0.85
    ballistics_exponent: float = 1.35
    edge_easing: bool = False

    # ── Click Anchor Stabilization (Anti Click-Slip) ────────────────────────
    # Locks cursor coordinate right as thumb+index make contact so clicking
    # never shifts the pointer, without freezing normal hand motion.
    anchor_lock_on_pinch: bool = True
    pinch_lock_ratio: float = 0.105

    # ── Left-Click & Drag-and-Drop (Thumb + Index Pinch) ────────────────────
    pinch_on_ratio: float = 0.15
    pinch_off_ratio: float = 0.18
    pinch_confirm_ms: int = 0          # instant activation on contact
    click_cooldown_ms: int = 250       # debounce between discrete clicks
    drag_hold_ms: int = 220            # pinch held >220ms enters DRAG mode (mouseDown)

    # ── Double-Click Detection ──────────────────────────────────────────────
    # Second click within this window triggers double-click.
    double_click_window_ms: int = 400
    double_click_min_interval_ms: int = 60

    # ── Right-Click (Thumb + Middle Pinch) ──────────────────────────────────
    right_pinch_on_ratio: float = 0.25
    right_pinch_off_ratio: float = 0.30
    right_pinch_confirm_ms: int = 50
    right_click_cooldown_ms: int = 500

    # ── Multi-Hand Architecture ─────────────────────────────────────────────
    # Right hand: Middle finger exclusively controls the cursor (+ clicks/drags).
    # Left hand: Strictly actions-only (scroll, clicks) — NEVER moves cursor
    #            as long as Right hand is present or within right_hand_gone_timeout_ms.
    multi_hand: bool = True
    num_hands: int = 2
    spatial_hand_sorting: bool = True
    right_hand_gone_timeout_ms: int = 450

    # ── Midas-Touch Clutch (Standby Gesture) ─────────────────────────────────
    # Closed fist pauses cursor tracking so users can rest or gesture freely.
    clutch_enabled: bool = True
    clutch_confirm_ms: int = 120

    # Accept results up to 250 ms old — prevents tracking dropouts on frame hiccup.
    max_result_age_ms: int = 250

    # MediaPipe confidence settings.
    min_hand_detection_confidence: float = 0.55
    min_hand_presence_confidence: float = 0.55
    min_tracking_confidence: float = 0.55

    display: bool = True
    show_fps: bool = True

    # Interval for ML gesture classifier inference (frames).
    gesture_infer_every: int = 3

    # ── Proportional Smooth Scrolling ───────────────────────────────────────
    # Activated when index + middle extended, ring + pinky folded.
    mode_confirm_ms: int = 120
    mode_exit_confirm_ms: int = 0
    scroll_speed_scale: float = 0.35    # continuous speed multiplier
    scroll_max_speed: int = 25          # maximum scroll velocity per tick
    scroll_deadzone_ratio: float = 0.025 # deadzone around neutral anchor
    scroll_tick_interval_ms: int = 25   # rapid tick rate for continuous fluid scroll
    scroll_smooth_min_cutoff: float = 0.6
    scroll_smooth_beta: float = 0.01