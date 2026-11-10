"""Vectorized biomechanical kinematics, 3D palm scaling, and finger extension analysis."""

import math
from typing import Dict, Tuple

import numpy as np


def extract_pts_px(landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    """Extract all 21 landmarks into a contiguous (21, 2) float64 pixel array."""
    pts = np.empty((21, 2), dtype=np.float64)
    fw = float(frame_w)
    fh = float(frame_h)
    for i, lm in enumerate(landmarks):
        pts[i, 0] = lm.x * fw
        pts[i, 1] = lm.y * fh
    return pts


def determine_middle_finger_point(
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


def compute_3d_palm_scale(pts: np.ndarray) -> float:
    """Compute 3D-invariant multi-point palm scale across landmarks (0, 5, 9, 17).

    Combines wrist->middle_MCP length with index_MCP->pinky_MCP knuckle width
    so forward/backward hand tilt does not distort pinch ratios.
    """
    wx, wy = float(pts[0, 0]), float(pts[0, 1])
    mmx, mmy = float(pts[9, 0]), float(pts[9, 1])
    palm_len = math.hypot(mmx - wx, mmy - wy)
    knuckle_span = math.hypot(pts[5, 0] - pts[17, 0], pts[5, 1] - pts[17, 1])
    return max(palm_len, knuckle_span * 0.90, 1.0)


def get_extended_fingers_from_pts(pts: np.ndarray) -> Dict[str, bool]:
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


def get_extended_fingers(landmarks, frame_w: int, frame_h: int) -> Dict[str, bool]:
    """Rotation-invariant finger extension analysis using joint chain vectors."""
    pts = extract_pts_px(landmarks, frame_w, frame_h)
    return get_extended_fingers_from_pts(pts)


def remap_and_clamp(
    val: float, in_min: float, in_max: float, out_min: float, out_max: float
) -> float:
    """Linearly remap value from [in_min, in_max] to [out_min, out_max] with clamping."""
    if in_max <= in_min:
        return (out_min + out_max) * 0.5
    if val <= in_min:
        return out_min
    if val >= in_max:
        return out_max
    return out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min)
