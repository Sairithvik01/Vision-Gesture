"""Modular gesture kinematics, finite state machine, and multi-hand arbitration package."""

from gestures.kinematics import (
    compute_3d_palm_scale,
    determine_middle_finger_point,
    extract_pts_px,
    get_extended_fingers,
    get_extended_fingers_from_pts,
    remap_and_clamp,
)
from gestures.multi_hand import assign_two_hand_roles, compute_palm_centroid
from gestures.state_machine import ControlMode, PinchState, _HandTracker

__all__ = [
    "ControlMode",
    "PinchState",
    "_HandTracker",
    "assign_two_hand_roles",
    "compute_3d_palm_scale",
    "compute_palm_centroid",
    "determine_middle_finger_point",
    "extract_pts_px",
    "get_extended_fingers",
    "get_extended_fingers_from_pts",
    "remap_and_clamp",
]
