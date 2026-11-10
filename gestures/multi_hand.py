"""Spatial multi-hand centroid computation and role assignment."""

import math
from typing import List, Optional, Tuple


def compute_palm_centroid(hand_lms) -> Tuple[float, float]:
    """Compute normalized (x, y) palm centroid using rigid palm landmarks (0, 5, 9, 13, 17)."""
    indices = (0, 5, 9, 13, 17)
    cx = sum(hand_lms[i].x for i in indices) * 0.2
    cy = sum(hand_lms[i].y for i in indices) * 0.2
    return (cx, cy)


def assign_two_hand_roles(
    all_landmarks: List,
    handedness: Optional[List],
    spatial_hand_sorting: bool = True,
    last_right_palm: Optional[Tuple[float, float]] = None,
    last_left_palm: Optional[Tuple[float, float]] = None,
) -> List[Tuple[object, str, Tuple[float, float]]]:
    """Assign 'Right' (cursor) and 'Left' (actions-only) roles to 2 detected hands."""
    lms_0, lms_1 = all_landmarks[0], all_landmarks[1]
    p0 = compute_palm_centroid(lms_0)
    p1 = compute_palm_centroid(lms_1)

    # 1. If both hands have temporal history and are very close horizontally,
    # use minimum-distance centroid matching to prevent cross-over swaps.
    if (
        last_right_palm is not None
        and last_left_palm is not None
        and abs(p0[0] - p1[0]) < 0.08
    ):
        d0_r = math.hypot(p0[0] - last_right_palm[0], p0[1] - last_right_palm[1])
        d0_l = math.hypot(p0[0] - last_left_palm[0], p0[1] - last_left_palm[1])
        d1_r = math.hypot(p1[0] - last_right_palm[0], p1[1] - last_right_palm[1])
        d1_l = math.hypot(p1[0] - last_left_palm[0], p1[1] - last_left_palm[1])
        if d0_r + d1_l <= d0_l + d1_r:
            return [(lms_0, "Right", p0), (lms_1, "Left", p1)]
        return [(lms_1, "Right", p1), (lms_0, "Left", p0)]

    # 2. Spatial sorting in the mirrored webcam frame:
    # In a horizontally flipped (mirror) frame, the user's physical Right hand sits
    # on the right side of the image (larger X), and the Left hand sits on the left
    # side (smaller X).
    if spatial_hand_sorting and abs(p0[0] - p1[0]) >= 0.04:
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
