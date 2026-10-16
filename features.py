"""Shared landmark normalisation (used by train_model AND live inference)."""

import math

import numpy as np


def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """Normalise a (21, 3) landmark array to a pose-invariant (63,) feature vector.

    Three-step convention (mirrors HandGestureController.update hand_scale logic):

    1. **Translate** — subtract wrist (landmark 0) so it sits at the origin.
    2. **Scale**     — divide every coordinate by the wrist-to-middle-MCP
                       distance (landmarks 0→9, xy plane), i.e. the same
                       ``hand_scale`` used for the pinch ratio.  Clamped to
                       1e-6 to avoid division-by-zero on degenerate frames.
    3. **Rotate**    — apply a 2-D in-plane rotation (xy only) so that the
                       wrist→middle-MCP direction always points to (0, −1),
                       i.e. "straight up" in canonical image coordinates.
                       This removes orientation differences between datasets
                       that were captured with different hand tilts or
                       y-axis conventions (e.g. image-y-down vs y-up).
                       The z axis is rotated by the same angle to keep the
                       3-D shape consistent.

    Parameters
    ----------
    lm : np.ndarray, shape (21, 3)
        Raw MediaPipe landmark coordinates (x, y, z) in any consistent unit
        (normalised image coords, pixel coords, etc.).

    Returns
    -------
    np.ndarray, shape (63,)
        Flat, wrist-centred, scale-invariant, rotation-normalised feature
        vector ready for the classifier.  All 21 landmarks × 3 axes,
        row-major.  After normalisation lm9 ≈ (0, −1, ·) regardless of
        which direction the hand was pointing when captured.
    """
    lm = np.asarray(lm, dtype=np.float64)   # (21, 3)
    relative = lm - lm[0]                   # step 1: translate wrist to origin

    # Step 2: scale by wrist→middle-MCP distance (xy only)
    mx, my = relative[9, 0], relative[9, 1]
    hand_scale = max(math.hypot(mx, my), 1e-6)
    relative /= hand_scale
    mx, my = relative[9, 0], relative[9, 1]  # re-read after scaling (≈unit vec)

    # Step 3: rotate xy plane so wrist→middle-MCP aligns with (0, −1)
    #   current direction: (mx, my)  →  target: (0, −1)
    #   rotation angle θ satisfies: R(θ)·(mx, my) = (0, −1)
    #   i.e. θ = atan2(mx, −my)  (positive = counter-clockwise)
    cos_a = -my       # cos of angle between (mx,my) and (0,-1)
    sin_a = -mx       # sin of angle  (negative: rotating TO (0,-1), not away)
    norm  = math.hypot(cos_a, sin_a)
    if norm > 1e-9:
        cos_a /= norm
        sin_a /= norm
    # Apply 2-D rotation matrix [[cos -sin],[sin cos]] to x,y columns
    x_rot = cos_a * relative[:, 0] - sin_a * relative[:, 1]
    y_rot = sin_a * relative[:, 0] + cos_a * relative[:, 1]
    relative[:, 0] = x_rot
    relative[:, 1] = y_rot
    # z is also rotated by the same angle so depth profile stays consistent
    # (treated as a third orthogonal component in the same plane rotation)

    return relative.ravel()  # (63,)  — lm9 ≈ (0, −1, ·)
