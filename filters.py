"""High-precision adaptive filters for gesture interaction and cursor stabilization.

Includes:
1. LowPassFilter: Fundamental exponential smoother.
2. OneEuroFilter: 1-D adaptive cutoff low-pass filter (Casiez et al., CHI 2012).
3. Point2DOneEuroFilter: Isotropic 2-D One-Euro filter with unified velocity estimation,
   sub-pixel deadband gating, and duplicate-timestamp protection.
4. CursorBallistics: Human-factors-tuned precision damping & optional edge-easing remapper.
"""

import math
from typing import Optional, Tuple


class LowPassFilter:
    """First-order exponential low-pass filter."""

    def __init__(self, alpha: float = 1.0):
        self._y: Optional[float] = None
        self._alpha: float = alpha

    def apply(self, x: float, alpha: Optional[float] = None) -> float:
        a = self._alpha if alpha is None else alpha
        if self._y is None:
            self._y = x
        else:
            self._y = a * x + (1.0 - a) * self._y
        return self._y

    @property
    def last_value(self) -> Optional[float]:
        return self._y

    def reset(self) -> None:
        self._y = None


class OneEuroFilter:
    """1-D One-Euro filter adapting cutoff frequency to signal velocity.

    Parameters
    ----------
    min_cutoff : baseline cutoff frequency (Hz) when signal is still.
    beta : sensitivity to velocity. Higher values reduce lag during fast sweeps.
    d_cutoff : cutoff frequency for derivative (velocity) estimation.
    """

    def __init__(self, min_cutoff: float = 1.8, beta: float = 0.045, d_cutoff: float = 1.2):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time_s: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, x: float, timestamp_s: float) -> float:
        prev_x = self._x_filter.last_value
        if self._last_time_s is None or prev_x is None:
            self._last_time_s = timestamp_s
            return self._x_filter.apply(x, 1.0)

        dt = timestamp_s - self._last_time_s
        if dt <= 1e-6:
            # Duplicate timestamp: return current filtered state without corrupting derivative
            return prev_x

        dt = min(dt, 0.2)
        self._last_time_s = timestamp_s

        dx = (x - prev_x) / dt
        edx = self._dx_filter.apply(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x_filter.apply(x, self._alpha(cutoff, dt))

    def reset(self) -> None:
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time_s = None


class Point2DOneEuroFilter:
    """Isotropic 2-D One-Euro filter with unified velocity and deadband gating.

    Filtering X and Y with independent scalar cutoffs causes directional distortion
    (elliptical lag) when moving diagonally. Point2DOneEuroFilter computes Euclidean
    speed to drive an identical cutoff for both axes, maintaining 1:1 trajectory sync.
    """

    def __init__(
        self,
        min_cutoff: float = 1.8,
        beta: float = 0.045,
        d_cutoff: float = 1.2,
        deadband_px: float = 0.5,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.deadband_px = deadband_px

        self._filter_x = LowPassFilter()
        self._filter_y = LowPassFilter()
        self._filter_dx = LowPassFilter()
        self._filter_dy = LowPassFilter()
        self._last_time_s: Optional[float] = None
        self._last_output: Optional[Tuple[float, float]] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, x: float, y: float, timestamp_s: float) -> Tuple[float, float]:
        prev_x = self._filter_x.last_value
        prev_y = self._filter_y.last_value

        if self._last_time_s is None or prev_x is None or prev_y is None:
            self._last_time_s = timestamp_s
            fx = self._filter_x.apply(x, 1.0)
            fy = self._filter_y.apply(y, 1.0)
            self._last_output = (fx, fy)
            return (fx, fy)

        dt = timestamp_s - self._last_time_s
        if dt <= 1e-6:
            return self._last_output if self._last_output is not None else (prev_x, prev_y)

        dt = min(dt, 0.2)
        self._last_time_s = timestamp_s

        # Raw derivative based on true input change
        dx = (x - prev_x) / dt
        dy = (y - prev_y) / dt

        alpha_d = self._alpha(self.d_cutoff, dt)
        edx = self._filter_dx.apply(dx, alpha_d)
        edy = self._filter_dy.apply(dy, alpha_d)

        # Euclidean speed drives common cutoff frequency
        speed = math.hypot(edx, edy)
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(cutoff, dt)

        fx = self._filter_x.apply(x, alpha)
        fy = self._filter_y.apply(y, alpha)

        # Micro-tremor deadband gating: only suppress true sub-pixel resting jitter
        if self._last_output is not None and self.deadband_px > 0:
            last_x, last_y = self._last_output
            dist = math.hypot(fx - last_x, fy - last_y)
            if dist < self.deadband_px:
                return (last_x, last_y)

        self._last_output = (fx, fy)
        return (fx, fy)

    def reset(self) -> None:
        self._filter_x.reset()
        self._filter_y.reset()
        self._filter_dx.reset()
        self._filter_dy.reset()
        self._last_time_s = None
        self._last_output = None


class CursorBallistics:
    """Pointer ballistics and boundary remapper."""

    def __init__(
        self,
        precision_gain: float = 0.85,
        ballistic_exponent: float = 1.35,
        speed_threshold_low: float = 40.0,
        speed_threshold_high: float = 500.0,
    ):
        self.precision_gain = precision_gain
        self.ballistic_exponent = ballistic_exponent
        self.speed_threshold_low = speed_threshold_low
        self.speed_threshold_high = speed_threshold_high

    def compute_gain(self, speed_px_s: float) -> float:
        """Compute velocity-dependent dynamic gain factor."""
        if speed_px_s <= self.speed_threshold_low:
            return self.precision_gain
        if speed_px_s >= self.speed_threshold_high:
            norm_speed = 1.0
        else:
            norm_speed = (speed_px_s - self.speed_threshold_low) / (
                self.speed_threshold_high - self.speed_threshold_low
            )
        gain = 1.0 + (norm_speed ** self.ballistic_exponent) * 1.5
        return gain

    @staticmethod
    def ease_edge(val: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
        """Remap with gentle boundary softening while preserving linear feel."""
        if in_max <= in_min:
            return (out_min + out_max) * 0.5
        t = (val - in_min) / (in_max - in_min)
        t = max(0.0, min(1.0, t))
        t_curved = 0.75 * t + 0.25 * (t * t * (3.0 - 2.0 * t))
        return out_min + t_curved * (out_max - out_min)