"""Spatial lane validation and eight-observation temporal fallback.

Input mask IDs are those of iCPS Duckietown Lane Segmentation:
  0 background, 1 red, 2 white, 3 yellow.
The two ROI boundaries are configurable deployment choices. The paper specifies
near/far regions and N=8 memory, but does not publish exact pixel boundaries.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LaneEstimate:
    center_px: Optional[float]
    error_px: Optional[float]
    error_cm: Optional[float]
    lane_width_px: float
    source: str


class SpatioTemporalLaneEstimator:
    def __init__(self, *, history_size=8, near_roi=(0.70, 0.95),
                 far_roi=(0.45, 0.65), min_pixels=30,
                 initial_lane_width_px=400.0, lane_width_cm=20.32):
        self.history = deque(maxlen=history_size)
        self.near_roi = near_roi
        self.far_roi = far_roi
        self.min_pixels = min_pixels
        self.width_px = float(initial_lane_width_px)
        self.lane_width_cm = float(lane_width_cm)

    def _roi_center(self, mask, roi):
        height, width = mask.shape
        lo, hi = (max(0, min(height, round(float(f) * height))) for f in roi)
        if hi <= lo:
            return None
        white_y, white_x = np.nonzero(mask[lo:hi] == 2)
        yellow_y, yellow_x = np.nonzero(mask[lo:hi] == 3)
        has_white = white_x.size >= self.min_pixels
        has_yellow = yellow_x.size >= self.min_pixels
        if not has_white and not has_yellow:
            return None
        if has_white and has_yellow:
            wx = float(np.median(white_x))
            yx = float(np.median(yellow_x))
            measured_width = abs(wx - yx)
            if measured_width < width * 0.12:
                return None  # reject implausibly close / conflicting markings
            self.width_px = 0.8 * self.width_px + 0.2 * measured_width
            return float((wx + yx) / 2)
        # A single detected boundary can supply a temporary geometric center.
        half_width = self.width_px / 2.0
        center = float(np.median(white_x) - half_width) if has_white else float(np.median(yellow_x) + half_width)
        return float(np.clip(center, 0, width - 1))

    def estimate(self, segmentation_ids: np.ndarray) -> LaneEstimate:
        mask = np.asarray(segmentation_ids)
        if mask.ndim != 2:
            raise ValueError("Expected [height, width] integer segmentation class IDs")
        _, width = mask.shape
        center = self._roi_center(mask, self.near_roi)
        source = "near"
        if center is None:
            center = self._roi_center(mask, self.far_roi)
            source = "far"
        if center is not None:
            self.history.append(center)
        elif self.history:
            center = float(np.mean(self.history))
            source = "memory"
        else:
            return LaneEstimate(None, None, None, self.width_px, "unavailable")

        error_px = float(center - width / 2.0)
        error_cm = error_px * (self.lane_width_cm / max(1.0, self.width_px))
        return LaneEstimate(center, error_px, error_cm, self.width_px, source)
