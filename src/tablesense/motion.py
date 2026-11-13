"""Classic CV baseline: motion per zone and debounced occupancy.

No deep learning here. Background subtraction learns what the empty scene looks like and flags
pixels that differ. It's cheap, fast and explainable, and it's the baseline the YOLO pipeline
needs to beat.
"""

from __future__ import annotations

import cv2  # noqa: F401
import numpy as np


def clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Turn a raw background-subtractor mask into a clean binary mask.

    Input values: 0 (background), 127 (shadow, from MOG2 with detectShadows=True), 255 (foreground).
    Steps:
      1. Keep only true foreground (255): shadows become 0.
      2. Morphological **opening** (removes isolated noise pixels).
      3. Morphological **closing** (fills small holes inside blobs).
    Returns a uint8 mask containing only 0 and 255, same shape as the input.
    """
    raise NotImplementedError


def zone_activity(mask: np.ndarray, zone_mask: np.ndarray) -> float:
    """Fraction (0.0 to 1.0) of the zone's pixels that are foreground.

    `mask` and `zone_mask` have the same shape; a pixel is "on" when its value is > 0.
    Returns 0.0 if the zone is empty.
    """
    raise NotImplementedError


class MotionDetector:
    """Background subtraction (MOG2) followed by `clean_mask`.

    - `apply(frame)` takes a BGR frame and returns a clean binary mask (0/255, shape HxW).
    - Parameters go straight to cv2.createBackgroundSubtractorMOG2 (history, varThreshold,
      detectShadows) and to clean_mask (kernel_size).
    """

    def __init__(
        self,
        history: int = 100,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        kernel_size: int = 3,
    ) -> None:
        raise NotImplementedError

    def apply(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DebouncedSwitch:
    """An on/off state that ignores brief noise. Used for "is this table occupied?".

    - While OFF: turns ON once `value >= on_threshold` has held **continuously** for at least
      `min_on_seconds`. Any value below `on_threshold` resets that timer.
    - While ON: turns OFF once `value < off_threshold` has held continuously for at least
      `min_off_seconds`. Any value >= `off_threshold` resets that timer.
    - `update(value, t)` takes the new measurement and its timestamp (seconds), returns the state.
    - `state` property returns the current state.
    - Raise ValueError if off_threshold > on_threshold.

    Example (on=0.3, off=0.1, min_on=2 s):
        t=0 v=0.5 -> False   (timer starts)
        t=1 v=0.5 -> False
        t=2 v=0.5 -> True    (held for 2 s)
    """

    def __init__(
        self,
        on_threshold: float,
        off_threshold: float,
        min_on_seconds: float,
        min_off_seconds: float,
        initial: bool = False,
    ) -> None:
        raise NotImplementedError

    def update(self, value: float, t: float) -> bool:
        raise NotImplementedError

    @property
    def state(self) -> bool:
        raise NotImplementedError
