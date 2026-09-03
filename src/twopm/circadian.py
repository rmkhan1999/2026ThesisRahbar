import numpy as np
from numpy.typing import ArrayLike, NDArray


def circadian_thresholds(
    time: ArrayLike,
    *,
    upper_base: float,
    lower_base: float,
    amplitude: float,
    phase: float,
    period: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if amplitude < 0:
        raise ValueError("circadian amplitude must be nonnegative")
    if period <= 0:
        raise ValueError("circadian period must be positive")
    if lower_base >= upper_base:
        raise ValueError("lower threshold must be below upper threshold")

    time_array = np.asarray(time, dtype=np.float64)
    displacement = amplitude * np.cos(
        2 * np.pi * (time_array - phase) / period
    )
    return upper_base + displacement, lower_base + displacement
