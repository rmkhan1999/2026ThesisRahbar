import math


def thresholds_from_mean_gap(mean: float, gap: float) -> tuple[float, float]:
    if gap <= 0:
        raise ValueError("threshold gap must be positive")

    half_gap = gap / 2
    return mean + half_gap, mean - half_gap


def sleep_duration(chi_sleep: float, upper: float, lower: float) -> float:
    if chi_sleep <= 0:
        raise ValueError("sleep time constant must be positive")
    if not 0 < lower < upper:
        raise ValueError("thresholds must satisfy 0 < lower < upper")

    return chi_sleep * math.log(upper / lower)


def wake_duration(
    chi_wake: float,
    mu: float,
    upper: float,
    lower: float,
) -> float:
    if chi_wake <= 0:
        raise ValueError("wake time constant must be positive")
    if not lower < upper < mu:
        raise ValueError("thresholds must satisfy lower < upper < mu")

    return chi_wake * math.log((mu - lower) / (mu - upper))
