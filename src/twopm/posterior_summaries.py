from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


@dataclass(frozen=True)
class EntrainedTransition:

    onset: float
    offset: float
    converged: bool
    reason: str | None


@dataclass(frozen=True)
class PosteriorTransitions:

    onset: NDArray[np.float64]
    offset: NDArray[np.float64]
    failed: int
    total: int
    failure_reasons: dict[str, int]


def entrained_transition_equations(
    onset_duration: NDArray[np.float64],
    *,
    c1: float,
    c2: float,
    chi_sleep: float,
    chi_wake: float,
    threshold_gap: float,
    threshold_mean: float,
    mu: float,
    period: float,
) -> NDArray[np.float64]:
    onset, sleep_duration = onset_duration
    offset = onset + sleep_duration
    omega = 2 * np.pi / period

    def displacement(time: float) -> float:
        return c1 * np.cos(omega * time) + c2 * np.sin(omega * time)

    upper_onset = threshold_mean + threshold_gap / 2 + displacement(onset)
    lower_offset = (
        threshold_mean - threshold_gap / 2 + displacement(offset)
    )
    sleep_residual = lower_offset - upper_onset * np.exp(
        -sleep_duration / chi_sleep
    )
    wake_duration = period - sleep_duration
    next_onset_pressure = mu - (mu - lower_offset) * np.exp(
        -wake_duration / chi_wake
    )
    wake_residual = upper_onset - next_onset_pressure
    return np.asarray((sleep_residual, wake_residual), dtype=np.float64)


def solve_entrained_transition(
    *,
    c1: float,
    c2: float,
    chi_sleep: float,
    chi_wake: float,
    threshold_gap: float,
    threshold_mean: float = 0.42,
    mu: float = 1.0,
    period: float = 24.0,
    residual_tolerance: float = 1e-8,
) -> EntrainedTransition:
    values = np.asarray(
        (c1, c2, chi_sleep, chi_wake, threshold_gap),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        return EntrainedTransition(np.nan, np.nan, False, "non_finite_parameter")
    if chi_sleep <= 0 or chi_wake <= 0 or threshold_gap <= 0:
        return EntrainedTransition(np.nan, np.nan, False, "non_physical_parameter")

    phase = float(
        (period * np.arctan2(c2, c1) / (2 * np.pi)) % period
    )
    lower_base = threshold_mean - threshold_gap / 2
    upper_base = threshold_mean + threshold_gap / 2

    initial_onset = (phase + 7.4580834) % period
    initial_duration = 23.3977647 - 15.4580834
    keywords = {
        "c1": c1,
        "c2": c2,
        "chi_sleep": chi_sleep,
        "chi_wake": chi_wake,
        "threshold_gap": threshold_gap,
        "threshold_mean": threshold_mean,
        "mu": mu,
        "period": period,
    }
    solution = least_squares(
        lambda value: entrained_transition_equations(value, **keywords),
        x0=np.asarray((initial_onset, initial_duration)),
        bounds=(
            np.asarray((0.0, np.finfo(float).eps)),
            np.asarray((period, period - np.finfo(float).eps)),
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=500,
    )
    residual = entrained_transition_equations(solution.x, **keywords)
    if not solution.success or np.max(np.abs(residual)) > residual_tolerance:
        return EntrainedTransition(np.nan, np.nan, False, "root_non_convergence")

    onset, sleep_duration = map(float, solution.x)
    offset_unwrapped = onset + sleep_duration
    omega = 2 * np.pi / period

    def displacement(time: float) -> float:
        return c1 * np.cos(omega * time) + c2 * np.sin(omega * time)

    def displacement_derivative(time: float) -> float:
        return omega * (
            -c1 * np.sin(omega * time) + c2 * np.cos(omega * time)
        )

    upper_onset = upper_base + displacement(onset)
    lower_offset = lower_base + displacement(offset_unwrapped)
    if not (0 < lower_offset < mu and 0 < upper_onset < mu):
        return EntrainedTransition(
            np.nan,
            np.nan,
            False,
            "crossing_threshold_out_of_range",
        )
    onset_crossing = (
        (mu - upper_onset) / chi_wake - displacement_derivative(onset)
    )
    offset_crossing = (
        -lower_offset / chi_sleep - displacement_derivative(offset_unwrapped)
    )
    if onset_crossing <= 0 or offset_crossing >= 0:
        return EntrainedTransition(
            np.nan,
            np.nan,
            False,
            "wrong_crossing_direction",
        )
    return EntrainedTransition(
        onset=onset % period,
        offset=offset_unwrapped % period,
        converged=True,
        reason=None,
    )


def posterior_transition_times(
    posterior: Mapping[str, NDArray[np.float64]],
    *,
    threshold_mean: float = 0.42,
    mu: float = 1.0,
    period: float = 24.0,
) -> PosteriorTransitions:
    required = ("c1", "c2", "chi_sleep", "chi_wake", "threshold_gap")
    shape = np.asarray(posterior["c1"]).shape
    if any(np.asarray(posterior[name]).shape != shape for name in required):
        raise ValueError("all posterior parameter arrays must share one shape")
    onset = np.full(shape, np.nan, dtype=np.float64)
    offset = np.full(shape, np.nan, dtype=np.float64)
    reasons: dict[str, int] = {}
    for index in np.ndindex(shape):
        result = solve_entrained_transition(
            c1=float(posterior["c1"][index]),
            c2=float(posterior["c2"][index]),
            chi_sleep=float(posterior["chi_sleep"][index]),
            chi_wake=float(posterior["chi_wake"][index]),
            threshold_gap=float(posterior["threshold_gap"][index]),
            threshold_mean=threshold_mean,
            mu=mu,
            period=period,
        )
        if result.converged:
            onset[index] = result.onset
            offset[index] = result.offset
        else:
            reason = result.reason or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
    failed = int(np.sum(~np.isfinite(onset) | ~np.isfinite(offset)))
    return PosteriorTransitions(
        onset=onset,
        offset=offset,
        failed=failed,
        total=int(onset.size),
        failure_reasons=reasons,
    )


def variance_contraction(
    prior: NDArray[np.float64],
    posterior: NDArray[np.float64],
) -> float:
    prior_values = np.asarray(prior, dtype=np.float64)
    posterior_values = np.asarray(posterior, dtype=np.float64)
    prior_values = prior_values[np.isfinite(prior_values)]
    posterior_values = posterior_values[np.isfinite(posterior_values)]
    if prior_values.size < 2 or posterior_values.size < 2:
        return np.nan
    prior_variance = float(np.var(prior_values, ddof=1))
    if prior_variance <= 0:
        return np.nan
    return float(
        1 - np.var(posterior_values, ddof=1) / prior_variance
    )
