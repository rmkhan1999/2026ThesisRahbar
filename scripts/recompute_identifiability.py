from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit, polygamma

from twopm.config import load_config
from twopm.posterior_summaries import solve_entrained_transition


OUTPUT = Path("docs/identifiability_metrics.json")


def _root(
    *,
    phase: float,
    chi_sleep: float,
    chi_wake: float,
    amplitude: float,
    gap: float,
    mean: float,
    mu: float,
    period: float,
) -> np.ndarray:
    angle = 2 * np.pi * phase / period
    result = solve_entrained_transition(
        c1=amplitude * np.cos(angle),
        c2=amplitude * np.sin(angle),
        chi_sleep=chi_sleep,
        chi_wake=chi_wake,
        threshold_gap=gap,
        threshold_mean=mean,
        mu=mu,
        period=period,
    )
    if not result.converged:
        raise RuntimeError(f"physical root failed: {result.reason}")
    return np.asarray((result.onset, result.offset))


def _jacobian(function, point: np.ndarray, step: float = 1e-4) -> np.ndarray:
    columns = []
    for index in range(point.size):
        offset = np.zeros_like(point)
        offset[index] = step
        columns.append((function(point + offset) - function(point - offset)) / (2 * step))
    return np.column_stack(columns)


def _geometry(jacobian: np.ndarray, covariance: np.ndarray) -> dict:
    cholesky = np.linalg.cholesky(covariance)
    whitened = jacobian @ cholesky
    _, singular_values, right = np.linalg.svd(whitened, full_matrices=True)
    tolerance = np.max(whitened.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    row_projector = right[:rank].T @ right[:rank]
    coordinate_projector = (
        covariance
        @ jacobian.T
        @ np.linalg.inv(jacobian @ covariance @ jacobian.T)
        @ jacobian
    )
    return {
        "jacobian": jacobian.tolist(),
        "whitened_singular_values": singular_values.tolist(),
        "condition_number": float(singular_values[0] / singular_values[rank - 1]),
        "rank": rank,
        "kernel_dimension": int(jacobian.shape[1] - rank),
        "whitened_leverage": np.diag(row_projector).tolist(),
        "physical_coordinate_leverage": np.diag(coordinate_projector).tolist(),
    }


def _beta_moments(alpha: float, beta: float) -> tuple[float, float, float]:
    total = alpha + beta
    mean = alpha / total
    second = alpha * (alpha + 1) / (total * (total + 1))
    variance = second - mean**2
    return mean, variance, second


def main() -> None:
    config = load_config("config/model.yaml")
    model = config.section("model")
    fixed = config.section("fixed")
    priors = config.section("priors")
    design = config.section("designs")["entrained"]
    constrained = design["constrained_prior"]
    recovery = config.section("recovery")
    mean = float(fixed["threshold_mean"])
    mu = float(model["mu"])
    period = float(model["circadian_period"])
    phase = float(recovery["true_phase"])
    chi_sleep = float(model["chi_sleep"])
    chi_wake = float(model["chi_wake"])
    amplitude = float(model["circadian_amplitude"])
    gap = float(model["upper_base"]) - float(model["lower_base"])
    excursion = amplitude + gap / 2
    amplitude_fraction = amplitude / excursion

    alpha_s = float(constrained["excursion_concentration1"])
    beta_s = float(constrained["excursion_concentration0"])
    alpha_w = float(constrained["amplitude_concentration1"])
    beta_w = float(constrained["amplitude_concentration0"])
    s_fraction_mean, s_fraction_var, s_fraction_second = _beta_moments(
        alpha_s, beta_s
    )
    w_mean, w_var, w_second = _beta_moments(alpha_w, beta_w)

    angle = 2 * np.pi * phase / period
    sampled_point = np.asarray(
        (
            np.cos(angle),
            np.sin(angle),
            logit(excursion / mean),
            logit(amplitude_fraction),
            np.log(chi_sleep),
            np.log(chi_wake),
            logit(0.01 / float(priors["misclassification"]["maximum"])),
        )
    )

    def sampled_map(point):
        z1, z2, logit_s, logit_w, log_chi_s, log_chi_w, _ = point
        direction = np.asarray((z1, z2)) / np.hypot(z1, z2)
        s = mean * expit(logit_s)
        w = expit(logit_w)
        a = w * s
        sampled_phase = (
            period * np.arctan2(direction[1], direction[0]) / (2 * np.pi)
        ) % period
        return _root(
            phase=sampled_phase,
            chi_sleep=np.exp(log_chi_s),
            chi_wake=np.exp(log_chi_w),
            amplitude=a,
            gap=2 * (1 - w) * s,
            mean=mean,
            mu=mu,
            period=period,
        )

    sampled_scales = np.asarray(
        (
            1.0,
            1.0,
            np.sqrt(polygamma(1, alpha_s) + polygamma(1, beta_s)),
            np.sqrt(polygamma(1, alpha_w) + polygamma(1, beta_w)),
            float(priors["chi_sleep"]["log_sd"]),
            float(priors["chi_wake"]["log_sd"]),
            np.sqrt(
                polygamma(
                    1,
                    float(priors["misclassification"]["concentration1"]),
                )
                + polygamma(
                    1,
                    float(priors["misclassification"]["concentration0"]),
                )
            ),
        )
    )
    sampled_geometry = _geometry(
        _jacobian(sampled_map, sampled_point),
        np.diag(sampled_scales**2),
    )

    excursion_point = np.asarray(
        (phase, chi_sleep, chi_wake, excursion, amplitude_fraction)
    )

    def excursion_map(point):
        p, chi_s, chi_w, s, w = point
        return _root(
            phase=p,
            chi_sleep=chi_s,
            chi_wake=chi_w,
            amplitude=w * s,
            gap=2 * (1 - w) * s,
            mean=mean,
            mu=mu,
            period=period,
        )

    chi_sleep_mean = np.exp(
        float(priors["chi_sleep"]["log_mean"])
        + float(priors["chi_sleep"]["log_sd"]) ** 2 / 2
    )
    chi_wake_mean = np.exp(
        float(priors["chi_wake"]["log_mean"])
        + float(priors["chi_wake"]["log_sd"]) ** 2 / 2
    )
    chi_sleep_var = (
        np.exp(float(priors["chi_sleep"]["log_sd"]) ** 2) - 1
    ) * chi_sleep_mean**2
    chi_wake_var = (
        np.exp(float(priors["chi_wake"]["log_sd"]) ** 2) - 1
    ) * chi_wake_mean**2
    excursion_covariance = np.diag(
        (
            period**2 / 12,
            chi_sleep_var,
            chi_wake_var,
            mean**2 * s_fraction_var,
            w_var,
        )
    )
    excursion_geometry = _geometry(
        _jacobian(excursion_map, excursion_point),
        excursion_covariance,
    )

    s_mean = mean * s_fraction_mean
    s_second = mean**2 * s_fraction_second
    a_mean = s_mean * w_mean
    gap_mean = 2 * s_mean * (1 - w_mean)
    a_var = s_second * w_second - a_mean**2
    one_minus_w_second = 1 - 2 * w_mean + w_second
    gap_var = 4 * (s_second * one_minus_w_second - (gap_mean / 2) ** 2)
    aw_moment = alpha_w * beta_w / (
        (alpha_w + beta_w) * (alpha_w + beta_w + 1)
    )
    covariance_a_gap = 2 * s_second * aw_moment - a_mean * gap_mean
    physical_covariance = np.diag(
        (period**2 / 12, chi_sleep_var, chi_wake_var, 1.0, 1.0)
    )
    physical_covariance[3:, 3:] = np.asarray(
        ((a_var, covariance_a_gap), (covariance_a_gap, gap_var))
    )
    physical_point = np.asarray((phase, chi_sleep, chi_wake, amplitude, gap))

    def physical_map(point):
        p, chi_s, chi_w, a, threshold_gap = point
        return _root(
            phase=p,
            chi_sleep=chi_s,
            chi_wake=chi_w,
            amplitude=a,
            gap=threshold_gap,
            mean=mean,
            mu=mu,
            period=period,
        )

    physical_geometry = _geometry(
        _jacobian(physical_map, physical_point),
        physical_covariance,
    )
    output = {
        "config": str(config.source),
        "canonical_transition_times": sampled_map(sampled_point).tolist(),
        "sampled_unconstrained": {
            "coordinates": [
                "phase_z1",
                "phase_z2",
                "logit_excursion_fraction",
                "logit_amplitude_fraction",
                "log_chi_sleep",
                "log_chi_wake",
                "logit_scaled_misclassification",
            ],
            "prior_scales": sampled_scales.tolist(),
            **sampled_geometry,
        },
        "physical_excursion_split": {
            "coordinates": [
                "phase",
                "chi_sleep",
                "chi_wake",
                "total_excursion",
                "amplitude_fraction",
            ],
            **excursion_geometry,
        },
        "induced_physical_amplitude_gap": {
            "coordinates": [
                "phase",
                "chi_sleep",
                "chi_wake",
                "amplitude",
                "threshold_gap",
            ],
            "amplitude_gap_correlation": float(
                covariance_a_gap / np.sqrt(a_var * gap_var)
            ),
            **physical_geometry,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
