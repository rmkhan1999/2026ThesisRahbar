import argparse
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.likelihood import (
    likelihood_parameter_names,
    log_likelihood,
)
from twopm.soft_gate import (
    circadian_amplitude_phase,
    circadian_coefficients,
)


def circular_phase_error(estimate: float, truth: float) -> float:
    return abs((estimate - truth + 12.0) % 24.0 - 12.0)


def reporting_to_vector(
    reporting: np.ndarray,
    names: tuple[str, ...],
    period: float,
) -> np.ndarray:
    phase, chi_sleep, chi_wake, amplitude, gap, error = reporting
    c1, c2 = circadian_coefficients(amplitude, phase, period)
    values = {
        "c1": float(c1),
        "c2": float(c2),
        "chi_sleep": chi_sleep,
        "chi_wake": chi_wake,
        "threshold_gap": gap,
        "misclassification": error,
    }
    return np.asarray(tuple(values[name] for name in names), dtype=float)


def vector_to_reporting(
    parameters: np.ndarray,
    names: tuple[str, ...],
    period: float,
) -> np.ndarray:
    values = dict(zip(names, parameters))
    amplitude, phase = circadian_amplitude_phase(
        values["c1"],
        values["c2"],
        period,
    )
    return np.asarray(
        (
            float(phase),
            values["chi_sleep"],
            values["chi_wake"],
            float(amplitude),
            values["threshold_gap"],
            values["misclassification"],
        )
    )


def valid_parameters(
    parameters: np.ndarray,
    names: tuple[str, ...],
    threshold_mean: float,
) -> bool:
    values = dict(zip(names, parameters))
    chi_sleep = values["chi_sleep"]
    chi_wake = values["chi_wake"]
    gap = values["threshold_gap"]
    error = values["misclassification"]
    lower = threshold_mean - gap / 2
    upper = threshold_mean + gap / 2
    return (
        chi_sleep > 0
        and chi_wake > 0
        and gap > 0
        and 0 < lower < upper < 1
        and 0 < error < 0.5
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k",
        type=float,
        help="override the configured gate sharpness for this recovery run",
    )
    arguments = parser.parse_args()
    config = load_config("config/model.yaml")
    if arguments.k is not None:
        data = deepcopy(config.data)
        data["soft_gate"]["k"] = arguments.k
        config = ProjectConfig(data=data, source=config.source)
    model = config.section("model")
    fixed = config.section("fixed")
    soft_gate = config.section("soft_gate")
    recovery = config.section("recovery")
    names = likelihood_parameter_names(config)
    period = float(model["circadian_period"])
    threshold_mean = float(fixed["threshold_mean"])

    generating_parameters = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
        phase=float(recovery["true_phase"]),
    )
    truth_values = {
        "c1": generating_parameters.c1,
        "c2": generating_parameters.c2,
        "chi_sleep": generating_parameters.chi_sleep,
        "chi_wake": generating_parameters.chi_wake,
        "threshold_gap": (
            generating_parameters.upper - generating_parameters.lower
        ),
        "misclassification": generating_parameters.misclassification,
    }
    truth = np.asarray(
        tuple(truth_values[name] for name in names),
        dtype=float,
    )
    truth_reporting = vector_to_reporting(truth, names, period)
    recording = generate_recording(
        jax.random.PRNGKey(int(recovery["seed"])),
        config,
        generating_parameters,
    )
    print(
        f"gate: p0={float(soft_gate['p0']):g}, "
        f"k={float(soft_gate['k']):g}"
    )

    def objective(parameters: np.ndarray) -> float:
        physical = np.asarray(parameters, dtype=float).copy()
        if not valid_parameters(physical, names, threshold_mean):
            return 1e12
        value = log_likelihood(
            jnp.asarray(physical),
            recording.observations,
            config,
        )
        result = -float(value)
        return result if np.isfinite(result) else 1e12

    simplex_steps = np.asarray((0.01, 0.01, 0.2, 0.8, 0.02, 0.002))
    results: list[OptimizeResult] = []
    for restart, starting_point in enumerate(recovery["starting_points"], 1):
        start = reporting_to_vector(
            np.asarray(starting_point, dtype=float),
            names,
            period,
        )
        initial_simplex = np.vstack((start, start + np.diag(simplex_steps)))
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={
                "initial_simplex": initial_simplex,
                "maxiter": int(recovery["max_iterations"]),
                "xatol": 1e-3,
                "fatol": 1e-3,
            },
        )
        results.append(result)
        reporting = vector_to_reporting(result.x, names, period)
        values = ", ".join(
            f"{name}={value:.5g}"
            for name, value in zip(
                (
                    "phase",
                    "chi_sleep",
                    "chi_wake",
                    "circadian_amplitude",
                    "threshold_gap",
                    "misclassification",
                ),
                reporting,
            )
        )
        print(
            f"restart={restart}, success={result.success}, "
            f"evaluations={result.nfev}, negative_log_likelihood={result.fun:.6f}"
        )
        print(f"  {values}")

    estimates = np.vstack(
        [vector_to_reporting(result.x, names, period) for result in results]
    )
    phase_errors = np.asarray(
        [
            circular_phase_error(value, truth_reporting[0])
            for value in estimates[:, 0]
        ]
    )
    chi_sleep_errors = np.abs(
        estimates[:, 1] / truth_reporting[1] - 1
    )
    chi_wake_errors = np.abs(
        estimates[:, 2] / truth_reporting[2] - 1
    )
    strong_pass = (
        np.all(phase_errors <= float(recovery["phase_tolerance_hours"]))
        and np.all(
            chi_sleep_errors
            <= float(recovery["time_constant_relative_tolerance"])
        )
        and np.all(
            chi_wake_errors
            <= float(recovery["time_constant_relative_tolerance"])
        )
    )
    amplitude_spread = float(np.ptp(estimates[:, 3]))
    gap_spread = float(np.ptp(estimates[:, 4]))
    weak_scatter = (
        amplitude_spread >= float(recovery["weak_amplitude_spread"])
        and gap_spread >= float(recovery["weak_gap_spread"])
    )

    reporting_names = (
        "phase",
        "chi_sleep",
        "chi_wake",
        "circadian_amplitude",
        "threshold_gap",
        "misclassification",
    )
    print(f"truth={dict(zip(reporting_names, truth_reporting))}")
    print(
        f"strong_parameter_recovery={'PASS' if strong_pass else 'FAIL'}; "
        f"max phase error={np.max(phase_errors):.3f} h, "
        f"max chi_sleep error={np.max(chi_sleep_errors):.1%}, "
        f"max chi_wake error={np.max(chi_wake_errors):.1%}"
    )
    print(
        f"weak_parameter_scatter={'OBSERVED' if weak_scatter else 'NOT OBSERVED'}; "
        f"amplitude range={amplitude_spread:.5f}, gap range={gap_spread:.5f}"
    )
    if not strong_pass:
        raise RuntimeError("strong-parameter recovery criteria were not met")


if __name__ == "__main__":
    main()
