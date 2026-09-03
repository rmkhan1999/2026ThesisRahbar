from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from twopm.hard_switch import HardSwitchResult
from twopm.predictive import PriorPredictiveResult
from twopm.soft_gate import ConvergenceResult


def plot_hard_switch_trajectory(
    result: HardSwitchResult,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, (pressure_axis, state_axis) = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(10, 5),
        height_ratios=(3, 1),
    )
    pressure_axis.plot(result.time, result.pressure, label="Homeostatic pressure")
    pressure_axis.plot(
        result.time,
        result.upper_threshold,
        linestyle="--",
        label="Upper threshold",
    )
    pressure_axis.plot(
        result.time,
        result.lower_threshold,
        linestyle="--",
        label="Lower threshold",
    )
    pressure_axis.set_ylabel("H")
    pressure_axis.legend(loc="upper right")

    state_axis.step(
        result.time,
        result.asleep.astype(float),
        where="post",
        color="black",
    )
    state_axis.set_yticks((0, 1), labels=("wake", "sleep"))
    state_axis.set_xlabel("Time (hours)")
    state_axis.set_ylabel("State")

    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_smoothing_convergence(
    result: ConvergenceResult,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.loglog(
        result.k_values,
        result.mean_absolute_error,
        marker="o",
    )
    axis.set_xlabel("Gate sharpness (k)")
    axis.set_ylabel("Mean absolute transition error (hours)")
    axis.set_title("Soft-gate convergence to hard-switch oracle")
    axis.grid(visible=True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_prior_predictive(
    result: PriorPredictiveResult,
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    summaries = (
        (result.sleep_fraction, "Sleep fraction"),
        (result.mean_sleep_bout, "Mean sleep bout (hours)"),
        (result.mean_wake_bout, "Mean wake bout (hours)"),
        (result.transition_count, "Transition count"),
    )
    for axis, (values, label) in zip(axes.flat, summaries):
        finite_values = values[np.isfinite(values)]
        axis.hist(finite_values, bins=20)
        axis.set_xlabel(label)
        axis.set_ylabel("Prior draws")

    degenerate_fraction = np.mean(result.all_sleep | result.all_wake)
    figure.suptitle(
        f"Prior predictive checks (degenerate: {degenerate_fraction:.1%})"
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
