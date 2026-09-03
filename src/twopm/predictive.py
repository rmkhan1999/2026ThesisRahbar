from dataclasses import dataclass

import jax
import numpy as np
from numpy.typing import NDArray

from twopm.config import ProjectConfig
from twopm.generative import Parameters, generate_recording


@dataclass(frozen=True)
class RecordingSummary:

    sleep_fraction: float
    mean_sleep_bout: float
    mean_wake_bout: float
    transition_count: int
    all_sleep: bool
    all_wake: bool


@dataclass(frozen=True)
class PriorPredictiveResult:

    sleep_fraction: NDArray[np.float64]
    mean_sleep_bout: NDArray[np.float64]
    mean_wake_bout: NDArray[np.float64]
    transition_count: NDArray[np.int64]
    all_sleep: NDArray[np.bool_]
    all_wake: NDArray[np.bool_]
    lower_threshold_margin: NDArray[np.float64]
    upper_threshold_margin: NDArray[np.float64]
    normalized_threshold_margin: NDArray[np.float64]
    physical_domain: NDArray[np.bool_]


def threshold_domain_margins(
    parameters: Parameters,
    threshold_mean: float,
) -> tuple[float, float, float]:
    amplitude = float(parameters.amplitude)
    lower_margin = float(parameters.lower) - amplitude
    upper_margin = float(parameters.mu) - float(parameters.upper) - amplitude
    return (
        lower_margin,
        upper_margin,
        lower_margin / threshold_mean,
    )


def recording_summary(
    observations: jax.Array | NDArray[np.integer],
    epoch_times: jax.Array | NDArray[np.floating],
) -> RecordingSummary:
    labels = np.asarray(observations, dtype=np.bool_)
    times = np.asarray(epoch_times, dtype=np.float64)
    if labels.ndim != 1 or times.ndim != 1:
        raise ValueError("observations and epoch times must be one-dimensional")
    if labels.size != times.size or labels.size < 2:
        raise ValueError("observations and epoch times need equal nonzero length")
    if np.any(np.diff(times) <= 0):
        raise ValueError("epoch times must be strictly increasing")

    epoch_duration = float(np.median(np.diff(times)))
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([labels.size])))
    bout_states = labels[starts]
    bout_lengths = (ends - starts) * epoch_duration
    sleep_bouts = bout_lengths[bout_states]
    wake_bouts = bout_lengths[~bout_states]

    return RecordingSummary(
        sleep_fraction=float(np.mean(labels)),
        mean_sleep_bout=(
            float(np.mean(sleep_bouts)) if sleep_bouts.size else np.nan
        ),
        mean_wake_bout=(
            float(np.mean(wake_bouts)) if wake_bouts.size else np.nan
        ),
        transition_count=int(boundaries.size),
        all_sleep=bool(np.all(labels)),
        all_wake=bool(np.all(~labels)),
    )


def run_prior_predictive(
    key: jax.Array,
    config: ProjectConfig,
    draws: int | None = None,
    design: str | None = None,
) -> PriorPredictiveResult:
    settings = config.section("prior_predictive")
    draw_count = int(settings["draws"]) if draws is None else draws
    if draw_count <= 0:
        raise ValueError("prior-predictive draw count must be positive")

    summaries = []
    margins = []
    threshold_mean = float(config.section("fixed")["threshold_mean"])
    for draw_key in jax.random.split(key, draw_count):
        recording = generate_recording(draw_key, config, design=design)
        summaries.append(
            recording_summary(recording.observations, recording.time)
        )
        margins.append(
            threshold_domain_margins(recording.parameters, threshold_mean)
        )

    return PriorPredictiveResult(
        sleep_fraction=np.asarray(
            [summary.sleep_fraction for summary in summaries],
            dtype=np.float64,
        ),
        mean_sleep_bout=np.asarray(
            [summary.mean_sleep_bout for summary in summaries],
            dtype=np.float64,
        ),
        mean_wake_bout=np.asarray(
            [summary.mean_wake_bout for summary in summaries],
            dtype=np.float64,
        ),
        transition_count=np.asarray(
            [summary.transition_count for summary in summaries],
            dtype=np.int64,
        ),
        all_sleep=np.asarray(
            [summary.all_sleep for summary in summaries],
            dtype=np.bool_,
        ),
        all_wake=np.asarray(
            [summary.all_wake for summary in summaries],
            dtype=np.bool_,
        ),
        lower_threshold_margin=np.asarray(
            [margin[0] for margin in margins],
            dtype=np.float64,
        ),
        upper_threshold_margin=np.asarray(
            [margin[1] for margin in margins],
            dtype=np.float64,
        ),
        normalized_threshold_margin=np.asarray(
            [margin[2] for margin in margins],
            dtype=np.float64,
        ),
        physical_domain=np.asarray(
            [margin[0] > 0 and margin[1] > 0 for margin in margins],
            dtype=np.bool_,
        ),
    )
