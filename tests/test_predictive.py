import jax
import numpy as np
import pytest

from twopm.config import load_config
from twopm.generative import standard_parameters
from twopm.predictive import (
    recording_summary,
    run_prior_predictive,
    threshold_domain_margins,
)


def test_recording_summary_extracts_bouts_and_transitions():
    summary = recording_summary(
        observations=np.asarray((0, 0, 1, 1, 1, 0)),
        epoch_times=np.arange(6) * 0.5,
    )

    assert summary.sleep_fraction == pytest.approx(0.5)
    assert summary.mean_sleep_bout == pytest.approx(1.5)
    assert summary.mean_wake_bout == pytest.approx(0.75)
    assert summary.transition_count == 2
    assert not summary.all_sleep
    assert not summary.all_wake


def test_prior_predictive_draws_are_non_degenerate_and_bounded():
    config = load_config("config/model.yaml")
    settings = config.section("prior_predictive")
    result = run_prior_predictive(
        jax.random.PRNGKey(int(settings["seed"])),
        config,
        draws=20,
    )
    degenerate_fraction = np.mean(result.all_sleep | result.all_wake)
    outside_sleep_bounds = np.mean(
        (result.sleep_fraction < float(settings["sleep_fraction_min"]))
        | (result.sleep_fraction > float(settings["sleep_fraction_max"]))
    )

    assert degenerate_fraction <= float(settings["max_degenerate_fraction"])
    assert outside_sleep_bounds <= float(settings["max_degenerate_fraction"])
    assert np.all(np.isfinite(result.mean_sleep_bout))
    assert np.all(np.isfinite(result.mean_wake_bout))
    assert np.all(result.physical_domain)
    assert np.min(result.normalized_threshold_margin) > float(
        settings["minimum_normalized_threshold_margin"]
    )


def test_threshold_margin_matches_canonical_clearance():
    config = load_config("config/model.yaml")
    parameters = standard_parameters(config, amplitude=0.12)
    lower, upper, normalized = threshold_domain_margins(parameters, 0.42)

    assert lower == pytest.approx(0.05)
    assert upper == pytest.approx(0.21)
    assert normalized == pytest.approx(0.05 / 0.42)
