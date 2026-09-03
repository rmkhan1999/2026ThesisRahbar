from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.likelihood import (
    likelihood_parameter_names,
    log_likelihood,
    standard_likelihood_parameters,
)
from twopm.soft_gate import circadian_coefficients


def test_log_likelihood_propagates_solver_failure():
    base = load_config("config/model.yaml")
    label_data = deepcopy(base.data)
    label_data["designs"]["entrained"]["duration"] = 24.0
    label_data["designs"]["entrained"]["burn_in_hours"] = 48.0
    label_config = ProjectConfig(data=label_data, source=base.source)
    model = label_config.section("model")
    parameters = standard_parameters(
        label_config,
        amplitude=float(model["circadian_amplitude"]),
        phase=float(label_config.section("recovery")["true_phase"]),
    )
    recording = generate_recording(
        jax.random.PRNGKey(31415),
        label_config,
        parameters,
    )
    truth = standard_likelihood_parameters(label_config)

    fail_data = deepcopy(label_data)
    fail_data["soft_gate"]["max_steps"] = 100
    fail_config = ProjectConfig(data=fail_data, source=base.source)

    with pytest.raises(Exception) as caught:
        log_likelihood(truth, recording.observations, fail_config)
    assert not isinstance(caught.value, ValueError)
    message = str(caught.value).lower()
    assert "maximum number of solver steps" in message or "max_steps" in message


def test_log_likelihood_finite_near_excursion_boundary():
    config = load_config("config/model.yaml")
    model = config.section("model")
    fixed = config.section("fixed")
    threshold_mean = float(fixed["threshold_mean"])
    parameters = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
        phase=float(config.section("recovery")["true_phase"]),
    )
    recording = generate_recording(
        jax.random.PRNGKey(31415),
        config,
        parameters,
    )
    names = likelihood_parameter_names(config)
    s = 1.0 - 1e-6
    w = 0.24
    excursion = threshold_mean * s
    amplitude = w * excursion
    gap = 2.0 * (1.0 - w) * excursion
    c1, c2 = circadian_coefficients(
        amplitude,
        float(parameters.phase),
        float(model["circadian_period"]),
    )
    values = {
        "c1": float(c1),
        "c2": float(c2),
        "chi_sleep": float(parameters.chi_sleep),
        "chi_wake": float(parameters.chi_wake),
        "threshold_gap": float(gap),
        "misclassification": 0.01,
    }
    vector = jnp.asarray(tuple(values[name] for name in names))
    value = float(log_likelihood(vector, recording.observations, config))
    assert np.isfinite(value), f"non-finite log-likelihood at s→1: {value}"

    interior = float(
        log_likelihood(
            standard_likelihood_parameters(config),
            recording.observations,
            config,
        )
    )
    assert np.isfinite(interior)
