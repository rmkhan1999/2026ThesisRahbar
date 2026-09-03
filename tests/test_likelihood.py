import jax
import jax.numpy as jnp
import numpy as np
import pytest

from twopm.config import load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.likelihood import (
    likelihood_parameter_names,
    log_likelihood,
    standard_likelihood_parameters,
)
from twopm.soft_gate import (
    circadian_amplitude_phase,
    circadian_coefficients,
)


def test_true_parameters_have_finite_better_likelihood_than_perturbation():
    config = load_config("config/model.yaml")
    model = config.section("model")
    parameters = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
    )
    recording = generate_recording(
        jax.random.PRNGKey(31415),
        config,
        parameters,
    )
    truth = standard_likelihood_parameters(config)
    names = likelihood_parameter_names(config)
    values = dict(zip(names, np.asarray(truth)))
    c1, c2 = circadian_coefficients(
        0.5 * float(model["circadian_amplitude"]),
        8.0,
        float(model["circadian_period"]),
    )
    values.update(
        c1=float(c1),
        c2=float(c2),
        chi_sleep=1.4 * values["chi_sleep"],
        chi_wake=0.7 * values["chi_wake"],
        threshold_gap=1.3 * values["threshold_gap"],
        misclassification=0.08,
    )
    perturbed = jnp.asarray(tuple(values[name] for name in names))

    true_log_likelihood = log_likelihood(
        truth,
        recording.observations,
        config,
    )
    perturbed_log_likelihood = log_likelihood(
        perturbed,
        recording.observations,
        config,
    )

    assert np.isfinite(float(true_log_likelihood))
    assert np.isfinite(float(perturbed_log_likelihood))
    assert float(true_log_likelihood) > float(perturbed_log_likelihood)


def test_initial_pressure_is_inferred_only_for_weak_forcing():
    config = load_config("config/model.yaml")

    assert "initial_pressure" not in likelihood_parameter_names(
        config,
        "entrained",
    )
    assert likelihood_parameter_names(config, "weak_forcing")[-1] == (
        "initial_pressure"
    )
    weak_names = likelihood_parameter_names(config, "weak_forcing")
    weak_values = dict(
        zip(
            weak_names,
            np.asarray(
                standard_likelihood_parameters(config, "weak_forcing")
            ),
        )
    )
    weak_amplitude = circadian_amplitude_phase(
        weak_values["c1"],
        weak_values["c2"],
        24.0,
    )[0]
    assert float(weak_amplitude) == pytest.approx(0.02)


def test_cartesian_log_likelihood_gradient_matches_finite_differences():
    from copy import deepcopy

    from twopm.config import ProjectConfig

    base = load_config("config/model.yaml")
    data = deepcopy(base.data)
    data["designs"]["entrained"]["burn_in_hours"] = 12.0
    data["designs"]["entrained"]["duration"] = 12.0
    config = ProjectConfig(data=data, source=base.source)
    model = config.section("model")
    parameters = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
        phase=8.0,
    )
    recording = generate_recording(
        jax.random.PRNGKey(123),
        config,
        parameters,
    )
    names = likelihood_parameter_names(config)
    values = {
        "c1": parameters.c1,
        "c2": parameters.c2,
        "chi_sleep": parameters.chi_sleep,
        "chi_wake": parameters.chi_wake,
        "threshold_gap": parameters.upper - parameters.lower,
        "misclassification": parameters.misclassification,
    }
    point = jnp.asarray(tuple(values[name] for name in names))
    function = lambda vector: log_likelihood(
        vector,
        recording.observations,
        config,
    )
    automatic = np.asarray(jax.grad(function)(point))
    step = float(config.section("validation")["finite_difference_step"])
    finite_difference = []
    for index in range(point.size):
        offset = jnp.zeros_like(point).at[index].set(step)
        finite_difference.append(
            float((function(point + offset) - function(point - offset)) / (2 * step))
        )
    finite_difference = np.asarray(finite_difference)
    relative_error = np.abs(automatic - finite_difference) / np.maximum(
        np.abs(finite_difference),
        1e-10,
    )

    assert np.max(relative_error) < 0.01
