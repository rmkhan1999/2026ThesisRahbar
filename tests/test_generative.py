from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from twopm.config import ProjectConfig, load_config
from twopm.generative import (
    generate_recording,
    sample_parameters,
    standard_parameters,
)
from twopm.observation import sample_observations, sleep_probabilities
from twopm.soft_gate import (
    simulate_soft_gate_from_config,
    soft_transition_times,
)


def test_standard_parameters_requires_explicit_amplitude():
    config = load_config("config/model.yaml")

    with pytest.raises(TypeError):
        standard_parameters(config)


def test_fixed_mode_reproduces_observation_pipeline():
    config = load_config("config/model.yaml")
    observation = config.section("observation")
    design = config.section("designs")["entrained"]
    soft = config.section("soft_gate")
    parameters = standard_parameters(config, amplitude=0.0)
    key = jax.random.PRNGKey(int(observation["seed"]))
    generated = generate_recording(key, config, parameters)
    burn_in = float(design["burn_in_hours"])

    trajectory = simulate_soft_gate_from_config(
        config,
        chi_sleep=parameters.chi_sleep,
        chi_wake=parameters.chi_wake,
        mu=parameters.mu,
        upper=parameters.upper,
        lower=parameters.lower,
        c1=parameters.c1,
        c2=parameters.c2,
        initial_pressure=parameters.initial_pressure,
        k=float(soft["k"]),
        p0=float(soft["p0"]),
        duration=burn_in + float(design["duration"]),
        output_step=float(observation["epoch_hours"]),
    )
    retained_gate = trajectory.gate[trajectory.time >= burn_in]
    expected_probabilities = sleep_probabilities(
        retained_gate,
        parameters.misclassification,
    )
    expected_observations = sample_observations(
        key,
        expected_probabilities,
    )

    assert np.array_equal(generated.gate, retained_gate)
    assert np.array_equal(
        generated.probabilities,
        expected_probabilities,
    )
    assert np.array_equal(generated.observations, expected_observations)


def test_prior_draw_mode_is_reproducible_varied_and_plausible():
    config = load_config("config/model.yaml")
    first = generate_recording(jax.random.PRNGKey(0), config)
    repeated = generate_recording(jax.random.PRNGKey(0), config)
    second = generate_recording(jax.random.PRNGKey(1), config)

    assert first.parameters == repeated.parameters
    assert np.array_equal(first.observations, repeated.observations)
    assert first.parameters != second.parameters

    for recording in (first, second):
        parameters = recording.parameters
        sleep_fraction = float(np.mean(recording.observations))
        assert 0 < parameters.lower < parameters.upper < parameters.mu
        assert parameters.lower - parameters.amplitude > 0
        assert parameters.upper + parameters.amplitude < parameters.mu
        assert 0 < parameters.misclassification < 0.5
        assert 0.1 < sleep_fraction < 0.6


@pytest.mark.parametrize(
    ("design", "amplitude_mean", "amplitude_sd"),
    [
        ("entrained", 0.12, 0.025),
        ("weak_forcing", 0.02 * np.sqrt(np.pi / 2), 0.02 * np.sqrt((4 - np.pi) / 2)),
    ],
)
def test_constrained_prior_has_expected_moments_and_exact_domain(
    design,
    amplitude_mean,
    amplitude_sd,
):
    config = load_config("config/model.yaml")
    keys = jax.random.split(jax.random.PRNGKey(20260728), 20_000)
    draws = jax.vmap(lambda key: sample_parameters(key, config, design))(keys)
    amplitude = np.asarray(jnp.hypot(draws.c1, draws.c2))
    gap = np.asarray(draws.upper - draws.lower)

    assert np.all(np.asarray(draws.lower) - amplitude > 0)
    assert np.all(np.asarray(draws.upper) + amplitude < np.asarray(draws.mu))
    assert np.mean(amplitude) == pytest.approx(amplitude_mean, abs=0.001)
    assert np.std(amplitude) == pytest.approx(amplitude_sd, abs=0.001)
    assert np.mean(gap) == pytest.approx(0.5, abs=0.002)
    assert np.std(gap) == pytest.approx(0.0607, abs=0.002)


def test_weak_forcing_draw_includes_initial_pressure():
    config = load_config("config/model.yaml")
    design = config.section("designs")["weak_forcing"]
    parameters = sample_parameters(
        jax.random.PRNGKey(7),
        config,
        design="weak_forcing",
    )
    pressure_prior = design["initial_pressure_prior"]

    assert pressure_prior["lower"] <= parameters.initial_pressure
    assert parameters.initial_pressure <= pressure_prior["upper"]
    assert np.isfinite(float(parameters.amplitude))
    assert np.isfinite(float(parameters.phase))


def test_burn_in_removes_circadian_initial_transient():
    base = load_config("config/model.yaml")
    data = deepcopy(base.data)
    data["designs"]["entrained"]["duration"] = 240.0
    config = ProjectConfig(data=data, source=base.source)
    model = config.section("model")
    recovery = config.section("recovery")
    parameters = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
        phase=float(recovery["true_phase"]),
    )
    recording = generate_recording(
        jax.random.PRNGKey(0),
        config,
        parameters,
    )
    first_day = recording.gate[recording.time < 24.0]
    settled = recording.gate[recording.time >= 24.0]

    assert float(jnp.mean(first_day)) == pytest.approx(
        float(jnp.mean(settled)),
        abs=0.01,
    )


def test_forced_recording_is_invariant_to_initial_pressure_after_burn_in():
    base = load_config("config/model.yaml")
    data = deepcopy(base.data)
    data["designs"]["entrained"]["duration"] = 240.0
    config = ProjectConfig(data=data, source=base.source)
    model = config.section("model")
    recovery = config.section("recovery")
    level = float(config.section("validation")["transition_gate_level"])
    recordings = []
    transitions = []
    for initial_pressure in (0.2, 0.4, 0.6, 0.8):
        parameters = standard_parameters(
            config,
            amplitude=float(model["circadian_amplitude"]),
            phase=float(recovery["true_phase"]),
            initial_pressure=initial_pressure,
        )
        recording = generate_recording(
            jax.random.PRNGKey(1),
            config,
            parameters,
        )
        recordings.append(recording)
        transitions.append(
            soft_transition_times(recording.time, recording.gate, level)
        )

    for candidate in transitions[1:]:
        assert np.allclose(candidate, transitions[0], atol=0.01)
    for candidate in recordings[1:]:
        assert np.array_equal(
            candidate.observations,
            recordings[0].observations,
        )
