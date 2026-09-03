import jax.numpy as jnp
import numpy as np
import pytest

from twopm.config import load_config
from twopm.soft_gate import (
    cartesian_circadian_displacement,
    circadian_coefficients,
    gate_offset,
    gate_target,
    match_transition_times,
    simulate_soft_gate_from_config,
    smoothing_convergence_study,
    soft_gate_vector_field,
    soft_transition_times,
)


@pytest.fixture(scope="module")
def project_config():
    return load_config("config/model.yaml")


@pytest.fixture(scope="module")
def soft_result(project_config):
    return simulate_soft_gate_from_config(project_config)


def test_active_threshold_distance_preserves_hysteresis():
    args = {
        "chi_sleep": 4.2,
        "chi_wake": 18.2,
        "mu": 1.0,
        "upper": 0.67,
        "lower": 0.17,
        "c1": 0.0,
        "c2": 0.0,
        "period": 24.0,
        "k": 1000.0,
        "p0": 0.05,
        "tau_gate": 0.05,
    }
    awake_derivative = soft_gate_vector_field(
        jnp.asarray(0.0),
        jnp.asarray((0.42, 0.0)),
        args,
    )
    asleep_derivative = soft_gate_vector_field(
        jnp.asarray(0.0),
        jnp.asarray((0.42, 1.0)),
        args,
    )

    assert awake_derivative[1] == pytest.approx(0.0, abs=1e-12)
    assert asleep_derivative[1] == pytest.approx(0.0, abs=1e-12)
    assert awake_derivative[0] > 0
    assert asleep_derivative[0] < 0


def test_cartesian_circadian_displacement_matches_amplitude_phase():
    time = jnp.linspace(0.0, 48.0, 97)
    amplitude = 0.12
    phase = 6.0
    period = 24.0
    c1, c2 = circadian_coefficients(amplitude, phase, period)
    cartesian = cartesian_circadian_displacement(time, c1, c2, period)
    physiological = amplitude * jnp.cos(
        2 * jnp.pi * (time - phase) / period
    )

    assert np.allclose(cartesian, physiological, atol=1e-12)


def test_gate_target_equals_configured_p0_at_active_threshold(project_config):
    soft = project_config.section("soft_gate")
    p0 = float(soft["p0"])

    assert float(gate_target(0.0, float(soft["k"]), p0)) == pytest.approx(
        p0,
        abs=1e-12,
    )


def test_gate_transition_spans_configured_pressure_width(project_config):
    soft = project_config.section("soft_gate")
    validation = project_config.section("validation")
    p0 = float(soft["p0"])
    k = float(soft["k"])
    theta = float(gate_offset(p0))
    distance_5 = (np.log(0.05 / 0.95) - theta) / k
    distance_95 = (np.log(0.95 / 0.05) - theta) / k

    assert float(gate_target(distance_5, k, p0)) == pytest.approx(0.05)
    assert float(gate_target(distance_95, k, p0)) == pytest.approx(0.95)
    assert distance_95 - distance_5 == pytest.approx(
        float(validation["transition_pressure_width_5_95"]),
        abs=1e-6,
    )


def test_soft_gate_is_smooth_bounded_sleep_sequence(
    project_config,
    soft_result,
):
    validation = project_config.section("validation")
    transitions = soft_transition_times(
        soft_result.time,
        soft_result.gate,
        float(validation["transition_gate_level"]),
    )

    assert np.all(np.isfinite(soft_result.gate))
    assert float(np.min(soft_result.gate)) >= -1e-8
    assert float(np.max(soft_result.gate)) <= 1.0 + 1e-7
    assert transitions.size >= 8


def test_configured_k_is_within_observation_error_budget(
    project_config,
):
    result = smoothing_convergence_study(project_config)
    configured_k = float(project_config.section("soft_gate")["k"])
    budget = float(
        project_config.section("validation")["transition_bias_budget_hours"]
    )
    chosen_index = int(np.where(result.k_values == configured_k)[0][0])

    assert np.all(np.isfinite(result.mean_absolute_error))
    assert result.mean_absolute_error[chosen_index] < budget


def test_transition_matching_ignores_one_extra_event_without_index_shift():
    match = match_transition_times(
        reference=np.asarray((10.0, 20.0, 40.0)),
        candidate=np.asarray((10.1, 19.9, 30.0, 40.1)),
    )

    assert np.allclose(match.reference, (10.0, 20.0, 40.0))
    assert np.allclose(match.candidate, (10.1, 19.9, 40.1))
    assert np.mean(match.absolute_error) == pytest.approx(0.1)
    assert match.unmatched_reference == 0
    assert match.unmatched_candidate == 1
