import jax.numpy as jnp
import pytest

from twopm.config import load_config
from twopm.hard_switch import simulate_hard_switch_from_config
from twopm.likelihood import LIKELIHOOD_PARAMETER_NAMES


@pytest.fixture(scope="module")
def project_config():
    return load_config("config/model.yaml")


def test_package_enables_float64_by_default():
    assert jnp.zeros(1).dtype == jnp.float64


def test_config_contains_standard_model_values(project_config):
    model = project_config.section("model")

    assert model["chi_sleep"] == 4.2
    assert model["chi_wake"] == 18.2
    assert model["upper_base"] == 0.67
    assert model["lower_base"] == 0.17


def test_hard_switch_simulator_consumes_config(project_config):
    result = simulate_hard_switch_from_config(project_config, duration=48.0)

    assert result.time[-1] == pytest.approx(48.0)
    assert result.switch_times.size > 1


def test_priors_exactly_match_base_inferred_parameters(project_config):
    priors = project_config.section("priors")
    inference = project_config.section("inference")

    assert set(priors) == set(inference["parameters"])
    assert tuple(inference["likelihood_parameters"]) == LIKELIHOOD_PARAMETER_NAMES
