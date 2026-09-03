import jax
import jax.numpy as jnp
import numpy as np

from twopm.config import load_config
from twopm.hard_switch import simulate_hard_switch_from_config
from twopm.observation import sample_observations, sleep_probabilities
from twopm.soft_gate import circadian_coefficients, simulate_soft_gate_from_config


def test_sleep_probabilities_are_deterministic_label_error_map():
    probabilities = sleep_probabilities(
        jnp.asarray((0.0, 0.5, 1.0)),
        misclassification=0.01,
    )

    assert np.allclose(probabilities, (0.01, 0.5, 0.99))


def test_sampled_recording_is_reproducible_and_tracks_hard_labels():
    config = load_config("config/model.yaml")
    observation = config.section("observation")
    design = config.section("designs")["entrained"]
    soft_settings = config.section("soft_gate")
    model = config.section("model")
    amplitude = 0.0
    c1, c2 = circadian_coefficients(
        amplitude,
        float(model["phase"]),
        float(model["circadian_period"]),
    )
    soft = simulate_soft_gate_from_config(
        config,
        c1=c1,
        c2=c2,
        duration=float(design["duration"]),
        output_step=float(observation["epoch_hours"]),
        k=float(soft_settings["k"]),
        p0=float(soft_settings["p0"]),
    )
    probabilities = sleep_probabilities(
        soft.gate,
        float(observation["misclassification"]),
    )
    key = jax.random.PRNGKey(int(observation["seed"]))
    first = sample_observations(key, probabilities)
    second = sample_observations(key, probabilities)

    hard = simulate_hard_switch_from_config(
        config,
        amplitude=amplitude,
        duration=float(design["duration"]),
    )
    hard_indices = np.searchsorted(hard.time, np.asarray(soft.time))
    hard_labels = hard.asleep[hard_indices]
    agreement = np.mean(np.asarray(first) == hard_labels)
    observed_sleep_fraction = float(jnp.mean(first))
    agreement_floor = float(
        config.section("validation")["flat_hard_label_agreement_min"]
    )

    assert np.array_equal(first, second)
    assert agreement >= agreement_floor
    assert np.isclose(observed_sleep_fraction, 0.26, atol=0.03)
