from collections.abc import Sequence

import jax
import jax.numpy as jnp

from twopm.config import ProjectConfig
from twopm.observation import sleep_probabilities
from twopm.soft_gate import (
    circadian_coefficients,
    simulate_soft_gate_from_config,
)


LIKELIHOOD_PARAMETER_NAMES = (
    "c1",
    "c2",
    "chi_sleep",
    "chi_wake",
    "threshold_gap",
    "misclassification",
)


def likelihood_parameter_names(
    config: ProjectConfig,
    design: str | None = None,
) -> tuple[str, ...]:
    inference = config.section("inference")
    designs = config.section("designs")
    design_name = (
        str(inference["default_design"]) if design is None else design
    )
    names = tuple(
        str(name)
        for name in inference.get(
            "likelihood_parameters",
            LIKELIHOOD_PARAMETER_NAMES,
        )
    )
    if bool(designs[design_name]["infer_initial_pressure"]):
        names += ("initial_pressure",)
    return names


def standard_likelihood_parameters(
    config: ProjectConfig,
    design: str | None = None,
) -> jax.Array:
    model = config.section("model")
    observation = config.section("observation")
    initial = config.section("initial_state")
    inference = config.section("inference")
    designs = config.section("designs")
    design_name = (
        str(inference["default_design"]) if design is None else design
    )
    c1, c2 = circadian_coefficients(
        float(designs[design_name]["reference_amplitude"]),
        float(model["phase"]),
        float(model["circadian_period"]),
    )
    values = {
        "c1": c1,
        "c2": c2,
        "chi_sleep": model["chi_sleep"],
        "chi_wake": model["chi_wake"],
        "threshold_gap": (
            float(model["upper_base"]) - float(model["lower_base"])
        ),
        "misclassification": observation["misclassification"],
        "initial_pressure": initial["pressure"],
    }
    return jnp.asarray(
        tuple(values[name] for name in likelihood_parameter_names(config, design))
    )


def log_likelihood(
    parameter_vector: jax.Array | Sequence[float],
    observed_labels: jax.Array | Sequence[int],
    config: ProjectConfig,
    design: str | None = None,
) -> jax.Array:
    parameters = jnp.asarray(parameter_vector)
    labels = jnp.asarray(observed_labels)
    names = likelihood_parameter_names(config, design)
    if parameters.shape != (len(names),):
        raise ValueError(
            f"parameter_vector must have shape ({len(names)},)"
        )

    fixed = config.section("fixed")
    initial = config.section("initial_state")
    observation = config.section("observation")
    inference = config.section("inference")
    designs = config.section("designs")
    design_name = (
        str(inference["default_design"]) if design is None else design
    )
    design_settings = designs[design_name]
    burn_in = float(design_settings["burn_in_hours"])
    epoch_hours = float(observation["epoch_hours"])
    values = dict(zip(names, parameters))
    threshold_mean = float(fixed["threshold_mean"])
    initial_pressure = values.get(
        "initial_pressure",
        jnp.asarray(initial["pressure"]),
    )
    upper = threshold_mean + values["threshold_gap"] / 2
    lower = threshold_mean - values["threshold_gap"] / 2
    retained_start = int(round(burn_in / epoch_hours))
    retained_points = (
        int(round(float(design_settings["duration"]) / epoch_hours)) + 1
    )
    if labels.shape != (retained_points,):
        raise ValueError(
            "observed_labels must match the configured retained epoch grid"
        )

    trajectory = simulate_soft_gate_from_config(
        config,
        c1=values["c1"],
        c2=values["c2"],
        chi_sleep=values["chi_sleep"],
        chi_wake=values["chi_wake"],
        upper=upper,
        lower=lower,
        initial_pressure=initial_pressure,
        duration=burn_in + float(design_settings["duration"]),
        output_step=epoch_hours,
        throw=True,
    )
    gate = trajectory.gate[retained_start:]
    probabilities = sleep_probabilities(gate, values["misclassification"])
    epsilon = jnp.finfo(probabilities.dtype).eps
    probabilities = jnp.clip(probabilities, epsilon, 1 - epsilon)
    return jnp.sum(
        labels * jnp.log(probabilities)
        + (1 - labels) * jnp.log1p(-probabilities)
    )
