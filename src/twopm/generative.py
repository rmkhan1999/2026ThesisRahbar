from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from twopm.config import ProjectConfig
from twopm.observation import sample_observations, sleep_probabilities
from twopm.soft_gate import (
    circadian_amplitude_phase,
    circadian_coefficients,
    simulate_soft_gate_from_config,
)


class Parameters(NamedTuple):

    chi_sleep: jax.Array
    chi_wake: jax.Array
    mu: jax.Array
    upper: jax.Array
    lower: jax.Array
    c1: jax.Array
    c2: jax.Array
    period: jax.Array
    misclassification: jax.Array
    initial_pressure: jax.Array
    phase_z1: jax.Array
    phase_z2: jax.Array
    excursion_fraction: jax.Array
    amplitude_fraction: jax.Array

    @property
    def amplitude(self) -> jax.Array:
        return circadian_amplitude_phase(self.c1, self.c2, self.period)[0]

    @property
    def phase(self) -> jax.Array:
        return circadian_amplitude_phase(self.c1, self.c2, self.period)[1]


class GeneratedRecording(NamedTuple):

    parameters: Parameters
    time: jax.Array
    pressure: jax.Array
    gate: jax.Array
    probabilities: jax.Array
    observations: jax.Array


def standard_parameters(
    config: ProjectConfig,
    *,
    amplitude: float | jax.Array,
    phase: float | jax.Array | None = None,
    misclassification: float | jax.Array | None = None,
    initial_pressure: float | jax.Array | None = None,
) -> Parameters:
    model = config.section("model")
    observation = config.section("observation")
    initial = config.section("initial_state")
    period = jnp.asarray(model["circadian_period"])
    configured_phase = (
        jnp.asarray(model["phase"]) if phase is None else jnp.asarray(phase)
    )
    c1, c2 = circadian_coefficients(
        amplitude,
        configured_phase,
        period,
    )
    threshold_mean = jnp.asarray(config.section("fixed")["threshold_mean"])
    threshold_gap = jnp.asarray(model["upper_base"]) - jnp.asarray(
        model["lower_base"]
    )
    total_excursion = threshold_gap / 2 + jnp.asarray(amplitude)
    direction_norm = jnp.hypot(c1, c2)
    phase_z1 = jnp.where(direction_norm > 0, c1 / direction_norm, 1.0)
    phase_z2 = jnp.where(direction_norm > 0, c2 / direction_norm, 0.0)
    return Parameters(
        chi_sleep=jnp.asarray(model["chi_sleep"]),
        chi_wake=jnp.asarray(model["chi_wake"]),
        mu=jnp.asarray(model["mu"]),
        upper=jnp.asarray(model["upper_base"]),
        lower=jnp.asarray(model["lower_base"]),
        c1=c1,
        c2=c2,
        period=period,
        misclassification=jnp.asarray(
            observation["misclassification"]
            if misclassification is None
            else misclassification
        ),
        initial_pressure=jnp.asarray(
            initial["pressure"] if initial_pressure is None else initial_pressure
        ),
        phase_z1=phase_z1,
        phase_z2=phase_z2,
        excursion_fraction=total_excursion / threshold_mean,
        amplitude_fraction=jnp.asarray(amplitude) / total_excursion,
    )


def sample_parameters(
    key: jax.Array,
    config: ProjectConfig,
    design: str | None = None,
) -> Parameters:
    model = config.section("model")
    fixed = config.section("fixed")
    initial = config.section("initial_state")
    priors = config.section("priors")
    inference = config.section("inference")
    designs = config.section("designs")
    design_name = (
        str(inference["default_design"]) if design is None else design
    )
    design_settings = designs[design_name]
    infer_initial_pressure = bool(
        design_settings["infer_initial_pressure"]
    )
    keys = jax.random.split(key, 8 if infer_initial_pressure else 7)

    chi_sleep_prior = priors["chi_sleep"]
    chi_wake_prior = priors["chi_wake"]
    phase_z1_prior = priors["phase_z1"]
    phase_z2_prior = priors["phase_z2"]
    error_prior = priors["misclassification"]
    constrained_prior = design_settings["constrained_prior"]

    phase_z1 = dist.Normal(
        phase_z1_prior["mean"],
        phase_z1_prior["sd"],
    ).sample(keys[0])
    phase_z2 = dist.Normal(
        phase_z2_prior["mean"],
        phase_z2_prior["sd"],
    ).sample(keys[1])
    excursion_fraction = dist.Beta(
        constrained_prior["excursion_concentration1"],
        constrained_prior["excursion_concentration0"],
    ).sample(keys[2])
    amplitude_fraction = dist.Beta(
        constrained_prior["amplitude_concentration1"],
        constrained_prior["amplitude_concentration0"],
    ).sample(keys[3])
    chi_sleep = dist.LogNormal(
        chi_sleep_prior["log_mean"],
        chi_sleep_prior["log_sd"],
    ).sample(keys[4])
    chi_wake = dist.LogNormal(
        chi_wake_prior["log_mean"],
        chi_wake_prior["log_sd"],
    ).sample(keys[5])
    misclassification = error_prior["maximum"] * dist.Beta(
        error_prior["concentration1"],
        error_prior["concentration0"],
    ).sample(keys[6])
    if infer_initial_pressure:
        pressure_prior = design_settings["initial_pressure_prior"]
        initial_pressure = dist.Uniform(
            pressure_prior["lower"],
            pressure_prior["upper"],
        ).sample(keys[7])
    else:
        initial_pressure = jnp.asarray(initial["pressure"])
    threshold_mean = jnp.asarray(fixed["threshold_mean"])
    total_excursion = threshold_mean * excursion_fraction
    amplitude = amplitude_fraction * total_excursion
    threshold_gap = 2 * (1 - amplitude_fraction) * total_excursion
    half_gap = threshold_gap / 2
    direction_norm = jnp.hypot(phase_z1, phase_z2)
    safe_norm = jnp.where(direction_norm > 0, direction_norm, 1.0)
    c1 = jnp.where(
        direction_norm > 0,
        amplitude * phase_z1 / safe_norm,
        amplitude,
    )
    c2 = jnp.where(
        direction_norm > 0,
        amplitude * phase_z2 / safe_norm,
        0.0,
    )

    return Parameters(
        chi_sleep=chi_sleep,
        chi_wake=chi_wake,
        mu=jnp.asarray(model["mu"]),
        upper=threshold_mean + half_gap,
        lower=threshold_mean - half_gap,
        c1=c1,
        c2=c2,
        period=jnp.asarray(model["circadian_period"]),
        misclassification=misclassification,
        initial_pressure=initial_pressure,
        phase_z1=phase_z1,
        phase_z2=phase_z2,
        excursion_fraction=excursion_fraction,
        amplitude_fraction=amplitude_fraction,
    )


def generate_recording(
    key: jax.Array,
    config: ProjectConfig,
    parameters: Parameters | None = None,
    design: str | None = None,
) -> GeneratedRecording:
    if parameters is None:
        parameter_key, observation_key = jax.random.split(key)
        parameters = sample_parameters(parameter_key, config, design)
    else:
        observation_key = key

    observation = config.section("observation")
    soft = config.section("soft_gate")
    inference = config.section("inference")
    designs = config.section("designs")
    design_name = (
        str(inference["default_design"]) if design is None else design
    )
    settings = designs[design_name]
    burn_in = float(settings["burn_in_hours"])
    recording_duration = float(settings["duration"])
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
        duration=burn_in + recording_duration,
        output_step=float(observation["epoch_hours"]),
    )
    retained = trajectory.time >= burn_in
    retained_time = trajectory.time[retained] - burn_in
    retained_pressure = trajectory.pressure[retained]
    retained_gate = trajectory.gate[retained]
    probabilities = sleep_probabilities(
        retained_gate,
        parameters.misclassification,
    )
    observations = sample_observations(observation_key, probabilities)
    return GeneratedRecording(
        parameters=parameters,
        time=retained_time,
        pressure=retained_pressure,
        gate=retained_gate,
        probabilities=probabilities,
        observations=observations,
    )
