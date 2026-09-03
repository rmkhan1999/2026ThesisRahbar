from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from twopm.config import ProjectConfig
from twopm.likelihood import likelihood_parameter_names, log_likelihood
from twopm.soft_gate import circadian_amplitude_phase


def free_running_period(
    chi_sleep: float | jax.Array,
    chi_wake: float | jax.Array,
    threshold_gap: float | jax.Array,
    threshold_mean: float | jax.Array,
    mu: float | jax.Array,
) -> jax.Array:
    gap = jnp.asarray(threshold_gap)
    mean = jnp.asarray(threshold_mean)
    upper = mean + gap / 2
    lower = mean - gap / 2
    return jnp.asarray(chi_sleep) * jnp.log(upper / lower) + jnp.asarray(
        chi_wake
    ) * jnp.log((jnp.asarray(mu) - lower) / (jnp.asarray(mu) - upper))


def prior_distribution(
    name: str,
    config: ProjectConfig,
    design: str | None = None,
) -> dist.Distribution:
    inference = config.section("inference")
    designs = config.section("designs")
    priors = config.section("priors")
    design_name = (
        str(inference["default_design"]) if design is None else design
    )

    if name in {"phase_z1", "phase_z2"}:
        settings = priors[name]
        return dist.Normal(
            float(settings["mean"]),
            float(settings["sd"]),
        )
    if name in {"excursion_fraction", "amplitude_fraction"}:
        settings = designs[design_name]["constrained_prior"]
        prefix = "excursion" if name == "excursion_fraction" else "amplitude"
        return dist.Beta(
            float(settings[f"{prefix}_concentration1"]),
            float(settings[f"{prefix}_concentration0"]),
        )
    if name in {"chi_sleep", "chi_wake"}:
        settings = priors[name]
        return dist.LogNormal(
            float(settings["log_mean"]),
            float(settings["log_sd"]),
        )
    if name == "misclassification":
        settings = priors[name]
        base = dist.Beta(
            float(settings["concentration1"]),
            float(settings["concentration0"]),
        )
        return dist.TransformedDistribution(
            base,
            dist.transforms.AffineTransform(
                loc=0.0,
                scale=float(settings["maximum"]),
            ),
        )
    if name == "initial_pressure":
        settings = designs[design_name]["initial_pressure_prior"]
        return dist.Uniform(
            float(settings["lower"]),
            float(settings["upper"]),
        )
    raise KeyError(f"no configured prior for {name!r}")


def sampled_parameter_names(
    config: ProjectConfig,
    design: str | None = None,
) -> tuple[str, ...]:
    inference = config.section("inference")
    designs = config.section("designs")
    design_name = (
        str(inference["default_design"]) if design is None else design
    )
    names = tuple(str(name) for name in inference["parameters"])
    if bool(designs[design_name]["infer_initial_pressure"]):
        names += ("initial_pressure",)
    return names


def constrained_physical_parameters(
    sampled: Mapping[str, jax.Array],
    config: ProjectConfig,
) -> dict[str, jax.Array]:
    fixed = config.section("fixed")
    model = config.section("model")
    mean = jnp.asarray(fixed["threshold_mean"])
    z1 = jnp.asarray(sampled["phase_z1"])
    z2 = jnp.asarray(sampled["phase_z2"])
    radius = jnp.hypot(z1, z2)
    safe_radius = jnp.where(radius > 0, radius, jnp.asarray(1.0))
    excursion = mean * jnp.asarray(sampled["excursion_fraction"])
    amplitude_fraction = jnp.asarray(sampled["amplitude_fraction"])
    amplitude = amplitude_fraction * excursion
    threshold_gap = 2 * (1 - amplitude_fraction) * excursion
    c1 = amplitude * z1 / safe_radius
    c2 = amplitude * z2 / safe_radius
    c1 = jnp.where(radius > 0, c1, amplitude)
    c2 = jnp.where(radius > 0, c2, jnp.asarray(0.0))
    amplitude, phase = circadian_amplitude_phase(
        c1,
        c2,
        float(model["circadian_period"]),
    )
    return {
        **{name: jnp.asarray(value) for name, value in sampled.items()},
        "c1": c1,
        "c2": c2,
        "amplitude": amplitude,
        "phase": phase,
        "threshold_gap": threshold_gap,
        "total_excursion": excursion,
        "threshold_margin": (mean - excursion) / mean,
    }


def numpyro_model(
    observed_labels: jax.Array | Sequence[int],
    config: ProjectConfig,
    design: str | None = None,
) -> None:
    sampled: dict[str, jax.Array] = {}
    for name in sampled_parameter_names(config, design):
        sampled[name] = numpyro.sample(
            name,
            prior_distribution(name, config, design),
        )
    if "misclassification" not in sampled:
        fixed_error = jnp.asarray(
            float(config.section("observation")["misclassification"])
        )
        sampled["misclassification"] = fixed_error
        numpyro.deterministic("misclassification", fixed_error)
    values = constrained_physical_parameters(sampled, config)

    model = config.section("model")
    fixed = config.section("fixed")
    tau = free_running_period(
        values["chi_sleep"],
        values["chi_wake"],
        values["threshold_gap"],
        float(fixed["threshold_mean"]),
        float(model["mu"]),
    )
    numpyro.deterministic("c1", values["c1"])
    numpyro.deterministic("c2", values["c2"])
    numpyro.deterministic("amplitude", values["amplitude"])
    numpyro.deterministic("phase", values["phase"])
    numpyro.deterministic("threshold_gap", values["threshold_gap"])
    numpyro.deterministic("total_excursion", values["total_excursion"])
    numpyro.deterministic("threshold_margin", values["threshold_margin"])
    numpyro.deterministic("tau", tau)

    parameter_vector = parameter_mapping_to_vector(values, config, design)
    numpyro.factor(
        "obs_log_likelihood",
        log_likelihood(parameter_vector, observed_labels, config, design),
    )


def parameter_mapping_to_vector(
    parameters: Mapping[str, jax.Array],
    config: ProjectConfig,
    design: str | None = None,
) -> jax.Array:
    return jnp.stack(
        [
            jnp.asarray(parameters[name])
            for name in likelihood_parameter_names(config, design)
        ]
    )
