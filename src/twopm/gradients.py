from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from twopm.config import ProjectConfig
from twopm.soft_gate import simulate_soft_gate_from_config


@dataclass(frozen=True)
class GradientCheck:

    automatic: float
    finite_difference: float
    absolute_error: float


def sleep_fraction(gate: jax.Array) -> jax.Array:
    return jnp.mean(gate)


def sleep_fraction_for_chi_sleep(
    config: ProjectConfig,
    chi_sleep: float | jax.Array,
    duration: float | None = None,
) -> jax.Array:
    result = simulate_soft_gate_from_config(
        config,
        chi_sleep=chi_sleep,
        duration=duration,
    )
    return sleep_fraction(result.gate)


def finite_difference_gradient(
    function: Callable[[float], float | jax.Array],
    value: float,
    step: float,
) -> float:
    if step <= 0:
        raise ValueError("finite-difference step must be positive")
    upper = float(function(value + step))
    lower = float(function(value - step))
    return (upper - lower) / (2 * step)


def check_sleep_fraction_gradient(config: ProjectConfig) -> GradientCheck:
    model = config.section("model")
    validation = config.section("validation")
    value = float(model["chi_sleep"])
    step = float(validation["finite_difference_step"])
    duration = float(validation["gradient_duration"])

    summary = lambda parameter: sleep_fraction_for_chi_sleep(
        config,
        parameter,
        duration=duration,
    )
    automatic = float(jax.grad(summary)(jnp.asarray(value)))
    finite_difference = finite_difference_gradient(summary, value, step)
    return GradientCheck(
        automatic=automatic,
        finite_difference=finite_difference,
        absolute_error=abs(automatic - finite_difference),
    )
