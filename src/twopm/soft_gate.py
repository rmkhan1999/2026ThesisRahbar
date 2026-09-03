from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import diffrax
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from twopm.config import ProjectConfig
from twopm.hard_switch import simulate_hard_switch_from_config


class SoftGateResult(NamedTuple):

    time: jax.Array
    pressure: jax.Array
    gate: jax.Array
    solver_steps: jax.Array
    successful: jax.Array


@dataclass(frozen=True)
class ConvergenceResult:

    k_values: NDArray[np.float64]
    mean_absolute_error: NDArray[np.float64]


@dataclass(frozen=True)
class SmoothingCalibrationResult:

    p0_values: NDArray[np.float64]
    k_values: NDArray[np.float64]
    mean_absolute_error: NDArray[np.float64]
    solver_steps: NDArray[np.int64]
    unmatched_reference: NDArray[np.int64]
    unmatched_candidate: NDArray[np.int64]


@dataclass(frozen=True)
class TransitionMatch:

    reference: NDArray[np.float64]
    candidate: NDArray[np.float64]
    absolute_error: NDArray[np.float64]
    unmatched_reference: int
    unmatched_candidate: int


def circadian_coefficients(
    amplitude: float | jax.Array,
    phase: float | jax.Array,
    period: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    angle = 2 * jnp.pi * jnp.asarray(phase) / jnp.asarray(period)
    magnitude = jnp.asarray(amplitude)
    return magnitude * jnp.cos(angle), magnitude * jnp.sin(angle)


def circadian_amplitude_phase(
    c1: float | jax.Array,
    c2: float | jax.Array,
    period: float | jax.Array,
) -> tuple[jax.Array, jax.Array]:
    first = jnp.asarray(c1)
    second = jnp.asarray(c2)
    amplitude = jnp.sqrt(first**2 + second**2)
    phase = (
        jnp.asarray(period)
        * jnp.arctan2(second, first)
        / (2 * jnp.pi)
    ) % jnp.asarray(period)
    return amplitude, phase


def cartesian_circadian_displacement(
    time: float | jax.Array,
    c1: float | jax.Array,
    c2: float | jax.Array,
    period: float | jax.Array,
) -> jax.Array:
    angle = 2 * jnp.pi * jnp.asarray(time) / jnp.asarray(period)
    return jnp.asarray(c1) * jnp.cos(angle) + jnp.asarray(c2) * jnp.sin(
        angle
    )


def gate_offset(p0: float | jax.Array) -> jax.Array:
    probability = jnp.asarray(p0)
    return -jnp.log(1 / probability - 1)


def gate_target(
    signed_distance: float | jax.Array,
    k: float | jax.Array,
    p0: float | jax.Array,
) -> jax.Array:
    return jax.nn.sigmoid(
        jnp.asarray(k) * jnp.asarray(signed_distance) + gate_offset(p0)
    )


def soft_gate_vector_field(
    time: jax.Array,
    state: jax.Array,
    args: Mapping[str, Any],
) -> jax.Array:
    pressure, gate = state
    displacement = cartesian_circadian_displacement(
        time,
        args["c1"],
        args["c2"],
        args["period"],
    )
    upper = args["upper"] + displacement
    lower = args["lower"] + displacement

    sleep_flow = -pressure / args["chi_sleep"]
    wake_flow = (args["mu"] - pressure) / args["chi_wake"]
    pressure_derivative = gate * sleep_flow + (1 - gate) * wake_flow

    signed_distance = (1 - gate) * (pressure - upper) + gate * (
        pressure - lower
    )
    target_gate = gate_target(signed_distance, args["k"], args["p0"])
    gate_derivative = (target_gate - gate) / args["tau_gate"]
    return jnp.stack((pressure_derivative, gate_derivative))


@eqx.filter_jit
def simulate_soft_gate(
    *,
    save_times: jax.Array,
    chi_sleep: float | jax.Array,
    chi_wake: float | jax.Array,
    mu: float | jax.Array,
    upper: float | jax.Array,
    lower: float | jax.Array,
    c1: float | jax.Array,
    c2: float | jax.Array,
    period: float | jax.Array,
    k: float | jax.Array,
    p0: float | jax.Array,
    tau_gate: float | jax.Array,
    initial_pressure: float | jax.Array,
    initial_gate: float | jax.Array,
    dt0: float,
    integration_mode: str,
    fixed_step_size: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    max_steps: int,
    adjoint_mode: str,
    adjoint_checkpoints: int,
    throw: bool = True,
) -> SoftGateResult:
    if integration_mode == "fixed":
        initial_step = fixed_step_size
        step_controller = diffrax.ConstantStepSize()
    elif integration_mode == "adaptive":
        initial_step = dt0
        step_controller = diffrax.PIDController(
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    else:
        raise ValueError("integration_mode must be 'adaptive' or 'fixed'")
    if adjoint_mode == "direct":
        adjoint = diffrax.DirectAdjoint()
    elif adjoint_mode == "recursive_checkpoint":
        adjoint = diffrax.RecursiveCheckpointAdjoint(
            checkpoints=adjoint_checkpoints
        )
    else:
        raise ValueError(
            "adjoint_mode must be 'recursive_checkpoint' or 'direct'"
        )
    vector_field_args = {
        "chi_sleep": chi_sleep,
        "chi_wake": chi_wake,
        "mu": mu,
        "upper": upper,
        "lower": lower,
        "c1": c1,
        "c2": c2,
        "period": period,
        "k": k,
        "p0": p0,
        "tau_gate": tau_gate,
    }
    initial_state = jnp.stack(
        (jnp.asarray(initial_pressure), jnp.asarray(initial_gate))
    )
    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(soft_gate_vector_field),
        solver=diffrax.Tsit5(),
        t0=save_times[0],
        t1=save_times[-1],
        dt0=initial_step,
        y0=initial_state,
        args=vector_field_args,
        saveat=diffrax.SaveAt(ts=save_times),
        stepsize_controller=step_controller,
        adjoint=adjoint,
        max_steps=max_steps,
        throw=throw,
    )
    return SoftGateResult(
        time=save_times,
        pressure=solution.ys[:, 0],
        gate=solution.ys[:, 1],
        solver_steps=solution.stats["num_steps"],
        successful=solution.result == diffrax.RESULTS.successful,
    )


def simulate_soft_gate_from_config(
    config: ProjectConfig,
    *,
    k: float | jax.Array | None = None,
    p0: float | jax.Array | None = None,
    tau_gate: float | jax.Array | None = None,
    chi_sleep: float | jax.Array | None = None,
    chi_wake: float | jax.Array | None = None,
    mu: float | jax.Array | None = None,
    upper: float | jax.Array | None = None,
    lower: float | jax.Array | None = None,
    c1: float | jax.Array | None = None,
    c2: float | jax.Array | None = None,
    initial_pressure: float | jax.Array | None = None,
    initial_gate: float | jax.Array | None = None,
    duration: float | None = None,
    output_step: float | None = None,
    throw: bool = True,
) -> SoftGateResult:
    model = config.section("model")
    hard = config.section("hard_switch")
    soft = config.section("soft_gate")
    initial = config.section("initial_state")
    configured_duration = (
        float(hard["duration"]) if duration is None else duration
    )
    start_time = float(hard["start_time"])
    configured_output_step = (
        float(hard["dt"]) if output_step is None else output_step
    )
    step_count = int(
        np.floor(configured_duration / configured_output_step)
    )
    save_times = jnp.asarray(
        start_time + configured_output_step * np.arange(step_count + 1)
    )

    configured_k = jnp.asarray(
        float(soft["k"]) if k is None else k
    )
    configured_p0 = jnp.asarray(
        float(soft["p0"]) if p0 is None else p0
    )
    configured_tau_gate = jnp.asarray(
        float(soft["tau_gate"]) if tau_gate is None else tau_gate
    )
    configured_chi_sleep = jnp.asarray(
        float(model["chi_sleep"]) if chi_sleep is None else chi_sleep
    )
    configured_chi_wake = jnp.asarray(
        float(model["chi_wake"]) if chi_wake is None else chi_wake
    )
    configured_mu = jnp.asarray(float(model["mu"]) if mu is None else mu)
    configured_upper = jnp.asarray(
        float(model["upper_base"]) if upper is None else upper
    )
    configured_lower = jnp.asarray(
        float(model["lower_base"]) if lower is None else lower
    )
    default_c1, default_c2 = circadian_coefficients(
        float(model["circadian_amplitude"]),
        float(model["phase"]),
        float(model["circadian_period"]),
    )
    if (c1 is None) != (c2 is None):
        raise ValueError("c1 and c2 must be provided together")
    configured_c1 = default_c1 if c1 is None else jnp.asarray(c1)
    configured_c2 = default_c2 if c2 is None else jnp.asarray(c2)
    configured_initial_pressure = jnp.asarray(
        float(initial["pressure"])
        if initial_pressure is None
        else initial_pressure
    )
    configured_initial_gate = jnp.asarray(
        float(initial["soft_gate"]) if initial_gate is None else initial_gate
    )
    return simulate_soft_gate(
        save_times=save_times,
        chi_sleep=configured_chi_sleep,
        chi_wake=configured_chi_wake,
        mu=configured_mu,
        upper=configured_upper,
        lower=configured_lower,
        c1=configured_c1,
        c2=configured_c2,
        period=float(model["circadian_period"]),
        k=configured_k,
        p0=configured_p0,
        tau_gate=configured_tau_gate,
        initial_pressure=configured_initial_pressure,
        initial_gate=configured_initial_gate,
        dt0=float(soft["dt0"]),
        integration_mode=str(soft["integration_mode"]),
        fixed_step_size=float(soft["fixed_step_size"]),
        relative_tolerance=float(soft["relative_tolerance"]),
        absolute_tolerance=float(soft["absolute_tolerance"]),
        max_steps=int(soft["max_steps"]),
        adjoint_mode=str(soft["adjoint_mode"]),
        adjoint_checkpoints=int(soft["adjoint_checkpoints"]),
        throw=throw,
    )


def soft_transition_times(
    time: jax.Array | NDArray[np.float64],
    gate: jax.Array | NDArray[np.float64],
    level: float,
) -> NDArray[np.float64]:
    time_array = np.asarray(time, dtype=np.float64)
    gate_array = np.asarray(gate, dtype=np.float64)
    offsets = gate_array - level
    crossing_indices = np.flatnonzero(offsets[:-1] * offsets[1:] < 0)
    transitions = []

    for index in crossing_indices:
        fraction = -offsets[index] / (
            offsets[index + 1] - offsets[index]
        )
        transitions.append(
            time_array[index]
            + fraction * (time_array[index + 1] - time_array[index])
        )

    return np.asarray(transitions, dtype=np.float64)


def match_transition_times(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
) -> TransitionMatch:
    reference_times = np.asarray(reference, dtype=np.float64)
    candidate_times = np.asarray(candidate, dtype=np.float64)
    if reference_times.ndim != 1 or candidate_times.ndim != 1:
        raise ValueError("transition arrays must be one-dimensional")
    if np.any(np.diff(reference_times) <= 0) or np.any(
        np.diff(candidate_times) <= 0
    ):
        raise ValueError("transition times must be strictly increasing")

    if reference_times.size == 0 or candidate_times.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return TransitionMatch(
            reference=empty,
            candidate=empty,
            absolute_error=empty,
            unmatched_reference=int(reference_times.size),
            unmatched_candidate=int(candidate_times.size),
        )

    cost = np.abs(reference_times[:, None] - candidate_times[None, :])
    reference_indices, candidate_indices = linear_sum_assignment(cost)
    order = np.argsort(reference_indices)
    matched_reference = reference_times[reference_indices[order]]
    matched_candidate = candidate_times[candidate_indices[order]]
    return TransitionMatch(
        reference=matched_reference,
        candidate=matched_candidate,
        absolute_error=np.abs(matched_reference - matched_candidate),
        unmatched_reference=int(
            reference_times.size - matched_reference.size
        ),
        unmatched_candidate=int(
            candidate_times.size - matched_candidate.size
        ),
    )


def smoothing_convergence_study(
    config: ProjectConfig,
) -> ConvergenceResult:
    validation = config.section("validation")
    level = float(validation["transition_gate_level"])
    k_values = np.asarray(
        validation["smoothing_k_values"],
        dtype=np.float64,
    )
    hard_result = simulate_hard_switch_from_config(config)
    errors = []

    for k in k_values:
        soft_result = simulate_soft_gate_from_config(
            config,
            k=float(k),
        )
        soft_times = soft_transition_times(
            soft_result.time,
            soft_result.gate,
            level,
        )
        match = match_transition_times(
            hard_result.switch_times,
            soft_times,
        )
        if match.absolute_error.size == 0:
            errors.append(np.nan)
            continue
        errors.append(float(np.mean(match.absolute_error)))

    return ConvergenceResult(
        k_values=k_values,
        mean_absolute_error=np.asarray(errors, dtype=np.float64),
    )


def calibrate_smoothing_grid(
    config: ProjectConfig,
) -> SmoothingCalibrationResult:
    validation = config.section("validation")
    level = float(validation["transition_gate_level"])
    p0_values = np.asarray(
        validation["smoothing_p0_values"],
        dtype=np.float64,
    )
    k_values = np.asarray(
        validation["smoothing_k_values"],
        dtype=np.float64,
    )
    hard_result = simulate_hard_switch_from_config(config)
    errors = np.empty((p0_values.size, k_values.size), dtype=np.float64)
    solver_steps = np.empty_like(errors, dtype=np.int64)
    unmatched_reference = np.empty_like(errors, dtype=np.int64)
    unmatched_candidate = np.empty_like(errors, dtype=np.int64)

    for p0_index, p0 in enumerate(p0_values):
        for k_index, k in enumerate(k_values):
            soft_result = simulate_soft_gate_from_config(
                config,
                p0=float(p0),
                k=float(k),
            )
            soft_times = soft_transition_times(
                soft_result.time,
                soft_result.gate,
                level,
            )
            match = match_transition_times(
                hard_result.switch_times,
                soft_times,
            )
            errors[p0_index, k_index] = (
                float(np.mean(match.absolute_error))
                if match.absolute_error.size
                else np.nan
            )
            solver_steps[p0_index, k_index] = int(soft_result.solver_steps)
            unmatched_reference[p0_index, k_index] = match.unmatched_reference
            unmatched_candidate[p0_index, k_index] = match.unmatched_candidate

    return SmoothingCalibrationResult(
        p0_values=p0_values,
        k_values=k_values,
        mean_absolute_error=errors,
        solver_steps=solver_steps,
        unmatched_reference=unmatched_reference,
        unmatched_candidate=unmatched_candidate,
    )
