from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.distributions.transforms import biject_to
from numpyro.infer import NUTS
from numpyro.infer.initialization import init_to_median
from numpyro.infer.util import potential_energy

from twopm.config import ProjectConfig
from twopm.inference import (
    constrained_physical_parameters,
    free_running_period,
    numpyro_model,
    prior_distribution,
    sampled_parameter_names,
)


DEFAULT_DIAGNOSTIC_PARAMETERS = (
    "phase_z1",
    "phase_z2",
    "excursion_fraction",
    "amplitude_fraction",
    "chi_sleep",
    "chi_wake",
    "misclassification",
    "tau",
)


@dataclass(frozen=True)
class SamplerResult:

    posterior: dict[str, np.ndarray]
    sample_stats: dict[str, np.ndarray]
    timings: dict[str, Any]
    diagnostics: dict[str, Any]
    inference_data: az.InferenceData | None
    adapt_states: list[dict[str, Any]] | None = None


def _block_tree(tree: Any) -> Any:
    return jax.tree.map(lambda value: jax.block_until_ready(value), tree)


def _physical_sample(
    unconstrained: dict[str, jax.Array],
    config: ProjectConfig,
    design: str | None,
) -> dict[str, jax.Array]:
    names = sampled_parameter_names(config, design)
    sampled = {
        name: biject_to(prior_distribution(name, config, design).support)(
            unconstrained[name]
        )
        for name in names
    }
    if "misclassification" not in sampled:
        sampled["misclassification"] = jnp.asarray(
            float(config.section("observation")["misclassification"])
        )
    values = constrained_physical_parameters(sampled, config)
    model = config.section("model")
    fixed = config.section("fixed")
    values["tau"] = free_running_period(
        values["chi_sleep"],
        values["chi_wake"],
        values["threshold_gap"],
        float(fixed["threshold_mean"]),
        float(model["mu"]),
    )
    return values


def _finite_extreme(values: np.ndarray, operation: str) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return None
    if operation == "max":
        return float(np.max(finite))
    return float(np.min(finite))


def _circular_summary(phases: np.ndarray, period: float) -> dict[str, float]:
    angles = 2 * np.pi * np.asarray(phases) / period
    resultant = np.mean(np.exp(1j * angles))
    mean = float((np.angle(resultant) * period / (2 * np.pi)) % period)
    magnitude = max(float(abs(resultant)), np.finfo(float).tiny)
    standard_deviation = float(
        math.sqrt(max(0.0, -2 * math.log(magnitude))) * period / (2 * np.pi)
    )
    return {"circular_mean_hours": mean, "circular_sd_hours": standard_deviation}


def _tree_depth(num_steps: int) -> int:
    return int(math.ceil(math.log2(num_steps + 1)))


def _open_iteration_log(path: Path | None) -> TextIO | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=(
            "chain",
            "phase",
            "iteration",
            "step_size",
            "tree_depth",
            "diverging",
            "accept_prob",
            "num_steps",
            "seconds",
        ),
    )
    writer.writeheader()
    handle.flush()
    setattr(handle, "_twopm_writer", writer)
    return handle


def _write_iteration(
    handle: TextIO | None,
    *,
    chain: int,
    phase: str,
    iteration: int,
    step_size: float,
    tree_depth: int,
    diverging: bool,
    num_steps: int,
    seconds: float,
    accept_prob: float | None = None,
) -> None:
    if handle is None:
        return
    writer = getattr(handle, "_twopm_writer")
    writer.writerow(
        {
            "chain": chain,
            "phase": phase,
            "iteration": iteration,
            "step_size": f"{step_size:.16e}",
            "tree_depth": tree_depth,
            "diverging": int(diverging),
            "accept_prob": (
                "" if accept_prob is None else f"{float(accept_prob):.8f}"
            ),
            "num_steps": num_steps,
            "seconds": f"{seconds:.6f}",
        }
    )
    handle.flush()


def _time_potential_gradients(
    *,
    labels: jax.Array,
    config: ProjectConfig,
    design: str | None,
    unconstrained: dict[str, jax.Array],
    repeats: int = 10,
) -> dict[str, float]:
    model_args = (labels, config, design)

    def potential(parameters: dict[str, jax.Array]) -> jax.Array:
        return potential_energy(numpyro_model, model_args, {}, parameters)

    value_and_grad = jax.jit(jax.value_and_grad(potential))
    compile_start = time.perf_counter()
    first_value, first_grad = value_and_grad(unconstrained)
    _block_tree((first_value, first_grad))
    compile_seconds = time.perf_counter() - compile_start
    reps: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        value, gradient = value_and_grad(unconstrained)
        _block_tree((value, gradient))
        reps.append(time.perf_counter() - start)
    array = np.asarray(reps, dtype=float)
    return {
        "gradient_compile_seconds": compile_seconds,
        "gradient_median_seconds": float(np.median(array)),
        "gradient_mean_seconds": float(np.mean(array)),
        "gradient_min_seconds": float(np.min(array)),
        "gradient_max_seconds": float(np.max(array)),
        "gradient_repeats": float(repeats),
    }


def _progress_line(
    *,
    chain_index: int,
    phase: str,
    iteration: int,
    total: int,
    step_size: float,
    mean_tree_depth: float,
    divergences_so_far: int,
    seconds_per_iteration: float,
) -> str:
    return (
        f"chain={chain_index} phase={phase} "
        f"iteration={iteration}/{total} "
        f"step_size={step_size:.6e} "
        f"mean_tree_depth={mean_tree_depth:.2f} "
        f"divergences_so_far={divergences_so_far} "
        f"seconds_per_iteration={seconds_per_iteration:.3f}"
    )


def _summarize_diagnostics(
    inference_data: az.InferenceData | None,
    posterior: dict[str, np.ndarray],
    sample_stats: dict[str, np.ndarray],
    step_sizes: list[float],
    max_tree_depth: int,
    period: float,
    diagnostic_variables: tuple[str, ...],
) -> dict[str, Any]:
    total = int(sample_stats["diverging"].size) if sample_stats["diverging"].size else 0
    divergences = int(np.sum(sample_stats["diverging"])) if total else 0
    saturation = None
    median_steps = None
    if total:
        saturation_threshold = 2**max_tree_depth - 1
        saturation = float(
            np.mean(sample_stats["n_steps"] >= saturation_threshold)
        )
        median_steps = float(np.median(sample_stats["n_steps"]))
    diagnostics: dict[str, Any] = {
        "rhat_max": None,
        "ess_bulk_min": None,
        "ess_tail_min": None,
        "divergences": divergences,
        "total_draws": total,
        "tree_depth_saturation_fraction": saturation,
        "bfmi_by_chain": None,
        "adapted_step_size_by_chain": step_sizes,
        "median_num_steps": median_steps,
        "diagnostic_variables": list(diagnostic_variables),
        "phase_diagnostics_excluded": True,
    }
    if inference_data is not None and total > 0 and "phase" in posterior:
        variables = [name for name in diagnostic_variables if name in posterior]
        if variables:
            rhat = az.rhat(inference_data, var_names=variables).to_array().values
            ess_bulk = (
                az.ess(inference_data, var_names=variables, method="bulk")
                .to_array()
                .values
            )
            ess_tail = (
                az.ess(inference_data, var_names=variables, method="tail")
                .to_array()
                .values
            )
            diagnostics["rhat_max"] = _finite_extreme(rhat, "max")
            diagnostics["ess_bulk_min"] = _finite_extreme(ess_bulk, "min")
            diagnostics["ess_tail_min"] = _finite_extreme(ess_tail, "min")
            diagnostics["bfmi_by_chain"] = [
                float(value) for value in az.bfmi(inference_data)
            ]
        diagnostics["phase"] = _circular_summary(posterior["phase"], period)
    return diagnostics


def run_sequential_nuts(
    *,
    labels: jax.Array,
    config: ProjectConfig,
    seed: int,
    chains: int,
    warmup: int,
    draws: int,
    target_accept_prob: float,
    max_tree_depth: int = 10,
    dense_mass: bool = True,
    design: str | None = None,
    progress_every: int = 10,
    iteration_log_path: Path | None = None,
) -> SamplerResult:
    if warmup < 0 or draws < 0:
        raise ValueError("warmup and draws must be non-negative")
    chain_keys = jax.random.split(jax.random.PRNGKey(seed), chains)
    posterior_chains: list[dict[str, list[np.ndarray]]] = []
    statistic_chains: list[dict[str, list[float | bool]]] = []
    step_sizes: list[float] = []
    adapt_states: list[dict[str, Any]] = []
    compile_times: list[float] = []
    warmup_times: list[float] = []
    sampling_times: list[float] = []
    gradient_timing: dict[str, float] | None = None
    sampled_names = sampled_parameter_names(config, design)
    diagnostic_variables = tuple(
        name
        for name in DEFAULT_DIAGNOSTIC_PARAMETERS
        if name == "tau" or name in sampled_names
    )
    iteration_log = _open_iteration_log(iteration_log_path)

    try:
        for chain_index, chain_key in enumerate(chain_keys, 1):
            kernel = NUTS(
                numpyro_model,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                dense_mass=dense_mass,
                init_strategy=init_to_median(),
            )
            model_args = (labels, config, design)
            compile_start = time.perf_counter()
            state = kernel.init(
                chain_key,
                max(warmup, 1),
                model_args=model_args,
                model_kwargs={},
            )

            def transition(current_state):
                return kernel.sample(current_state, model_args, {})

            compiled_transition = jax.jit(transition).lower(state).compile()
            _block_tree(state)
            compile_times.append(time.perf_counter() - compile_start)
            print(
                f"chain={chain_index} phase=compile "
                f"seconds={compile_times[-1]:.3f} dense_mass={dense_mass}",
                flush=True,
            )

            if gradient_timing is None:
                gradient_timing = _time_potential_gradients(
                    labels=labels,
                    config=config,
                    design=design,
                    unconstrained=state.z,
                )
                print(
                    "gradient_timing "
                    f"median_seconds={gradient_timing['gradient_median_seconds']:.6f} "
                    f"compile_seconds={gradient_timing['gradient_compile_seconds']:.3f}",
                    flush=True,
                )

            warmup_start = time.perf_counter()
            divergences_so_far = 0
            recent_depths: list[int] = []
            window_seconds: list[float] = []
            for iteration in range(1, warmup + 1):
                iter_start = time.perf_counter()
                state = compiled_transition(state)
                _block_tree(state)
                seconds = time.perf_counter() - iter_start
                steps = int(state.num_steps)
                depth = _tree_depth(steps)
                diverging = bool(state.diverging)
                step_size = float(state.adapt_state.step_size)
                recent_depths.append(depth)
                window_seconds.append(seconds)
                if len(recent_depths) > progress_every:
                    recent_depths.pop(0)
                    window_seconds.pop(0)
                if diverging:
                    divergences_so_far += 1
                _write_iteration(
                    iteration_log,
                    chain=chain_index,
                    phase="warmup",
                    iteration=iteration,
                    step_size=step_size,
                    tree_depth=depth,
                    diverging=diverging,
                    num_steps=steps,
                    seconds=seconds,
                    accept_prob=float(state.accept_prob),
                )
                if (
                    iteration == 1
                    or iteration % progress_every == 0
                    or iteration == warmup
                ):
                    print(
                        _progress_line(
                            chain_index=chain_index,
                            phase="warmup",
                            iteration=iteration,
                            total=warmup,
                            step_size=step_size,
                            mean_tree_depth=float(np.mean(recent_depths)),
                            divergences_so_far=divergences_so_far,
                            seconds_per_iteration=float(np.mean(window_seconds)),
                        ),
                        flush=True,
                    )
            warmup_times.append(time.perf_counter() - warmup_start)
            step_sizes.append(float(state.adapt_state.step_size))

            chain_posterior: dict[str, list[np.ndarray]] = {}
            chain_statistics: dict[str, list[float | bool]] = {
                "diverging": [],
                "energy": [],
                "n_steps": [],
                "acceptance_rate": [],
                "tree_depth": [],
            }
            sampling_start = time.perf_counter()
            divergences_so_far = 0
            recent_depths = []
            window_seconds = []
            for iteration in range(1, draws + 1):
                iter_start = time.perf_counter()
                state = compiled_transition(state)
                _block_tree(state)
                seconds = time.perf_counter() - iter_start
                physical = _physical_sample(state.z, config, design)
                for name, value in physical.items():
                    chain_posterior.setdefault(name, []).append(np.asarray(value))
                steps = int(state.num_steps)
                depth = _tree_depth(steps)
                diverging = bool(state.diverging)
                step_size = float(state.adapt_state.step_size)
                recent_depths.append(depth)
                window_seconds.append(seconds)
                if len(recent_depths) > progress_every:
                    recent_depths.pop(0)
                    window_seconds.pop(0)
                if diverging:
                    divergences_so_far += 1
                chain_statistics["diverging"].append(diverging)
                chain_statistics["energy"].append(float(state.energy))
                chain_statistics["n_steps"].append(steps)
                chain_statistics["acceptance_rate"].append(float(state.accept_prob))
                chain_statistics["tree_depth"].append(depth)
                _write_iteration(
                    iteration_log,
                    chain=chain_index,
                    phase="sampling",
                    iteration=iteration,
                    step_size=step_size,
                    tree_depth=depth,
                    diverging=diverging,
                    num_steps=steps,
                    seconds=seconds,
                    accept_prob=float(state.accept_prob),
                )
                if (
                    iteration == 1
                    or iteration % progress_every == 0
                    or iteration == draws
                ):
                    print(
                        _progress_line(
                            chain_index=chain_index,
                            phase="sampling",
                            iteration=iteration,
                            total=draws,
                            step_size=step_size,
                            mean_tree_depth=float(np.mean(recent_depths)),
                            divergences_so_far=divergences_so_far,
                            seconds_per_iteration=float(np.mean(window_seconds)),
                        ),
                        flush=True,
                    )
            sampling_times.append(time.perf_counter() - sampling_start)
            posterior_chains.append(chain_posterior)
            statistic_chains.append(chain_statistics)
            adapt = state.adapt_state
            adapt_states.append(
                {
                    "step_size": float(adapt.step_size),
                    "inverse_mass_matrix": np.asarray(adapt.inverse_mass_matrix),
                    "mass_matrix_sqrt": np.asarray(adapt.mass_matrix_sqrt),
                    "mass_matrix_sqrt_inv": np.asarray(adapt.mass_matrix_sqrt_inv),
                    "z": {key: np.asarray(value) for key, value in state.z.items()},
                    "sampled_parameter_names": list(sampled_names),
                    "dense_mass": bool(dense_mass),
                    "max_tree_depth": int(max_tree_depth),
                }
            )
    finally:
        if iteration_log is not None:
            iteration_log.close()

    if draws == 0:
        posterior = {}
        sample_stats = {
            "diverging": np.zeros((chains, 0), dtype=bool),
            "energy": np.zeros((chains, 0), dtype=float),
            "n_steps": np.zeros((chains, 0), dtype=int),
            "acceptance_rate": np.zeros((chains, 0), dtype=float),
            "tree_depth": np.zeros((chains, 0), dtype=int),
        }
        inference_data = None
    else:
        posterior = {
            name: np.asarray(
                [np.stack(chain[name], axis=0) for chain in posterior_chains]
            )
            for name in posterior_chains[0]
        }
        sample_stats = {
            name: np.asarray([chain[name] for chain in statistic_chains])
            for name in statistic_chains[0]
        }
        inference_data = az.from_dict(
            posterior=posterior,
            sample_stats=sample_stats,
        )
    timings: dict[str, Any] = {
        "compile_seconds_by_chain": compile_times,
        "warmup_seconds_by_chain": warmup_times,
        "sampling_seconds_by_chain": sampling_times,
        "compile_seconds": float(sum(compile_times)),
        "warmup_seconds": float(sum(warmup_times)),
        "sampling_seconds": float(sum(sampling_times)),
    }
    if gradient_timing is not None:
        timings.update(gradient_timing)
    diagnostics = _summarize_diagnostics(
        inference_data,
        posterior,
        sample_stats,
        step_sizes,
        max_tree_depth,
        float(config.section("model")["circadian_period"]),
        diagnostic_variables,
    )
    return SamplerResult(
        posterior=posterior,
        sample_stats=sample_stats,
        timings=timings,
        diagnostics=diagnostics,
        inference_data=inference_data,
        adapt_states=adapt_states,
    )


def effective_config_hash(config: ProjectConfig) -> str:
    encoded = json.dumps(
        config.data,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_provenance(repository: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "working_tree_dirty": bool(status.strip())}
