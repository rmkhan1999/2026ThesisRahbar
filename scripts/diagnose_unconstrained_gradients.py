from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import potential_energy

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.inference import (
    constrained_physical_parameters,
    numpyro_model,
    prior_distribution,
    sampled_parameter_names,
)
from twopm.soft_gate import simulate_soft_gate_from_config


REPOSITORY = Path(__file__).resolve().parents[1]
RAW_TAYLOR_STEPS = (1e-4, 1e-5, 1e-6)
UNIT_TAYLOR_STEPS = (1e-5, 1e-6, 1e-7)
COSINE_MIN = 0.9999
TAYLOR_TOL = 0.05
NORM_REL_MAX = 0.05


def _flatten(tree: dict[str, jnp.ndarray]) -> tuple[callable, callable]:
    keys = sorted(tree)
    sizes = [int(np.size(tree[key])) for key in keys]

    def pack(values: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return jnp.concatenate([jnp.ravel(values[key]) for key in keys])

    def unpack(vector: jnp.ndarray) -> dict[str, jnp.ndarray]:
        pieces = []
        offset = 0
        for key, size in zip(keys, sizes):
            pieces.append(
                (key, vector[offset : offset + size].reshape(tree[key].shape))
            )
            offset += size
        return {key: value for key, value in pieces}

    return pack, unpack


def _gate_extremes(parameters, config: ProjectConfig) -> dict[str, float | bool]:
    design = config.section("designs")["entrained"]
    trajectory = simulate_soft_gate_from_config(
        config,
        c1=parameters["c1"],
        c2=parameters["c2"],
        chi_sleep=parameters["chi_sleep"],
        chi_wake=parameters["chi_wake"],
        upper=float(config.section("fixed")["threshold_mean"])
        + 0.5 * parameters["threshold_gap"],
        lower=float(config.section("fixed")["threshold_mean"])
        - 0.5 * parameters["threshold_gap"],
        initial_pressure=float(config.section("initial_state")["pressure"]),
        duration=float(design["burn_in_hours"]) + float(design["duration"]),
        output_step=float(config.section("observation")["epoch_hours"]),
        throw=True,
    )
    gate = np.asarray(trajectory.gate)
    min_g = float(np.min(gate))
    max_g = float(np.max(gate))
    return {
        "min_g": min_g,
        "max_g": max_g,
        "gate_escape": bool(min_g < 0.0 or max_g > 1.0),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _raw_taylor_ratios(
    scalar, flat0: jnp.ndarray, grad: np.ndarray, steps: tuple[float, ...]
) -> list[dict[str, float]]:
    u0 = float(scalar(flat0))
    grad_norm_sq = float(np.dot(grad, grad))
    grad_norm = float(np.sqrt(grad_norm_sq))
    rows = []
    for h in steps:
        observed = float(scalar(flat0 + h * grad) - u0)
        predicted = h * grad_norm_sq
        ratio = observed / predicted if abs(predicted) > 1e-30 else float("nan")
        rows.append(
            {
                "h": float(h),
                "step_length": float(h * grad_norm),
                "observed_delta": observed,
                "predicted_delta": predicted,
                "taylor_ratio": float(ratio),
                "abs_ratio_minus_one": float(abs(ratio - 1.0)),
            }
        )
    return rows


def _unit_taylor_ratios(
    scalar, flat0: jnp.ndarray, grad: np.ndarray, steps: tuple[float, ...]
) -> list[dict[str, float]]:
    u0 = float(scalar(flat0))
    grad_norm = float(np.linalg.norm(grad))
    if grad_norm < 1e-30:
        return [
            {
                "epsilon": float(eps),
                "taylor_ratio": float("nan"),
                "abs_ratio_minus_one": float("nan"),
            }
            for eps in steps
        ]
    uhat = grad / grad_norm
    rows = []
    for eps in steps:
        observed = float(scalar(flat0 + eps * uhat) - u0)
        predicted = eps * grad_norm
        ratio = observed / predicted if abs(predicted) > 1e-30 else float("nan")
        rows.append(
            {
                "epsilon": float(eps),
                "observed_delta": observed,
                "predicted_delta": predicted,
                "taylor_ratio": float(ratio),
                "abs_ratio_minus_one": float(abs(ratio - 1.0)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-step-size", type=float, default=None)
    parser.add_argument("--tau-gate", type=float, default=None)
    parser.add_argument("--adjoint-mode", type=str, default=None)
    parser.add_argument("--adjoint-checkpoints", type=int, default=None)
    parser.add_argument("--duration", type=float, default=48.0)
    parser.add_argument("--burn-in", type=float, default=48.0)
    parser.add_argument("--n-draws", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "docs" / "unconstrained_gradient_audit.json",
    )
    arguments = parser.parse_args()

    base = load_config(REPOSITORY / "config" / "model.yaml")
    data = deepcopy(base.data)
    data["designs"]["entrained"]["duration"] = float(arguments.duration)
    data["designs"]["entrained"]["burn_in_hours"] = float(arguments.burn_in)
    data["observation"]["misclassification"] = 0.01
    data["inference"]["parameters"] = [
        name
        for name in data["inference"]["parameters"]
        if name != "misclassification"
    ]
    if arguments.fixed_step_size is not None:
        data["soft_gate"]["fixed_step_size"] = float(arguments.fixed_step_size)
        horizon = (
            data["designs"]["entrained"]["burn_in_hours"]
            + data["designs"]["entrained"]["duration"]
        )
        data["soft_gate"]["max_steps"] = int(
            max(
                float(data["soft_gate"]["max_steps"]),
                2 * horizon / float(arguments.fixed_step_size),
            )
        )
    if arguments.tau_gate is not None:
        data["soft_gate"]["tau_gate"] = float(arguments.tau_gate)
    if arguments.adjoint_mode is not None:
        data["soft_gate"]["adjoint_mode"] = arguments.adjoint_mode
    if arguments.adjoint_checkpoints is not None:
        data["soft_gate"]["adjoint_checkpoints"] = int(arguments.adjoint_checkpoints)
    config = ProjectConfig(data=data, source=base.source)
    soft = config.section("soft_gate")
    model = config.section("model")
    recovery = config.section("recovery")
    truth = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
        phase=float(recovery["true_phase"]),
    )
    recording = generate_recording(
        jax.random.PRNGKey(int(recovery["seed"])),
        config,
        truth,
        "entrained",
    )
    labels = recording.observations
    model_args = (labels, config, "entrained")

    def potential(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return potential_energy(numpyro_model, model_args, {}, params)

    names = sampled_parameter_names(config, "entrained")
    draw_rows = []
    for index in range(arguments.n_draws):
        keys = jax.random.split(
            jax.random.PRNGKey(arguments.seed_base + index), len(names)
        )
        unconstrained = {}
        constrained = {}
        for key, name in zip(keys, names):
            distribution = prior_distribution(name, config, "entrained")
            value = distribution.sample(key)
            constrained[name] = value
            unconstrained[name] = biject_to(distribution.support).inv(value)
        physical = constrained_physical_parameters(constrained, config)
        gate_stats = _gate_extremes(physical, config)

        pack, unpack = _flatten(unconstrained)
        flat0 = pack(unconstrained)

        def scalar(vector):
            return potential(unpack(vector))

        automatic = np.asarray(jax.grad(scalar)(flat0), dtype=float)
        step = 1e-5
        finite = []
        for coordinate in range(flat0.size):
            offset = jnp.zeros_like(flat0).at[coordinate].set(step)
            finite.append(
                float((scalar(flat0 + offset) - scalar(flat0 - offset)) / (2 * step))
            )
        finite = np.asarray(finite, dtype=float)
        relative = np.abs(automatic - finite) / np.maximum(np.abs(finite), 1e-10)
        worst = int(np.argmax(relative))
        ad_norm = float(np.linalg.norm(automatic))
        fd_norm = float(np.linalg.norm(finite))
        norm_rel = abs(ad_norm - fd_norm) / max(fd_norm, 1e-10)
        cosine = _cosine(automatic, finite)
        taylor_raw = _raw_taylor_ratios(scalar, flat0, automatic, RAW_TAYLOR_STEPS)
        taylor_unit = _unit_taylor_ratios(scalar, flat0, automatic, UNIT_TAYLOR_STEPS)
        best_taylor = min(
            row["abs_ratio_minus_one"]
            for row in taylor_unit
            if np.isfinite(row["abs_ratio_minus_one"])
        )
        pass_cosine = bool(np.isfinite(cosine) and cosine >= COSINE_MIN)
        pass_taylor = bool(best_taylor <= TAYLOR_TOL)
        pass_norm = bool(norm_rel <= NORM_REL_MAX)
        passed = pass_cosine and pass_taylor and pass_norm

        row = {
            "draw": index,
            "seed": arguments.seed_base + index,
            "names": list(names),
            "automatic": automatic.tolist(),
            "finite_difference": finite.tolist(),
            "relative_error_per_coordinate": relative.tolist(),
            "max_relative_error_per_coordinate": float(np.max(relative)),
            "worst_coordinate": names[worst],
            "worst_automatic": float(automatic[worst]),
            "worst_finite_difference": float(finite[worst]),
            "worst_absolute_difference": float(abs(automatic[worst] - finite[worst])),
            "cosine_similarity": cosine,
            "ad_norm": ad_norm,
            "fd_norm": fd_norm,
            "norm_relative_error": float(norm_rel),
            "taylor_raw_h_times_grad": taylor_raw,
            "taylor_unit_direction": taylor_unit,
            "best_abs_unit_taylor_ratio_minus_one": float(best_taylor),
            "pass_cosine": pass_cosine,
            "pass_taylor": pass_taylor,
            "pass_norm": pass_norm,
            "pass": passed,
            **gate_stats,
        }
        draw_rows.append(row)
        print(
            f"draw={index} dt={soft['fixed_step_size']} "
            f"tau_g={soft['tau_gate']} adjoint={soft['adjoint_mode']} "
            f"cosine={cosine:.6f} unit_|r-1|={best_taylor:.3e} "
            f"norm_rel={norm_rel:.3e} pass={passed} "
            f"worst={names[worst]} AD={automatic[worst]:.3e} FD={finite[worst]:.3e} "
            f"gate_escape={gate_stats['gate_escape']}",
            flush=True,
        )

    payload = {
        "retained_hours": 48.0,
        "burn_in_hours": 72.0,
        "fixed_misclassification": 0.01,
        "fixed_step_size": float(soft["fixed_step_size"]),
        "tau_gate": float(soft["tau_gate"]),
        "k": float(soft["k"]),
        "dt_over_tau_gate": float(soft["fixed_step_size"]) / float(soft["tau_gate"]),
        "adjoint_mode": soft["adjoint_mode"],
        "criteria": {
            "cosine_min": COSINE_MIN,
            "unit_taylor_abs_ratio_minus_one_max": TAYLOR_TOL,
            "norm_relative_error_max": NORM_REL_MAX,
            "unit_taylor_steps": list(UNIT_TAYLOR_STEPS),
            "raw_taylor_steps_diagnostic_only": list(RAW_TAYLOR_STEPS),
        },
        "draws": draw_rows,
        "min_cosine_across_draws": min(row["cosine_similarity"] for row in draw_rows),
        "max_best_abs_unit_taylor_ratio_minus_one": max(
            row["best_abs_unit_taylor_ratio_minus_one"] for row in draw_rows
        ),
        "max_norm_relative_error": max(row["norm_relative_error"] for row in draw_rows),
        "all_draws_pass": all(row["pass"] for row in draw_rows),
        "gate_escape_draws": [
            row["draw"] for row in draw_rows if row["gate_escape"]
        ],
        "ad_fd": draw_rows,
        "max_relative_error_across_draws": max(
            row["max_relative_error_per_coordinate"] for row in draw_rows
        ),
    }
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fixed_step_size": payload["fixed_step_size"],
                "tau_gate": payload["tau_gate"],
                "dt_over_tau_gate": payload["dt_over_tau_gate"],
                "min_cosine_across_draws": payload["min_cosine_across_draws"],
                "max_best_abs_unit_taylor_ratio_minus_one": payload[
                    "max_best_abs_unit_taylor_ratio_minus_one"
                ],
                "max_norm_relative_error": payload["max_norm_relative_error"],
                "all_draws_pass": payload["all_draws_pass"],
                "gate_escape_draws": payload["gate_escape_draws"],
            },
            indent=2,
        )
    )
    print(f"Saved {arguments.output}")


if __name__ == "__main__":
    main()
