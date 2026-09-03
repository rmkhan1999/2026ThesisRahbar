from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from twopm.config import load_config
from twopm.generative import generate_recording, sample_parameters, standard_parameters
from twopm.inference import prior_distribution, sampled_parameter_names
from twopm.likelihood import likelihood_parameter_names, log_likelihood
from twopm.soft_gate import simulate_soft_gate_from_config


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "docs" / "run004_diagnosis.json"
N_DRAWS = 500
SEED = 20260728


def _physical_vector(parameters, names: tuple[str, ...]) -> jnp.ndarray:
    values = {
        "c1": parameters.c1,
        "c2": parameters.c2,
        "chi_sleep": parameters.chi_sleep,
        "chi_wake": parameters.chi_wake,
        "threshold_gap": parameters.upper - parameters.lower,
        "misclassification": parameters.misclassification,
        "initial_pressure": parameters.initial_pressure,
    }
    return jnp.asarray(tuple(float(values[name]) for name in names))


def _log_prior(parameters, config) -> float:
    sampled = {
        "phase_z1": parameters.phase_z1,
        "phase_z2": parameters.phase_z2,
        "excursion_fraction": parameters.excursion_fraction,
        "amplitude_fraction": parameters.amplitude_fraction,
        "chi_sleep": parameters.chi_sleep,
        "chi_wake": parameters.chi_wake,
        "misclassification": parameters.misclassification,
    }
    total = 0.0
    for name in sampled_parameter_names(config):
        total += float(prior_distribution(name, config).log_prob(sampled[name]))
    return total


def _parameter_record(parameters, names: tuple[str, ...]) -> dict[str, float]:
    vector = _physical_vector(parameters, names)
    record = {name: float(value) for name, value in zip(names, vector)}
    record.update(
        {
            "phase_z1": float(parameters.phase_z1),
            "phase_z2": float(parameters.phase_z2),
            "excursion_fraction": float(parameters.excursion_fraction),
            "amplitude_fraction": float(parameters.amplitude_fraction),
            "amplitude": float(parameters.amplitude),
            "phase": float(parameters.phase),
            "upper": float(parameters.upper),
            "lower": float(parameters.lower),
        }
    )
    return record


def main() -> None:
    config = load_config(REPOSITORY / "config" / "model.yaml")
    design = "entrained"
    names = likelihood_parameter_names(config, design)
    recovery = config.section("recovery")
    model = config.section("model")
    design_settings = config.section("designs")[design]
    burn_in = float(design_settings["burn_in_hours"])
    duration = float(design_settings["duration"])
    epoch = float(config.section("observation")["epoch_hours"])

    truth = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
        phase=float(recovery["true_phase"]),
    )
    recording = generate_recording(
        jax.random.PRNGKey(int(recovery["seed"])),
        config,
        truth,
        design,
    )
    labels = recording.observations

    def objective(vector):
        return log_likelihood(vector, labels, config, design)

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    keys = jax.random.split(jax.random.PRNGKey(SEED), N_DRAWS)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for index, key in enumerate(keys):
        parameters = sample_parameters(key, config, design)
        trajectory = simulate_soft_gate_from_config(
            config,
            c1=parameters.c1,
            c2=parameters.c2,
            chi_sleep=parameters.chi_sleep,
            chi_wake=parameters.chi_wake,
            upper=parameters.upper,
            lower=parameters.lower,
            initial_pressure=parameters.initial_pressure,
            duration=burn_in + duration,
            output_step=epoch,
            throw=True,
        )
        gate = np.asarray(trajectory.gate)
        min_g = float(np.min(gate))
        max_g = float(np.max(gate))
        gate_escape = bool(min_g < 0.0 or max_g > 1.0)

        vector = _physical_vector(parameters, names)
        raised = None
        try:
            log_like, gradient = value_and_grad(vector)
            log_like_value = float(log_like)
            gradient_np = np.asarray(gradient, dtype=float)
        except Exception as error:  # noqa: BLE001 — diagnostic must record any trip
            raised = f"{type(error).__name__}: {error}"
            log_like_value = float("nan")
            gradient_np = np.full(len(names), np.nan)
        log_prior = _log_prior(parameters, config)
        log_joint = log_prior + log_like_value

        nonfinite_log_like = not np.isfinite(log_like_value)
        nonfinite_log_joint = not np.isfinite(log_joint)
        nonfinite_grad_components = [
            name
            for name, value in zip(names, gradient_np)
            if not np.isfinite(value)
        ]
        nonfinite_gradient = bool(nonfinite_grad_components)
        failure = (
            gate_escape
            or nonfinite_log_joint
            or nonfinite_gradient
            or raised is not None
        )

        row = {
            "index": index,
            "min_g": min_g,
            "max_g": max_g,
            "gate_escape": gate_escape,
            "solver_successful": True,
            "raised_exception": raised,
            "log_likelihood": log_like_value,
            "log_prior": log_prior,
            "log_joint": log_joint,
            "nonfinite_log_likelihood": nonfinite_log_like,
            "nonfinite_log_joint": nonfinite_log_joint,
            "nonfinite_gradient": nonfinite_gradient,
            "nonfinite_gradient_components": nonfinite_grad_components,
            "gradient": {name: float(value) for name, value in zip(names, gradient_np)},
        }
        rows.append(row)
        if failure:
            failures.append({**row, "parameters": _parameter_record(parameters, names)})

        if (index + 1) % 50 == 0:
            print(f"completed {index + 1}/{N_DRAWS}", flush=True)

    gate_escape = np.asarray([row["gate_escape"] for row in rows])
    nonfinite_joint = np.asarray([row["nonfinite_log_joint"] for row in rows])
    nonfinite_grad = np.asarray([row["nonfinite_gradient"] for row in rows])
    raised = np.asarray([row["raised_exception"] is not None for row in rows])
    nonfinite_either = nonfinite_joint | nonfinite_grad | raised

    cross = {
        "gate_escape_and_nonfinite": int(np.sum(gate_escape & nonfinite_either)),
        "gate_escape_and_finite": int(np.sum(gate_escape & ~nonfinite_either)),
        "no_escape_and_nonfinite": int(np.sum(~gate_escape & nonfinite_either)),
        "no_escape_and_finite": int(np.sum(~gate_escape & ~nonfinite_either)),
    }
    component_counts = {
        name: int(
            sum(
                name in row["nonfinite_gradient_components"]
                for row in rows
            )
        )
        for name in names
    }

    payload = {
        "n_draws": N_DRAWS,
        "seed": SEED,
        "design": design,
        "retained_hours": duration,
        "burn_in_hours": burn_in,
        "k": float(config.section("soft_gate")["k"]),
        "fixed_step_size": float(config.section("soft_gate")["fixed_step_size"]),
        "dtype": str(jnp.zeros(1).dtype),
        "throw_on_solver_failure": True,
        "likelihood_parameter_names": list(names),
        "fraction_gate_escape": float(np.mean(gate_escape)),
        "fraction_nonfinite_log_joint": float(np.mean(nonfinite_joint)),
        "fraction_nonfinite_log_likelihood": float(
            np.mean([row["nonfinite_log_likelihood"] for row in rows])
        ),
        "fraction_nonfinite_gradient": float(np.mean(nonfinite_grad)),
        "fraction_raised_exception": float(np.mean(raised)),
        "cross_tabulation_gate_escape_vs_nonfinite": cross,
        "nonfinite_gradient_component_counts": component_counts,
        "min_g_summary": {
            "min": float(np.min([row["min_g"] for row in rows])),
            "median": float(np.median([row["min_g"] for row in rows])),
            "max": float(np.max([row["min_g"] for row in rows])),
        },
        "max_g_summary": {
            "min": float(np.min([row["max_g"] for row in rows])),
            "median": float(np.median([row["max_g"] for row in rows])),
            "max": float(np.max([row["max_g"] for row in rows])),
        },
        "n_failures": len(failures),
        "failures": failures,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "dtype",
        "fraction_gate_escape",
        "fraction_nonfinite_log_joint",
        "fraction_nonfinite_gradient",
        "fraction_raised_exception",
        "cross_tabulation_gate_escape_vs_nonfinite",
        "n_failures",
    )}, indent=2))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
