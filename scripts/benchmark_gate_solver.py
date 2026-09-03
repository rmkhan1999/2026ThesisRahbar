from copy import deepcopy
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from twopm.config import ProjectConfig, load_config
from twopm.hard_switch import simulate_hard_switch_from_config
from twopm.soft_gate import (
    match_transition_times,
    simulate_soft_gate_from_config,
    soft_transition_times,
)


OUTPUT = Path("docs/gate_solver_benchmark.json")


def _configured(base: ProjectConfig, k: float, tau_gate: float) -> ProjectConfig:
    data = deepcopy(base.data)
    data["soft_gate"]["k"] = k
    data["soft_gate"]["tau_gate"] = tau_gate
    return ProjectConfig(data=data, source=base.source)


def _gradient_metrics(config: ProjectConfig) -> dict[str, float]:
    model = config.section("model")
    validation = config.section("validation")
    duration = float(validation["gradient_duration"])

    def sleep_fraction(chi_sleep):
        result = simulate_soft_gate_from_config(
            config,
            chi_sleep=chi_sleep,
            duration=duration,
        )
        return jnp.mean(result.gate)

    gradient = jax.jit(jax.grad(sleep_fraction))
    point = jnp.asarray(model["chi_sleep"])
    start = time.perf_counter()
    automatic = gradient(point)
    jax.block_until_ready(automatic)
    compile_and_first = time.perf_counter() - start
    start = time.perf_counter()
    automatic = gradient(point)
    jax.block_until_ready(automatic)
    evaluation = time.perf_counter() - start
    step = float(validation["finite_difference_step"])
    finite = (sleep_fraction(point + step) - sleep_fraction(point - step)) / (
        2 * step
    )
    jax.block_until_ready(finite)
    return {
        "compile_and_first_gradient_seconds": compile_and_first,
        "gradient_evaluation_seconds": evaluation,
        "automatic_gradient": float(automatic),
        "finite_difference_gradient": float(finite),
        "absolute_gradient_error": float(abs(automatic - finite)),
    }


def _transition_metrics(config: ProjectConfig) -> dict[str, float | int]:
    validation = config.section("validation")
    level = float(validation["transition_gate_level"])
    hard = simulate_hard_switch_from_config(config)
    soft = simulate_soft_gate_from_config(config)
    transitions = soft_transition_times(soft.time, soft.gate, level)
    match = match_transition_times(hard.switch_times, transitions)
    return {
        "transition_mae_hours": float(np.mean(match.absolute_error)),
        "unmatched_hard": match.unmatched_reference,
        "unmatched_soft": match.unmatched_candidate,
        "solver_steps": int(soft.solver_steps),
    }


def _flat_metrics(config: ProjectConfig) -> dict[str, float | int]:
    level = float(config.section("validation")["transition_gate_level"])
    duration = 240.0
    hard = simulate_hard_switch_from_config(
        config,
        duration=duration,
        amplitude=0.0,
    )
    soft = simulate_soft_gate_from_config(
        config,
        c1=0.0,
        c2=0.0,
        duration=duration,
        output_step=0.5,
    )
    hard_stride = int(round(0.5 / float(config.section("hard_switch")["dt"])))
    hard_labels = hard.asleep[::hard_stride]
    soft_labels = np.asarray(soft.gate) >= level
    transitions = soft_transition_times(soft.time, soft.gate, level)
    periods = transitions[2:] - transitions[:-2]
    return {
        "flat_label_agreement": float(np.mean(hard_labels == soft_labels)),
        "flat_period_hours": float(np.median(periods)),
        "flat_transition_count": int(transitions.size),
        "solver_steps": int(soft.solver_steps),
    }


def main() -> None:
    base = load_config("config/model.yaml")
    output = {
        "integration_mode": base.section("soft_gate")["integration_mode"],
        "fixed_step_size": base.section("soft_gate")["fixed_step_size"],
        "variants": [],
    }
    for tau_gate in (0.05, 0.5):
        for k in (5500.0, 2000.0, 650.0):
            config = _configured(base, k, tau_gate)
            result = {
                "k": k,
                "tau_gate": tau_gate,
                **_gradient_metrics(config),
                **{
                    f"forced_{name}": value
                    for name, value in _transition_metrics(config).items()
                },
                **{
                    f"global_{name}": value
                    for name, value in _flat_metrics(config).items()
                },
            }
            output["variants"].append(result)
            print(json.dumps(result, indent=2), flush=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
