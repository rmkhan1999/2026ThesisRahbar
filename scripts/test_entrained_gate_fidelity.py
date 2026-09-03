from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from twopm.config import ProjectConfig, load_config
from twopm.hard_switch import simulate_hard_switch_from_config
from twopm.soft_gate import (
    match_transition_times,
    simulate_soft_gate_from_config,
    soft_transition_times,
)


OUTPUT = Path("docs/entrained_gate_fidelity.json")


def _config_with_k(base: ProjectConfig, k: float) -> ProjectConfig:
    data = deepcopy(base.data)
    data["soft_gate"]["k"] = k
    return ProjectConfig(data=data, source=base.source)


def evaluate(config: ProjectConfig, duration: float) -> dict:
    validation = config.section("validation")
    observation = config.section("observation")
    soft_settings = config.section("soft_gate")
    model = config.section("model")
    level = float(validation["transition_gate_level"])
    epoch = float(observation["epoch_hours"])
    hard_dt = float(config.section("hard_switch")["dt"])

    hard = simulate_hard_switch_from_config(
        config,
        duration=duration,
        amplitude=float(model["circadian_amplitude"]),
    )
    soft = simulate_soft_gate_from_config(
        config,
        duration=duration,
        output_step=epoch,
    )
    soft_times = soft_transition_times(soft.time, soft.gate, level)
    match = match_transition_times(hard.switch_times, soft_times)

    stride = int(round(epoch / hard_dt))
    hard_labels = np.asarray(hard.asleep[::stride], dtype=bool)
    soft_labels = np.asarray(soft.gate) >= level
    n = min(hard_labels.size, soft_labels.size)
    hard_labels = hard_labels[:n]
    soft_labels = soft_labels[:n]

    onset = hard.switch_times[hard.switch_states]
    if onset.size >= 2:
        forced_period = float(np.median(np.diff(onset)))
    else:
        forced_period = float("nan")

    return {
        "k": float(soft_settings["k"]),
        "tau_gate": float(soft_settings["tau_gate"]),
        "duration_hours": duration,
        "fixed_step_size": float(soft_settings["fixed_step_size"]),
        "circadian_amplitude": float(model["circadian_amplitude"]),
        "forced_transition_mae_hours": float(np.mean(match.absolute_error)),
        "forced_transition_max_hours": float(np.max(match.absolute_error)),
        "matched_transitions": int(match.absolute_error.size),
        "unmatched_hard": int(match.unmatched_reference),
        "unmatched_soft": int(match.unmatched_candidate),
        "epoch_label_agreement": float(np.mean(hard_labels == soft_labels)),
        "soft_sleep_fraction": float(np.mean(soft_labels)),
        "hard_sleep_fraction": float(np.mean(hard_labels)),
        "hard_forced_onset_period_hours": forced_period,
        "solver_steps": int(soft.solver_steps),
    }


def main() -> None:
    base = load_config("config/model.yaml")
    duration = float(base.section("designs")["entrained"]["duration"])
    results = []
    for k in (650.0, 2000.0, 5500.0):
        result = evaluate(_config_with_k(base, k), duration)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    payload = {
        "regime": "entrained",
        "duration_hours": duration,
        "variants": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
