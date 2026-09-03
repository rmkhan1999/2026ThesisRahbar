from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import jax
import numpy as np

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.soft_gate import soft_transition_times


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "docs" / "burnin_tau015.json"
BURN_IN_CANDIDATES = (24.0, 48.0, 72.0, 96.0)
INITIAL_PRESSURES = (0.2, 0.4, 0.6, 0.8)
ATOL_HOURS = 0.01
RETAINED = 48.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained", type=float, default=RETAINED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()

    base = load_config(REPOSITORY / "config" / "model.yaml")
    model = base.section("model")
    recovery = base.section("recovery")
    soft = base.section("soft_gate")
    level = float(base.section("validation")["transition_gate_level"])
    retained = arguments.retained
    results = []

    for burn_in in BURN_IN_CANDIDATES:
        data = deepcopy(base.data)
        data["designs"]["entrained"]["burn_in_hours"] = burn_in
        data["designs"]["entrained"]["duration"] = retained
        config = ProjectConfig(data=data, source=base.source)
        recordings = []
        transitions = []
        for pressure in INITIAL_PRESSURES:
            parameters = standard_parameters(
                config,
                amplitude=float(model["circadian_amplitude"]),
                phase=float(recovery["true_phase"]),
                initial_pressure=pressure,
            )
            recording = generate_recording(
                jax.random.PRNGKey(1),
                config,
                parameters,
                "entrained",
            )
            recordings.append(recording)
            transitions.append(
                soft_transition_times(recording.time, recording.gate, level)
            )

        reference = transitions[0]
        max_delta = 0.0
        labels_identical = True
        comparable = True
        for candidate, recording in zip(transitions[1:], recordings[1:]):
            if candidate.shape != reference.shape:
                comparable = False
                labels_identical = False
                max_delta = float("inf")
                break
            max_delta = max(
                max_delta, float(np.max(np.abs(candidate - reference)))
            )
            labels_identical = labels_identical and bool(
                np.array_equal(
                    recording.observations, recordings[0].observations
                )
            )
        passed = comparable and max_delta <= ATOL_HOURS and labels_identical
        block = {
            "burn_in_hours": burn_in,
            "retained_hours": retained,
            "atol_hours": ATOL_HOURS,
            "max_transition_delta_hours": (
                None if not np.isfinite(max_delta) else max_delta
            ),
            "labels_identical": labels_identical,
            "comparable_transition_counts": comparable,
            "pass": passed,
            "initial_pressures": list(INITIAL_PRESSURES),
            "reference_transition_count": int(reference.size),
        }
        results.append(block)
        print(
            f"burn_in={burn_in} maxΔ={block['max_transition_delta_hours']} "
            f"labels={labels_identical} pass={passed}",
            flush=True,
        )

    shortest = next((r["burn_in_hours"] for r in results if r["pass"]), None)
    payload = {
        "gate": {
            "p0": float(soft["p0"]),
            "tau_gate": float(soft["tau_gate"]),
            "k": float(soft["k"]),
            "fixed_step_size": float(soft["fixed_step_size"]),
        },
        "results": results,
        "shortest_pass_hours": shortest,
    }
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"shortest pass: {shortest}", flush=True)
    print(f"Saved {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
