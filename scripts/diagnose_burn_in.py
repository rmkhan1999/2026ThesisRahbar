from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import jax
import numpy as np

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.soft_gate import soft_transition_times


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "docs" / "burn_in_invariance.json"
BURN_IN_CANDIDATES = (24.0, 48.0, 72.0, 120.0)
INITIAL_PRESSURES = (0.2, 0.4, 0.6, 0.8)
ATOL_HOURS = 0.01


def main() -> None:
    base = load_config(REPOSITORY / "config" / "model.yaml")
    model = base.section("model")
    validation = base.section("validation")
    level = float(validation["transition_gate_level"])
    retained = float(base.section("designs")["entrained"]["duration"])
    results = []

    for burn_in in BURN_IN_CANDIDATES:
        data = deepcopy(base.data)
        data["designs"]["entrained"]["burn_in_hours"] = burn_in
        config = ProjectConfig(data=data, source=base.source)
        recordings = []
        transitions = []
        for pressure in INITIAL_PRESSURES:
            parameters = standard_parameters(
                config,
                amplitude=float(model["circadian_amplitude"]),
                initial_pressure=pressure,
            )
            recording = generate_recording(
                jax.random.PRNGKey(1),
                config,
                parameters,
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
            max_delta = max(max_delta, float(np.max(np.abs(candidate - reference))))
            labels_identical = labels_identical and bool(
                np.array_equal(recording.observations, recordings[0].observations)
            )
        passed = comparable and max_delta <= ATOL_HOURS and labels_identical
        block = {
            "burn_in_hours": burn_in,
            "retained_hours": retained,
            "atol_hours": ATOL_HOURS,
            "max_transition_delta_hours": None if not np.isfinite(max_delta) else max_delta,
            "labels_identical": labels_identical,
            "comparable_transition_counts": comparable,
            "pass": passed,
            "initial_pressures": list(INITIAL_PRESSURES),
            "reference_transition_count": int(reference.size),
        }
        results.append(block)
        print(json.dumps(block, indent=2), flush=True)

    chosen = next((block for block in results if block["pass"]), None)
    payload = {
        "candidates": results,
        "chosen_burn_in_hours": None if chosen is None else chosen["burn_in_hours"],
        "criterion": (
            "shortest burn-in among tested values with max post-burn-in "
            f"transition delta <= {ATOL_HOURS} h and identical epoch labels "
            "across four initial pressures"
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
