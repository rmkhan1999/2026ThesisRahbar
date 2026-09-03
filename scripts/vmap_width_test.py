from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.inference import numpyro_model


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "docs" / "vmap_scaling.json"
N_WARMUP = 20
N_SAMPLES = 1
SEED = 20260808
MAX_TREE_DEPTH = 7
WIDTHS = (4, 8, 16)


def _config() -> ProjectConfig:
    base = load_config(REPOSITORY / "config" / "model.yaml")
    data = deepcopy(base.data)
    data["designs"]["entrained"]["duration"] = 24.0
    data["designs"]["entrained"]["burn_in_hours"] = 48.0
    data["observation"]["misclassification"] = 0.01
    data["inference"]["parameters"] = [
        name
        for name in data["inference"]["parameters"]
        if name != "misclassification"
    ]
    soft = data["soft_gate"]
    horizon = 72.0
    soft["max_steps"] = int(
        max(float(soft["max_steps"]), 2 * horizon / float(soft["fixed_step_size"]))
    )
    return ProjectConfig(data=data, source=base.source)


def _run(method: str, n_chains: int, labels, config: ProjectConfig) -> dict:
    def model(observed_labels):
        numpyro_model(observed_labels, config, "entrained")

    kernel = NUTS(
        model,
        target_accept_prob=0.8,
        max_tree_depth=MAX_TREE_DEPTH,
        dense_mass=True,
        init_strategy=init_to_median(),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=N_WARMUP,
        num_samples=N_SAMPLES,
        num_chains=n_chains,
        chain_method=method,
        progress_bar=False,
        jit_model_args=True,
    )
    start = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(SEED), labels)
    jax.block_until_ready(mcmc.get_samples(group_by_chain=True))
    wall = time.perf_counter() - start
    return {
        "chain_method": method,
        "num_chains": n_chains,
        "wall_seconds_including_compile": wall,
        "num_warmup": N_WARMUP,
        "num_samples": N_SAMPLES,
    }


def main() -> None:
    config = _config()
    soft = config.section("soft_gate")
    model = config.section("model")
    recovery = config.section("recovery")
    truth = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
        phase=float(recovery["true_phase"]),
    )
    labels = generate_recording(
        jax.random.PRNGKey(int(recovery["seed"])),
        config,
        truth,
        "entrained",
    ).observations

    print("sequential-1...", flush=True)
    seq1 = _run("sequential", 1, labels, config)
    print(f"  {seq1['wall_seconds_including_compile']:.3f}s", flush=True)

    width_rows = []
    for width in WIDTHS:
        print(f"vectorized-{width}...", flush=True)
        row = _run("vectorized", width, labels, config)
        print(f"  {row['wall_seconds_including_compile']:.3f}s", flush=True)
        ratio = (
            row["wall_seconds_including_compile"]
            / seq1["wall_seconds_including_compile"]
        )
        width_rows.append({**row, "ratio_over_sequential1": ratio})

    chosen = None
    for row in width_rows:
        if row["ratio_over_sequential1"] <= 1.25:
            chosen = row["num_chains"]
    if chosen is None:
        for row in width_rows:
            if row["ratio_over_sequential1"] <= 2.5:
                chosen = row["num_chains"]
                break
    if chosen is None:
        chosen = 1

    previous = {}
    if OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text())

    payload = {
        **previous,
        "date_width_sweep": "2026-08-08",
        "width_sweep": {
            "n_warmup": N_WARMUP,
            "seed": SEED,
            "max_tree_depth": MAX_TREE_DEPTH,
            "adjoint_mode": soft["adjoint_mode"],
            "adjoint_checkpoints": int(soft["adjoint_checkpoints"]),
            "gate": {
                "p0": float(soft["p0"]),
                "tau_gate": float(soft["tau_gate"]),
                "k": float(soft["k"]),
                "fixed_step_size": float(soft["fixed_step_size"]),
                "burn_in_hours": 48.0,
                "retained_hours": 24.0,
            },
            "sequential_1": seq1,
            "vectorized_by_width": width_rows,
            "chosen_batch_width": chosen,
        },
    }
    best = min(width_rows, key=lambda r: r["ratio_over_sequential1"])
    payload["wall_clock_ratio_vectorized4_over_sequential1"] = next(
        (r["ratio_over_sequential1"] for r in width_rows if r["num_chains"] == 4),
        previous.get("wall_clock_ratio_vectorized4_over_sequential1"),
    )
    payload["chosen_batch_width"] = chosen
    payload["best_width_ratio"] = best["ratio_over_sequential1"]
    payload["best_width"] = best["num_chains"]
    if best["ratio_over_sequential1"] <= 1.25:
        payload["meaning"] = "near_ideal_parallelism_SBC_weekend_local"
    elif best["ratio_over_sequential1"] <= 2.5:
        payload["meaning"] = "partial_win_usable_for_SBC"
    else:
        payload["meaning"] = "little_parallelism_SBC_needs_fewer_replicates_or_accelerator"

    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["width_sweep"], indent=2), flush=True)
    print(f"Saved {OUTPUT}; chosen_batch_width={chosen}", flush=True)


if __name__ == "__main__":
    main()
