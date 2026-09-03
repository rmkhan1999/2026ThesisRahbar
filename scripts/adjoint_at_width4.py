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


def _config(adjoint_mode: str, checkpoints: int = 180) -> ProjectConfig:
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
    soft["adjoint_mode"] = adjoint_mode
    soft["adjoint_checkpoints"] = checkpoints
    horizon = 72.0
    soft["max_steps"] = int(
        max(float(soft["max_steps"]), 2 * horizon / float(soft["fixed_step_size"]))
    )
    return ProjectConfig(data=data, source=base.source)


def _run(method: str, n_chains: int, labels, config: ProjectConfig) -> float:
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
    return time.perf_counter() - start


def main() -> None:
    previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    live = _config("recursive_checkpoint", 180)
    model = live.section("model")
    recovery = live.section("recovery")
    labels = generate_recording(
        jax.random.PRNGKey(int(recovery["seed"])),
        live,
        standard_parameters(
            live,
            amplitude=float(model["circadian_amplitude"]),
            phase=float(recovery["true_phase"]),
        ),
        "entrained",
    ).observations

    results = {}
    for adjoint, checkpoints in (
        ("direct", 16),
        ("recursive_checkpoint", 180),
    ):
        config = _config(adjoint, checkpoints)
        print(f"{adjoint}: sequential-1...", flush=True)
        seq1 = _run("sequential", 1, labels, config)
        print(f"  {seq1:.3f}s", flush=True)
        print(f"{adjoint}: vectorized-4...", flush=True)
        vec4 = _run("vectorized", 4, labels, config)
        print(f"  {vec4:.3f}s", flush=True)
        results[adjoint] = {
            "sequential_1_seconds": seq1,
            "vectorized_4_seconds": vec4,
            "ratio_vec4_over_seq1": vec4 / seq1,
            "effective_seconds_per_replicate_at_width4": vec4 / 4.0,
            "serial4_seconds": 4.0 * seq1,
            "speedup_vs_serial4": (4.0 * seq1) / vec4,
        }

    direct = results["direct"]
    recursive = results["recursive_checkpoint"]
    if (
        direct["effective_seconds_per_replicate_at_width4"]
        <= recursive["effective_seconds_per_replicate_at_width4"]
    ):
        winner = "direct"
        reason = (
            "Lower effective s/replicate at vectorized-4 despite worse "
            "single-chain gradient cost; checkpoint recomputation scales with batch."
        )
    else:
        winner = "recursive_checkpoint"
        reason = (
            "Lower effective s/replicate at vectorized-4; single-chain 1.88× "
            "advantage survives batching."
        )

    block = {
        "date": "2026-08-08",
        "n_warmup": N_WARMUP,
        "seed": SEED,
        "max_tree_depth": MAX_TREE_DEPTH,
        "burn_in_hours": 48.0,
        "retained_hours": 24.0,
        "by_adjoint": results,
        "sbc_adjoint": winner,
        "reason": reason,
    }
    previous["adjoint_at_width4"] = block
    previous["sbc_adjoint"] = winner
    previous["chosen_batch_width"] = 4
    OUTPUT.write_text(json.dumps(previous, indent=2) + "\n")
    print(json.dumps(block, indent=2), flush=True)
    print(f"SBC adjoint={winner}; saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
