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
N_CHAINS = 4
SEED = 20260802
MAX_TREE_DEPTH = 7


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
    soft["p0"] = 0.5
    soft["tau_gate"] = 0.15
    soft["k"] = 650.0
    soft["fixed_step_size"] = 0.003
    soft["adjoint_mode"] = "direct"
    horizon = 72.0
    soft["max_steps"] = int(max(float(soft["max_steps"]), 2 * horizon / 0.003))
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
    key = jax.random.PRNGKey(SEED)
    start = time.perf_counter()
    mcmc.run(key, labels)
    jax.block_until_ready(mcmc.get_samples(group_by_chain=True))
    wall = time.perf_counter() - start
    step_sizes = None
    if hasattr(mcmc, "last_state") and mcmc.last_state is not None:
        adapt = getattr(mcmc.last_state, "adapt_state", None)
        if adapt is not None and hasattr(adapt, "step_size"):
            step_sizes = np.asarray(adapt.step_size).tolist()
    return {
        "chain_method": method,
        "num_chains": n_chains,
        "wall_seconds_including_compile": wall,
        "num_warmup": N_WARMUP,
        "num_samples": N_SAMPLES,
        "adapted_step_size": step_sizes,
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
    recording = generate_recording(
        jax.random.PRNGKey(int(recovery["seed"])),
        config,
        truth,
        "entrained",
    )
    labels = recording.observations

    print("Pass 1: sequential 1-chain...", flush=True)
    seq1_a = _run("sequential", 1, labels, config)
    print(f"  wall={seq1_a['wall_seconds_including_compile']:.3f}s", flush=True)
    print("Pass 1: vectorized 4-chain...", flush=True)
    vec4_a = _run("vectorized", N_CHAINS, labels, config)
    print(f"  wall={vec4_a['wall_seconds_including_compile']:.3f}s", flush=True)

    print("Pass 2: sequential 1-chain...", flush=True)
    seq1_b = _run("sequential", 1, labels, config)
    print(f"  wall={seq1_b['wall_seconds_including_compile']:.3f}s", flush=True)
    print("Pass 2: vectorized 4-chain...", flush=True)
    vec4_b = _run("vectorized", N_CHAINS, labels, config)
    print(f"  wall={vec4_b['wall_seconds_including_compile']:.3f}s", flush=True)

    print("Sequential 4-chain (serial baseline)...", flush=True)
    seq4 = _run("sequential", N_CHAINS, labels, config)
    print(f"  wall={seq4['wall_seconds_including_compile']:.3f}s", flush=True)

    seq1 = seq1_b["wall_seconds_including_compile"]
    vec4 = vec4_b["wall_seconds_including_compile"]
    vs_one = vec4 / seq1 if seq1 > 0 else float("nan")
    vs_four = (
        vec4 / seq4["wall_seconds_including_compile"]
        if seq4["wall_seconds_including_compile"] > 0
        else float("nan")
    )

    if vs_one <= 1.25:
        meaning = "vectorized_4_near_sequential_1_SBC_weekend_local"
    elif vs_one <= 2.5:
        meaning = "partial_win_usable_for_SBC"
    else:
        meaning = "little_parallelism_SBC_needs_fewer_replicates_or_accelerator"

    payload = {
        "date": "2026-08-02",
        "method": "numpyro_chain_method_vectorized_vs_sequential",
        "n_warmup": N_WARMUP,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "max_tree_depth": MAX_TREE_DEPTH,
        "dense_mass": True,
        "gate": {
            "p0": float(soft["p0"]),
            "tau_gate": float(soft["tau_gate"]),
            "k": float(soft["k"]),
            "fixed_step_size": float(soft["fixed_step_size"]),
            "adjoint_mode": soft["adjoint_mode"],
            "burn_in_hours": 48.0,
            "retained_hours": 24.0,
            "integration_steps": int(round(72.0 / float(soft["fixed_step_size"]))),
        },
        "sequential_1_pass1": seq1_a,
        "vectorized_4_pass1": vec4_a,
        "sequential_1_pass2": seq1_b,
        "vectorized_4_pass2": vec4_b,
        "sequential_4": seq4,
        "wall_clock_ratio_vectorized4_over_sequential1": vs_one,
        "wall_clock_ratio_vectorized4_over_sequential4": vs_four,
        "effective_speedup_vs_serial_4x": (
            seq4["wall_seconds_including_compile"] / vec4 if vec4 > 0 else None
        ),
        "meaning": meaning,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
