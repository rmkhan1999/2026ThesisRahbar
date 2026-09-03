from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from numpyro.infer import NUTS
from numpyro.infer.initialization import init_to_median

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, sample_parameters
from twopm.inference import numpyro_model


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "docs" / "vmap_scaling.json"
N_ITERATIONS = 20
N_BATCH = 4
SEED = 20260802
MAX_TREE_DEPTH = 7
TARGET_ACCEPT = 0.8


def _live_config() -> ProjectConfig:
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


def _make_labels(config: ProjectConfig, key: jax.Array) -> jax.Array:
    parameters = sample_parameters(key, config, "entrained")
    recording = generate_recording(key, config, parameters, "entrained")
    return jnp.asarray(recording.observations)


def main() -> None:
    config = _live_config()
    soft = config.section("soft_gate")
    design = "entrained"

    def model(observed_labels):
        numpyro_model(observed_labels, config, design)

    kernel = NUTS(
        model,
        target_accept_prob=TARGET_ACCEPT,
        max_tree_depth=MAX_TREE_DEPTH,
        dense_mass=True,
        init_strategy=init_to_median(),
    )

    def run_fit(rng_key, labels):
        state = kernel.init(
            rng_key,
            N_ITERATIONS,
            model_args=(labels,),
            model_kwargs={},
        )

        def body(current, _):
            return kernel.sample(current, (labels,), {}), None

        state, _ = jax.lax.scan(body, state, xs=None, length=N_ITERATIONS)
        return state.adapt_state.step_size, state.num_steps, state.accept_prob

    single_fit = jax.jit(run_fit)
    batch_fit = jax.jit(jax.vmap(run_fit))

    keys = jax.random.split(jax.random.PRNGKey(SEED), N_BATCH + 1)
    label_list = [_make_labels(config, keys[i]) for i in range(N_BATCH)]
    labels_batch = jnp.stack(label_list)
    fit_keys = jax.random.split(keys[-1], N_BATCH)

    print("Compiling single-replicate fit...", flush=True)
    compile_single_start = time.perf_counter()
    step0, steps0, accept0 = single_fit(fit_keys[0], labels_batch[0])
    jax.block_until_ready((step0, steps0, accept0))
    compile_single_seconds = time.perf_counter() - compile_single_start

    print("Timing single-replicate fit (1 pass)...", flush=True)
    start = time.perf_counter()
    step, steps, accept = single_fit(fit_keys[0], labels_batch[0])
    jax.block_until_ready((step, steps, accept))
    single_seconds = time.perf_counter() - start
    single_reps = [single_seconds]

    print("Compiling 4-replicate vmapped fit...", flush=True)
    compile_batch_start = time.perf_counter()
    step_b, steps_b, accept_b = batch_fit(fit_keys, labels_batch)
    jax.block_until_ready((step_b, steps_b, accept_b))
    compile_batch_seconds = time.perf_counter() - compile_batch_start

    print("Timing 4-replicate vmapped fit (1 pass)...", flush=True)
    start = time.perf_counter()
    step_b, steps_b, accept_b = batch_fit(fit_keys, labels_batch)
    jax.block_until_ready((step_b, steps_b, accept_b))
    batch_seconds = time.perf_counter() - start
    batch_reps = [batch_seconds]

    ratio = batch_seconds / single_seconds if single_seconds > 0 else float("nan")
    if ratio <= 1.25:
        meaning = "near_ideal_parallelism_SBC_weekend_local"
    elif ratio <= 2.5:
        meaning = "partial_win_usable_for_SBC"
    else:
        meaning = "little_parallelism_SBC_needs_fewer_replicates_or_accelerator"

    payload = {
        "date": "2026-08-02",
        "n_iterations": N_ITERATIONS,
        "n_batch": N_BATCH,
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
        "single_replicate": {
            "compile_seconds": compile_single_seconds,
            "wall_seconds_median": single_seconds,
            "wall_seconds_reps": single_reps,
            "final_step_size": float(step),
            "final_num_steps": int(steps),
            "final_accept_prob": float(accept),
        },
        "vmapped_4_replicates": {
            "compile_seconds": compile_batch_seconds,
            "wall_seconds_median": batch_seconds,
            "wall_seconds_reps": batch_reps,
            "final_step_size": [float(x) for x in np.asarray(step_b)],
            "final_num_steps": [int(x) for x in np.asarray(steps_b)],
            "final_accept_prob": [float(x) for x in np.asarray(accept_b)],
        },
        "wall_clock_ratio_4_over_1": ratio,
        "effective_speedup_vs_serial_4x": (
            (4.0 * single_seconds) / batch_seconds if batch_seconds > 0 else None
        ),
        "meaning": meaning,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
