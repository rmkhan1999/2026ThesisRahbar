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


def _adapted_step_sizes(mcmc: MCMC, n_chains: int) -> tuple[list[float], str]:
    try:
        state = mcmc.last_state
    except Exception as exc:  # noqa: BLE001
        return [], f"not_available: last_state ({type(exc).__name__}: {exc})"
    if state is None:
        return [], "not_available: last_state is None"
    try:
        step = state.adapt_state.step_size
        step_np = np.asarray(jax.device_get(step)).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        return [], f"not_available: adapt_state.step_size ({type(exc).__name__}: {exc})"
    if step_np.size == 0:
        return [], "not_available: empty step_size"
    if int(step_np.size) != n_chains:
        return step_np.tolist(), (
            f"mismatch: got {int(step_np.size)} step sizes for {n_chains} chains"
        )
    return step_np.tolist(), "ok"


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
    mcmc.run(jax.random.PRNGKey(SEED), labels, extra_fields=("diverging",))
    samples = mcmc.get_samples(group_by_chain=True)
    jax.block_until_ready(samples)
    wall = time.perf_counter() - start

    extras = mcmc.get_extra_fields(group_by_chain=True) or {}
    step_list, step_check = _adapted_step_sizes(mcmc, n_chains)

    sample_shapes = {
        name: list(np.asarray(val).shape) for name, val in samples.items()
    }
    ok_shapes = all(
        shape[0] == n_chains and shape[1] == N_SAMPLES
        for shape in sample_shapes.values()
    )
    distinct = True
    for name, val in samples.items():
        arr = np.asarray(val)
        if arr.ndim >= 2 and n_chains > 1:
            flats = arr.reshape(n_chains, -1)
            if all(np.allclose(flats[0], flats[i]) for i in range(1, n_chains)):
                distinct = False
                break

    n_div = None
    if "diverging" in extras:
        n_div = int(np.sum(np.asarray(extras["diverging"])))

    return {
        "chain_method": method,
        "num_chains": n_chains,
        "wall_seconds_including_compile": wall,
        "num_warmup": N_WARMUP,
        "num_samples": N_SAMPLES,
        "sample_shapes": sample_shapes,
        "shapes_ok": ok_shapes,
        "adapted_step_size": step_list,
        "n_adapted_step_sizes": len(step_list),
        "step_sizes_check": step_check,
        "chains_not_all_identical": distinct,
        "n_divergences": n_div,
        "extra_field_keys": sorted(extras.keys()),
        "wall_per_replicate": wall / float(n_chains),
        "relative_work_vs_seq1_unit": None,
    }


def choose_batch_width(rows: list[dict]) -> int:
    valid = [r for r in rows if r["shapes_ok"]]
    if not valid:
        raise RuntimeError(
            "No vectorized width produced shapes_ok=True; "
            "refuse to fail-closed to width 1."
        )
    best = min(valid, key=lambda r: r["wall_per_replicate"])
    return int(best["num_chains"])


def main() -> None:
    config = _config()
    soft = config.section("soft_gate")
    model = config.section("model")
    recovery = config.section("recovery")
    labels = generate_recording(
        jax.random.PRNGKey(int(recovery["seed"])),
        config,
        standard_parameters(
            config,
            amplitude=float(model["circadian_amplitude"]),
            phase=float(recovery["true_phase"]),
        ),
        "entrained",
    ).observations

    print("sequential-1...", flush=True)
    seq1 = _run("sequential", 1, labels, config)
    print(
        f"  {seq1['wall_seconds_including_compile']:.3f}s "
        f"shapes_ok={seq1['shapes_ok']} step_check={seq1['step_sizes_check']}",
        flush=True,
    )

    rows = []
    for width in WIDTHS:
        print(f"vectorized-{width}...", flush=True)
        row = _run("vectorized", width, labels, config)
        ratio = (
            row["wall_seconds_including_compile"]
            / seq1["wall_seconds_including_compile"]
        )
        row["ratio_over_sequential1"] = ratio
        row["relative_work_vs_seq1_unit"] = ratio / float(width)
        rows.append(row)
        print(
            f"  {row['wall_seconds_including_compile']:.3f}s "
            f"ratio={ratio:.3f} wall/rep={row['wall_per_replicate']:.1f}s "
            f"shapes_ok={row['shapes_ok']} "
            f"step_check={row['step_sizes_check']} "
            f"distinct={row['chains_not_all_identical']}",
            flush=True,
        )

    chosen = choose_batch_width(rows)

    previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    block = {
        "date": "2026-08-09",
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
        "vectorized_by_width": rows,
        "all_shapes_ok": all(r["shapes_ok"] for r in rows),
        "step_sizes_status": {
            str(r["num_chains"]): r["step_sizes_check"] for r in rows
        },
        "chosen_batch_width": chosen,
        "previous_unverified_width16_wall": 2654.3921037078835,
    }
    previous["width_sweep_verified"] = block
    previous["chosen_batch_width"] = chosen
    best = min((r for r in rows if r["shapes_ok"]), key=lambda r: r["wall_per_replicate"])
    previous["best_width"] = best["num_chains"]
    previous["best_width_ratio"] = best["ratio_over_sequential1"]
    previous["best_wall_per_replicate"] = best["wall_per_replicate"]
    OUTPUT.write_text(json.dumps(previous, indent=2) + "\n")
    print(json.dumps(block, indent=2), flush=True)
    print(f"Saved {OUTPUT}; chosen_batch_width={chosen}", flush=True)


if __name__ == "__main__":
    main()
