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
WIDTHS = (4, 16)


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
    samples = mcmc.get_samples(group_by_chain=True)
    jax.block_until_ready(samples)
    wall = time.perf_counter() - start
    try:
        step = np.asarray(
            jax.device_get(mcmc.last.adapt_state.step_size)
        ).reshape(-1)
    except Exception as exc:  # noqa: BLE001 — diagnostic harness
        step = np.asarray([])
        print(f"  warn: step_size unavailable ({type(exc).__name__}: {exc})", flush=True)
    shapes = {name: list(np.asarray(val).shape) for name, val in samples.items()}
    shapes_ok = all(
        shape[0] == n_chains and shape[1] == N_SAMPLES for shape in shapes.values()
    )
    return {
        "wall_seconds": wall,
        "num_chains": n_chains,
        "num_warmup": N_WARMUP,
        "num_samples": N_SAMPLES,
        "sample_shapes": shapes,
        "shapes_ok": shapes_ok,
        "n_adapted_step_sizes": int(step.size),
        "step_sizes_match_chains": int(step.size) == n_chains,
        "adapted_step_size": step.tolist(),
        "wall_per_replicate": wall / float(n_chains),
    }


def main() -> None:
    previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    deferred = {
        "date": "2026-08-08",
        "status": "deferred",
        "sbc_adjoint": "recursive_checkpoint",
    }
    previous["adjoint_at_batch_widths"] = deferred
    previous["sbc_adjoint"] = "recursive_checkpoint"
    OUTPUT.write_text(json.dumps(previous, indent=2) + "\n")
    print(json.dumps(deferred, indent=2), flush=True)
    print(f"Adjoint sweep deferred; saved {OUTPUT}", flush=True)
    return

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

    results: dict = {}
    for adjoint, checkpoints in (
        ("direct", 16),
        ("recursive_checkpoint", 180),
    ):
        config = _config(adjoint, checkpoints)
        print(f"{adjoint}: sequential-1...", flush=True)
        seq1 = _run("sequential", 1, labels, config)
        print(
            f"  {seq1['wall_seconds']:.3f}s shapes_ok={seq1['shapes_ok']}",
            flush=True,
        )
        by_width = {}
        for width in WIDTHS:
            print(f"{adjoint}: vectorized-{width}...", flush=True)
            row = _run("vectorized", width, labels, config)
            row["ratio_over_seq1"] = row["wall_seconds"] / seq1["wall_seconds"]
            row["relative_work"] = row["ratio_over_seq1"] / float(width)
            by_width[str(width)] = row
            print(
                f"  {row['wall_seconds']:.3f}s ratio={row['ratio_over_seq1']:.3f} "
                f"wall/rep={row['wall_per_replicate']:.1f}s "
                f"shapes_ok={row['shapes_ok']} step_n={row['n_adapted_step_sizes']}",
                flush=True,
            )
        results[adjoint] = {"sequential_1": seq1, "vectorized_by_width": by_width}

    operating = previous.get("chosen_batch_width")
    if operating not in (4, 16):
        rec_widths = results["recursive_checkpoint"]["vectorized_by_width"]
        operating = int(
            min(rec_widths, key=lambda w: rec_widths[w]["wall_per_replicate"])
        )

    def effective_at(adjoint: str, width: int) -> float:
        return results[adjoint]["vectorized_by_width"][str(width)][
            "wall_per_replicate"
        ]

    if effective_at("direct", operating) <= effective_at(
        "recursive_checkpoint", operating
    ):
        winner = "direct"
        reason = (
            f"Lower wall/replicate at vectorized-{operating}; checkpoint "
            "recomputation can dominate at batch width even if Recursive wins "
            "single-chain gradient timing."
        )
    else:
        winner = "recursive_checkpoint"
        reason = (
            f"Lower wall/replicate at vectorized-{operating}; single-chain "
            "Recursive advantage survives at this batch width."
        )

    winners_by_width = {}
    for width in WIDTHS:
        d = effective_at("direct", width)
        r = effective_at("recursive_checkpoint", width)
        winners_by_width[str(width)] = "direct" if d <= r else "recursive_checkpoint"

    block = {
        "date": "2026-08-08",
        "n_warmup": N_WARMUP,
        "seed": SEED,
        "max_tree_depth": MAX_TREE_DEPTH,
        "burn_in_hours": 48.0,
        "retained_hours": 24.0,
        "widths": list(WIDTHS),
        "by_adjoint": results,
        "operating_batch_width": operating,
        "sbc_adjoint": winner,
        "winners_by_width": winners_by_width,
        "reason": reason,
    }
    previous["adjoint_at_batch_widths"] = block
    previous["sbc_adjoint"] = winner
    if "chosen_batch_width" not in previous:
        previous["chosen_batch_width"] = operating
    OUTPUT.write_text(json.dumps(previous, indent=2) + "\n")
    print(json.dumps({
        "operating_batch_width": operating,
        "sbc_adjoint": winner,
        "winners_by_width": winners_by_width,
        "reason": reason,
    }, indent=2), flush=True)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
