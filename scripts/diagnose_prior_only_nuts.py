from __future__ import annotations

import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import NUTS
from numpyro.infer.initialization import init_to_median

from twopm.config import ProjectConfig, load_config
from twopm.inference import (
    constrained_physical_parameters,
    free_running_period,
    prior_distribution,
    sampled_parameter_names,
)


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPOSITORY / "docs"
CSV_PATH = OUTPUT_DIR / "prior_only_nuts_iterations.csv"
SUMMARY_PATH = OUTPUT_DIR / "prior_only_nuts.json"


def prior_only_model(config: ProjectConfig, design: str = "entrained") -> None:
    sampled = {
        name: numpyro.sample(name, prior_distribution(name, config, design))
        for name in sampled_parameter_names(config, design)
    }
    if "misclassification" not in sampled:
        fixed_error = jnp.asarray(
            float(config.section("observation")["misclassification"])
        )
        sampled["misclassification"] = fixed_error
        numpyro.deterministic("misclassification", fixed_error)
    values = constrained_physical_parameters(sampled, config)
    model = config.section("model")
    fixed = config.section("fixed")
    tau = free_running_period(
        values["chi_sleep"],
        values["chi_wake"],
        values["threshold_gap"],
        float(fixed["threshold_mean"]),
        float(model["mu"]),
    )
    numpyro.deterministic("amplitude", values["amplitude"])
    numpyro.deterministic("phase", values["phase"])
    numpyro.deterministic("threshold_gap", values["threshold_gap"])
    numpyro.deterministic("tau", tau)


def main() -> None:
    base = load_config(REPOSITORY / "config" / "model.yaml")
    data = deepcopy(base.data)
    data["observation"]["misclassification"] = 0.01
    data["inference"]["parameters"] = [
        name
        for name in data["inference"]["parameters"]
        if name != "misclassification"
    ]
    config = ProjectConfig(data=data, source=base.source)

    warmup = 50
    seed = 0
    kernel = NUTS(
        prior_only_model,
        target_accept_prob=0.8,
        max_tree_depth=10,
        dense_mass=False,
        init_strategy=init_to_median(),
    )
    key = jax.random.PRNGKey(seed)
    state = kernel.init(key, warmup, model_args=(config, "entrained"), model_kwargs={})

    def transition(current):
        return kernel.sample(current, (config, "entrained"), {})

    compiled = jax.jit(transition).lower(state).compile()
    jax.tree.map(lambda x: jax.block_until_ready(x), state)

    rows = []
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "iteration",
                "step_size",
                "tree_depth",
                "diverging",
                "num_steps",
                "accept_prob",
                "seconds",
            ),
        )
        writer.writeheader()
        for iteration in range(1, warmup + 1):
            t0 = time.perf_counter()
            state = compiled(state)
            jax.tree.map(lambda x: jax.block_until_ready(x), state)
            seconds = time.perf_counter() - t0
            steps = int(state.num_steps)
            depth = int(math.ceil(math.log2(steps + 1)))
            row = {
                "iteration": iteration,
                "step_size": float(state.adapt_state.step_size),
                "tree_depth": depth,
                "diverging": int(bool(state.diverging)),
                "num_steps": steps,
                "accept_prob": float(state.accept_prob),
                "seconds": seconds,
            }
            rows.append(row)
            writer.writerow(
                {
                    **row,
                    "step_size": f"{row['step_size']:.16e}",
                    "seconds": f"{seconds:.6f}",
                }
            )
            handle.flush()
            print(
                f"prior_only iteration={iteration}/{warmup} "
                f"step_size={row['step_size']:.6e} "
                f"tree_depth={depth} diverging={row['diverging']} "
                f"accept={row['accept_prob']:.3f} seconds={seconds:.3f}",
                flush=True,
            )

    final_eps = rows[-1]["step_size"]
    summary = {
        "warmup": warmup,
        "dense_mass": False,
        "fixed_misclassification": 0.01,
        "sampled_parameters": list(config.section("inference")["parameters"]),
        "final_step_size": final_eps,
        "min_step_size": min(row["step_size"] for row in rows),
        "max_step_size": max(row["step_size"] for row in rows),
        "divergence_fraction": float(np.mean([row["diverging"] for row in rows])),
        "mean_accept_prob": float(np.mean([row["accept_prob"] for row in rows])),
        "collapse_like_run006": bool(final_eps < 1e-4),
        "healthy_like_prior": bool(final_eps > 0.1),
        "iterations": rows,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "final_step_size",
        "min_step_size",
        "divergence_fraction",
        "mean_accept_prob",
        "collapse_like_run006",
        "healthy_like_prior",
    )}, indent=2))
    print(f"Saved {CSV_PATH} and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
