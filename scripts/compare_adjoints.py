from __future__ import annotations

import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import potential_energy

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.inference import (
    numpyro_model,
    prior_distribution,
    sampled_parameter_names,
)


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "docs" / "adjoint_comparison.json"
N_REPEATS = 10
SEED = 20260802


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
    soft["p0"] = 0.5
    soft["tau_gate"] = 0.15
    soft["k"] = 650.0
    soft["fixed_step_size"] = 0.003
    soft["adjoint_mode"] = adjoint_mode
    soft["adjoint_checkpoints"] = checkpoints
    horizon = 72.0
    soft["max_steps"] = int(max(float(soft["max_steps"]), 2 * horizon / 0.003))
    return ProjectConfig(data=data, source=base.source)


def _median_gradient_seconds(config: ProjectConfig) -> dict[str, float]:
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
    model_args = (labels, config, "entrained")
    names = sampled_parameter_names(config, "entrained")
    keys = jax.random.split(jax.random.PRNGKey(SEED), len(names))
    unconstrained = {}
    for key, name in zip(keys, names):
        distribution = prior_distribution(name, config, "entrained")
        unconstrained[name] = biject_to(distribution.support).inv(
            distribution.sample(key)
        )

    def potential(params):
        return potential_energy(numpyro_model, model_args, {}, params)

    value_and_grad = jax.jit(jax.value_and_grad(potential))
    compile_start = time.perf_counter()
    value, grad = value_and_grad(unconstrained)
    jax.block_until_ready((value, grad))
    compile_seconds = time.perf_counter() - compile_start

    reps = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        value, grad = value_and_grad(unconstrained)
        jax.block_until_ready((value, grad))
        reps.append(time.perf_counter() - start)
    array = np.asarray(reps, dtype=float)
    return {
        "compile_seconds": float(compile_seconds),
        "median_seconds": float(np.median(array)),
        "mean_seconds": float(np.mean(array)),
        "min_seconds": float(np.min(array)),
        "max_seconds": float(np.max(array)),
        "repeats": reps,
        "potential": float(value),
        "grad_norm": float(
            np.linalg.norm(
                np.concatenate([np.ravel(np.asarray(g)) for g in grad.values()])
            )
        ),
    }


def main() -> None:
    direct_config = _config("direct")
    recursive_config = _config("recursive_checkpoint", checkpoints=180)
    soft = direct_config.section("soft_gate")

    print("Timing DirectAdjoint...", flush=True)
    direct = _median_gradient_seconds(direct_config)
    print(
        f"Direct median={direct['median_seconds']:.4f}s "
        f"compile={direct['compile_seconds']:.3f}s",
        flush=True,
    )

    print("Timing RecursiveCheckpointAdjoint(checkpoints=180)...", flush=True)
    recursive = _median_gradient_seconds(recursive_config)
    print(
        f"Recursive median={recursive['median_seconds']:.4f}s "
        f"compile={recursive['compile_seconds']:.3f}s",
        flush=True,
    )

    speedup = (
        direct["median_seconds"] / recursive["median_seconds"]
        if recursive["median_seconds"] > 0
        else None
    )
    adopt_candidate = bool(
        speedup is not None and speedup >= 1.1
    )

    audit_path = None
    audit_passed = None
    if adopt_candidate:
        audit_path = str(
            REPOSITORY / "docs" / "unconstrained_gradient_audit_recursive64.json"
        )
        print("Recursive faster — re-running 20-draw gradient audit...", flush=True)
        command = [
            sys.executable,
            str(REPOSITORY / "scripts" / "diagnose_unconstrained_gradients.py"),
            "--n-draws",
            "20",
            "--seed-base",
            "1000",
            "--adjoint-mode",
            "recursive_checkpoint",
            "--adjoint-checkpoints",
            "180",
            "--fixed-step-size",
            "0.003",
            "--tau-gate",
            "0.15",
            "--burn-in",
            "48",
            "--duration",
            "24",
            "--output",
            audit_path,
        ]
        completed = subprocess.run(command, check=False, cwd=REPOSITORY)
        if completed.returncode == 0 and Path(audit_path).exists():
            audit = json.loads(Path(audit_path).read_text())
            if "all_draws_pass" in audit:
                audit_passed = bool(audit["all_draws_pass"])
            else:
                draw_rows = audit.get("draws", [])
                audit_passed = all(
                    bool(row.get("pass", False)) for row in draw_rows
                )
        else:
            audit_passed = False

    payload = {
        "date": "2026-08-02",
        "n_repeats": N_REPEATS,
        "seed": SEED,
        "gate": {
            "p0": float(soft["p0"]),
            "tau_gate": float(soft["tau_gate"]),
            "k": float(soft["k"]),
            "fixed_step_size": float(soft["fixed_step_size"]),
            "burn_in_hours": 48.0,
            "retained_hours": 24.0,
        },
        "direct": {"adjoint_mode": "direct", **direct},
        "recursive_checkpoint": {
            "adjoint_mode": "recursive_checkpoint",
            "checkpoints": 180,
            **recursive,
        },
        "speedup_direct_over_recursive": speedup,
        "recursive_faster": adopt_candidate,
        "gradient_audit_path": audit_path,
        "gradient_audit_passed": audit_passed,
        "recommendation": (
            "adopt_recursive_after_audit"
            if adopt_candidate and audit_passed
            else (
                "keep_direct_recursive_not_faster"
                if not adopt_candidate
                else "keep_direct_audit_failed"
            )
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
