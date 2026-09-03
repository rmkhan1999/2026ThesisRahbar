from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.sampling import (
    SamplerResult,
    effective_config_hash,
    git_provenance,
    run_sequential_nuts,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_snapshot_hash() -> str:
    digest = hashlib.sha256()
    roots = ("src", "scripts", "config")
    files = [REPOSITORY / "pyproject.toml", REPOSITORY / "requirements-lock.txt"]
    for root in roots:
        files.extend(
            path
            for path in (REPOSITORY / root).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    for path in sorted(files):
        digest.update(str(path.relative_to(REPOSITORY)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--run-date", default=date.today().isoformat())
    parser.add_argument("--design", default="entrained")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--burn-in", type=float, default=120.0)
    parser.add_argument("--chains", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--dense-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fix-misclassification",
        type=float,
        default=None,
        help="If set, remove lambda from the sampled sites and fix it.",
    )
    parser.add_argument(
        "--tau-gate",
        type=float,
        default=None,
        help="Override soft_gate.tau_gate for this run.",
    )
    parser.add_argument(
        "--fixed-step-size",
        type=float,
        default=None,
        help="Override soft_gate.fixed_step_size for this run.",
    )
    parser.add_argument(
        "--p0",
        type=float,
        default=None,
        help="Override soft_gate.p0 (θ = logit(p0); p0=0.5 ⇒ θ=0).",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=None,
        help="Override soft_gate.k for this run.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    run_directory = REPOSITORY / "runs" / arguments.run_id
    if run_directory.exists():
        raise FileExistsError(
            f"{run_directory} already exists; run identifiers are immutable"
        )
    run_directory.mkdir(parents=True)

    base_config = load_config(REPOSITORY / "config" / "model.yaml")
    data = deepcopy(base_config.data)
    data["designs"][arguments.design]["duration"] = arguments.duration
    data["designs"][arguments.design]["burn_in_hours"] = arguments.burn_in
    if arguments.fix_misclassification is not None:
        data["observation"]["misclassification"] = float(
            arguments.fix_misclassification
        )
        data["inference"]["parameters"] = [
            name
            for name in data["inference"]["parameters"]
            if name != "misclassification"
        ]
    if arguments.tau_gate is not None:
        data["soft_gate"]["tau_gate"] = float(arguments.tau_gate)
    if arguments.p0 is not None:
        data["soft_gate"]["p0"] = float(arguments.p0)
    if arguments.k is not None:
        data["soft_gate"]["k"] = float(arguments.k)
    if arguments.fixed_step_size is not None:
        data["soft_gate"]["fixed_step_size"] = float(arguments.fixed_step_size)
        horizon = (
            float(data["designs"][arguments.design]["burn_in_hours"])
            + float(data["designs"][arguments.design]["duration"])
        )
        data["soft_gate"]["max_steps"] = int(
            max(
                float(data["soft_gate"]["max_steps"]),
                2 * horizon / float(arguments.fixed_step_size),
            )
        )
    config = ProjectConfig(data=data, source=base_config.source)
    provenance = git_provenance(REPOSITORY)
    manifest: dict[str, object] = {
        "run_id": arguments.run_id,
        "date": arguments.run_date,
        "outcome": "running",
        "git": provenance,
        "source_snapshot_sha256": _source_snapshot_hash(),
        "effective_config_sha256": effective_config_hash(config),
        "seed": arguments.seed,
        "sampler": {
            "design": arguments.design,
            "retained_hours": arguments.duration,
            "burn_in_hours": arguments.burn_in,
            "chains": arguments.chains,
            "warmup": arguments.warmup,
            "draws": arguments.draws,
            "target_accept_prob": arguments.target_accept,
            "max_tree_depth": arguments.max_tree_depth,
            "dense_mass": arguments.dense_mass,
            "fixed_misclassification": arguments.fix_misclassification,
            "tau_gate": float(config.section("soft_gate")["tau_gate"]),
            "p0": float(config.section("soft_gate")["p0"]),
            "k": float(config.section("soft_gate")["k"]),
            "fixed_step_size": float(config.section("soft_gate")["fixed_step_size"]),
            "sampled_parameters": list(
                config.section("inference")["parameters"]
            ),
            "chain_method": "sequential",
            "init_strategy": "init_to_median",
            "thread_stack_bytes": 256 * 1024 * 1024,
        },
    }
    _write_json(run_directory / "manifest.json", manifest)

    timings: dict[str, object] = {}
    diagnostics: dict[str, object] | None = None
    try:
        generation_start = time.perf_counter()
        recovery = config.section("recovery")
        model = config.section("model")
        parameters = standard_parameters(
            config,
            amplitude=float(model["circadian_amplitude"]),
            phase=float(recovery["true_phase"]),
        )
        recording = generate_recording(
            jax.random.PRNGKey(arguments.seed),
            config,
            parameters,
            arguments.design,
        )
        jax.block_until_ready(recording.observations)
        timings["data_generation_seconds"] = (
            time.perf_counter() - generation_start
        )

        result_holder: dict[str, SamplerResult] = {}
        error_holder: dict[str, BaseException | str] = {}

        def run_mcmc() -> None:
            try:
                result_holder["result"] = run_sequential_nuts(
                    labels=recording.observations,
                    config=config,
                    seed=arguments.seed,
                    chains=arguments.chains,
                    warmup=arguments.warmup,
                    draws=arguments.draws,
                    target_accept_prob=arguments.target_accept,
                    max_tree_depth=arguments.max_tree_depth,
                    dense_mass=arguments.dense_mass,
                    design=arguments.design,
                    iteration_log_path=run_directory / "iterations.csv",
                )
            except BaseException as error:
                error_holder["exception"] = error
                error_holder["traceback"] = traceback.format_exc()

        threading.stack_size(256 * 1024 * 1024)
        sampler_thread = threading.Thread(target=run_mcmc, name=arguments.run_id)
        sampler_thread.start()
        sampler_thread.join()
        if error_holder:
            raise error_holder["exception"]  # type: ignore[misc]

        result = result_holder["result"]
        timings.update(result.timings)
        diagnostics = result.diagnostics
        if result.inference_data is not None:
            result.inference_data.to_netcdf(run_directory / "posterior.nc")
        if result.adapt_states:
            payload: dict[str, object] = {"n_chains": len(result.adapt_states)}
            for index, adapt in enumerate(result.adapt_states):
                prefix = f"chain{index}_"
                payload[prefix + "step_size"] = np.asarray(adapt["step_size"])
                payload[prefix + "inverse_mass_matrix"] = adapt[
                    "inverse_mass_matrix"
                ]
                payload[prefix + "mass_matrix_sqrt"] = adapt["mass_matrix_sqrt"]
                payload[prefix + "mass_matrix_sqrt_inv"] = adapt[
                    "mass_matrix_sqrt_inv"
                ]
                for name, value in adapt["z"].items():
                    payload[prefix + f"z_{name}"] = value
                payload[prefix + "sampled_parameter_names"] = np.asarray(
                    adapt["sampled_parameter_names"]
                )
                payload[prefix + "dense_mass"] = np.asarray(adapt["dense_mass"])
                payload[prefix + "max_tree_depth"] = np.asarray(
                    adapt["max_tree_depth"]
                )
            np.savez_compressed(run_directory / "adapt_state.npz", **payload)
        outcome = (
            "diverged"
            if diagnostics and diagnostics.get("divergences")
            else "completed"
        )
        if arguments.draws == 0:
            outcome = "warmup_only"
        manifest.update(
            {
                "outcome": outcome,
                "timings": timings,
                "diagnostics": diagnostics,
                "observed_epochs": int(recording.observations.size),
                "adapt_state_path": (
                    "adapt_state.npz" if result.adapt_states else None
                ),
            }
        )
        _write_json(run_directory / "timings.json", timings)
        _write_json(run_directory / "manifest.json", manifest)
    except BaseException as error:
        trace = str(error_holder.get("traceback", traceback.format_exc())) if "error_holder" in locals() else traceback.format_exc()
        manifest.update(
            {
                "outcome": "crashed",
                "timings": timings,
                "diagnostics": diagnostics,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": trace,
                },
            }
        )
        _write_json(run_directory / "timings.json", timings)
        _write_json(run_directory / "manifest.json", manifest)
        print(trace, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
