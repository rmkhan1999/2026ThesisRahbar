from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from datetime import date
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import arviz as az
import jax.numpy as jnp
import numpy as np
from numpyro.infer import NUTS
from numpyro.infer.initialization import init_to_value

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.inference import numpyro_model, sampled_parameter_names
from twopm.sampling import (
    _block_tree,
    _open_iteration_log,
    _physical_sample,
    _summarize_diagnostics,
    _tree_depth,
    _write_iteration,
    effective_config_hash,
    git_provenance,
)


REPOSITORY = Path(__file__).resolve().parents[1]
RUN012 = REPOSITORY / "runs" / "RUN-012"


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
    return ProjectConfig(data=data, source=base.source)


def _init_and_step(config: ProjectConfig) -> tuple[dict[str, jnp.ndarray], float]:
    manifest = json.loads((RUN012 / "manifest.json").read_text())
    step = float(manifest["diagnostics"]["adapted_step_size_by_chain"][0])
    idata = az.from_netcdf(RUN012 / "posterior.nc")
    names = sampled_parameter_names(config, "entrained")
    init_constrained = {
        name: jnp.asarray(idata.posterior[name].values.reshape(-1)[-1])
        for name in names
    }
    return init_constrained, step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="RUN-013")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    run_directory = REPOSITORY / "runs" / arguments.run_id
    if run_directory.exists():
        raise FileExistsError(f"{run_directory} already exists")
    run_directory.mkdir(parents=True)

    config = _config()
    init_z, step_size = _init_and_step(config)
    model = config.section("model")
    recovery = config.section("recovery")
    truth = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
        phase=float(recovery["true_phase"]),
    )
    labels = generate_recording(
        jax.random.PRNGKey(arguments.seed),
        config,
        truth,
        "entrained",
    ).observations

    def model_fn(observed_labels):
        numpyro_model(observed_labels, config, "entrained")

    kernel = NUTS(
        model_fn,
        step_size=step_size,
        adapt_step_size=False,
        adapt_mass_matrix=True,
        dense_mass=True,
        max_tree_depth=arguments.max_tree_depth,
        target_accept_prob=0.8,
        init_strategy=init_to_value(values=init_z),
    )
    model_args = (labels,)
    key = jax.random.PRNGKey(arguments.seed)
    compile_start = time.perf_counter()
    state = kernel.init(
        key,
        max(arguments.warmup, 1),
        model_args=model_args,
        model_kwargs={},
    )
    state = state._replace(
        adapt_state=state.adapt_state._replace(step_size=jnp.asarray(step_size))
    )

    def transition(current):
        return kernel.sample(current, model_args, {})

    compiled = jax.jit(transition).lower(state).compile()
    _block_tree(state)
    compile_seconds = time.perf_counter() - compile_start
    print(
        f"compile={compile_seconds:.3f}s frozen_step_size={step_size:.6e} "
        f"depth={arguments.max_tree_depth} adapt_mass=True",
        flush=True,
    )

    iteration_log = _open_iteration_log(run_directory / "iterations.csv")
    divergences = 0
    warmup_start = time.perf_counter()
    for iteration in range(1, arguments.warmup + 1):
        t0 = time.perf_counter()
        state = compiled(state)
        _block_tree(state)
        seconds = time.perf_counter() - t0
        steps = int(state.num_steps)
        depth = _tree_depth(steps)
        diverging = bool(state.diverging)
        divergences += int(diverging)
        _write_iteration(
            iteration_log,
            chain=1,
            phase="warmup",
            iteration=iteration,
            step_size=float(state.adapt_state.step_size),
            tree_depth=depth,
            diverging=diverging,
            num_steps=steps,
            seconds=seconds,
            accept_prob=float(state.accept_prob),
        )
        if iteration == 1 or iteration % 10 == 0 or iteration == arguments.warmup:
            print(
                f"warmup {iteration}/{arguments.warmup} "
                f"eps={float(state.adapt_state.step_size):.3e} "
                f"depth={depth} s={seconds:.1f} div={divergences}",
                flush=True,
            )
    warmup_seconds = time.perf_counter() - warmup_start

    posterior: dict[str, list[np.ndarray]] = {}
    sample_stats = {
        "diverging": [],
        "energy": [],
        "n_steps": [],
        "acceptance_rate": [],
        "tree_depth": [],
    }
    sampling_start = time.perf_counter()
    divergences = 0
    for iteration in range(1, arguments.draws + 1):
        t0 = time.perf_counter()
        state = compiled(state)
        _block_tree(state)
        seconds = time.perf_counter() - t0
        physical = _physical_sample(state.z, config, "entrained")
        for name, value in physical.items():
            posterior.setdefault(name, []).append(np.asarray(value))
        steps = int(state.num_steps)
        depth = _tree_depth(steps)
        diverging = bool(state.diverging)
        divergences += int(diverging)
        sample_stats["diverging"].append(diverging)
        sample_stats["energy"].append(float(state.energy))
        sample_stats["n_steps"].append(steps)
        sample_stats["acceptance_rate"].append(float(state.accept_prob))
        sample_stats["tree_depth"].append(depth)
        _write_iteration(
            iteration_log,
            chain=1,
            phase="sampling",
            iteration=iteration,
            step_size=float(state.adapt_state.step_size),
            tree_depth=depth,
            diverging=diverging,
            num_steps=steps,
            seconds=seconds,
            accept_prob=float(state.accept_prob),
        )
        if iteration == 1 or iteration % 10 == 0 or iteration == arguments.draws:
            print(
                f"sampling {iteration}/{arguments.draws} "
                f"eps={float(state.adapt_state.step_size):.3e} "
                f"depth={depth} s={seconds:.1f} div={divergences}",
                flush=True,
            )
    sampling_seconds = time.perf_counter() - sampling_start
    iteration_log.close()

    adapt = state.adapt_state
    np.savez_compressed(
        run_directory / "adapt_state.npz",
        n_chains=np.asarray(1),
        chain0_step_size=np.asarray(float(adapt.step_size)),
        chain0_inverse_mass_matrix=np.asarray(adapt.inverse_mass_matrix),
        chain0_mass_matrix_sqrt=np.asarray(adapt.mass_matrix_sqrt),
        chain0_mass_matrix_sqrt_inv=np.asarray(adapt.mass_matrix_sqrt_inv),
        **{f"chain0_z_{name}": np.asarray(value) for name, value in state.z.items()},
        chain0_sampled_parameter_names=np.asarray(list(state.z.keys())),
        chain0_dense_mass=np.asarray(True),
        chain0_max_tree_depth=np.asarray(arguments.max_tree_depth),
    )

    posterior_arr = {
        name: np.asarray([np.stack(values, axis=0)])
        for name, values in posterior.items()
    }
    stats_arr = {name: np.asarray([values]) for name, values in sample_stats.items()}
    inference_data = az.from_dict(posterior=posterior_arr, sample_stats=stats_arr)
    diagnostics = _summarize_diagnostics(
        inference_data,
        posterior_arr,
        stats_arr,
        [float(step_size)],
        arguments.max_tree_depth,
        float(model["circadian_period"]),
        tuple(
            name
            for name in (
                "phase_z1",
                "phase_z2",
                "excursion_fraction",
                "amplitude_fraction",
                "chi_sleep",
                "chi_wake",
                "tau",
            )
            if name in posterior_arr
        ),
    )
    timings = {
        "compile_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        "sampling_seconds": sampling_seconds,
    }
    inference_data.to_netcdf(run_directory / "posterior.nc")
    manifest = {
        "run_id": arguments.run_id,
        "date": date.today().isoformat(),
        "outcome": "completed",
        "git": git_provenance(REPOSITORY),
        "effective_config_sha256": effective_config_hash(config),
        "warm_start": {
            "source": "RUN-012",
            "step_size": step_size,
            "adapt_step_size": False,
            "adapt_mass_matrix": True,
        },
        "adapt_state_path": "adapt_state.npz",
        "sampler": {
            "design": "entrained",
            "retained_hours": 24.0,
            "burn_in_hours": 48.0,
            "chains": 1,
            "warmup": arguments.warmup,
            "draws": arguments.draws,
            "max_tree_depth": arguments.max_tree_depth,
            "dense_mass": True,
            "fixed_misclassification": 0.01,
            "tau_gate": float(config.section("soft_gate")["tau_gate"]),
            "p0": float(config.section("soft_gate")["p0"]),
            "k": float(config.section("soft_gate")["k"]),
            "fixed_step_size": float(config.section("soft_gate")["fixed_step_size"]),
            "adjoint_mode": config.section("soft_gate")["adjoint_mode"],
            "adjoint_checkpoints": int(
                config.section("soft_gate")["adjoint_checkpoints"]
            ),
        },
        "timings": timings,
        "diagnostics": diagnostics,
        "seed": arguments.seed,
    }
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (run_directory / "timings.json").write_text(
        json.dumps(timings, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Saved {run_directory}", flush=True)


if __name__ == "__main__":
    main()
