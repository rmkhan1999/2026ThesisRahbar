from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from datetime import date
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import arviz as az
import numpy as np
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, sample_parameters
from twopm.inference import free_running_period, numpyro_model
from twopm.soft_gate import soft_transition_times


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPOSITORY / "docs" / "sbc"

RANK_SITES = (
    "phase_z1",
    "phase_z2",
    "excursion_fraction",
    "amplitude_fraction",
    "chi_sleep",
    "chi_wake",
    "tau",
)
CIRCULAR_RANK_SITES = ("phase",)


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


def _config_hash(config: ProjectConfig) -> str:
    return hashlib.sha256(
        json.dumps(config.data, sort_keys=True, default=str).encode()
    ).hexdigest()


def _rank(true_value: float, samples: np.ndarray) -> float:
    arr = np.asarray(samples, dtype=float).ravel()
    return float(np.mean(arr < true_value))


def _circular_rank(true_phase: float, samples: np.ndarray, period: float) -> float:
    arr = np.asarray(samples, dtype=float).ravel()
    diffs = (arr - true_phase + period / 2.0) % period - period / 2.0
    return float(np.mean(diffs < 0.0) + 0.5 * np.mean(diffs == 0.0))


def _thin_indices(n_draws: int, n_keep: int) -> np.ndarray:
    n_keep = max(1, min(int(n_keep), int(n_draws)))
    if n_keep >= n_draws:
        return np.arange(n_draws, dtype=int)
    return np.unique(np.linspace(0, n_draws - 1, n_keep).astype(int))


def _ranks_for_samples(
    generating: dict,
    samples_by_site: dict[str, np.ndarray],
    period: float,
) -> dict[str, float]:
    ranks: dict[str, float] = {}
    for name in RANK_SITES:
        if name in generating and name in samples_by_site:
            ranks[name] = _rank(generating[name], samples_by_site[name])
    for name in CIRCULAR_RANK_SITES:
        if name in generating and name in samples_by_site:
            ranks[name] = _circular_rank(
                generating[name], samples_by_site[name], period
            )
            ranks[f"{name}_linear_ecdf"] = _rank(
                generating[name], samples_by_site[name]
            )
    return ranks


def _generating_and_truth(
    seed: int, config: ProjectConfig
) -> tuple[dict, np.ndarray, np.ndarray, jax.Array]:
    key = jax.random.PRNGKey(seed)
    gen_key, label_key, mcmc_key = jax.random.split(key, 3)
    parameters = sample_parameters(gen_key, config, "entrained")
    recording = generate_recording(label_key, config, parameters, "entrained")
    labels = np.asarray(recording.observations)
    level = float(config.section("validation")["transition_gate_level"])
    true_transitions = np.asarray(
        soft_transition_times(recording.time, recording.gate, level)
    )
    model = config.section("model")
    fixed = config.section("fixed")
    true_tau = float(
        free_running_period(
            parameters.chi_sleep,
            parameters.chi_wake,
            parameters.upper - parameters.lower,
            float(fixed["threshold_mean"]),
            float(model["mu"]),
        )
    )
    generating = {
        "chi_sleep": float(parameters.chi_sleep),
        "chi_wake": float(parameters.chi_wake),
        "amplitude": float(parameters.amplitude),
        "phase": float(parameters.phase),
        "threshold_gap": float(parameters.upper - parameters.lower),
        "c1": float(parameters.c1),
        "c2": float(parameters.c2),
        "tau": true_tau,
        "phase_z1": float(parameters.phase_z1),
        "phase_z2": float(parameters.phase_z2),
        "excursion_fraction": float(parameters.excursion_fraction),
        "amplitude_fraction": float(parameters.amplitude_fraction),
        "upper": float(parameters.upper),
        "lower": float(parameters.lower),
    }
    return generating, labels, true_transitions, mcmc_key


def _transition_record(true_transitions: np.ndarray) -> dict:
    times = np.asarray(true_transitions, dtype=float)
    return {
        "true_transition_count": int(times.size),
        "true_transition_times": times.tolist(),
        "true_first_transition": float(times[0]) if times.size else None,
        "true_onset_times": times[0::2].tolist() if times.size else [],
        "true_offset_times": times[1::2].tolist() if times.size else [],
    }


def _payload_from_posterior(
    *,
    replicate: int,
    seed: int,
    config: ProjectConfig,
    generating: dict,
    true_transitions: np.ndarray,
    posterior: dict[str, np.ndarray],
    diverging: np.ndarray,
    n_steps: np.ndarray,
    wall: float,
    warmup: int,
    draws: int,
    max_tree_depth: int,
) -> dict:
    period = float(config.section("model")["circadian_period"])
    fixed = config.section("fixed")
    model = config.section("model")

    if "tau" not in posterior and "chi_sleep" in posterior:
        posterior["tau"] = np.asarray(
            free_running_period(
                posterior["chi_sleep"],
                posterior["chi_wake"],
                posterior["threshold_gap"],
                float(fixed["threshold_mean"]),
                float(model["mu"]),
            )
        )

    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 1:
            return a.reshape(1, -1)
        return a

    posterior_2d = {k: _ensure_2d(v) for k, v in posterior.items()}
    diverging = _ensure_2d(np.asarray(diverging))
    n_steps = _ensure_2d(np.asarray(n_steps))

    idata = az.from_dict(
        posterior={k: v for k, v in posterior_2d.items() if v.ndim >= 2},
        sample_stats={"diverging": diverging, "n_steps": n_steps},
    )
    diagnostic_names = [name for name in RANK_SITES if name in idata.posterior]
    ess_bulk = {
        name: float(az.ess(idata, var_names=[name], method="bulk")[name].values)
        for name in diagnostic_names
    }

    ranks = _ranks_for_samples(generating, posterior_2d, period)

    draw_sites = list(
        dict.fromkeys(
            list(RANK_SITES)
            + list(CIRCULAR_RANK_SITES)
            + ["threshold_gap", "amplitude", "c1", "c2"]
        )
    )
    posterior_draws = {
        name: np.asarray(posterior_2d[name]).ravel().tolist()
        for name in draw_sites
        if name in posterior_2d
    }
    n_draws = int(next(iter(posterior_draws.values())).__len__()) if posterior_draws else 0

    ranks_thinned: dict[str, float] = {}
    thinned_n: dict[str, int] = {}
    for name in list(RANK_SITES) + list(CIRCULAR_RANK_SITES):
        if name not in posterior_draws or name not in generating:
            continue
        ess_n = ess_bulk.get(name)
        if ess_n is None or not np.isfinite(ess_n):
            ess_n = min(
                (v for v in ess_bulk.values() if np.isfinite(v)),
                default=1.0,
            )
        n_keep = max(1, int(ess_n))
        idx = _thin_indices(n_draws, n_keep)
        thinned_n[name] = int(idx.size)
        thinned = {name: np.asarray(posterior_draws[name])[idx]}
        if name in CIRCULAR_RANK_SITES:
            ranks_thinned[name] = _circular_rank(
                generating[name], thinned[name], period
            )
            ranks_thinned[f"{name}_linear_ecdf"] = _rank(
                generating[name], thinned[name]
            )
        else:
            ranks_thinned[name] = _rank(generating[name], thinned[name])

    prior_key = jax.random.PRNGKey(seed + 10_000)
    prior_draws = []
    for k in jax.random.split(prior_key, 64):
        p = sample_parameters(k, config, "entrained")
        prior_draws.append(
            {
                "chi_sleep": float(p.chi_sleep),
                "chi_wake": float(p.chi_wake),
                "amplitude": float(p.amplitude),
                "phase": float(p.phase),
                "threshold_gap": float(p.upper - p.lower),
                "tau": float(
                    free_running_period(
                        p.chi_sleep,
                        p.chi_wake,
                        p.upper - p.lower,
                        float(fixed["threshold_mean"]),
                        float(model["mu"]),
                    )
                ),
            }
        )
    contractions = {}
    for name in ("chi_sleep", "chi_wake", "amplitude", "tau", "threshold_gap"):
        if name not in posterior_2d:
            continue
        prior_var = float(np.var([row[name] for row in prior_draws], ddof=1))
        post_var = float(np.var(np.asarray(posterior_2d[name]).ravel(), ddof=1))
        contractions[name] = (
            None if prior_var <= 0 else float(1.0 - post_var / prior_var)
        )

    transition_summary = _transition_record(true_transitions)
    transition_summary["posterior_mean_params"] = {
        name: float(np.mean(np.asarray(posterior_2d[name])))
        for name in ("c1", "c2", "chi_sleep", "chi_wake", "threshold_gap", "phase")
        if name in posterior_2d
    }

    max_steps = 2**max_tree_depth - 1
    saturation = float(np.mean(n_steps >= max_steps)) if n_steps.size else None

    return {
        "replicate": replicate,
        "date": date.today().isoformat(),
        "seed": seed,
        "config_sha256": _config_hash(config),
        "sampler": {
            "design": "entrained",
            "retained_hours": 24.0,
            "burn_in_hours": 48.0,
            "warmup": warmup,
            "draws": draws,
            "chains": 1,
            "max_tree_depth": max_tree_depth,
            "dense_mass": True,
            "fixed_misclassification": 0.01,
            "adjoint_mode": config.section("soft_gate")["adjoint_mode"],
            "adjoint_checkpoints": int(
                config.section("soft_gate")["adjoint_checkpoints"]
            ),
            "chain_method": "sequential",
        },
        "generating_parameters": generating,
        "posterior_draws": posterior_draws,
        "ranks": ranks,
        "ranks_thinned": ranks_thinned,
        "ranks_thinned_n": thinned_n,
        "rank_sites": list(RANK_SITES) + list(CIRCULAR_RANK_SITES),
        "rank_definition": (
            "For each site, fractional rank of the prior draw θ_true among "
            "posterior samples: mean(θ_post < θ_true). Phase uses signed-wrap "
            "circular rank; phase_linear_ecdf is the cut-circle linear ECDF. "
            "ranks_thinned uses int(ESS_bulk) evenly spaced draws per site "
            "(Talts et al.: autocorrelation can induce artefactual U-shape)."
        ),
        "contraction": contractions,
        "transition_times": transition_summary,
        "diagnostics": {
            "ess_bulk": ess_bulk,
            "ess_bulk_min": min(ess_bulk.values()) if ess_bulk else None,
            "divergences": int(np.sum(diverging)),
            "total_draws": int(diverging.size),
            "tree_depth_saturation_fraction": saturation,
            "median_num_steps": float(np.median(n_steps)) if n_steps.size else None,
            "wall_seconds": wall,
        },
        "entrainment_failed": bool(true_transitions.size == 0),
        "true_transition_count": int(true_transitions.size),
    }


def _run_one(
    *,
    replicate: int,
    seed: int,
    config: ProjectConfig,
    warmup: int,
    draws: int,
    max_tree_depth: int,
) -> dict:
    generating, labels, true_transitions, mcmc_key = _generating_and_truth(
        seed, config
    )

    def model_fn(observed_labels):
        numpyro_model(observed_labels, config, "entrained")

    kernel = NUTS(
        model_fn,
        target_accept_prob=0.8,
        max_tree_depth=max_tree_depth,
        dense_mass=True,
        init_strategy=init_to_median(),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=draws,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )
    start = time.perf_counter()
    mcmc.run(mcmc_key, labels, extra_fields=("diverging", "num_steps"))
    wall = time.perf_counter() - start
    samples = mcmc.get_samples(group_by_chain=True)
    extras = mcmc.get_extra_fields(group_by_chain=True) or {}
    posterior = {name: np.asarray(values) for name, values in samples.items()}
    diverging = np.asarray(extras.get("diverging", np.zeros((1, draws), dtype=bool)))
    n_steps = np.asarray(extras.get("num_steps", np.zeros((1, draws))))
    return _payload_from_posterior(
        replicate=replicate,
        seed=seed,
        config=config,
        generating=generating,
        true_transitions=true_transitions,
        posterior=posterior,
        diverging=diverging,
        n_steps=n_steps,
        wall=wall,
        warmup=warmup,
        draws=draws,
        max_tree_depth=max_tree_depth,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-replicates", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=20260808)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--max-tree-depth", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    config = _config()

    for offset in range(arguments.n_replicates):
        index = arguments.start_index + offset
        path = arguments.output_dir / f"replicate_{index:03d}.json"
        if path.exists():
            print(f"skip existing {path}", flush=True)
            continue
        seed = arguments.seed_base + index
        print(f"running replicate {index} seed={seed}", flush=True)
        partial = path.with_suffix(".partial.json")
        partial.write_text(
            json.dumps({"replicate": index, "status": "running", "seed": seed}) + "\n"
        )
        try:
            payload = _run_one(
                replicate=index,
                seed=seed,
                config=config,
                warmup=arguments.warmup,
                draws=arguments.draws,
                max_tree_depth=arguments.max_tree_depth,
            )
            path.write_text(json.dumps(payload, indent=2) + "\n")
            partial.unlink(missing_ok=True)
            print(
                f"saved {path} ess_min={payload['diagnostics']['ess_bulk_min']} "
                f"div={payload['diagnostics']['divergences']} "
                f"wall_h={payload['diagnostics']['wall_seconds']/3600:.3f}",
                flush=True,
            )
        except Exception as error:
            partial.write_text(
                json.dumps(
                    {
                        "replicate": index,
                        "status": "crashed",
                        "seed": seed,
                        "error": f"{type(error).__name__}: {error}",
                    },
                    indent=2,
                )
                + "\n"
            )
            raise


if __name__ == "__main__":
    main()
