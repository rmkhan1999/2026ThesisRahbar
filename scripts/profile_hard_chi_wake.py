from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_likelihood import (  # noqa: E402
    INTERNAL_NAMES,
    NLL_SOLVER_BOUND,
    _apply_fixed_physical,
    derived_quantities,
    internal_from_reporting,
    optimise,
    reporting_from_internal,
    reporting_to_likelihood_vector,
)

from twopm.config import ProjectConfig, load_config
from twopm.generative import standard_parameters
from twopm.likelihood import likelihood_parameter_names, log_likelihood


REPOSITORY = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPOSITORY / "docs" / "profile_hard"
SUMMARY = REPOSITORY / "docs" / "profile_hard_chi_wake.json"
HARD_NPZ = REPOSITORY / "docs" / "hard_generated_recording.npz"
MLE_NLL_MAX = 15.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--chi-wake-min", type=float, default=8.0)
    parser.add_argument("--chi-wake-max", type=float, default=40.0)
    parser.add_argument("--max-fev", type=int, default=1000)
    parser.add_argument("--mle-nll-max", type=float, default=MLE_NLL_MAX)
    parser.add_argument("--retained-hours", type=float, default=24.0)
    parser.add_argument("--burn-in-hours", type=float, default=48.0)
    arguments = parser.parse_args()

    if not HARD_NPZ.exists():
        raise FileNotFoundError(f"missing {HARD_NPZ}; run generate_hard_recording.py")

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    base = load_config(REPOSITORY / "config" / "model.yaml")
    data = deepcopy(base.data)
    data["designs"]["entrained"]["duration"] = arguments.retained_hours
    data["designs"]["entrained"]["burn_in_hours"] = arguments.burn_in_hours
    data["observation"]["misclassification"] = 0.01
    config = ProjectConfig(data=data, source=base.source)

    model = config.section("model")
    fixed = config.section("fixed")
    recovery = config.section("recovery")
    names = likelihood_parameter_names(config)
    period = float(model["circadian_period"])
    threshold_mean = float(fixed["threshold_mean"])

    payload = np.load(HARD_NPZ)
    observations = jnp.asarray(payload["observations"])

    truth = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
        phase=float(recovery["true_phase"]),
    )
    truth_reporting = {
        "phase": float(truth.phase),
        "chi_sleep": float(truth.chi_sleep),
        "chi_wake": float(truth.chi_wake),
        "amplitude": float(truth.amplitude),
        "threshold_gap": float(truth.upper - truth.lower),
        "misclassification": 0.01,
        "excursion_fraction": float(
            (truth.amplitude + 0.5 * (truth.upper - truth.lower)) / threshold_mean
        ),
        "amplitude_fraction": float(
            truth.amplitude
            / (truth.amplitude + 0.5 * (truth.upper - truth.lower))
        ),
    }
    truth_internal = internal_from_reporting(truth_reporting, threshold_mean)

    def negative_log_likelihood(internal: np.ndarray) -> float:
        reporting = reporting_from_internal(internal, threshold_mean)
        value = log_likelihood(
            jnp.asarray(reporting_to_likelihood_vector(reporting, names, period)),
            observations,
            config,
        )
        result = -float(value)
        if not np.isfinite(result):
            return NLL_SOLVER_BOUND
        return result

    starts = [truth_internal.copy()]
    for point in recovery["starting_points"]:
        phase, chi_s, chi_w, amp, gap, error = map(float, point)
        excursion = amp + 0.5 * gap
        if excursion <= 0 or excursion >= threshold_mean:
            continue
        starts.append(
            internal_from_reporting(
                {
                    "phase": phase,
                    "chi_sleep": chi_s,
                    "chi_wake": chi_w,
                    "amplitude": amp,
                    "threshold_gap": gap,
                    "misclassification": error,
                    "excursion_fraction": excursion / threshold_mean,
                    "amplitude_fraction": amp / excursion,
                },
                threshold_mean,
            )
        )

    print("Finding unrestricted MLE on hard-generated labels...", flush=True)
    mle_candidates = []
    for start in starts:
        estimate, nll, success, nfev, hit = optimise(
            objective=negative_log_likelihood,
            start_internal=start,
            free_indices=list(range(len(INTERNAL_NAMES))),
            max_fev=arguments.max_fev,
        )
        mle_candidates.append((estimate, nll, success, nfev, hit))
        print(
            f"  start nll={nll:.6f} success={success} nfev={nfev} hit_bound={hit}",
            flush=True,
        )
    mle_internal, mle_nll, _, _, _ = min(mle_candidates, key=lambda item: item[1])
    print(f"MLE nll={mle_nll:.6f}", flush=True)
    if mle_nll > arguments.mle_nll_max:
        raise RuntimeError(
            f"unrestricted MLE nll={mle_nll:.6f} exceeds assert "
            f"≤{arguments.mle_nll_max}; aborting before χ_w grid"
        )

    mle_reporting = reporting_from_internal(mle_internal, threshold_mean)
    grid = np.linspace(
        arguments.chi_wake_min, arguments.chi_wake_max, arguments.grid_size
    )
    locked = INTERNAL_NAMES.index("log_chi_wake")
    free_indices = [i for i in range(len(INTERNAL_NAMES)) if i != locked]
    continuation = mle_internal.copy()
    cold = starts[0].copy()
    points = []

    for grid_index, value in enumerate(grid):
        path = PROFILE_DIR / f"chi_wake_{grid_index:02d}.json"
        if path.exists():
            best = json.loads(path.read_text())
            points.append(best)
            continuation = internal_from_reporting(best["reporting"], threshold_mean)
            print(
                f"chi_wake[{grid_index}]={value:.5g} RESUMED "
                f"nll={best['negative_log_likelihood']:.6f} "
                f"t_on={best.get('t_on')} tau={best.get('tau')}",
                flush=True,
            )
            continue

        candidates = []
        for label, start in (("continuation", continuation), ("cold", cold)):
            trial = _apply_fixed_physical(
                start,
                name="chi_wake",
                value=float(value),
                threshold_mean=threshold_mean,
            )
            estimate, nll, success, nfev, hit = optimise(
                objective=negative_log_likelihood,
                start_internal=trial,
                free_indices=free_indices,
                max_fev=arguments.max_fev,
            )
            estimate = _apply_fixed_physical(
                estimate,
                name="chi_wake",
                value=float(value),
                threshold_mean=threshold_mean,
            )
            reporting = reporting_from_internal(estimate, threshold_mean)
            nll = float(negative_log_likelihood(estimate))
            derived = derived_quantities(reporting, config)
            candidates.append(
                {
                    "fixed_parameter": "chi_wake",
                    "fixed_value": float(value),
                    "grid_index": grid_index,
                    "seed": label,
                    "negative_log_likelihood": nll,
                    "delta_nll_from_mle": nll - mle_nll,
                    "success": success,
                    "nfev": nfev,
                    "hit_solver_bound": hit,
                    "reporting": reporting,
                    **derived,
                }
            )
        best = min(candidates, key=lambda item: item["negative_log_likelihood"])
        if best["negative_log_likelihood"] >= 0.99 * NLL_SOLVER_BOUND:
            raise RuntimeError(
                f"accepted profile point on solver bound at chi_wake[{grid_index}]"
            )
        continuation = internal_from_reporting(best["reporting"], threshold_mean)
        path.write_text(json.dumps(best, indent=2) + "\n")
        points.append(best)
        print(
            f"chi_wake[{grid_index}]={value:.5g} "
            f"nll={best['negative_log_likelihood']:.6f} "
            f"dnll={best['delta_nll_from_mle']:.6f} "
            f"t_on={best['t_on']} tau={best['tau']}",
            flush=True,
        )

    summary = {
        "date": "2026-08-10",
        "generator": "hard_switch",
        "observations": str(HARD_NPZ.relative_to(REPOSITORY)),
        "retained_hours": arguments.retained_hours,
        "burn_in_hours": arguments.burn_in_hours,
        "grid": [float(v) for v in grid],
        "mle_negative_log_likelihood": mle_nll,
        "mle_reporting": mle_reporting,
        "mle_derived": derived_quantities(mle_reporting, config),
        "points": points,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
