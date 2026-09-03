from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.inference import free_running_period
from twopm.likelihood import likelihood_parameter_names, log_likelihood
from twopm.posterior_summaries import solve_entrained_transition
from twopm.soft_gate import circadian_coefficients


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "docs" / "profile_likelihood.json"
PROFILE_DIR = REPOSITORY / "docs" / "profile"
NLL_SOLVER_BOUND = 1e4
MLE_NLL_MAX = 10.6

INTERNAL_NAMES = (
    "phase",
    "log_chi_sleep",
    "log_chi_wake",
    "logit_excursion_fraction",
    "logit_amplitude_fraction",
    "logit_misclassification",
)
PROFILED = ("chi_wake", "chi_sleep", "threshold_gap", "amplitude")
REPORTING_NAMES = (
    "phase",
    "chi_sleep",
    "chi_wake",
    "amplitude",
    "threshold_gap",
    "misclassification",
    "excursion_fraction",
    "amplitude_fraction",
)


def _clip01(value: float, eps: float = 1e-8) -> float:
    return float(np.clip(value, eps, 1.0 - eps))


def reporting_from_internal(
    internal: np.ndarray,
    threshold_mean: float,
) -> dict[str, float]:
    phase, log_chi_s, log_chi_w, logit_s, logit_w, logit_lam = map(float, internal)
    s = float(expit(logit_s))
    w = float(expit(logit_w))
    lam = 0.5 * float(expit(logit_lam))
    excursion = threshold_mean * s
    amplitude = w * excursion
    gap = 2.0 * (1.0 - w) * excursion
    return {
        "phase": phase,
        "chi_sleep": float(np.exp(log_chi_s)),
        "chi_wake": float(np.exp(log_chi_w)),
        "amplitude": amplitude,
        "threshold_gap": gap,
        "misclassification": lam,
        "excursion_fraction": s,
        "amplitude_fraction": w,
        "total_excursion": excursion,
    }


def internal_from_reporting(
    reporting: dict[str, float],
    threshold_mean: float,
) -> np.ndarray:
    s = reporting.get("excursion_fraction")
    w = reporting.get("amplitude_fraction")
    if s is None or w is None:
        amp = float(reporting["amplitude"])
        gap = float(reporting["threshold_gap"])
        excursion = amp + 0.5 * gap
        s = excursion / threshold_mean
        w = amp / excursion if excursion > 0 else 0.5
    s = _clip01(float(s))
    w = _clip01(float(w))
    lam = float(np.clip(reporting["misclassification"], 1e-8, 0.5 - 1e-8))
    return np.asarray(
        (
            float(reporting["phase"]),
            float(np.log(reporting["chi_sleep"])),
            float(np.log(reporting["chi_wake"])),
            float(logit(s)),
            float(logit(w)),
            float(logit(2.0 * lam)),
        ),
        dtype=float,
    )


def reporting_to_likelihood_vector(
    reporting: dict[str, float],
    names: tuple[str, ...],
    period: float,
) -> np.ndarray:
    c1, c2 = circadian_coefficients(
        reporting["amplitude"],
        reporting["phase"],
        period,
    )
    values = {
        "c1": float(c1),
        "c2": float(c2),
        "chi_sleep": reporting["chi_sleep"],
        "chi_wake": reporting["chi_wake"],
        "threshold_gap": reporting["threshold_gap"],
        "misclassification": reporting["misclassification"],
    }
    return np.asarray(tuple(values[name] for name in names), dtype=float)


def prior_scales(config: ProjectConfig) -> dict[str, float]:
    priors = config.section("priors")
    scales = {}
    for name in ("chi_sleep", "chi_wake"):
        log_mean = float(priors[name]["log_mean"])
        log_sd = float(priors[name]["log_sd"])
        scales[name] = float(
            np.sqrt((np.exp(log_sd**2) - 1.0) * np.exp(2 * log_mean + log_sd**2))
        )
    scales["amplitude"] = 0.025
    scales["threshold_gap"] = 0.0607
    return scales


def derived_quantities(
    reporting: dict[str, float],
    config: ProjectConfig,
) -> dict[str, float | bool | str | None]:
    model = config.section("model")
    fixed = config.section("fixed")
    period = float(model["circadian_period"])
    c1, c2 = circadian_coefficients(
        reporting["amplitude"],
        reporting["phase"],
        period,
    )
    transition = solve_entrained_transition(
        c1=float(c1),
        c2=float(c2),
        chi_sleep=float(reporting["chi_sleep"]),
        chi_wake=float(reporting["chi_wake"]),
        threshold_gap=float(reporting["threshold_gap"]),
        threshold_mean=float(fixed["threshold_mean"]),
        mu=float(model["mu"]),
        period=period,
    )
    tau = float(
        free_running_period(
            reporting["chi_sleep"],
            reporting["chi_wake"],
            reporting["threshold_gap"],
            float(fixed["threshold_mean"]),
            float(model["mu"]),
        )
    )
    return {
        "t_on": None if not transition.converged else float(transition.onset),
        "t_off": None if not transition.converged else float(transition.offset),
        "tau": tau,
        "transition_converged": bool(transition.converged),
        "transition_reason": transition.reason,
    }


def optimise(
    *,
    objective,
    start_internal: np.ndarray,
    free_indices: list[int],
    max_fev: int,
) -> tuple[np.ndarray, float, bool, int, bool]:
    free_start = start_internal[free_indices]
    hit_bound = False

    def wrapped(free: np.ndarray) -> float:
        nonlocal hit_bound
        candidate = start_internal.copy()
        candidate[free_indices] = free
        value = float(objective(candidate))
        if value >= 0.99 * NLL_SOLVER_BOUND:
            hit_bound = True
        return value

    steps = np.asarray([0.2, 0.2, 0.8, 0.3, 0.3, 0.2], dtype=float)
    simplex = np.vstack((free_start, free_start + np.diag(steps[free_indices])))
    result = minimize(
        wrapped,
        free_start,
        method="Nelder-Mead",
        options={
            "initial_simplex": simplex,
            "maxfev": max_fev,
            "xatol": 1e-3,
            "fatol": 1e-3,
        },
    )
    best = start_internal.copy()
    best[free_indices] = result.x
    value = float(result.fun)
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite profile optimum: nll={value}")
    if value >= 0.99 * NLL_SOLVER_BOUND:
        raise RuntimeError(
            f"profile optimum stuck on solver bound: nll={value}, nfev={result.nfev}"
        )
    return best, value, bool(result.success), int(result.nfev), hit_bound


def profile_grid(
    name: str,
    centre: float,
    scale: float,
    grid_size: int,
    threshold_mean: float,
    *,
    chi_wake_min: float | None = None,
    chi_wake_max: float | None = None,
    chi_sleep_min: float | None = None,
    chi_sleep_max: float | None = None,
    threshold_gap_min: float | None = None,
    threshold_gap_max: float | None = None,
    amplitude_min: float | None = None,
    amplitude_max: float | None = None,
) -> np.ndarray:
    if name == "chi_wake":
        if chi_wake_min is not None or chi_wake_max is not None:
            lo = 8.0 if chi_wake_min is None else float(chi_wake_min)
            hi = 40.0 if chi_wake_max is None else float(chi_wake_max)
            return np.linspace(lo, hi, grid_size)
        return np.linspace(max(1e-3, centre - 3 * scale), 40.0, grid_size)
    if name == "chi_sleep":
        if chi_sleep_min is not None or chi_sleep_max is not None:
            lo = 1.5 if chi_sleep_min is None else float(chi_sleep_min)
            hi = 10.0 if chi_sleep_max is None else float(chi_sleep_max)
            return np.linspace(lo, hi, grid_size)
        return np.linspace(max(1e-3, centre - 3 * scale), 10.0, grid_size)
    if name == "amplitude":
        if amplitude_min is not None or amplitude_max is not None:
            lo = 0.02 if amplitude_min is None else float(amplitude_min)
            hi = 0.25 if amplitude_max is None else float(amplitude_max)
            return np.clip(
                np.linspace(lo, hi, grid_size),
                1e-4,
                threshold_mean - 1e-3,
            )
        return np.clip(
            np.linspace(centre - 3 * scale, centre + 3 * scale, grid_size),
            1e-4,
            threshold_mean - 1e-3,
        )
    if name == "threshold_gap":
        if threshold_gap_min is not None or threshold_gap_max is not None:
            lo = 0.25 if threshold_gap_min is None else float(threshold_gap_min)
            hi = 0.75 if threshold_gap_max is None else float(threshold_gap_max)
            return np.clip(
                np.linspace(lo, hi, grid_size),
                1e-3,
                2 * threshold_mean - 1e-3,
            )
        return np.clip(
            np.linspace(centre - 3 * scale, centre + 3 * scale, grid_size),
            1e-3,
            2 * threshold_mean - 1e-3,
        )
    raise KeyError(name)


def _point_path(name: str, index: int, profile_dir: Path | None = None) -> Path:
    directory = PROFILE_DIR if profile_dir is None else profile_dir
    return directory / f"{name}_{index:02d}.json"


def _write_point(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_point(path: Path) -> dict:
    return json.loads(path.read_text())


def _apply_fixed_physical(
    internal: np.ndarray,
    *,
    name: str,
    value: float,
    threshold_mean: float,
) -> np.ndarray:
    reporting = reporting_from_internal(internal, threshold_mean)
    if name == "chi_wake":
        reporting["chi_wake"] = float(value)
    elif name == "chi_sleep":
        reporting["chi_sleep"] = float(value)
    elif name == "amplitude":
        excursion = max(reporting["total_excursion"], 1e-8)
        reporting["amplitude"] = float(value)
        reporting["amplitude_fraction"] = _clip01(float(value) / excursion)
        reporting["threshold_gap"] = 2.0 * (1.0 - reporting["amplitude_fraction"]) * excursion
    elif name == "threshold_gap":
        excursion = max(reporting["total_excursion"], 1e-8)
        reporting["threshold_gap"] = float(value)
        reporting["amplitude_fraction"] = _clip01(
            1.0 - float(value) / (2.0 * excursion)
        )
        reporting["amplitude"] = reporting["amplitude_fraction"] * excursion
    else:
        raise KeyError(name)
    return internal_from_reporting(reporting, threshold_mean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--max-fev", type=int, default=1000)
    parser.add_argument("--retained-hours", type=float, default=72.0)
    parser.add_argument("--burn-in-hours", type=float, default=72.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mle-nll-max",
        type=float,
        default=MLE_NLL_MAX,
        help="Abort if unrestricted MLE nll exceeds this (catches bad optima).",
    )
    parser.add_argument(
        "--mle-relative-tol",
        type=float,
        default=0.05,
        help=(
            "After all grids, unrestricted MLE must lie within this relative "
            "tolerance of the campaign-wide profile minimum "
            "(mle <= best * (1 + tol)). Toothless absolute bounds fail at "
            "short horizons; this catches under-converged unrestricted NM."
        ),
    )
    parser.add_argument(
        "--fail-on-underconverged-mle",
        action="store_true",
        help="Exit non-zero if unrestricted MLE fails the relative check.",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        default=list(PROFILED),
        choices=list(PROFILED),
        help="Which coordinates to profile (default: all four).",
    )
    parser.add_argument("--chi-wake-min", type=float, default=None)
    parser.add_argument("--chi-wake-max", type=float, default=None)
    parser.add_argument("--chi-sleep-min", type=float, default=None)
    parser.add_argument("--chi-sleep-max", type=float, default=None)
    parser.add_argument("--threshold-gap-min", type=float, default=None)
    parser.add_argument("--threshold-gap-max", type=float, default=None)
    parser.add_argument("--amplitude-min", type=float, default=None)
    parser.add_argument("--amplitude-max", type=float, default=None)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Per-point JSON directory (default: docs/profile).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary JSON path (default: docs/profile_likelihood.json).",
    )
    arguments = parser.parse_args()

    global PROFILE_DIR, OUTPUT
    if arguments.profile_dir is not None:
        PROFILE_DIR = (
            arguments.profile_dir
            if arguments.profile_dir.is_absolute()
            else REPOSITORY / arguments.profile_dir
        )
    if arguments.output is not None:
        OUTPUT = (
            arguments.output
            if arguments.output.is_absolute()
            else REPOSITORY / arguments.output
        )
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    base = load_config(REPOSITORY / "config" / "model.yaml")
    data = deepcopy(base.data)
    data["designs"]["entrained"]["duration"] = arguments.retained_hours
    data["designs"]["entrained"]["burn_in_hours"] = arguments.burn_in_hours
    config = ProjectConfig(data=data, source=base.source)

    model = config.section("model")
    fixed = config.section("fixed")
    recovery = config.section("recovery")
    names = likelihood_parameter_names(config)
    period = float(model["circadian_period"])
    threshold_mean = float(fixed["threshold_mean"])
    seed = int(recovery["seed"] if arguments.seed is None else arguments.seed)
    scales = prior_scales(config)

    truth_parameters = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
        phase=float(recovery["true_phase"]),
    )
    recording = generate_recording(
        jax.random.PRNGKey(seed),
        config,
        truth_parameters,
    )
    truth_reporting = {
        "phase": float(truth_parameters.phase),
        "chi_sleep": float(truth_parameters.chi_sleep),
        "chi_wake": float(truth_parameters.chi_wake),
        "amplitude": float(truth_parameters.amplitude),
        "threshold_gap": float(truth_parameters.upper - truth_parameters.lower),
        "misclassification": float(truth_parameters.misclassification),
        "excursion_fraction": float(
            (truth_parameters.amplitude + 0.5 * (truth_parameters.upper - truth_parameters.lower))
            / threshold_mean
        ),
        "amplitude_fraction": float(
            truth_parameters.amplitude
            / (
                truth_parameters.amplitude
                + 0.5 * (truth_parameters.upper - truth_parameters.lower)
            )
        ),
    }
    truth_internal = internal_from_reporting(truth_reporting, threshold_mean)

    bound_hits: list[str] = []

    def negative_log_likelihood(internal: np.ndarray) -> float:
        reporting = reporting_from_internal(internal, threshold_mean)
        value = log_likelihood(
            jnp.asarray(reporting_to_likelihood_vector(reporting, names, period)),
            recording.observations,
            config,
        )
        result = -float(value)
        if not np.isfinite(result):
            bound_hits.append(
                "nonfinite@"
                + ",".join(f"{k}={reporting[k]:.4g}" for k in ("chi_wake", "amplitude"))
            )
            return NLL_SOLVER_BOUND
        return result

    starts = [truth_internal.copy()]
    for point in recovery["starting_points"]:
        phase, chi_s, chi_w, amp, gap, error = map(float, point)
        excursion = amp + 0.5 * gap
        if excursion <= 0 or excursion >= threshold_mean:
            continue
        reporting = {
            "phase": phase,
            "chi_sleep": chi_s,
            "chi_wake": chi_w,
            "amplitude": amp,
            "threshold_gap": gap,
            "misclassification": error,
            "excursion_fraction": excursion / threshold_mean,
            "amplitude_fraction": amp / excursion,
        }
        starts.append(internal_from_reporting(reporting, threshold_mean))

    print("Finding unrestricted MLE neighbourhood...", flush=True)
    print(
        f"coordinate system: unconstrained internals over (s,w) "
        f"with s,w∈(0,1) — circadian extrema feasible by construction; "
        f"solver-failure bound={NLL_SOLVER_BOUND:g}",
        flush=True,
    )
    mle_candidates = []
    for start in starts:
        estimate, nll, success, nfev, hit = optimise(
            objective=negative_log_likelihood,
            start_internal=start,
            free_indices=list(range(len(INTERNAL_NAMES))),
            max_fev=arguments.max_fev,
        )
        mle_candidates.append((estimate, nll, success, nfev, hit))
        print(f"  start nll={nll:.6f} success={success} nfev={nfev} hit_bound={hit}", flush=True)
    mle_internal, mle_nll, _, _, _ = min(mle_candidates, key=lambda item: item[1])
    print(f"MLE nll={mle_nll:.6f}", flush=True)
    if mle_nll > arguments.mle_nll_max:
        raise RuntimeError(
            f"unrestricted MLE nll={mle_nll:.6f} exceeds assert "
            f"≤{arguments.mle_nll_max}; aborting before grids "
            "(likely wrong coordinate system or likelihood path)"
        )

    mle_reporting = reporting_from_internal(mle_internal, threshold_mean)
    profiles: dict[str, object] = {}

    for name in arguments.parameters:
        if name in ("chi_wake", "chi_sleep", "amplitude", "threshold_gap"):
            centre = float(mle_reporting[name if name != "threshold_gap" else "threshold_gap"])
        else:
            raise KeyError(name)
        scale = scales[name]
        grid = profile_grid(
            name,
            centre,
            scale,
            arguments.grid_size,
            threshold_mean,
            chi_wake_min=arguments.chi_wake_min,
            chi_wake_max=arguments.chi_wake_max,
            chi_sleep_min=arguments.chi_sleep_min,
            chi_sleep_max=arguments.chi_sleep_max,
            threshold_gap_min=arguments.threshold_gap_min,
            threshold_gap_max=arguments.threshold_gap_max,
            amplitude_min=arguments.amplitude_min,
            amplitude_max=arguments.amplitude_max,
        )
        free_indices = list(range(len(INTERNAL_NAMES)))
        points = []
        continuation = mle_internal.copy()
        cold = starts[0].copy()

        for grid_index, value in enumerate(grid):
            path = _point_path(name, grid_index, PROFILE_DIR)
            if path.exists():
                best = _load_point(path)
                points.append(best)
                continuation = internal_from_reporting(best["reporting"], threshold_mean)
                print(
                    f"{name}[{grid_index}]={value:.5g} RESUMED nll={best['negative_log_likelihood']:.6f} "
                    f"t_on={best.get('t_on')} tau={best.get('tau')}",
                    flush=True,
                )
                continue

            candidates = []
            for label, start in (("continuation", continuation), ("cold", cold)):
                trial = _apply_fixed_physical(
                    start,
                    name=name,
                    value=float(value),
                    threshold_mean=threshold_mean,
                )
                if name in ("chi_wake", "chi_sleep"):
                    locked = {
                        "chi_wake": INTERNAL_NAMES.index("log_chi_wake"),
                        "chi_sleep": INTERNAL_NAMES.index("log_chi_sleep"),
                    }[name]
                    trial_free = [i for i in free_indices if i != locked]
                elif name in ("amplitude", "threshold_gap"):
                    locked = INTERNAL_NAMES.index("logit_amplitude_fraction")
                    trial_free = [i for i in free_indices if i != locked]
                else:
                    trial_free = free_indices

                estimate, nll, success, nfev, hit = optimise(
                    objective=negative_log_likelihood,
                    start_internal=trial,
                    free_indices=trial_free,
                    max_fev=arguments.max_fev,
                )
                estimate = _apply_fixed_physical(
                    estimate,
                    name=name,
                    value=float(value),
                    threshold_mean=threshold_mean,
                )
                reporting = reporting_from_internal(estimate, threshold_mean)
                nll = float(negative_log_likelihood(estimate))
                derived = derived_quantities(reporting, config)
                record = {
                    "fixed_parameter": name,
                    "fixed_value": float(value),
                    "grid_index": grid_index,
                    "seed": label,
                    "negative_log_likelihood": nll,
                    "log_likelihood": -nll,
                    "delta_nll_from_mle": nll - mle_nll,
                    "success": success,
                    "nfev": nfev,
                    "hit_solver_bound": hit,
                    "reporting": reporting,
                    **derived,
                }
                if hit:
                    bound_hits.append(f"{name}[{grid_index}]/{label}")
                    print(
                        f"WARNING: solver-bound hit at {name}[{grid_index}] seed={label}",
                        flush=True,
                    )
                candidates.append(record)

            best = min(candidates, key=lambda item: item["negative_log_likelihood"])
            if best["negative_log_likelihood"] >= 0.99 * NLL_SOLVER_BOUND:
                raise RuntimeError(
                    f"accepted profile point on solver bound at {name}[{grid_index}]"
                )
            continuation = internal_from_reporting(best["reporting"], threshold_mean)
            _write_point(path, best)
            points.append(best)
            print(
                f"{name}[{grid_index}]={value:.5g} nll={best['negative_log_likelihood']:.6f} "
                f"success={best['success']} nfev={best['nfev']} "
                f"t_on={best['t_on']} tau={best['tau']}",
                flush=True,
            )

        profiles[name] = {
            "prior_sd": scale,
            "centre": centre,
            "grid": [float(v) for v in grid],
            "points": points,
        }

    campaign_nlls = [
        float(point["negative_log_likelihood"])
        for profile in profiles.values()
        for point in profile["points"]  # type: ignore[index]
        if point.get("negative_log_likelihood") is not None
        and point["negative_log_likelihood"] < 0.99 * NLL_SOLVER_BOUND
    ]
    campaign_best_nll = float(min(campaign_nlls)) if campaign_nlls else float("nan")
    relative_excess = (
        (mle_nll - campaign_best_nll) / max(abs(campaign_best_nll), 1e-12)
        if campaign_nlls
        else float("nan")
    )
    underconverged = bool(
        campaign_nlls and relative_excess > float(arguments.mle_relative_tol)
    )
    for profile in profiles.values():
        for point in profile["points"]:  # type: ignore[index]
            point["delta_nll_from_campaign_best"] = (
                float(point["negative_log_likelihood"]) - campaign_best_nll
            )

    payload = {
        "retained_hours": arguments.retained_hours,
        "burn_in_hours": arguments.burn_in_hours,
        "seed": seed,
        "grid_size": arguments.grid_size,
        "max_fev": arguments.max_fev,
        "coordinate_system": "constrained_(s,w)_via_unconstrained_internals",
        "solver_bound": NLL_SOLVER_BOUND,
        "mle_nll_max_assert": arguments.mle_nll_max,
        "mle_relative_tol": float(arguments.mle_relative_tol),
        "prior_scales": scales,
        "truth_reporting": truth_reporting,
        "mle_reporting": mle_reporting,
        "mle_negative_log_likelihood": mle_nll,
        "campaign_best_negative_log_likelihood": campaign_best_nll,
        "mle_relative_excess_over_campaign_best": relative_excess,
        "unrestricted_mle_underconverged": underconverged,
        "raue_baseline": "campaign_best_across_all_profiled_grids",
        "mle_derived": derived_quantities(mle_reporting, config),
        "solver_bound_hits": bound_hits,
        "profiles": profiles,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {OUTPUT}")
    print(
        f"campaign_best_nll={campaign_best_nll:.6f} "
        f"unrestricted_mle={mle_nll:.6f} "
        f"relative_excess={relative_excess:.4f} "
        f"underconverged={underconverged}",
        flush=True,
    )
    if bound_hits:
        print(f"WARNING: {len(bound_hits)} solver-bound encounters: {bound_hits[:10]}")
    if underconverged:
        message = (
            f"unrestricted MLE nll={mle_nll:.6f} exceeds campaign profile "
            f"minimum {campaign_best_nll:.6f} by {100 * relative_excess:.1f}% "
            f"(tol {100 * arguments.mle_relative_tol:.1f}%). "
            "Use campaign_best as the Raue baseline; unrestricted NM is "
            "under-converged on the flat fibre."
        )
        print(f"WARNING: {message}", flush=True)
        if arguments.fail_on_underconverged_mle:
            raise RuntimeError(message)


if __name__ == "__main__":
    main()
