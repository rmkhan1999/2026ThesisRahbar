from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import arviz as az
import jax
import numpy as np

from twopm.config import load_config
from twopm.generative import sample_parameters, standard_parameters
from twopm.hard_switch import simulate_hard_switch
from twopm.inference import free_running_period
from twopm.posterior_summaries import (
    posterior_transition_times,
    solve_entrained_transition,
    variance_contraction,
)
from twopm.sampling import _circular_summary
from twopm.soft_gate import simulate_soft_gate_from_config, soft_transition_times


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_POSTERIOR = REPOSITORY / "runs" / "RUN-009" / "posterior.nc"
OUTPUT = REPOSITORY / "docs" / "run009_transition_bias.json"
N_PRIOR = 500
PRIOR_SEED = 20260730
PERIOD = 24.0


def _circular_interval(values: np.ndarray, period: float = PERIOD) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "n_finite": 0,
            "mean": None,
            "sd": None,
            "q05": None,
            "q95": None,
        }
    summary = _circular_summary(finite, period)
    mean = summary["circular_mean_hours"]
    unwrapped = (finite - mean + period / 2) % period + mean - period / 2
    q05, q95 = map(float, np.percentile(unwrapped, [5, 95]))
    return {
        "n_finite": int(finite.size),
        "mean": float(mean),
        "sd": float(summary["circular_sd_hours"]),
        "q05": q05 % period,
        "q95": q95 % period,
        "q05_unwrapped": q05,
        "q95_unwrapped": q95,
    }


def _oracle_in_circular_interval(
    oracle: float, q05_u: float, q95_u: float, mean: float, period: float = PERIOD
) -> bool:
    o = (float(oracle) - mean + period / 2) % period + mean - period / 2
    return bool(q05_u <= o <= q95_u)


def _bias_block(values: np.ndarray, oracle: float | None, *, circular: bool) -> dict:
    if circular:
        stats = _circular_interval(values)
    else:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            stats = {"n_finite": 0, "mean": None, "sd": None, "q05": None, "q95": None}
        else:
            stats = {
                "n_finite": int(finite.size),
                "mean": float(np.mean(finite)),
                "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "q05": float(np.percentile(finite, 5)),
                "q95": float(np.percentile(finite, 95)),
            }
    bias = None if oracle is None or stats["mean"] is None else stats["mean"] - float(oracle)
    if oracle is None or stats["mean"] is None:
        in90 = None
    elif circular:
        in90 = _oracle_in_circular_interval(
            float(oracle),
            stats["q05_unwrapped"],
            stats["q95_unwrapped"],
            stats["mean"],
        )
    else:
        in90 = bool(stats["q05"] <= float(oracle) <= stats["q95"])
    return {
        **{k: v for k, v in stats.items() if not k.endswith("_unwrapped")},
        "oracle": oracle,
        "bias_hours": bias,
        "bias_minutes": None if bias is None else 60.0 * bias,
        "oracle_in_90pct_interval": in90,
        "circular": circular,
    }


def _hard_oracle(config, *, burn_in: float, retained: float) -> dict:
    model = config.section("model")
    recovery = config.section("recovery")
    truth = standard_parameters(
        config,
        amplitude=float(model["circadian_amplitude"]),
        phase=float(recovery["true_phase"]),
    )
    analytic = solve_entrained_transition(
        c1=float(truth.c1),
        c2=float(truth.c2),
        chi_sleep=float(truth.chi_sleep),
        chi_wake=float(truth.chi_wake),
        threshold_gap=float(truth.upper - truth.lower),
        threshold_mean=float(config.section("fixed")["threshold_mean"]),
        mu=float(truth.mu),
        period=float(model["circadian_period"]),
    )
    hard = simulate_hard_switch(
        duration=burn_in + retained,
        dt=float(config.section("hard_switch")["dt"]),
        chi_sleep=float(truth.chi_sleep),
        chi_wake=float(truth.chi_wake),
        mu=float(truth.mu),
        upper=float(truth.upper),
        lower=float(truth.lower),
        initial_pressure=float(config.section("initial_state")["pressure"]),
        initially_asleep=bool(config.section("initial_state")["hard_asleep"]),
        amplitude=float(truth.amplitude),
        phase=float(truth.phase),
        period=float(model["circadian_period"]),
        start_time=0.0,
    )
    times = np.asarray(hard.switch_times, dtype=float)
    states = np.asarray(hard.switch_states, dtype=bool)
    mask = (times >= burn_in) & (times < burn_in + retained)
    times_r = times[mask] - burn_in
    states_r = states[mask]
    onsets = times_r[states_r]
    offsets = times_r[~states_r]

    level = float(config.section("validation")["transition_gate_level"])
    soft = simulate_soft_gate_from_config(
        config,
        duration=burn_in + retained,
        output_step=float(config.section("observation")["epoch_hours"]),
        c1=truth.c1,
        c2=truth.c2,
        chi_sleep=truth.chi_sleep,
        chi_wake=truth.chi_wake,
        upper=truth.upper,
        lower=truth.lower,
        throw=True,
    )
    soft_on, soft_off = _classify_soft(
        soft.time, soft.gate, level, burn_in=burn_in, retained=retained
    )
    tau = float(
        free_running_period(
            float(truth.chi_sleep),
            float(truth.chi_wake),
            float(truth.upper - truth.lower),
            float(config.section("fixed")["threshold_mean"]),
            float(truth.mu),
        )
    )
    return {
        "hard_t_on_clock": None if onsets.size == 0 else float(onsets[0] % PERIOD),
        "hard_t_off_clock": None if offsets.size == 0 else float(offsets[0] % PERIOD),
        "hard_t_on_retained": onsets.tolist(),
        "hard_t_off_retained": offsets.tolist(),
        "analytic_t_on": None if not analytic.converged else float(analytic.onset),
        "analytic_t_off": None if not analytic.converged else float(analytic.offset),
        "analytic_converged": bool(analytic.converged),
        "soft_truth_t_on_clock": None if soft_on.size == 0 else float(soft_on[0] % PERIOD),
        "soft_truth_t_off_clock": None if soft_off.size == 0 else float(soft_off[0] % PERIOD),
        "forward_onset_bias_vs_hard_minutes": (
            None
            if soft_on.size == 0 or onsets.size == 0
            else 60.0 * (float(soft_on[0] % PERIOD) - float(onsets[0] % PERIOD))
        ),
        "tau": tau,
        "truth": {
            "phase": float(truth.phase),
            "chi_sleep": float(truth.chi_sleep),
            "chi_wake": float(truth.chi_wake),
            "amplitude": float(truth.amplitude),
            "threshold_gap": float(truth.upper - truth.lower),
        },
    }


def _classify_soft(time, gate, level, *, burn_in, retained):
    times = soft_transition_times(time, gate, level)
    in_window = (times >= burn_in) & (times < burn_in + retained)
    times = times[in_window] - burn_in
    gate_a = np.asarray(gate)
    tgrid = np.asarray(time)
    onsets, offsets = [], []
    for t in times:
        abs_t = t + burn_in
        idx = int(np.searchsorted(tgrid, abs_t))
        idx = min(max(idx, 1), gate_a.size - 1)
        if gate_a[idx] > gate_a[idx - 1]:
            onsets.append(t)
        else:
            offsets.append(t)
    return np.asarray(onsets, dtype=float), np.asarray(offsets, dtype=float)


def _soft_draw(config, *, c1, c2, chi_sleep, chi_wake, gap, burn_in, retained, level):
    mean = float(config.section("fixed")["threshold_mean"])
    soft = simulate_soft_gate_from_config(
        config,
        duration=burn_in + retained,
        output_step=float(config.section("observation")["epoch_hours"]),
        c1=c1,
        c2=c2,
        chi_sleep=chi_sleep,
        chi_wake=chi_wake,
        upper=mean + 0.5 * gap,
        lower=mean - 0.5 * gap,
        throw=True,
    )
    return _classify_soft(soft.time, soft.gate, level, burn_in=burn_in, retained=retained)


def _unwrap_to_mean(values: np.ndarray, mean: float, period: float = PERIOD) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    out = np.full_like(finite, np.nan, dtype=float)
    mask = np.isfinite(finite)
    out[mask] = (finite[mask] - mean + period / 2) % period + mean - period / 2
    return out


def _circular_variance_contraction(
    prior: np.ndarray, posterior: np.ndarray, period: float = PERIOD
) -> float:
    post_summary = _circular_summary(
        np.asarray(posterior, dtype=float)[np.isfinite(posterior)], period
    )
    mean = post_summary["circular_mean_hours"]
    return variance_contraction(
        _unwrap_to_mean(prior, mean, period),
        _unwrap_to_mean(posterior, mean, period),
    )


def _prior_draws(config, n, seed, burn_in, retained, level):
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    on_clock, off_clock, taus = [], [], []
    failed = 0
    for key in keys:
        parameters = sample_parameters(key, config, "entrained")
        gap = float(parameters.upper - parameters.lower)
        try:
            onsets, offsets = _soft_draw(
                config,
                c1=float(parameters.c1),
                c2=float(parameters.c2),
                chi_sleep=float(parameters.chi_sleep),
                chi_wake=float(parameters.chi_wake),
                gap=gap,
                burn_in=burn_in,
                retained=retained,
                level=level,
            )
            if onsets.size == 0 or offsets.size == 0:
                failed += 1
                on_clock.append(np.nan)
                off_clock.append(np.nan)
            else:
                on_clock.append(float(onsets[0] % PERIOD))
                off_clock.append(float(offsets[0] % PERIOD))
        except Exception:  # noqa: BLE001
            failed += 1
            on_clock.append(np.nan)
            off_clock.append(np.nan)
        taus.append(
            float(
                free_running_period(
                    float(parameters.chi_sleep),
                    float(parameters.chi_wake),
                    gap,
                    float(config.section("fixed")["threshold_mean"]),
                    float(parameters.mu),
                )
            )
        )
    return {
        "t_on_clock": np.asarray(on_clock),
        "t_off_clock": np.asarray(off_clock),
        "tau": np.asarray(taus),
        "failed_soft": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--n-prior", type=int, default=N_PRIOR)
    parser.add_argument("--burn-in", type=float, default=96.0)
    parser.add_argument("--retained", type=float, default=48.0)
    arguments = parser.parse_args()

    config = load_config(REPOSITORY / "config" / "model.yaml")
    model = config.section("model")
    level = float(config.section("validation")["transition_gate_level"])
    burn_in, retained = arguments.burn_in, arguments.retained

    idata = az.from_netcdf(arguments.posterior)
    n_draws = int(idata.posterior.sizes["draw"])
    oracle = _hard_oracle(config, burn_in=burn_in, retained=retained)

    oracle_on = oracle["hard_t_on_clock"]
    oracle_off = oracle["hard_t_off_clock"]
    if oracle_on is None:
        oracle_on = oracle["analytic_t_on"]
    if oracle_off is None:
        oracle_off = oracle["analytic_t_off"]

    analytic = posterior_transition_times(
        {
            name: np.asarray(idata.posterior[name].values)
            for name in ("c1", "c2", "chi_sleep", "chi_wake", "threshold_gap")
        },
        threshold_mean=float(config.section("fixed")["threshold_mean"]),
        mu=float(model["mu"]),
        period=float(model["circadian_period"]),
    )

    on_clock, off_clock = [], []
    soft_failed = 0
    for draw in range(n_draws):
        onsets, offsets = _soft_draw(
            config,
            c1=float(idata.posterior["c1"].values[0, draw]),
            c2=float(idata.posterior["c2"].values[0, draw]),
            chi_sleep=float(idata.posterior["chi_sleep"].values[0, draw]),
            chi_wake=float(idata.posterior["chi_wake"].values[0, draw]),
            gap=float(idata.posterior["threshold_gap"].values[0, draw]),
            burn_in=burn_in,
            retained=retained,
            level=level,
        )
        if onsets.size == 0 or offsets.size == 0:
            soft_failed += 1
            on_clock.append(np.nan)
            off_clock.append(np.nan)
        else:
            on_clock.append(float(onsets[0] % PERIOD))
            off_clock.append(float(offsets[0] % PERIOD))
    on_clock = np.asarray(on_clock)
    off_clock = np.asarray(off_clock)
    tau_post = np.asarray(idata.posterior["tau"].values).reshape(-1)

    bias = {
        "t_on": _bias_block(on_clock, oracle_on, circular=True),
        "t_off": _bias_block(off_clock, oracle_off, circular=True),
        "tau": _bias_block(tau_post, oracle["tau"], circular=False),
        "reference": (
            "soft-gate empirical clock times (circular) vs hard-oracle "
            "entrained transitions at generating parameters"
        ),
    }

    on_bias = abs(bias["t_on"]["bias_minutes"] or 0.0)
    off_bias = abs(bias["t_off"]["bias_minutes"] or 0.0)
    max_abs = max(on_bias, off_bias)
    oracle_inside = bool(
        bias["t_on"]["oracle_in_90pct_interval"]
        and bias["t_off"]["oracle_in_90pct_interval"]
    )
    if max_abs <= 5 and oracle_inside:
        gate = "safe"
    elif max_abs >= 25 or not oracle_inside:
        gate = "biased"
    else:
        gate = "marginal"

    print("=== BIAS (soft empirical vs hard oracle, circular) ===", flush=True)
    for name in ("t_on", "t_off"):
        b = bias[name]
        print(
            f"{name}: mean={b['mean']:.4f} oracle={b['oracle']:.4f} "
            f"bias_min={b['bias_minutes']:.1f} sd_h={b['sd']:.4f} "
            f"90%=[{b['q05']:.4f},{b['q95']:.4f}] in90={b['oracle_in_90pct_interval']}",
            flush=True,
        )
    print(
        f"analytic 1:1 failures: {analytic.failed}/{analytic.total} "
        f"reasons={analytic.failure_reasons}",
        flush=True,
    )
    print(f"GATE: {gate}", flush=True)

    print(f"Prior predictive ({arguments.n_prior}) for contraction...", flush=True)
    prior = _prior_draws(
        config, arguments.n_prior, PRIOR_SEED, burn_in, retained, level
    )
    contraction = {
        "t_on": _circular_variance_contraction(prior["t_on_clock"], on_clock),
        "t_off": _circular_variance_contraction(prior["t_off_clock"], off_clock),
        "tau": variance_contraction(prior["tau"], tau_post),
        "bars": {"t_on": 0.99, "t_off": 0.99, "tau_max": 0.25},
        "meets_bars": None,
        "prior_n": arguments.n_prior,
        "prior_soft_failures": prior["failed_soft"],
        "clock_contraction": "circular_unwrap_to_posterior_mean",
    }
    contraction["meets_bars"] = {
        "t_on": bool(contraction["t_on"] > 0.99),
        "t_off": bool(contraction["t_off"] > 0.99),
        "tau": bool(contraction["tau"] <= 0.25),
    }
    print(
        f"C(t_on)={contraction['t_on']:.4f}  C(t_off)={contraction['t_off']:.4f}  "
        f"C(tau)={contraction['tau']:.4f}  meets={contraction['meets_bars']}",
        flush=True,
    )

    ess = {}
    for name in (
        "phase_z1",
        "phase_z2",
        "excursion_fraction",
        "amplitude_fraction",
        "chi_sleep",
        "chi_wake",
        "tau",
        "phase",
        "amplitude",
        "threshold_gap",
    ):
        if name not in idata.posterior:
            continue
        try:
            ess[name] = {
                "bulk": float(
                    az.ess(idata, var_names=[name], method="bulk")[name].values
                ),
                "tail": float(
                    az.ess(idata, var_names=[name], method="tail")[name].values
                ),
            }
        except Exception as error:  # noqa: BLE001
            ess[name] = {"error": str(error)}

    try:
        bfmi_vals = np.asarray(az.bfmi(idata), dtype=float).reshape(-1)
        bfmi = {"per_chain": bfmi_vals.tolist(), "mean": float(np.mean(bfmi_vals))}
    except Exception as error:  # noqa: BLE001
        bfmi = {"error": str(error)}

    phase = np.asarray(idata.posterior["phase"].values).reshape(-1)
    phase_summary = _circular_summary(phase, PERIOD)

    payload = {
        "run_id": "RUN-009",
        "posterior_path": str(arguments.posterior),
        "burn_in_hours": burn_in,
        "retained_hours": retained,
        "caveats": [
            "50 posterior draws",
            "one chain",
            "48 h retained (two sleep bouts)",
            "max_tree_depth pinned at 7",
        ],
        "oracle": oracle,
        "bias": bias,
        "gate": {
            "decision": gate,
            "max_abs_bias_minutes": max_abs,
            "proceed_to_sbc": gate == "safe",
        },
        "analytic_entrainment": {
            "failed": analytic.failed,
            "total": analytic.total,
            "failure_fraction": analytic.failed / max(analytic.total, 1),
            "failure_reasons": analytic.failure_reasons,
        },
        "soft_empirical_failures": soft_failed,
        "contraction": contraction,
        "ess": ess,
        "bfmi": bfmi,
        "phase_circular": phase_summary,
        "posterior_clock_draws": {
            "t_on": on_clock.tolist(),
            "t_off": off_clock.tolist(),
            "tau": tau_post.tolist(),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {arguments.output}", flush=True)
    if gate != "safe":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
