from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from twopm.config import load_config
from twopm.generative import sample_parameters
from twopm.hard_switch import simulate_hard_switch
from twopm.soft_gate import simulate_soft_gate_from_config, soft_transition_times


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "docs" / "gate_bias_prior_sweep.json"
N_DRAWS = 40
SEED = 20260730
OUTPUT_STEP = 0.01
BURN_IN = 48.0
RETAINED = 48.0
PERIOD = 24.0


def _classify(time, gate, level, *, burn_in, retained):
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
            onsets.append(float(t))
        else:
            offsets.append(float(t))
    return np.asarray(onsets), np.asarray(offsets)


def _crossing_speed(pressure: np.ndarray, time: np.ndarray, t_cross: float) -> float:
    idx = int(np.searchsorted(time, t_cross))
    idx = min(max(idx, 1), pressure.size - 2)
    dt = float(time[idx] - time[idx - 1])
    if dt <= 0:
        return float("nan")
    return abs(float(pressure[idx] - pressure[idx - 1]) / dt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--burn-in", type=float, default=BURN_IN)
    parser.add_argument("--retained", type=float, default=RETAINED)
    arguments = parser.parse_args()

    config = load_config(REPOSITORY / "config" / "model.yaml")
    soft = config.section("soft_gate")
    level = float(config.section("validation")["transition_gate_level"])
    burn_in = arguments.burn_in
    retained = arguments.retained
    keys = jax.random.split(jax.random.PRNGKey(arguments.seed), arguments.n_draws)

    rows = []
    for index, key in enumerate(keys):
        parameters = sample_parameters(key, config, "entrained")
        hard = simulate_hard_switch(
            duration=burn_in + retained,
            dt=float(config.section("hard_switch")["dt"]),
            chi_sleep=float(parameters.chi_sleep),
            chi_wake=float(parameters.chi_wake),
            mu=float(parameters.mu),
            upper=float(parameters.upper),
            lower=float(parameters.lower),
            initial_pressure=float(parameters.initial_pressure),
            initially_asleep=bool(config.section("initial_state")["hard_asleep"]),
            amplitude=float(parameters.amplitude),
            phase=float(parameters.phase),
            period=PERIOD,
            start_time=0.0,
        )
        soft_traj = simulate_soft_gate_from_config(
            config,
            duration=burn_in + retained,
            output_step=OUTPUT_STEP,
            c1=float(parameters.c1),
            c2=float(parameters.c2),
            chi_sleep=float(parameters.chi_sleep),
            chi_wake=float(parameters.chi_wake),
            upper=float(parameters.upper),
            lower=float(parameters.lower),
            initial_pressure=float(parameters.initial_pressure),
            throw=True,
        )
        times = np.asarray(hard.switch_times, dtype=float)
        states = np.asarray(hard.switch_states, dtype=bool)
        mask = (times >= burn_in) & (times < burn_in + retained)
        hard_on = times[mask][states[mask]] - burn_in
        hard_off = times[mask][~states[mask]] - burn_in
        soft_on, soft_off = _classify(
            soft_traj.time,
            soft_traj.gate,
            level,
            burn_in=burn_in,
            retained=retained,
        )
        n_hard_on = int(hard_on.size)
        n_hard_off = int(hard_off.size)
        n_soft_on = int(soft_on.size)
        n_soft_off = int(soft_off.size)
        one_to_one = (
            n_hard_on == n_soft_on
            and n_hard_off == n_soft_off
            and n_hard_on > 0
            and n_hard_off > 0
        )
        if hard_on.size == 0 or soft_on.size == 0 or hard_off.size == 0 or soft_off.size == 0:
            rows.append(
                {
                    "draw": index,
                    "ok": False,
                    "reason": "missing_transition",
                    "n_hard_on": n_hard_on,
                    "n_hard_off": n_hard_off,
                    "n_soft_on": n_soft_on,
                    "n_soft_off": n_soft_off,
                    "one_to_one": False,
                }
            )
            print(f"draw={index} FAIL missing transition", flush=True)
            continue

        bias_on = 60.0 * (float(soft_on[0] % PERIOD) - float(hard_on[0] % PERIOD))
        bias_off = 60.0 * (float(soft_off[0] % PERIOD) - float(hard_off[0] % PERIOD))
        if abs(bias_off) > 12 * 60:
            bias_off = ((bias_off / 60.0 + 12) % 24 - 12) * 60.0
        if abs(bias_on) > 12 * 60:
            bias_on = ((bias_on / 60.0 + 12) % 24 - 12) * 60.0

        t_on_abs = float(hard_on[0] + burn_in)
        speed = _crossing_speed(
            np.asarray(hard.pressure, dtype=float),
            np.asarray(hard.time, dtype=float),
            t_on_abs,
        )
        row = {
            "draw": index,
            "ok": True,
            "bias_on_minutes": bias_on,
            "bias_off_minutes": bias_off,
            "max_abs_bias_minutes": max(abs(bias_on), abs(bias_off)),
            "dHdt_abs_at_onset": speed,
            "n_hard_on": n_hard_on,
            "n_hard_off": n_hard_off,
            "n_soft_on": n_soft_on,
            "n_soft_off": n_soft_off,
            "one_to_one": bool(one_to_one),
            "chi_sleep": float(parameters.chi_sleep),
            "chi_wake": float(parameters.chi_wake),
            "amplitude": float(parameters.amplitude),
            "phase": float(parameters.phase),
            "threshold_gap": float(parameters.upper - parameters.lower),
        }
        rows.append(row)
        print(
            f"draw={index} on={bias_on:+.2f} off={bias_off:+.2f} "
            f"max={row['max_abs_bias_minutes']:.2f} |dH/dt|={speed:.5f} "
            f"1:1={one_to_one} hard=({n_hard_on},{n_hard_off}) "
            f"soft=({n_soft_on},{n_soft_off})",
            flush=True,
        )

    ok_rows = [row for row in rows if row.get("ok")]
    one_to_one_rows = [row for row in ok_rows if row.get("one_to_one")]
    non_one_to_one_rows = [row for row in ok_rows if not row.get("one_to_one")]

    def _bias_stats(subset):
        if not subset:
            return None
        max_abs = np.asarray(
            [row["max_abs_bias_minutes"] for row in subset], dtype=float
        )
        bias_on = np.asarray(
            [row["bias_on_minutes"] for row in subset], dtype=float
        )
        speeds = np.asarray(
            [row["dHdt_abs_at_onset"] for row in subset], dtype=float
        )
        finite = np.isfinite(max_abs) & np.isfinite(speeds)
        correlation = (
            float(np.corrcoef(max_abs[finite], speeds[finite])[0, 1])
            if finite.sum() >= 3
            else None
        )
        return {
            "n": len(subset),
            "max_abs_bias_minutes": {
                "mean": float(np.mean(max_abs)),
                "sd": float(np.std(max_abs, ddof=1)) if max_abs.size > 1 else 0.0,
                "min": float(np.min(max_abs)),
                "max": float(np.max(max_abs)),
                "percentiles": {
                    "p50": float(np.percentile(max_abs, 50)),
                    "p75": float(np.percentile(max_abs, 75)),
                    "p90": float(np.percentile(max_abs, 90)),
                    "p95": float(np.percentile(max_abs, 95)),
                },
                "fraction_le_5min": float(np.mean(max_abs <= 5.0)),
                "fraction_le_10min": float(np.mean(max_abs <= 10.0)),
                "fraction_le_15min": float(np.mean(max_abs <= 15.0)),
            },
            "bias_on_minutes": {
                "mean": float(np.mean(bias_on)),
                "sd": float(np.std(bias_on, ddof=1)) if bias_on.size > 1 else 0.0,
                "min": float(np.min(bias_on)),
                "max": float(np.max(bias_on)),
            },
            "correlation_max_abs_bias_with_abs_dHdt": correlation,
        }

    all_stats = _bias_stats(ok_rows)
    one_to_one_stats = _bias_stats(one_to_one_rows)
    non_one_to_one_stats = _bias_stats(non_one_to_one_rows)
    max_abs = np.asarray(
        [row["max_abs_bias_minutes"] for row in ok_rows], dtype=float
    )
    outlier_draws = sorted(
        ok_rows, key=lambda row: row["max_abs_bias_minutes"], reverse=True
    )[:5]
    summary = {
        "n_ok": len(ok_rows),
        "n_failed": len(rows) - len(ok_rows),
        "n_one_to_one": len(one_to_one_rows),
        "n_non_one_to_one": len(non_one_to_one_rows),
        "fraction_one_to_one": (
            len(one_to_one_rows) / len(ok_rows) if ok_rows else None
        ),
        "all_ok_draws": all_stats,
        "one_to_one_subset": one_to_one_stats,
        "non_one_to_one_subset": non_one_to_one_stats,
        "max_abs_bias_minutes": all_stats["max_abs_bias_minutes"]
        if all_stats
        else None,
        "bias_on_minutes": all_stats["bias_on_minutes"] if all_stats else None,
        "correlation_max_abs_bias_with_abs_dHdt": (
            all_stats["correlation_max_abs_bias_with_abs_dHdt"]
            if all_stats
            else None
        ),
        "gate": {
            "p0": float(soft["p0"]),
            "tau_gate": float(soft["tau_gate"]),
            "k": float(soft["k"]),
            "fixed_step_size": float(soft["fixed_step_size"]),
        },
        "top_outliers": [
            {
                "draw": row["draw"],
                "max_abs_bias_minutes": row["max_abs_bias_minutes"],
                "one_to_one": row["one_to_one"],
                "n_hard_on": row["n_hard_on"],
                "n_hard_off": row["n_hard_off"],
                "n_soft_on": row["n_soft_on"],
                "n_soft_off": row["n_soft_off"],
                "dHdt_abs_at_onset": row["dHdt_abs_at_onset"],
            }
            for row in outlier_draws
        ],
        "stratification_note": (
            "one_to_one means equal onset and offset counts for soft and hard "
            "in the retained window. First-transition bias on non-1:1 draws "
            "can be a matching artefact (polyphasic / extra soft crossings)."
        ),
        "action": (
            "confirm"
            if max_abs.size and float(np.max(max_abs)) <= 5.0
            else (
                "consider_robust_alternative_tau0p1_k800"
                if max_abs.size and float(np.max(max_abs)) <= 15.0
                else "investigate"
            )
        ),
    }
    if all_stats is not None:
        summary["percentiles_max_abs_bias_minutes"] = all_stats[
            "max_abs_bias_minutes"
        ]["percentiles"]
        summary["fraction_max_abs_le_5min"] = all_stats["max_abs_bias_minutes"][
            "fraction_le_5min"
        ]
        summary["fraction_max_abs_le_10min"] = all_stats["max_abs_bias_minutes"][
            "fraction_le_10min"
        ]
        summary["fraction_max_abs_le_15min"] = all_stats["max_abs_bias_minutes"][
            "fraction_le_15min"
        ]
    payload = {
        "n_draws": arguments.n_draws,
        "seed": arguments.seed,
        "burn_in_hours": burn_in,
        "retained_hours": retained,
        "output_step_hours": OUTPUT_STEP,
        "summary": summary,
        "draws": rows,
    }
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
