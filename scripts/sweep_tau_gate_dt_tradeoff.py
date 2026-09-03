from __future__ import annotations

import argparse
import json
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
from twopm.hard_switch import simulate_hard_switch
from twopm.inference import (
    constrained_physical_parameters,
    numpyro_model,
    prior_distribution,
    sampled_parameter_names,
)
from twopm.posterior_summaries import solve_entrained_transition
from twopm.soft_gate import simulate_soft_gate_from_config, soft_transition_times


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "docs" / "tau_gate_dt_tradeoff.json"
TAU_GATES = (0.05, 0.1, 0.2, 0.5)
DT_OVER_TAU = 0.02
COSINE_MIN = 0.9999
BIAS_MAX_MINUTES = 5.0
N_DRAWS = 20
SEED_BASE = 1000
BURN_IN = 96.0
RETAINED = 48.0
PERIOD = 24.0


def _configured(
    base: ProjectConfig,
    *,
    tau_gate: float,
    dt: float,
    p0: float | None = None,
) -> ProjectConfig:
    data = deepcopy(base.data)
    data["soft_gate"]["tau_gate"] = float(tau_gate)
    data["soft_gate"]["fixed_step_size"] = float(dt)
    if p0 is not None:
        data["soft_gate"]["p0"] = float(p0)
    data["designs"]["entrained"]["duration"] = RETAINED
    data["designs"]["entrained"]["burn_in_hours"] = BURN_IN
    data["observation"]["misclassification"] = 0.01
    data["inference"]["parameters"] = [
        name
        for name in data["inference"]["parameters"]
        if name != "misclassification"
    ]
    horizon = BURN_IN + RETAINED
    data["soft_gate"]["max_steps"] = int(
        max(float(data["soft_gate"]["max_steps"]), 2 * horizon / dt)
    )
    return ProjectConfig(data=data, source=base.source)


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
            onsets.append(float(t))
        else:
            offsets.append(float(t))
    return np.asarray(onsets), np.asarray(offsets)


def _soft_bias_at_truth(config: ProjectConfig) -> dict:
    model = config.section("model")
    recovery = config.section("recovery")
    level = float(config.section("validation")["transition_gate_level"])
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
        duration=BURN_IN + RETAINED,
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
    mask = (times >= BURN_IN) & (times < BURN_IN + RETAINED)
    hard_on = times[mask][states[mask]] - BURN_IN
    hard_off = times[mask][~states[mask]] - BURN_IN

    output_step = min(
        0.01, float(config.section("soft_gate")["fixed_step_size"])
    )
    soft = simulate_soft_gate_from_config(
        config,
        duration=BURN_IN + RETAINED,
        output_step=output_step,
        c1=truth.c1,
        c2=truth.c2,
        chi_sleep=truth.chi_sleep,
        chi_wake=truth.chi_wake,
        upper=truth.upper,
        lower=truth.lower,
        throw=True,
    )
    soft_on, soft_off = _classify_soft(
        soft.time, soft.gate, level, burn_in=BURN_IN, retained=RETAINED
    )
    oracle_on = float(hard_on[0] % PERIOD) if hard_on.size else float(analytic.onset)
    oracle_off = float(hard_off[0] % PERIOD) if hard_off.size else float(analytic.offset)
    soft_on_c = float(soft_on[0] % PERIOD) if soft_on.size else float("nan")
    soft_off_c = float(soft_off[0] % PERIOD) if soft_off.size else float("nan")
    bias_on = soft_on_c - oracle_on
    bias_off = soft_off_c - oracle_off
    if abs(bias_off) > 12:
        bias_off = ((soft_off_c - oracle_off + 12) % 24) - 12
    return {
        "hard_t_on": oracle_on,
        "hard_t_off": oracle_off,
        "analytic_t_on": None if not analytic.converged else float(analytic.onset),
        "analytic_t_off": None if not analytic.converged else float(analytic.offset),
        "soft_t_on": soft_on_c,
        "soft_t_off": soft_off_c,
        "bias_on_minutes": 60.0 * bias_on,
        "bias_off_minutes": 60.0 * bias_off,
        "max_abs_bias_minutes": max(abs(60.0 * bias_on), abs(60.0 * bias_off)),
        "gate_min": float(np.min(soft.gate)),
        "gate_max": float(np.max(soft.gate)),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _flatten(tree: dict[str, jnp.ndarray]):
    keys = sorted(tree)
    sizes = [int(np.size(tree[key])) for key in keys]

    def pack(values):
        return jnp.concatenate([jnp.ravel(values[key]) for key in keys])

    def unpack(vector):
        pieces = []
        offset = 0
        for key, size in zip(keys, sizes):
            pieces.append(
                (key, vector[offset : offset + size].reshape(tree[key].shape))
            )
            offset += size
        return {key: value for key, value in pieces}

    return pack, unpack


def _gradient_audit(config: ProjectConfig, *, n_draws: int, seed_base: int) -> dict:
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

    def potential(params):
        return potential_energy(numpyro_model, model_args, {}, params)

    names = sampled_parameter_names(config, "entrained")
    cosines = []
    draw_rows = []
    t0 = time.perf_counter()
    for index in range(n_draws):
        keys = jax.random.split(jax.random.PRNGKey(seed_base + index), len(names))
        unconstrained = {}
        for key, name in zip(keys, names):
            distribution = prior_distribution(name, config, "entrained")
            value = distribution.sample(key)
            unconstrained[name] = biject_to(distribution.support).inv(value)
        pack, unpack = _flatten(unconstrained)
        flat0 = pack(unconstrained)

        def scalar(vector):
            return potential(unpack(vector))

        automatic = np.asarray(jax.grad(scalar)(flat0), dtype=float)
        step = 1e-5
        finite = []
        for coordinate in range(flat0.size):
            offset = jnp.zeros_like(flat0).at[coordinate].set(step)
            finite.append(
                float((scalar(flat0 + offset) - scalar(flat0 - offset)) / (2 * step))
            )
        finite = np.asarray(finite, dtype=float)
        cosine = _cosine(automatic, finite)
        cosines.append(cosine)
        draw_rows.append(
            {
                "draw": index,
                "cosine_similarity": cosine,
                "ad_norm": float(np.linalg.norm(automatic)),
                "fd_norm": float(np.linalg.norm(finite)),
                "pass_cosine": bool(np.isfinite(cosine) and cosine >= COSINE_MIN),
            }
        )
        print(
            f"  draw={index} cosine={cosine:.6f} pass={draw_rows[-1]['pass_cosine']}",
            flush=True,
        )
    return {
        "n_draws": n_draws,
        "min_cosine": float(np.nanmin(cosines)),
        "median_cosine": float(np.nanmedian(cosines)),
        "all_pass_cosine": all(row["pass_cosine"] for row in draw_rows),
        "seconds": time.perf_counter() - t0,
        "draws": draw_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--p0", type=float, default=None)
    parser.add_argument(
        "--tau-gates",
        type=float,
        nargs="+",
        default=list(TAU_GATES),
    )
    arguments = parser.parse_args()
    base = load_config(REPOSITORY / "config" / "model.yaml")
    rows = []
    for tau_gate in arguments.tau_gates:
        dt = DT_OVER_TAU * float(tau_gate)
        p0_note = (
            float(base.section("soft_gate")["p0"])
            if arguments.p0 is None
            else float(arguments.p0)
        )
        print(
            f"\n=== τ_g={tau_gate}  dt={dt}  p0={p0_note}  "
            f"(dt/τ_g={DT_OVER_TAU}) ===",
            flush=True,
        )
        config = _configured(
            base, tau_gate=float(tau_gate), dt=dt, p0=arguments.p0
        )
        t_bias = time.perf_counter()
        bias = _soft_bias_at_truth(config)
        bias_seconds = time.perf_counter() - t_bias
        print(
            f"  soft bias: on={bias['bias_on_minutes']:+.1f} min  "
            f"off={bias['bias_off_minutes']:+.1f} min  "
            f"max|b|={bias['max_abs_bias_minutes']:.1f} min  "
            f"({bias_seconds:.1f}s)",
            flush=True,
        )
        print(f"  gradient audit ({arguments.n_draws} draws)...", flush=True)
        audit = _gradient_audit(
            config, n_draws=arguments.n_draws, seed_base=SEED_BASE
        )
        print(
            f"  min cosine={audit['min_cosine']:.6f}  "
            f"all_pass={audit['all_pass_cosine']}  ({audit['seconds']:.1f}s)",
            flush=True,
        )
        bias_ok = bias["max_abs_bias_minutes"] <= BIAS_MAX_MINUTES
        cosine_ok = bool(audit["all_pass_cosine"])
        rows.append(
            {
                "tau_gate": float(tau_gate),
                "fixed_step_size": dt,
                "p0": float(config.section("soft_gate")["p0"]),
                "dt_over_tau_gate": DT_OVER_TAU,
                "bias": bias,
                "gradient_audit": {
                    "min_cosine": audit["min_cosine"],
                    "median_cosine": audit["median_cosine"],
                    "all_pass_cosine": audit["all_pass_cosine"],
                    "n_draws": audit["n_draws"],
                    "seconds": audit["seconds"],
                    "draws": audit["draws"],
                },
                "bias_ok": bias_ok,
                "cosine_ok": cosine_ok,
                "acceptable": bias_ok and cosine_ok,
            }
        )
        payload = _payload(rows)
        arguments.output.write_text(json.dumps(payload, indent=2) + "\n")

    payload = _payload(rows)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n=== SUMMARY ===", flush=True)
    for row in rows:
        flag = "ACCEPT" if row["acceptable"] else "reject"
        print(
            f"τ_g={row['tau_gate']:<5} dt={row['fixed_step_size']:<6} "
            f"|bias|={row['bias']['max_abs_bias_minutes']:5.1f} min  "
            f"min_cos={row['gradient_audit']['min_cosine']:.6f}  {flag}",
            flush=True,
        )
    print(f"Saved {arguments.output}", flush=True)
    if payload["recommendation"] is None:
        print(
            "No acceptable (bias≤5 min AND cosine≥0.9999) configuration. "
            "Logit-gate rewrite is justified on evidence.",
            flush=True,
        )
        raise SystemExit(2)


def _payload(rows: list[dict]) -> dict:
    acceptable = [row for row in rows if row["acceptable"]]
    recommendation = None
    if acceptable:
        recommendation = min(
            acceptable, key=lambda r: r["bias"]["max_abs_bias_minutes"]
        )
    p0_values = sorted({float(row["p0"]) for row in rows})
    return {
        "criteria": {
            "dt_over_tau_gate": DT_OVER_TAU,
            "bias_max_minutes": BIAS_MAX_MINUTES,
            "cosine_min": COSINE_MIN,
            "n_gradient_draws": N_DRAWS,
            "burn_in_hours": BURN_IN,
            "retained_hours": RETAINED,
            "p0_values": p0_values,
        },
        "configurations": rows,
        "any_acceptable": bool(acceptable),
        "recommendation": (
            None
            if recommendation is None
            else {
                "tau_gate": recommendation["tau_gate"],
                "fixed_step_size": recommendation["fixed_step_size"],
                "p0": recommendation["p0"],
                "max_abs_bias_minutes": recommendation["bias"]["max_abs_bias_minutes"],
                "min_cosine": recommendation["gradient_audit"]["min_cosine"],
            }
        ),
        "logit_gate_justified": False,
    }


if __name__ == "__main__":
    main()
