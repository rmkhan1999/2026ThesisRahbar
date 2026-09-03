from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.hard_switch import simulate_hard_switch
from twopm.observation import sample_observations, sleep_probabilities


REPOSITORY = Path(__file__).resolve().parents[1]
NPZ = REPOSITORY / "docs" / "hard_generated_recording.npz"
META = REPOSITORY / "docs" / "hard_generated_recording.json"


def _config(retained: float, burn_in: float) -> ProjectConfig:
    base = load_config(REPOSITORY / "config" / "model.yaml")
    data = deepcopy(base.data)
    data["designs"]["entrained"]["duration"] = retained
    data["designs"]["entrained"]["burn_in_hours"] = burn_in
    data["observation"]["misclassification"] = 0.01
    return ProjectConfig(data=data, source=base.source)


def main() -> None:
    retained = 24.0
    burn_in = 48.0
    dt = 0.001
    config = _config(retained, burn_in)
    model = config.section("model")
    recovery = config.section("recovery")
    observation = config.section("observation")
    initial = config.section("initial_state")
    epoch = float(observation["epoch_hours"])
    seed = int(recovery["seed"])
    period = float(model["circadian_period"])
    phase = float(recovery["true_phase"])
    amplitude = float(model["circadian_amplitude"])
    lam = 0.01

    truth = standard_parameters(config, amplitude=amplitude, phase=phase)

    hard = simulate_hard_switch(
        duration=burn_in + retained,
        dt=dt,
        chi_sleep=float(truth.chi_sleep),
        chi_wake=float(truth.chi_wake),
        mu=float(truth.mu),
        upper=float(truth.upper),
        lower=float(truth.lower),
        initial_pressure=float(truth.initial_pressure),
        initially_asleep=bool(initial["hard_asleep"]),
        amplitude=amplitude,
        phase=phase,
        period=period,
        start_time=0.0,
    )

    retained_points = int(round(retained / epoch)) + 1
    epoch_abs = burn_in + epoch * np.arange(retained_points, dtype=np.float64)
    idx = np.searchsorted(hard.time, epoch_abs, side="left")
    idx = np.clip(idx, 0, hard.time.size - 1)
    prev = np.clip(idx - 1, 0, hard.time.size - 1)
    use_prev = np.abs(hard.time[prev] - epoch_abs) < np.abs(hard.time[idx] - epoch_abs)
    idx = np.where(use_prev, prev, idx)
    gate = hard.asleep[idx].astype(np.float64)
    time = epoch_abs - burn_in
    probabilities = np.asarray(sleep_probabilities(jnp.asarray(gate), lam))
    observations = np.asarray(
        sample_observations(jax.random.PRNGKey(seed), jnp.asarray(probabilities))
    )

    soft = generate_recording(jax.random.PRNGKey(seed), config, truth, "entrained")
    soft_obs = np.asarray(soft.observations)
    if soft_obs.shape != observations.shape:
        raise RuntimeError(
            f"shape mismatch soft {soft_obs.shape} vs hard {observations.shape}"
        )
    n_diff = int(np.sum(soft_obs != observations))

    switch_mask = hard.switch_times >= burn_in
    true_switch_times = hard.switch_times[switch_mask] - burn_in
    true_switch_states = hard.switch_states[switch_mask]

    config_hash = hashlib.sha256(
        json.dumps(config.data, sort_keys=True, default=str).encode()
    ).hexdigest()

    np.savez_compressed(
        NPZ,
        time=time,
        gate=gate,
        probabilities=probabilities,
        observations=observations.astype(np.int8),
        soft_observations=soft_obs.astype(np.int8),
        true_switch_times=true_switch_times,
        true_switch_states=true_switch_states.astype(np.bool_),
    )
    meta = {
        "date": "2026-08-10",
        "generator": "hard_switch",
        "dt": dt,
        "burn_in_hours": burn_in,
        "retained_hours": retained,
        "epoch_hours": epoch,
        "misclassification": lam,
        "seed": seed,
        "phase": phase,
        "amplitude": amplitude,
        "chi_sleep": float(truth.chi_sleep),
        "chi_wake": float(truth.chi_wake),
        "upper": float(truth.upper),
        "lower": float(truth.lower),
        "config_sha256": config_hash,
        "n_epochs": int(observations.size),
        "n_label_diffs_vs_soft_same_key": n_diff,
        "true_switch_times": true_switch_times.tolist(),
        "true_switch_states_asleep": [bool(x) for x in true_switch_states],
        "npz_path": str(NPZ.relative_to(REPOSITORY)),
    }
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({
        "npz": str(NPZ),
        "meta": str(META),
        "n_epochs": int(observations.size),
        "n_label_diffs_vs_soft": n_diff,
        "n_hard_switches_retained": int(true_switch_times.size),
    }, indent=2))
    if n_diff > 4:
        raise SystemExit(
            f"SANITY FAIL: {n_diff} epoch diffs vs soft (expected 0–2). Abort."
        )
    print(f"SANITY OK: {n_diff} epoch diffs vs soft", flush=True)


if __name__ == "__main__":
    main()
