from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import potential_energy

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.inference import (
    numpyro_model,
    prior_distribution,
    sampled_parameter_names,
)


COSINE_MIN = 0.9999
TAYLOR_TOL = 0.05
NORM_REL_MAX = 0.05
UNIT_TAYLOR_STEPS = (1e-5, 1e-6, 1e-7)
N_DRAWS = 5
SEED_BASE = 1000


def _flatten(tree: dict[str, jnp.ndarray]):
    keys = sorted(tree)
    sizes = [int(np.size(tree[key])) for key in keys]

    def pack(values):
        return jnp.concatenate([jnp.ravel(values[key]) for key in keys])

    def unpack(vector):
        pieces = {}
        offset = 0
        for key, size in zip(keys, sizes):
            pieces[key] = vector[offset : offset + size].reshape(tree[key].shape)
            offset += size
        return pieces

    return pack, unpack


@pytest.fixture(scope="module")
def short_config():
    base = load_config("config/model.yaml")
    data = deepcopy(base.data)
    data["designs"]["entrained"]["duration"] = 12.0
    data["designs"]["entrained"]["burn_in_hours"] = 12.0
    return ProjectConfig(data=data, source=base.source)


def test_unconstrained_potential_gradients_across_prior_draws(short_config):
    config = short_config
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
    )
    model_args = (recording.observations, config, None)
    names = sampled_parameter_names(config)

    def potential(params):
        return potential_energy(numpyro_model, model_args, {}, params)

    failures = []
    for index in range(N_DRAWS):
        keys = jax.random.split(jax.random.PRNGKey(SEED_BASE + index), len(names))
        unconstrained = {}
        for key, name in zip(keys, names):
            distribution = prior_distribution(name, config)
            value = distribution.sample(key)
            unconstrained[name] = biject_to(distribution.support).inv(value)
        pack, unpack = _flatten(unconstrained)
        flat0 = pack(unconstrained)

        def scalar(vector):
            return potential(unpack(vector))

        automatic = np.asarray(jax.grad(scalar)(flat0), dtype=float)
        step = 1e-5
        finite = np.asarray(
            [
                float(
                    (
                        scalar(flat0 + jnp.zeros_like(flat0).at[i].set(step))
                        - scalar(flat0 - jnp.zeros_like(flat0).at[i].set(step))
                    )
                    / (2 * step)
                )
                for i in range(flat0.size)
            ],
            dtype=float,
        )
        ad_norm = float(np.linalg.norm(automatic))
        fd_norm = float(np.linalg.norm(finite))
        cosine = float(np.dot(automatic, finite) / (ad_norm * fd_norm))
        norm_rel = abs(ad_norm - fd_norm) / max(fd_norm, 1e-10)

        u0 = float(scalar(flat0))
        uhat = automatic / ad_norm
        best_taylor = float("inf")
        for eps in UNIT_TAYLOR_STEPS:
            observed = float(scalar(flat0 + eps * uhat) - u0)
            predicted = eps * ad_norm
            ratio = observed / predicted if abs(predicted) > 1e-30 else float("nan")
            best_taylor = min(best_taylor, abs(ratio - 1.0))

        if (
            not np.isfinite(cosine)
            or cosine < COSINE_MIN
            or best_taylor > TAYLOR_TOL
            or norm_rel > NORM_REL_MAX
        ):
            failures.append(
                {
                    "draw": index,
                    "cosine": cosine,
                    "best_abs_unit_taylor_ratio_minus_one": best_taylor,
                    "norm_relative_error": norm_rel,
                    "ad_norm": ad_norm,
                    "fd_norm": fd_norm,
                }
            )

    assert not failures, (
        "unconstrained gradient vector audit failed "
        f"(cosine≥{COSINE_MIN}, |unit-Taylor−1|≤{TAYLOR_TOL}, "
        f"norm relative≤{NORM_REL_MAX}): {failures}"
    )


@pytest.mark.slow
def test_unconstrained_potential_gradients_live_horizon_20draws():
    base = load_config("config/model.yaml")
    data = deepcopy(base.data)
    data["designs"]["entrained"]["duration"] = 24.0
    data["designs"]["entrained"]["burn_in_hours"] = 48.0
    config = ProjectConfig(data=data, source=base.source)
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
    )
    model_args = (recording.observations, config, None)
    names = sampled_parameter_names(config)

    def potential(params):
        return potential_energy(numpyro_model, model_args, {}, params)

    failures = []
    for index in range(20):
        keys = jax.random.split(jax.random.PRNGKey(SEED_BASE + index), len(names))
        unconstrained = {}
        for key, name in zip(keys, names):
            distribution = prior_distribution(name, config)
            value = distribution.sample(key)
            unconstrained[name] = biject_to(distribution.support).inv(value)
        pack, unpack = _flatten(unconstrained)
        flat0 = pack(unconstrained)

        def scalar(vector):
            return potential(unpack(vector))

        automatic = np.asarray(jax.grad(scalar)(flat0), dtype=float)
        step = 1e-5
        finite = np.asarray(
            [
                float(
                    (
                        scalar(flat0 + jnp.zeros_like(flat0).at[i].set(step))
                        - scalar(flat0 - jnp.zeros_like(flat0).at[i].set(step))
                    )
                    / (2 * step)
                )
                for i in range(flat0.size)
            ],
            dtype=float,
        )
        ad_norm = float(np.linalg.norm(automatic))
        fd_norm = float(np.linalg.norm(finite))
        cosine = float(np.dot(automatic, finite) / (ad_norm * fd_norm))
        norm_rel = abs(ad_norm - fd_norm) / max(fd_norm, 1e-10)
        u0 = float(scalar(flat0))
        uhat = automatic / ad_norm
        best_taylor = float("inf")
        for eps in UNIT_TAYLOR_STEPS:
            observed = float(scalar(flat0 + eps * uhat) - u0)
            predicted = eps * ad_norm
            ratio = observed / predicted if abs(predicted) > 1e-30 else float("nan")
            best_taylor = min(best_taylor, abs(ratio - 1.0))
        if (
            not np.isfinite(cosine)
            or cosine < COSINE_MIN
            or best_taylor > TAYLOR_TOL
            or norm_rel > NORM_REL_MAX
        ):
            failures.append(
                {
                    "draw": index,
                    "cosine": cosine,
                    "best_abs_unit_taylor_ratio_minus_one": best_taylor,
                    "norm_relative_error": norm_rel,
                }
            )

    assert not failures, (
        "live-horizon unconstrained gradient audit failed: " f"{failures}"
    )
