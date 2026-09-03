import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from twopm.config import load_config
from twopm.generative import sample_parameters


def _report(name: str, lower_margin, upper_margin, amplitude, gap) -> None:
    print(name)
    print(f"  draws: {lower_margin.size}")
    print(f"  P(min L < 0): {float(jnp.mean(lower_margin < 0)):.6f}")
    print(f"  P(max U > mu): {float(jnp.mean(upper_margin < 0)):.6f}")
    print(
        "  amplitude mean/sd: "
        f"{float(jnp.mean(amplitude)):.6f}/"
        f"{float(jnp.std(amplitude)):.6f}"
    )
    print(
        "  gap mean/sd: "
        f"{float(jnp.mean(gap)):.6f}/{float(jnp.std(gap)):.6f}"
    )
    print(
        "  corr(amplitude, gap): "
        f"{float(jnp.corrcoef(amplitude, gap)[0, 1]):.6f}"
    )
    print(f"  minimum lower margin: {float(jnp.min(lower_margin)):.8f}")


def main() -> None:
    config = load_config("config/model.yaml")
    settings = config.section("prior_domain_audit")
    fixed = config.section("fixed")
    model = config.section("model")
    draws = int(settings["draws"])
    root_key = jax.random.PRNGKey(int(settings["seed"]))

    for index, design in enumerate(("entrained", "weak_forcing")):
        keys = jax.random.split(jax.random.fold_in(root_key, index), draws)
        parameters = jax.vmap(
            lambda key: sample_parameters(key, config, design)
        )(keys)
        amplitude = jnp.hypot(parameters.c1, parameters.c2)
        gap = parameters.upper - parameters.lower
        _report(
            f"current constrained prior ({design})",
            parameters.lower - amplitude,
            parameters.mu - parameters.upper - amplitude,
            amplitude,
            gap,
        )

    keys = jax.random.split(jax.random.fold_in(root_key, 2), 3)
    sigma = float(settings["superseded_cartesian_sigma"])
    gap = dist.LogNormal(
        float(settings["superseded_gap_log_mean"]),
        float(settings["superseded_gap_log_sd"]),
    ).sample(keys[0], (draws,))
    c1 = dist.Normal(0.0, sigma).sample(keys[1], (draws,))
    c2 = dist.Normal(0.0, sigma).sample(keys[2], (draws,))
    amplitude = jnp.hypot(c1, c2)
    mean = float(fixed["threshold_mean"])
    mu = float(model["mu"])
    _report(
        "superseded independent Cartesian prior",
        mean - gap / 2 - amplitude,
        mu - mean - gap / 2 - amplitude,
        amplitude,
        gap,
    )


if __name__ == "__main__":
    main()
