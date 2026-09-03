from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

from twopm.config import ProjectConfig, load_config
from twopm.generative import (
    generate_recording,
    sample_parameters,
    standard_parameters,
)


def main() -> None:
    config = load_config("config/model.yaml")
    settings = config.section("prior_predictive")
    draw_count = int(settings["draws"])
    root_keys = jax.random.split(
        jax.random.PRNGKey(int(settings["seed"])),
        draw_count,
    )

    no_burn_data = deepcopy(config.data)
    no_burn_data["designs"]["entrained"]["burn_in_hours"] = 0.0
    no_burn_config = ProjectConfig(data=no_burn_data, source=config.source)

    circadian_fractions = []
    flat_fractions = []
    no_burn_fractions = []
    for root_key in root_keys:
        parameter_key, observation_key = jax.random.split(root_key)
        parameters = sample_parameters(parameter_key, config)
        circadian = generate_recording(observation_key, config, parameters)
        flat = generate_recording(
            observation_key,
            config,
            parameters._replace(
                c1=jnp.asarray(0.0),
                c2=jnp.asarray(0.0),
            ),
        )
        no_burn = generate_recording(
            observation_key,
            no_burn_config,
            parameters,
        )
        circadian_fractions.append(float(jnp.mean(circadian.observations)))
        flat_fractions.append(float(jnp.mean(flat.observations)))
        no_burn_fractions.append(float(jnp.mean(no_burn.observations)))

    fixed = generate_recording(
        jax.random.PRNGKey(int(config.section("observation")["seed"])),
        config,
        standard_parameters(config, amplitude=0.0),
    )
    print(f"canonical fixed: {float(jnp.mean(fixed.observations)):.3f}")
    print(
        "prior median, circadian + burn-in: "
        f"{np.median(circadian_fractions):.3f}"
    )
    print(
        "prior median, flat + burn-in: "
        f"{np.median(flat_fractions):.3f}"
    )
    print(
        "prior median, circadian + no burn-in: "
        f"{np.median(no_burn_fractions):.3f}"
    )


if __name__ == "__main__":
    main()
