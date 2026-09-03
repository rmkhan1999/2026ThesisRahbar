from copy import deepcopy

import jax
import jax.numpy as jnp
from numpyro.infer.util import log_density

from twopm.config import ProjectConfig, load_config
from twopm.generative import generate_recording, standard_parameters
from twopm.inference import (
    numpyro_model,
    prior_distribution,
    sampled_parameter_names,
)


def test_log_joint_and_gradient_are_finite_at_ten_prior_draws():
    project_config = load_config("config/model.yaml")
    data = deepcopy(project_config.data)
    data["designs"]["entrained"]["burn_in_hours"] = 12.0
    data["designs"]["entrained"]["duration"] = 12.0
    config = ProjectConfig(data=data, source=project_config.source)
    labels = generate_recording(
        jax.random.PRNGKey(100),
        config,
        standard_parameters(config, amplitude=0.12),
    ).observations
    names = sampled_parameter_names(config)

    def log_joint(parameters):
        return log_density(
            numpyro_model,
            (labels, config),
            {},
            parameters,
        )[0]

    value_and_gradient = jax.jit(jax.value_and_grad(log_joint))
    draw_keys = jax.random.split(jax.random.PRNGKey(101), 10)
    for draw_key in draw_keys:
        site_keys = jax.random.split(draw_key, len(names))
        parameters = {
            name: prior_distribution(name, config).sample(site_key)
            for name, site_key in zip(names, site_keys)
        }
        value, gradient = value_and_gradient(parameters)
        assert jnp.isfinite(value)
        assert all(jnp.isfinite(component) for component in gradient.values())
