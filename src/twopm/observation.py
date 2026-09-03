import jax
import jax.numpy as jnp


def sleep_probabilities(
    gate: jax.Array,
    misclassification: float | jax.Array,
) -> jax.Array:
    error_rate = jnp.asarray(misclassification)
    return error_rate + (1 - 2 * error_rate) * gate


def sample_observations(
    key: jax.Array,
    probabilities: jax.Array,
) -> jax.Array:
    return jax.random.bernoulli(key, probabilities).astype(jnp.int8)
