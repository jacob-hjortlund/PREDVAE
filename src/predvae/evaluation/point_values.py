import jax
import numpy as np
import jax.numpy as jnp

from jax.scipy.stats import norm
from functools import partial


def difference_matrix(a):
    x = jnp.reshape(a, (len(a), 1))
    return x - x.transpose()


@partial(jax.vmap, in_axes=(0, 0, 0, 0, None, None, None))
def calculate_point_values(weights, means, stds, ppfs, q_min=0.01, q_max=0.99, q_n=100):

    mean = jnp.sum(weights * means, axis=-1)
    std = jnp.sqrt(
        jnp.sum(weights * stds**2, axis=-1)
        + jnp.sum(weights * means**2, axis=-1)
        - jnp.sum(weights * means, axis=-1) ** 2
    )

    q = jnp.linspace(q_min, q_max, q_n)
    idx_median = jnp.argmin(jnp.abs(q - 0.5))
    idx_1sig_lower = jnp.argmin(jnp.abs(q - 0.16))
    idx_1sig_upper = jnp.argmin(jnp.abs(q - 0.84))
    idx_3sig_lower = jnp.argmin(jnp.abs(q - 0.00135))
    idx_3sig_upper = jnp.argmin(jnp.abs(q - 0.99865))

    medians = ppfs[idx_median]
    lower_1sig = ppfs[idx_1sig_lower]
    upper_1sig = ppfs[idx_1sig_upper]
    lower_3sig = ppfs[idx_3sig_lower]
    upper_3sig = ppfs[idx_3sig_upper]

    return (mean, std), (medians, lower_1sig, upper_1sig), (lower_3sig, upper_3sig)
