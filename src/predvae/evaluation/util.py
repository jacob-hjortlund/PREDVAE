import jax
import jax.numpy as jnp

from jax.scipy.stats import norm
from functools import partial


@partial(jax.vmap, in_axes=(0, None, None, None))
def gaussian_mixture_pdf(x, means, stds, weights):
    return jnp.sum(weights * norm.pdf(x, means, stds), axis=-1)
