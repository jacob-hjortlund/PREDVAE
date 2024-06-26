import jax

import jax.numpy as jnp

from equinox import Module
from jax.typing import ArrayLike
from jax.scipy import stats as jstats


class Gaussian(Module):
    mu: ArrayLike
    log_sigma: ArrayLike

    def __init__(self, mu: ArrayLike, log_sigma: ArrayLike, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.log_sigma = log_sigma

    def log_pdf(self, x: ArrayLike) -> ArrayLike:
        return jnp.sum(jstats.norm.logpdf(x, self.mu, jnp.exp(self.log_sigma)))

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.log_pdf(x)


class Categorical(Module):
    logits: ArrayLike

    def __init__(self, logits: ArrayLike, **kwargs):
        super().__init__(**kwargs)
        self.logits = logits

    def log_pdf(self, x: ArrayLike) -> ArrayLike:
        probs = jax.nn.softmax(self.logits)
        return jstats.multinomial.logpmf(x, n=1, p=probs)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.log_pdf(x)
