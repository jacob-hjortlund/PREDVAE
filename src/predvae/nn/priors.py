import jax

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from equinox import Module
from equinox.nn import Linear
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


class GaussianMixture(Module):
    input_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    mus: Linear
    log_sigmas: Linear

    def __init__(self, input_size: int, output_size: int, key: ArrayLike):
        self.input_size = input_size
        self.output_size = output_size
        mu_key, log_sigma_key = jr.split(key)
        self.mus = Linear(input_size, output_size, False, key=mu_key)
        self.log_sigmas = Linear(input_size, output_size, False, key=log_sigma_key)

    def log_pdf(self, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        mus = self.mus(y)
        log_sigmas = self.log_sigmas(y)
        return jnp.sum(jstats.norm.logpdf(z, mus, jnp.exp(log_sigmas)))

    def __call__(self, y: ArrayLike, z: ArrayLike) -> ArrayLike:
        return self.log_pdf(y, z)


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
