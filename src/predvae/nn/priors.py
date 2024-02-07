import jax
import time
import optax
import torch

import equinox as eqx
import numpy as np
import jax.numpy as jnp
import jax.random as jr

from tqdm import tqdm
from equinox import Module
from jax.typing import ArrayLike
from src.predvae.nn.mlp import MLP
from jax.random import PRNGKeyArray
from collections.abc import Callable
from jax.experimental import checkify
from jax.scipy import stats as jstats
from progress_table import ProgressTable


class Gaussian(Module):
    mu: ArrayLike
    log_sigma: ArrayLike

    def __init__(self, mu: ArrayLike, log_sigma: ArrayLike, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.log_sigma = log_sigma

    def log_pdf(self, x: ArrayLike) -> ArrayLike:
        return jstats.norm.logpdf(x, self.mu, jnp.exp(self.log_sigma))

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.log_pdf(x)


class Categorical(Module):
    logits: ArrayLike

    def __init__(self, logits: ArrayLike, **kwargs):
        super().__init__(**kwargs)
        self.logits = logits

    def log_pdf(self, x: ArrayLike) -> ArrayLike:
        return jstats.multinomial.logpmf(x, logits=self.logits)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.log_pdf(x)
