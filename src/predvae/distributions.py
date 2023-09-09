from abc import abstractmethod
from math import prod

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from jax import Array
from jax.experimental import checkify
from jax.scipy import stats as jstats
from jax.typing import ArrayLike

from flowjax.flowjax.distributions import Distribution, StandardNormal, Normal


def GaussianMixture(Distribution):
    def __init__(self, locs: ArrayLike = 0, scales: ArrayLike = 1, log_weights=None):
        self.locs = locs
        self.scales = scales
        self.log_weights = log_weights
