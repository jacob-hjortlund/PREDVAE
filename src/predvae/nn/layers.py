from abc import abstractmethod
from math import prod

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from jax import Array
from jax.experimental import checkify
from jax.scipy import stats as jstats
from jax.typing import ArrayLike
