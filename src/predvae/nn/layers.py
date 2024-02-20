import jax.numpy as jnp

from equinox import Module
from jax.typing import ArrayLike
from jax.lax import stop_gradient


class Frozen(Module):
    arr: ArrayLike

    def __init__(self, arr: ArrayLike):
        super().__init__()
        self.arr = jnp.asarray(arr)

    def __jax_array__(self):
        return stop_gradient(self.arr)
