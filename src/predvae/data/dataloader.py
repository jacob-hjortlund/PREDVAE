import jax

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from collections.abc import Callable
from jaxtyping import Array, PRNGKeyArray


class DataLoader(eqx.Module):
    dataset: eqx.Module
    batch_size: int = eqx.field(static=True)
    shuffle: bool = eqx.field(static=True)
    drop_last: bool = eqx.field(static=True)

    reset_index: eqx.nn.StateIndex
    indices_index: eqx.nn.StateIndex
    position_index: eqx.nn.StateIndex
    rng_key_index: eqx.nn.StateIndex

    def __init__(
        self,
        dataset: eqx.Module,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        rng_key: PRNGKeyArray = None,
    ):
        self.batch_size = batch_size
        self.shuffle = jnp.array(shuffle)
        self.drop_last = drop_last
        self.dataset = dataset

        if rng_key is None:
            rng_key = jr.PRNGKey(0)

        indices = jnp.arange(0, len(dataset), dtype=int)
        self.reset_index = eqx.nn.StateIndex(jnp.array(shuffle))
        self.indices_index = eqx.nn.StateIndex(indices)
        self.position_index = eqx.nn.StateIndex(jnp.array(0))
        self.rng_key_index = eqx.nn.StateIndex(rng_key)

    def __call__(self, state: eqx.nn.State) -> tuple[Array, eqx.nn.State, bool]:

        reset = state.get(self.reset_index)
        rng_key = state.get(self.rng_key_index)

        def shuffle_indices(indices, rng_key):
            shuffle_key, rng_key = jr.split(rng_key)
            return jr.permutation(shuffle_key, indices), rng_key

        indices = state.get(self.indices_index)
        indices, rng_key = jax.lax.cond(
            reset,
            lambda indices, rng_key: shuffle_indices(indices, rng_key),
            lambda indices, rng_key: (indices, rng_key),
            indices,
            rng_key,
        )

        position = state.get(self.position_index)
        position = jax.lax.cond(
            reset, lambda pos: jnp.zeros_like(pos), lambda pos: pos, position
        )

        to_subtract_from_indices = jax.lax.cond(
            self.drop_last,
            lambda: jax.lax.rem(len(indices), self.batch_size),
            lambda: 0,
        )
        reset_condition = jax.lax.cond(
            position >= len(indices) - to_subtract_from_indices,
            lambda: True,
            lambda: False,
        )

        batch_indices = jax.lax.dynamic_slice_in_dim(indices, position, self.batch_size)
        batch = jax.vmap(lambda i: self.dataset(i))(batch_indices)

        state = state.set(self.rng_key_index, rng_key)
        state = state.set(self.reset_index, jnp.array(reset_condition))
        state = state.set(self.indices_index, indices)
        state = state.set(self.position_index, position + self.batch_size)

        return batch, state, reset_condition
