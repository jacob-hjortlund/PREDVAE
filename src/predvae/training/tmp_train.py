import optax

import numpy as np
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from tqdm import tqdm
from jaxtyping import PyTree
from functools import partial
from jax.typing import ArrayLike
from jax.tree_util import tree_map
from collections.abc import Callable

# from optax.contrib import reduce_on_plateau
from optax import contrib as optax_contrib


def save(filename, model):
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load(filename, model):
    with open(filename, "rb") as f:
        model = eqx.tree_deserialise_leaves(f, model)
    return model


def make_train_step(
    optimizer: optax.GradientTransformation,
    loss_fn: Callable,
    filter_spec: eqx.Module,
    vectorize: bool = False,
):

    def train_step(
        x: ArrayLike,
        y: ArrayLike,
        rng_key: ArrayLike,
        model: eqx.Module,
        input_state: eqx.nn.State,
        optimizer_state: PyTree,
    ):

        free_params, frozen_params = eqx.partition(model, filter_spec)

        (loss_value, (aux, output_state)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(free_params, frozen_params, input_state, x, y, rng_key)

        updates, optimizer_state = optimizer.update(grads, optimizer_state, model)
        model = eqx.apply_updates(model, updates)

        return model, output_state, optimizer_state, loss_value, aux

    if vectorize:
        train_step = eqx.filter_vmap(train_step)

    return train_step


def make_eval_step(
    filter_spec: eqx.Module,
    loss_fn: Callable,
    vectorize: bool = False,
):
    def eval_step(
        x: ArrayLike,
        y: ArrayLike,
        rng_key: ArrayLike,
        model: eqx.Module,
        input_state: eqx.nn.State,
    ):
        free_params, frozen_params = eqx.partition(model, filter_spec)
        loss_value, (aux, output_state) = loss_fn(
            free_params, frozen_params, input_state, x, y, rng_key
        )
        return loss_value, output_state, aux

    if vectorize:
        eval_step = eqx.filter_vmap(eval_step)

    return eval_step


def initialize_optimizer(
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    lr_reducer: optax.GradientTransformation,
):

    filtered_model = eqx.filter(model, eqx.is_array)
    lr_reducer_state = lr_reducer.init(filtered_model)
    optimizer_state = optimizer.init(filtered_model)

    return optimizer_state, lr_reducer_state


def train_ssvae(
    model: eqx.Module,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    train_iterator: Callable,
    eval_iterator: Callable,
    train_iterator_states: tuple,
    eval_iterator_states: tuple,
    epochs: int = 100,
    filter_spec: eqx.Module = None,
    reduce_lr_on_plateau: bool = False,
    lr_reduction_kwargs: dict = {},
    loss_fn_kwargs: dict = {},
    jit: bool = False,
    vectorize: bool = False,
):

    loss_fn = partial(loss_fn, **loss_fn_kwargs)

    optimizer, optimizer_state, lr_reducer, lr_reducer_state = initialize_optimizer(
        model=model,
        optimizer=optimizer,
        reduce_lr_on_plateau=reduce_lr_on_plateau,
        lr_reduction_kwargs=lr_reduction_kwargs,
    )

    if filter_spec is None:
        filter_spec = tree_map(lambda x: True, model)

    train_step = make_train_step(
        optimizer=optimizer,
        lr_reducer=lr_reducer,
        loss_fn=loss_fn,
        filter_spec=filter_spec,
        vectorize=vectorize,
    )
    eval_step = make_eval_step(
        loss_fn=loss_fn,
        filter_spec=filter_spec,
        vectorize=vectorize,
    )

    if jit:
        train_step = eqx.filter_jit(train_step)
        eval_step = eqx.filter_jit(eval_step)
        train_iterator = eqx.filter_jit(train_iterator)
        eval_iterator = eqx.filter_jit(eval_iterator)

    train_photometric_dataloader_state, train_spectroscopic_dataloader_state = (
        train_iterator_states
    )
    eval_photometric_dataloader_state, eval_spectroscopic_dataloader_state = (
        eval_iterator_states
    )

    end_of_train_split = False
    end_of_val_split = False
    val_loss = jnp.inf

    for epoch in range(epochs):

        while not end_of_train_split:

            (
                x,
                y,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                reset_condition,
            ) = train_iterator(
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                rng_key=jr.PRNGKey(epoch),
            )

            model, input_state, optimizer_state, loss_value, aux = train_step(
                x,
                y,
                jr.PRNGKey(epoch),
                model,
                input_state,
                optimizer_state,
                lr_reducer_state,
                val_loss,
            )

            end_of_train_split = jnp.all(reset_condition)

        end_of_train_split = False

        while not end_of_val_split:

            (
                x,
                y,
                eval_photometric_dataloader_state,
                eval_spectroscopic_dataloader_state,
                reset_condition,
            ) = eval_iterator(
                eval_photometric_dataloader_state,
                eval_spectroscopic_dataloader_state,
                rng_key=jr.PRNGKey(epoch),
            )

            loss_value, output_state, aux = eval_step(
                x, y, jr.PRNGKey(epoch), model, input_state
            )

            end_of_val_split = jnp.all(reset_condition)

        end_of_val_split = False
