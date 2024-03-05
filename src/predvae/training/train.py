import time
import optax
import torch

import numpy as np
import equinox as eqx
import jax.random as jr

from equinox import Module
from jaxtyping import PyTree
from functools import partial
from jax.typing import ArrayLike
from jax.tree_util import tree_map
from collections.abc import Callable
from progress_table import ProgressTable


def train(
    rng_key,
    model: Module,
    state: eqx.nn.State,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    loss_fn: Callable,
    epochs: int,
    train_batches_per_epoch: int,
    test_batches_per_epoch: int,
    print_every: int,
    loss_kwargs: dict = {},
    filter_spec: PyTree = None,
) -> Module:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    loss_fn = partial(loss_fn, **loss_kwargs)

    if filter_spec is None:
        filter_spec = tree_map(lambda x: True, model)

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_train_step(
        model: Module,
        input_state: eqx.nn.State,
        opt_state: PyTree,
        x: ArrayLike,
        y: ArrayLike,
        rng_key: ArrayLike,
    ):

        free_params, frozen_params = eqx.partition(model, filter_spec)
        (loss_value, (aux, output_state)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(free_params, frozen_params, input_state, x, y, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, output_state, opt_state, loss_value, aux

    # @eqx.filter_jit
    def make_eval_step(
        model: Module,
        input_state: eqx.nn.State,
        x: ArrayLike,
        y: ArrayLike,
        rng_key: ArrayLike,
    ):
        free_params, frozen_params = eqx.partition(model, filter_spec)
        loss_value, (aux, output_state) = loss_fn(
            free_params, frozen_params, input_state, x, y, rng_key
        )
        return loss_value, output_state, aux

    train_losses = []
    train_auxes = []
    test_losses = []
    test_auxes = []

    prog_table = ProgressTable(
        columns=["Epoch", "Train Loss", "Test Loss", "Epoch Rate [Epochs / Hour]"],
        refresh_rate=100,
        num_decimal_places=4,
        default_column_width=8,
        default_column_alignment="center",
        print_row_on_update=True,
        reprint_header_every_n_rows=30,
        custom_format=None,
        embedded_progress_bar=False,
        table_style="normal",
    )

    epoch_rates = []
    for epoch in range(epochs):
        t0 = time.time()

        batch_losses = []
        batch_auxes = []

        iterable_trainloader = iter(trainloader)
        iterable_testloader = iter(testloader)

        for _ in range(train_batches_per_epoch):

            batch_key, rng_key = jr.split(rng_key, 2)

            batch = next(iterable_trainloader)
            if len(batch) == 2:
                x, y = batch
            else:
                x, y, _ = batch
            x = x.numpy()
            y = y.numpy()
            y[np.isnan(y)] = -9999.0

            model, state, opt_state, train_loss, aux = make_train_step(
                model, state, opt_state, x, y, batch_key
            )
            batch_losses.append(train_loss)
            batch_auxes.append(aux)

        t1 = time.time()

        epoch_train_loss = np.mean(np.asarray(batch_losses), axis=0)
        train_losses.append(epoch_train_loss)

        epoch_train_aux = np.mean(np.asarray(batch_auxes), axis=0)
        train_auxes.append(epoch_train_aux)

        epoch_rate = 60**2 / (t1 - t0)
        epoch_rates.append(epoch_rate)

        if epoch % print_every == 0:
            test_loss = []
            test_aux = []

            for _ in range(test_batches_per_epoch):

                batch_key, rng_key = jr.split(rng_key, 2)

                batch = next(iterable_testloader)
                if len(batch) == 2:
                    x, y = batch
                else:
                    x, y, _ = batch
                x = x.numpy()
                y = y.numpy()
                y[np.isnan(y)] = -9999.0

                loss, state, aux = make_eval_step(model, state, x, y, batch_key)
                test_loss.append(loss)
                test_aux.append(aux)

            test_loss = np.mean(np.asarray(test_loss), axis=0)
            test_losses.append(test_loss)
            test_aux = np.mean(np.asarray(test_aux), axis=0)
            test_auxes.append(test_aux)

            prog_table.update("Epoch", epoch)
            prog_table.update("Train Loss", train_loss)
            prog_table.update("Test Loss", test_loss)
            prog_table.update("Epoch Rate [Epochs / Hour]", np.mean(epoch_rates))
            prog_table.next_row()

    prog_table.close()

    return model, state, train_losses, test_losses, train_auxes, test_auxes
