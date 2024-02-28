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
from jax.random import PRNGKeyArray
from collections.abc import Callable
from progress_table import ProgressTable


def train(
    rng_key,
    model: Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    loss_fn: Callable,
    epochs: int,
    test_epochs: int,
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
    def make_step(
        model: Module,
        opt_state: PyTree,
        x: ArrayLike,
        y: ArrayLike,
        rng_key: PRNGKeyArray,
    ):

        free_params, frozen_params = eqx.partition(model, filter_spec)
        (loss_value, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            free_params, frozen_params, x, y, rng_key
        )
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value, aux

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    def infinite_testloader():
        while True:
            yield from testloader

    train_losses = []
    train_auxes = []
    test_losses = []
    test_auxes = []

    prog_table = ProgressTable(
        columns=["Epoch", "Train Loss", "Test Loss", "Epoch Rate [Epochs / Hour]"],
    )

    t0 = time.time()
    for epoch, batch in zip(range(epochs), infinite_trainloader()):

        epoch_key, eval_key, rng_key = jr.split(rng_key, 3)

        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch
        x = x.numpy()
        y = y.numpy()
        y[np.isnan(y)] = -9999.0

        model, opt_state, train_loss, aux = make_step(model, opt_state, x, y, epoch_key)
        train_losses.append(train_loss)
        train_auxes.append(aux)

        t1 = time.time()

        if epoch % print_every == 0:
            test_loss = []
            test_aux = []

            for test_epoch, batch in zip(range(test_epochs), testloader):

                if len(batch) == 2:
                    x, y = batch
                else:
                    x, y, _ = batch
                x = x.numpy()
                y = y.numpy()
                y[np.isnan(y)] = -9999.0

                free_params, frozen_params = eqx.partition(model, filter_spec)
                loss, aux = loss_fn(free_params, frozen_params, x, y, eval_key)
                test_loss.append(loss)
                test_aux.append(aux)

            test_loss = np.sum(test_loss) / len(test_loss)
            test_losses.append(test_loss)
            test_auxes.append(test_aux)
            prog_table.update("Epoch", epoch)
            prog_table.update("Train Loss", train_loss)
            prog_table.update("Test Loss", test_loss)
            prog_table.update(
                "Epoch Rate [Epochs / Hour]", (epoch + 1) * 60**2 / (t1 - t0)
            )
            prog_table.next_row()

    prog_table.close()

    return model, train_losses, test_losses, train_auxes, test_auxes
