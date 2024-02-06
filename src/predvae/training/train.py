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
from jax import Array, vmap
from jax.typing import ArrayLike
from src.predvae.nn.mlp import MLP
from jax.random import PRNGKeyArray
from collections.abc import Callable
from jax.experimental import checkify
from jax.scipy import stats as jstats
from progress_table import ProgressTable
from jaxtyping import Array, Float, Int, PyTree


def gaussian_kl_divergence(mu: ArrayLike, log_sigma: ArrayLike) -> ArrayLike:
    """
    Compute the KL divergence between a diagonal Gaussian and the standard normal.

    Args:
        mu (ArrayLike): Mean array
        log_sigma (ArrayLike): Log standard deviation array

    Returns:
        ArrayLike: KL divergence
    """

    return -0.5 * jnp.sum(1 + 2 * log_sigma - mu**2 - jnp.exp(2 * log_sigma), axis=-1)


def gaussian_log_likelihood(
    x: ArrayLike, mu: ArrayLike, log_sigma: ArrayLike
) -> ArrayLike:
    """
    Compute the log likelihood of a diagonal Gaussian.

    Args:
        x (ArrayLike): Input array
        mu (ArrayLike): Mean array
        log_sigma (ArrayLike): Log standard deviation array

    Returns:
        ArrayLike: Log likelihood
    """

    return jnp.sum(jstats.norm.logpf(x, loc=mu, scale=jnp.exp(log_sigma)), axis=-1)


def train(
    rng_key,
    model: Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    loss_fn: Callable,
    epochs: int,
    print_every: int,
) -> Module:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: Module,
        opt_state: PyTree,
        x: Float[Array, "batch 784"],
        rng_key: PRNGKeyArray,
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    train_losses = []
    test_losses = []

    prog_table = ProgressTable(
        columns=["Epoch", "Train Loss", "Test Loss", "Epoch Rate [Epochs / Hour]"],
    )

    t0 = time.time()
    for epoch, (x, _) in zip(range(epochs), infinite_trainloader()):
        epoch_key, eval_key, rng_key = jr.split(rng_key, 3)
        # for x, _ in prog_table(trainloader):
        #     x = x.numpy()
        #     model, opt_state, train_loss = make_step(model, opt_state, x, epoch_key)
        # train_losses.append(train_loss)

        x = x.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, epoch_key)
        train_losses.append(train_loss)

        t1 = time.time()

        if epoch % print_every == 0:
            test_loss = []
            for x, _ in testloader:
                x = x.numpy()
                test_loss.append(loss_fn(model, x, eval_key))
            test_loss = np.sum(test_loss) / len(test_loss)
            test_losses.append(test_loss)
            prog_table.update("Epoch", epoch)
            prog_table.update("Train Loss", train_loss)
            prog_table.update("Test Loss", test_loss)
            prog_table.update(
                "Epoch Rate [Epochs / Hour]", (epoch + 1) * 60**2 / (t1 - t0)
            )
            prog_table.next_row()

    prog_table.close()

    return model, train_losses, test_losses
