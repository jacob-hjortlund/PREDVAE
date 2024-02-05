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


class GaussianCoder(Module):
    mlp: Module
    input_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    width: Array = eqx.field(static=True, converter=jnp.asarray)
    depth: int = eqx.field(static=True)
    activation: Callable

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: Array,
        depth: int,
        activation: Callable,
        key: PRNGKeyArray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlp = MLP(
            in_size=input_size,
            out_size=output_size + 1,
            width_size=width,
            depth=depth,
            key=key,
            activation=activation,
            **kwargs,
        )
        self.input_size = input_size
        self.output_size = output_size
        self.width = width
        self.depth = depth
        self.activation = activation

    def sample(self, mu, log_sigma, rng_key):
        return mu + jnp.exp(log_sigma) * jr.normal(rng_key, mu.shape)

    def __call__(self, x: ArrayLike, rng_key: ArrayLike):
        output = self.mlp(x)
        mu = output[..., : self.output_size]
        log_sigma = output[..., self.output_size :] * jnp.ones_like(mu)
        z = self.sample(mu, log_sigma, rng_key)

        return z, mu, log_sigma


class CategoricalCoder(Module):
    mlp: Module
    input_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    width: Array = eqx.field(static=True, converter=jnp.asarray)
    depth: int = eqx.field(static=True)
    activation: Callable

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: Array,
        depth: int,
        activation: Callable,
        key: PRNGKeyArray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlp = MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=width,
            depth=depth,
            key=key,
            activation=activation,
            **kwargs,
        )
        self.input_size = input_size
        self.output_size = output_size
        self.width = width
        self.depth = depth
        self.activation = activation

    def sample(self, probs, rng_key):
        return jr.categorical(rng_key, probs)

    def __call__(self, x: ArrayLike, rng_key: ArrayLike):
        logits = self.mlp(x)
        probs = jax.nn.softmax(logits)
        z = self.sample(probs, rng_key)

        return z, logits


class VAE(Module):
    encoder: Module
    decoder: Module

    def __init__(
        self,
        encoder: Module,
        decoder: Module,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: ArrayLike, rng_key: ArrayLike):
        z, z_mu, z_log_sigma = self.encoder(x, rng_key)

        return z, z_mu, z_log_sigma

    def decode(self, z: ArrayLike, rng_key: ArrayLike):
        x_hat, x_mu, x_log_sigma = self.decoder(z, rng_key)

        return x_hat, x_mu, x_log_sigma

    def __call__(self, x: ArrayLike, rng_key: ArrayLike):
        encoder_key, decoder_key = jr.split(rng_key)
        z, z_mu, z_log_sigma = self.encode(x, encoder_key)
        x_hat, x_mu, x_log_sigma = self.decode(z, decoder_key)

        return x_hat, z_mu, z_log_sigma


def kl_divergence(mu, log_sigma):
    return -0.5 * jnp.sum(1 + 2 * log_sigma - mu**2 - jnp.exp(2 * log_sigma), axis=-1)


def reproduction_loss(x_obs, mu, log_sigma):
    return jnp.sum(jstats.norm.logpdf(x_obs, loc=mu, scale=jnp.exp(log_sigma)), axis=-1)


def vae_loss(
    model: VAE,
    x: ArrayLike,
    rng_key: PRNGKeyArray,
):
    @eqx.filter_jit
    def _vae_loss(
        model: VAE,
        x: ArrayLike,
        rng_key: PRNGKeyArray,
    ):
        encoder_key, decoder_key = jr.split(rng_key)
        z, z_mu, z_log_sigma = model.encode(x, encoder_key)
        x_hat, x_mu, x_log_sigma = model.decode(z, decoder_key)

        kl_div = kl_divergence(z_mu, z_log_sigma)
        rep_loss = reproduction_loss(x, x_mu, x_log_sigma)
        loss = -(rep_loss - kl_div)

        return loss

    loss = vmap(_vae_loss, in_axes=(None, 0, None))(model, x, rng_key)
    batch_loss = jnp.mean(loss)

    return batch_loss


def train(
    rng_key,
    model: VAE,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    loss_fn: Callable,
    epochs: int,
    print_every: int,
) -> VAE:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(model: VAE, opt_state: PyTree, x: Float[Array, "batch 784"], rng_key):
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
