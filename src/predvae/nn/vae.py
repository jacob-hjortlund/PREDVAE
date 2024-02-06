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
        z, z_pars = self.encoder(x, rng_key)

        return z, z_pars

    def decode(self, z: ArrayLike, rng_key: ArrayLike):
        x_hat, x_pars = self.decoder(z, rng_key)

        return x_hat, x_pars

    def __call__(self, x: ArrayLike, rng_key: ArrayLike):
        encoder_key, decoder_key = jr.split(rng_key)
        z, z_pars = self.encode(x, encoder_key)
        x_hat, x_pars = self.decode(z, decoder_key)

        return x_hat, z_pars


class SSVAE(Module):
    encoder: Module
    decoder: Module
    classifier: Module

    def __init__(
        self,
        encoder: Module,
        decoder: Module,
        classifier: Module,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

    def classify(self, x: ArrayLike, rng_key: ArrayLike):
        y, y_pars = self.classifier(x, rng_key)

        return y, y_pars

    def encode(self, x: ArrayLike, y: ArrayLike, rng_key: ArrayLike):
        _x = jnp.column_stack([x, y])
        z, z_pars = self.encoder(_x, rng_key)

        return z, z_pars

    def decode(self, z: ArrayLike, y: ArrayLike, rng_key: ArrayLike):
        _z = jnp.column_stack([z, y])
        x_hat, x_pars = self.decoder(_z, rng_key)

        return x_hat, x_pars

    def __call__(self, x: ArrayLike, y: ArrayLike, rng_key: ArrayLike):
        classifier_key, encoder_key, decoder_key = jr.split(rng_key, 3)
        y, y_pars = self.classify(x, classifier_key)
        z, z_pars = self.encode(x, y, encoder_key)
        x_hat, x_pars = self.decode(z, y, decoder_key)

        return x_hat, z_pars, y_pars
