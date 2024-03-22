import jax

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from equinox import Module
from jax.typing import ArrayLike
from src.predvae.nn.mlp import MLP
from collections.abc import Callable
from jax.scipy import stats as jstats


class GaussianCoder(Module):
    mlp: Module
    input_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    width: ArrayLike = eqx.field(static=True, converter=jnp.asarray)
    depth: int = eqx.field(static=True)
    activation: Callable

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: ArrayLike,
        depth: int,
        activation: Callable,
        key: ArrayLike,
        use_spectral_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.mlp = MLP(
            in_size=input_size,
            out_size=output_size + 1,
            width_size=width,
            depth=depth,
            use_spectral_norm=use_spectral_norm,
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

    def log_prob(self, x, mu, log_sigma):
        return jnp.sum(jstats.norm.logpdf(x, loc=mu, scale=jnp.exp(log_sigma)))

    def __call__(self, x: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike):
        output, output_state = self.mlp(x, input_state)
        mu = output[..., : self.output_size]
        log_sigma = output[..., self.output_size :] * jnp.ones_like(mu)
        z = self.sample(mu, log_sigma, rng_key)

        return z, (mu, log_sigma), output_state


class CategoricalCoder(Module):
    mlp: Module
    input_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    width: ArrayLike = eqx.field(static=True, converter=jnp.asarray)
    depth: int = eqx.field(static=True)
    activation: Callable

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: ArrayLike,
        depth: int,
        activation: Callable,
        key: ArrayLike,
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

    def sample(self, logits, rng_key):
        categorical_sample = jr.categorical(rng_key, logits)
        one_hot_sample = jax.nn.one_hot(categorical_sample, self.output_size)

        return one_hot_sample

    def log_prob(self, x, logits):
        probs = jax.nn.softmax(logits)
        return jstats.multinomial.logpmf(x, 1, probs)

    def __call__(self, x: ArrayLike, state: eqx.nn.State, rng_key: ArrayLike):
        logits, state = self.mlp(x, state)
        z = self.sample(logits, rng_key)
        z = z.astype(jnp.int32)

        return z, (logits,), state


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

    def encode(self, x: ArrayLike, state: eqx.nn.State, rng_key: ArrayLike):
        z, z_pars, state = self.encoder(x, state, rng_key)

        return z, z_pars, state

    def decode(self, z: ArrayLike, state: eqx.nn.State, rng_key: ArrayLike):
        x_hat, x_pars, state = self.decoder(z, state, rng_key)

        return x_hat, x_pars, state

    def __call__(self, x: ArrayLike, state: eqx.nn.State, rng_key: ArrayLike):
        encoder_key, decoder_key = jr.split(rng_key)
        z, z_pars, state = self.encode(x, state, encoder_key)
        x_hat, x_pars, state = self.decode(z, state, decoder_key)

        return x_hat, z_pars, state


class SSVAE(Module):
    encoder: Module
    decoder: Module
    predictor: Module
    latent_prior: Module
    target_prior: Module
    encoder_input_layer: Module
    encoder_target_layer: Module
    decoder_input_layer: Module
    decoder_target_layer: Module

    def __init__(
        self,
        encoder: Module,
        decoder: Module,
        predictor: Module,
        latent_prior: Module,
        target_prior: Module,
        encoder_input_layer: Module = eqx.nn.Identity(),
        encoder_target_layer: Module = eqx.nn.Identity(),
        decoder_input_layer: Module = eqx.nn.Identity(),
        decoder_target_layer: Module = eqx.nn.Identity(),
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.latent_prior = latent_prior
        self.target_prior = target_prior
        self.encoder_input_layer = encoder_input_layer
        self.encoder_target_layer = encoder_target_layer
        self.decoder_input_layer = decoder_input_layer
        self.decoder_target_layer = decoder_target_layer

    def predict(self, x: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike):
        y, y_pars, output_state = self.predictor(x, input_state, rng_key)

        return y, y_pars, output_state

    def encode(
        self, x: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):
        x_tilde = self.encoder_input_layer(x)
        y_tilde = self.encoder_target_layer(y)
        _x = jnp.column_stack(
            [jnp.atleast_2d(x_tilde), jnp.atleast_2d(y_tilde)]
        ).squeeze()
        z, z_pars, output_state = self.encoder(_x, input_state, rng_key)

        return z, z_pars, output_state

    def decode(
        self, z: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):
        z_tilde = self.decoder_input_layer(z)
        y_tilde = self.decoder_target_layer(y)
        _z = jnp.column_stack(
            [jnp.atleast_2d(z_tilde), jnp.atleast_2d(y_tilde)]
        ).squeeze()
        x_hat, x_pars, output_state = self.decoder(_z, input_state, rng_key)

        return x_hat, x_pars, output_state

    def __call__(
        self, x: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):
        predictor_key, encoder_key, decoder_key = jr.split(rng_key, 3)
        y, y_pars, predictor_state = self.predict(x, input_state, predictor_key)
        z, z_pars, encoder_state = self.encode(x, y, predictor_state, encoder_key)
        x_hat, x_pars, decoder_state = self.decode(z, y, encoder_state, decoder_key)

        return y, z, x_hat, y_pars, z_pars, x_pars, decoder_state
