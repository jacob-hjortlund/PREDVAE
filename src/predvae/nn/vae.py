import jax
import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from equinox import Module
from jax.typing import ArrayLike
from src.predvae.nn.mlp import MLP
from collections.abc import Callable
from jax.scipy import stats as jstats


class InputLayer(Module):

    x_weight: ArrayLike
    y_weight: ArrayLike
    x_features: int = eqx.field(static=True)
    y_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(
        self, x_features: int, y_features: int, out_features: int, key: ArrayLike
    ):

        self.x_features = x_features
        self.y_features = y_features
        self.out_features = out_features

        x_lim = 1 / math.sqrt(x_features)
        y_lim = 1 / math.sqrt(y_features)
        x_key, y_key = jr.split(key)
        self.x_weight = jr.uniform(
            x_key, (out_features, x_features), minval=-x_lim, maxval=x_lim
        )
        self.y_weight = jr.uniform(
            y_key, (out_features, y_features), minval=-y_lim, maxval=y_lim
        )

    def __call__(self, x: ArrayLike, y: ArrayLike):
        return self.x_weight @ x + self.y_weight @ jnp.atleast_1d(y)


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
        num_power_iterations: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.mlp = MLP(
            in_size=input_size,
            out_size=output_size + 1,
            width_size=width,
            depth=depth,
            use_spectral_norm=use_spectral_norm,
            num_power_iterations=num_power_iterations,
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


class GaussianMixtureCoder(Module):
    mlp: Module
    input_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    width: ArrayLike = eqx.field(static=True, converter=jnp.asarray)
    depth: int = eqx.field(static=True)
    num_components: int = eqx.field(static=True)
    activation: Callable

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: ArrayLike,
        depth: int,
        num_components: int,
        activation: Callable,
        key: ArrayLike,
        use_spectral_norm: bool = False,
        num_power_iterations: int = 1,
    ):
        super().__init__()
        self.mlp = MLP(
            in_size=input_size,
            out_size=(1 + 2 * output_size) * num_components,
            width_size=width,
            depth=depth,
            key=key,
            activation=activation,
            use_spectral_norm=use_spectral_norm,
            num_power_iterations=num_power_iterations,
        )
        self.input_size = input_size
        self.output_size = output_size
        self.width = width
        self.depth = depth
        self.num_components = num_components
        self.activation = activation

    def sample(self, logits, mu, log_sigma, rng_key):

        categorical_key, normal_key = jr.split(rng_key)
        idx = jr.categorical(categorical_key, logits)
        mu, log_sigma = mu[..., idx], log_sigma[..., idx]
        z = mu + jnp.exp(log_sigma) * jr.normal(normal_key, mu.shape)

        return z

    def log_prob(self, x, logits, mu, log_sigma):
        log_normals = jstats.norm.logpdf(x, loc=mu, scale=jnp.exp(log_sigma))
        log_probs = jax.nn.log_softmax(logits)
        return jax.scipy.special.logsumexp(log_probs + log_normals, axis=-1)

    def __call__(self, x: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike):
        output, output_state = self.mlp(x, input_state)
        logits = output[..., : self.num_components]
        mu = output[
            ..., self.num_components : self.output_size * self.num_components * 2
        ]
        log_sigma = output[..., self.output_size * self.num_components * 2 :]
        z = self.sample(logits, mu, log_sigma, rng_key)

        return z, (logits, mu, log_sigma), output_state


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
    encoder_input_layer: InputLayer
    decoder_input_layer: InputLayer

    def __init__(
        self,
        encoder: Module,
        decoder: Module,
        predictor: Module,
        latent_prior: Module,
        target_prior: Module,
        encoder_input_layer: Module = eqx.nn.Identity(),
        decoder_input_layer: Module = eqx.nn.Identity(),
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
        self.decoder_input_layer = decoder_input_layer

    def predict(self, x: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike):
        y, y_pars, output_state = self.predictor(x, input_state, rng_key)

        return y, y_pars, output_state

    def encode(
        self, x: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):
        _x = self.encoder_input_layer(x, y)
        z, z_pars, output_state = self.encoder(_x, input_state, rng_key)

        return z, z_pars, output_state

    def decode(
        self, z: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):
        _z = self.decoder_input_layer(z, y)
        x_hat, x_pars, output_state = self.decoder(_z, input_state, rng_key)

        return x_hat, x_pars, output_state

    def unsupervised_call(
        self, x: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):

        return self(x, y, input_state, rng_key)

    def supervised_call(
        self, x: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):

        predictor_key, encoder_key, decoder_key = jr.split(rng_key, 3)
        _, y_pars, predictor_state = self.predict(x, input_state, predictor_key)
        z, z_pars, encoder_state = self.encode(x, y, predictor_state, encoder_key)
        x_hat, x_pars, decoder_state = self.decode(z, y, encoder_state, decoder_key)

        return (y, z, x_hat, y_pars, z_pars, x_pars), decoder_state

    def __call__(
        self, x: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):
        predictor_key, encoder_key, decoder_key = jr.split(rng_key, 3)
        y, y_pars, predictor_state = self.predict(x, input_state, predictor_key)
        z, z_pars, encoder_state = self.encode(x, y, predictor_state, encoder_key)
        x_hat, x_pars, decoder_state = self.decode(z, y, encoder_state, decoder_key)

        return (y, z, x_hat, y_pars, z_pars, x_pars), decoder_state


class SSVAEv2(Module):
    encoder: Module
    decoder: Module
    predictor: Module
    latent_prior: Module
    target_prior: Module
    predictor_input_layer: InputLayer
    decoder_input_layer: InputLayer

    def __init__(
        self,
        encoder: Module,
        decoder: Module,
        predictor: Module,
        latent_prior: Module,
        target_prior: Module,
        predictor_input_layer: Module = eqx.nn.Identity(),
        decoder_input_layer: Module = eqx.nn.Identity(),
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.latent_prior = latent_prior
        self.target_prior = target_prior
        self.predictor_input_layer = predictor_input_layer
        self.decoder_input_layer = decoder_input_layer

    def predict(
        self, x: ArrayLike, z: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):
        _x = self.predictor_input_layer(z, x)
        y, y_pars, output_state = self.predictor(_x, input_state, rng_key)

        return y, y_pars, output_state

    def encode(self, x: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike):
        z, z_pars, output_state = self.encoder(x, input_state, rng_key)

        return z, z_pars, output_state

    def decode(
        self, z: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):
        _z = self.decoder_input_layer(z, y)
        x_hat, x_pars, output_state = self.decoder(_z, input_state, rng_key)

        return x_hat, x_pars, output_state

    def unsupervised_call(
        self, x: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):

        return self(x, y, input_state, rng_key)

    def supervised_call(
        self, x: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):

        predictor_key, encoder_key, decoder_key = jr.split(rng_key, 3)
        z, z_pars, encoder_state = self.encode(x, input_state, encoder_key)
        _, y_pars, predictor_state = self.predict(x, z, encoder_state, predictor_key)
        x_hat, x_pars, decoder_state = self.decode(z, y, predictor_state, decoder_key)

        return (y, z, x_hat, y_pars, z_pars, x_pars), decoder_state

    def __call__(
        self, x: ArrayLike, y: ArrayLike, input_state: eqx.nn.State, rng_key: ArrayLike
    ):
        predictor_key, encoder_key, decoder_key = jr.split(rng_key, 3)
        z, z_pars, encoder_state = self.encode(x, input_state, encoder_key)
        y, y_pars, predictor_state = self.predict(x, z, encoder_state, predictor_key)
        x_hat, x_pars, decoder_state = self.decode(z, y, predictor_state, decoder_key)

        return (y, z, x_hat, y_pars, z_pars, x_pars), decoder_state
