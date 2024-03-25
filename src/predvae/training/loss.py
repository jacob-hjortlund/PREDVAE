import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from jax.lax import cond
from jax import vmap, pmap
from equinox import Module
from .util import filter_cond
from jax.typing import ArrayLike
from jax.scipy import stats as jstats
from collections.abc import Callable


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

    return jnp.sum(jstats.norm.logpdf(x, loc=mu, scale=jnp.exp(log_sigma)), axis=-1)


def gaussian_vae_loss(
    free_params: Module,
    frozen_params: Module,
    x: ArrayLike,
    y: ArrayLike,
    rng_key: ArrayLike,
    *args,
    **kwargs,
) -> Module:
    """
    Batch loss function for a gaussian VAE.

    Args:
        free_params (Module): VAE Model containing free parameters
        frozen_params (Module): VAE Model containing frozen parameters
        x (ArrayLike): Data batch
        rng_key (PRNGKeyArray): RNG key with leading dimension equal to the batch size

    Returns:
        Module: Batch loss
    """

    def _sample_loss(
        model: Module,
        x: ArrayLike,
        rng_key: ArrayLike,
    ):
        encoder_key, decoder_key = jr.split(rng_key)
        z, z_pars = model.encode(x, encoder_key)
        x_hat, x_pars = model.decode(z, decoder_key)

        kl_divergence = gaussian_kl_divergence(*z_pars)
        reproduction_loss = gaussian_log_likelihood(x, *x_pars)
        loss = -(reproduction_loss - kl_divergence)

        return loss

    model = eqx.combine(free_params, frozen_params)
    loss = vmap(_sample_loss, in_axes=(None, 0, None))(model, x, rng_key)
    batch_loss = jnp.sum(loss)

    return batch_loss, jnp.array([jnp.nan])


def _supervised_sample_loss(
    model: Module,
    input_state: eqx.nn.State,
    x: ArrayLike,
    y: ArrayLike,
    use_target: bool,
    rng_key: ArrayLike,
):
    encoder_key, decoder_key = jr.split(rng_key, 2)
    _, y_pars, predictor_state = model.predict(x, input_state, encoder_key)
    z, z_pars, encoder_state = model.encode(x, y, predictor_state, encoder_key)
    x_hat, x_pars, decoder_state = model.decode(z, y, encoder_state, decoder_key)

    target_log_prior = cond(
        use_target,
        lambda *_: model.target_prior(y),
        lambda *_: jnp.array(0.0),
    )
    target_log_prob = cond(
        use_target,
        lambda *_: model.predictor.log_prob(y, *y_pars),
        lambda *_: jnp.array(0.0),
    )

    latent_log_prior = model.latent_prior(z)
    latent_log_prob = model.encoder.log_prob(z, *z_pars)

    reconstruction_log_prob = model.decoder.log_prob(x, *x_pars)

    supervised_loss = (
        latent_log_prob - latent_log_prior - target_log_prior - reconstruction_log_prob
    )

    loss = jnp.array(
        [
            -9999.0,
            supervised_loss,
            target_log_prob,
            target_log_prior,
            latent_log_prior,
            latent_log_prob,
            reconstruction_log_prob,
        ]
    )

    return loss, decoder_state


def _unsupervised_sample_loss(
    model: Module,
    input_state: eqx.nn.State,
    x: ArrayLike,
    y: ArrayLike,
    use_target: bool,
    rng_key: ArrayLike,
):
    encoder_key, predictor_key, decoder_key = jr.split(rng_key, 3)
    y, y_pars, predictor_state = model.predict(x, input_state, predictor_key)
    z, z_pars, encoder_state = model.encode(x, y, predictor_state, encoder_key)
    x_hat, x_pars, decoder_state = model.decode(z, y, encoder_state, decoder_key)

    target_log_prior = cond(
        use_target,
        lambda *_: model.target_prior(y),
        lambda *_: jnp.array(0.0),
    )
    target_log_prob = cond(
        use_target,
        lambda *_: model.predictor.log_prob(y, *y_pars),
        lambda *_: jnp.array(0.0),
    )

    latent_log_prior = model.latent_prior(z)
    latent_log_prob = model.encoder.log_prob(z, *z_pars)

    reconstruction_log_prob = model.decoder.log_prob(x, *x_pars)

    unsupervised_loss = (
        latent_log_prob
        + target_log_prob
        - latent_log_prior
        - target_log_prior
        - reconstruction_log_prob
    )

    loss = jnp.array(
        [
            unsupervised_loss,
            0.0,
            target_log_prob,
            target_log_prior,
            latent_log_prior,
            latent_log_prob,
            reconstruction_log_prob,
        ]
    )

    return loss, decoder_state


def _sample_loss(
    model: Module,
    input_state: eqx.nn.State,
    x: ArrayLike,
    y: ArrayLike,
    rng_key: ArrayLike,
    missing_target_value: ArrayLike = -9999.0,
    target_transform: Callable = lambda x: x,
    use_target: bool = True,
):

    unsupervised_loss_args = [
        model,
        input_state,
        x,
        target_transform(y),
        use_target,
        rng_key,
    ]
    (dynamic_unsupervised_loss, dynamic_unsupervised_state), (
        static_unsupervised_loss,
        static_unsupervised_state,
    ) = eqx.partition(_unsupervised_sample_loss(*unsupervised_loss_args), eqx.is_array)
    unsupervised_state = eqx.combine(
        dynamic_unsupervised_state, static_unsupervised_state
    )

    supervised_loss_args = [
        model,
        unsupervised_state,
        x,
        target_transform(y),
        use_target,
        rng_key,
    ]
    (dynamic_supervised_loss, dynamic_supervised_state), (
        static_supervised_loss,
        static_supervised_state,
    ) = eqx.partition(_supervised_sample_loss(*supervised_loss_args), eqx.is_array)
    supervised_state = eqx.combine(dynamic_supervised_state, static_supervised_state)

    static_loss = eqx.error_if(
        static_unsupervised_loss,
        static_unsupervised_loss != static_supervised_loss,
        "Filtered conditional loss functions should have the same static component",
    )

    dynamic_loss = cond(
        y != missing_target_value,
        lambda *_: dynamic_supervised_loss,
        lambda *_: dynamic_unsupervised_loss,
    )

    loss_components = eqx.combine(dynamic_loss, static_loss)

    return loss_components, supervised_state


def ssvae_loss(
    free_params: Module,
    frozen_params: Module,
    input_state: eqx.nn.State,
    x: ArrayLike,
    y: ArrayLike,
    rng_key: ArrayLike,
    alpha: ArrayLike,
    vae_factor: int = 1.0,
    missing_target_value: ArrayLike = -9999.0,
    target_transform: Callable = lambda x: x,
    use_target: bool = True,
) -> Module:
    """
    Batch loss function for a semi-supervised VAE classifier.

    Args:
        free_params (Module): SSVAE model containing free parameters
        frozen_params (Module): SSVAE model containing frozen parameters
        x (ArrayLike): Data batch
        y (ArrayLike): Target batch
        rng_key (PRNGKeyArray): RNG key with leading dimension equal to the batch size
        alpha (ArrayLike): Target loss weight

    Returns:
        Module: Batch loss
    """

    model = eqx.combine(free_params, frozen_params)

    vmapped_sample_loss = vmap(
        _sample_loss,
        in_axes=(None, None, 0, 0, None, None, None, None),
        out_axes=(0, None),
    )
    loss_components, output_state = vmapped_sample_loss(
        model,
        input_state,
        x,
        y,
        rng_key,
        missing_target_value,
        target_transform,
        use_target,
    )

    batch_unsupervised_loss = vae_factor * jnp.sum(
        loss_components[:, 0], where=y.squeeze() != missing_target_value
    )
    batch_unsupervised_target_log_prob = jnp.sum(
        loss_components[:, 2], where=y.squeeze() != missing_target_value
    )
    batch_unsupervised_target_log_prior = jnp.sum(
        loss_components[:, 3], where=y.squeeze() != missing_target_value
    )
    batch_unsupervised_latent_log_prior = jnp.sum(
        loss_components[:, 4], where=y.squeeze() != missing_target_value
    )
    batch_unsupervised_latent_log_prob = jnp.sum(
        loss_components[:, 5], where=y.squeeze() != missing_target_value
    )
    batch_unsupervised_reconstruction_log_prob = jnp.sum(
        loss_components[:, 6], where=y.squeeze() != missing_target_value
    )

    batch_supervised_loss = vae_factor * jnp.sum(
        loss_components[:, 1], where=y.squeeze() == missing_target_value
    )
    batch_supervised_target_log_prob_loss = -alpha * jnp.mean(
        loss_components[:, 2], where=y.squeeze() == missing_target_value
    )
    batch_supervised_target_log_prob = jnp.sum(
        loss_components[:, 2], where=y.squeeze() == missing_target_value
    )
    batch_supervised_target_log_prior = jnp.sum(
        loss_components[:, 3], where=y.squeeze() == missing_target_value
    )
    batch_supervised_latent_log_prior = jnp.sum(
        loss_components[:, 4], where=y.squeeze() == missing_target_value
    )
    batch_supervised_latent_log_prob = jnp.sum(
        loss_components[:, 5], where=y.squeeze() == missing_target_value
    )
    batch_supervised_reconstruction_log_prob = jnp.sum(
        loss_components[:, 6], where=y.squeeze() == missing_target_value
    )

    sum_array = jnp.asarray(
        [
            batch_unsupervised_loss,
            batch_supervised_loss,
            batch_supervised_target_log_prob_loss
            #- alpha * batch_supervised_target_log_prob,
        ]
    )
    batch_loss = jnp.sum(sum_array, where=~jnp.isnan(sum_array))

    aux_values = jnp.array(
        [
            batch_unsupervised_loss,
            batch_unsupervised_target_log_prob,
            batch_unsupervised_target_log_prior,
            batch_unsupervised_latent_log_prior,
            batch_unsupervised_latent_log_prob,
            batch_unsupervised_reconstruction_log_prob,
            batch_supervised_loss,
            batch_supervised_target_log_prob_loss,
            #-alpha * batch_supervised_target_log_prob,
            batch_supervised_target_log_prob,
            batch_supervised_target_log_prior,
            batch_supervised_latent_log_prior,
            batch_supervised_latent_log_prob,
            batch_supervised_reconstruction_log_prob,
        ]
    )

    return (
        batch_loss,
        (
            aux_values,
            output_state,
        ),
    )
