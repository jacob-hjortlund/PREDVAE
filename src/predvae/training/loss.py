import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from jax import vmap
from jax.lax import cond
from equinox import Module
from jax.typing import ArrayLike
from jax.random import PRNGKeyArray
from jax.scipy import stats as jstats


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
    rng_key: PRNGKeyArray,
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
        rng_key: PRNGKeyArray,
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
    batch_loss = jnp.mean(loss)

    return batch_loss, jnp.array([jnp.nan])


def ssvae_loss(
    free_params: Module,
    frozen_params: Module,
    x: ArrayLike,
    y: ArrayLike,
    rng_key: PRNGKeyArray,
    alpha: ArrayLike,
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

    def _supervised_sample_loss(
        model: Module,
        x: ArrayLike,
        y: ArrayLike,
        rng_key: PRNGKeyArray,
    ):
        encoder_key, decoder_key = jr.split(rng_key, 2)
        _, y_pars = model.predictor(x, encoder_key)
        z, z_pars = model.encoder(x, y, encoder_key)
        x_hat, x_pars = model.decoder(z, y, decoder_key)

        target_log_prior = model.target_prior(y)
        target_log_prob = model.predictor.log_prob(y, *y_pars)

        latent_log_prior = model.latent_prior(z)
        latent_log_prob = model.encoder.log_prob(z, *z_pars)

        reconstruction_log_prob = model.decoder.log_prob(x, x_hat, *x_pars)

        supervised_loss = (
            latent_log_prob
            - latent_log_prior
            - target_log_prior
            - reconstruction_log_prob
        )

        loss = jnp.array([jnp.nan, supervised_loss, target_log_prob])

        return loss

    def _unsupervised_sample_loss(
        model: Module,
        x: ArrayLike,
        y: ArrayLike,
        rng_key: PRNGKeyArray,
    ):
        encoder_key, predictor_key, decoder_key = jr.split(rng_key, 3)
        y, y_pars = model.predictor(x, predictor_key)
        z, z_pars = model.encoder(x, y, encoder_key)
        x_hat, x_pars = model.decoder(z, y, decoder_key)

        target_log_prior = model.target_prior(y)
        target_log_prob = model.predictor.log_prob(y, *y_pars)

        latent_log_prior = model.latent_prior(z)
        latent_log_prob = model.encoder.log_prob(z, *z_pars)

        reconstruction_log_prob = model.decoder.log_prob(x, x_hat, *x_pars)

        unsupervised_loss = (
            latent_log_prob
            + target_log_prob
            - latent_log_prior
            - target_log_prior
            - reconstruction_log_prob
        )

        loss = jnp.array([unsupervised_loss, jnp.nan, jnp.nan])

        return loss

    def _sample_loss(
        model: Module,
        x: ArrayLike,
        y: ArrayLike,
        rng_key: PRNGKeyArray,
    ):
        loss_components = cond(
            y == -1,
            _unsupervised_sample_loss,
            _supervised_sample_loss,
            model,
            x,
            y,
            rng_key,
        )

        return loss_components

    model = eqx.combine(free_params, frozen_params)
    loss_components = vmap(_sample_loss, in_axes=(None, 0, 0, None))(
        model, x, y, rng_key
    )
    batch_unsupervised_loss = jnp.nanmean(loss_components[:, 0])
    batch_supervised_loss = jnp.nanmean(loss_components[:, 1])
    batch_target_loss = -alpha * jnp.nanmean(loss_components[:, 2])

    batch_loss = batch_unsupervised_loss + batch_supervised_loss + batch_target_loss

    return batch_loss, jnp.array(
        [
            batch_unsupervised_loss,
            batch_supervised_loss,
            batch_target_loss,
        ]
    )
