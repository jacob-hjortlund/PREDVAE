import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from tqdm import tqdm
from equinox import Module
from jax import Array, vmap
from jax.typing import ArrayLike
from jax.random import PRNGKeyArray
from predvae.training import gaussian_kl_divergence, gaussian_log_likelihood


def gaussian_vae_loss(
    model: Module,
    x: ArrayLike,
    rng_key: PRNGKeyArray,
) -> Module:
    """
    Batch loss function for a gaussian VAE.

    Args:
        model (Module): VAE model
        x (ArrayLike): Data batch
        rng_key (PRNGKeyArray): RNG key with leading dimension equal to the batch size

    Returns:
        Module: Batch loss
    """

    @eqx.filter_jit
    def _sample_loss(
        model: Module,
        x: ArrayLike,
        rng_key: PRNGKeyArray,
    ):
        encoder_key, decoder_key = jr.split(rng_key)
        z, z_mu, z_log_sigma = model.encode(x, encoder_key)
        x_hat, x_mu, x_log_sigma = model.decode(z, decoder_key)

        kl_divergence = gaussian_kl_divergence(z_mu, z_log_sigma)
        reproduction_loss = gaussian_log_likelihood(x, x_mu, x_log_sigma)
        loss = -(reproduction_loss - kl_divergence)

        return loss

    loss = vmap(_sample_loss, in_axes=(None, 0, None))(model, x, rng_key)
    batch_loss = jnp.mean(loss)

    return batch_loss


def ssvae_classifier_loss(
    model: Module,
    x: ArrayLike,
    y: ArrayLike,
    rng_key: PRNGKeyArray,
) -> Module:
    """
    Batch loss function for a semi-supervised VAE classifier.

    Args:
        model (Module): SSVAE model
        x (ArrayLike): Data batch
        y (ArrayLike): Target batch
        rng_key (PRNGKeyArray): RNG key with leading dimension equal to the batch size

    Returns:
        Module: Batch loss
    """

    @eqx.filter_jit
    def _sample_loss(
        model: Module,
        x: ArrayLike,
        y: ArrayLike,
        rng_key: PRNGKeyArray,
    ):
        pass

    loss = vmap(_sample_loss, in_axes=(None, 0, 0, None))(model, x, y, rng_key)
    batch_loss = jnp.mean(loss)

    return batch_loss
