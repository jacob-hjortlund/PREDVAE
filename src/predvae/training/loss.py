import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from jax import vmap
from jax.lax import cond
from equinox import Module
from jax.typing import ArrayLike
from collections.abc import Callable


def _sample_loss(
    model: Module,
    input_state: eqx.nn.State,
    x: ArrayLike,
    y: ArrayLike,
    rng_key: ArrayLike,
    missing_target_value: ArrayLike = -9999.0,
    target_transform: Callable = lambda x: x,
):

    y = target_transform(y)
    (dynamic_unsupervised_call, dynamic_unsupervised_state), (
        static_unsupervised_call,
        static_unsupervised_state,
    ) = eqx.partition(
        model.unsupervised_call(
            x,
            y,
            input_state,
            rng_key,
        ),
        eqx.is_array,
    )
    unsupervised_state = eqx.combine(
        dynamic_unsupervised_state, static_unsupervised_state
    )

    (dynamic_supervised_call, dynamic_supervised_state), (
        static_supervised_call,
        static_supervised_state,
    ) = eqx.partition(
        model.supervised_call(
            x,
            y,
            unsupervised_state,
            rng_key,
        ),
        eqx.is_array,
    )
    supervised_state = eqx.combine(dynamic_supervised_state, static_supervised_state)

    static_call = eqx.error_if(
        static_unsupervised_call,
        static_unsupervised_call != static_supervised_call,
        "Filtered conditional loss functions should have the same static component",
    )

    dynamic_call = cond(
        y != missing_target_value,
        lambda *_: dynamic_supervised_call,
        lambda *_: dynamic_unsupervised_call,
    )

    call_components = eqx.combine(dynamic_call, static_call)
    y, z, x_hat, y_pars, z_pars, x_pars = call_components

    target_log_prior = model.target_prior(y)
    target_log_prob = model.predictor.log_prob(y, *y_pars)

    latent_log_prior = model.latent_prior(z)
    latent_log_prob = model.encoder.log_prob(z, *z_pars)

    reconstruction_log_prob = model.decoder.log_prob(x, *x_pars)

    loss_components = jnp.array(
        [
            target_log_prob,
            target_log_prior,
            latent_log_prior,
            latent_log_prob,
            reconstruction_log_prob,
        ]
    )

    return loss_components, supervised_state


def _loss(
    model: Module,
    input_state: eqx.nn.State,
    x: ArrayLike,
    y: ArrayLike,
    rng_key: ArrayLike,
    n_samples: ArrayLike = 1,
    missing_target_value: ArrayLike = -9999.0,
    target_transform: Callable = lambda x: x,
):

    _vmapped_sample_loss = eqx.filter_vmap(
        _sample_loss,
        in_axes=(None, None, None, None, eqx.if_array(0), None, None),
        out_axes=(eqx.if_array(0), None)
    )

    rng_keys = jr.split(rng_key, n_samples)
    rng_keys = jnp.atleast_2d(rng_keys)
    loss_components, output_state = _vmapped_sample_loss(
        model,
        input_state,
        x,
        y,
        rng_keys,
        missing_target_value,
        target_transform,
    )

    loss_components = jnp.mean(loss_components, axis=0)

    return loss_components, output_state


def ssvae_loss(
    free_params: Module,
    frozen_params: Module,
    input_state: eqx.nn.State,
    x: ArrayLike,
    y: ArrayLike,
    rng_key: ArrayLike,
    alpha: ArrayLike,
    beta: int = 1.0,
    vae_factor: int = 1.0,
    predictor_factor: int = 1.0,
    n_samples: int = 1,
    missing_target_value: ArrayLike = -9999.0,
    target_transform: Callable = lambda x: x,
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

    vmapped_sample_loss = eqx.filter_vmap(
        _loss,
        in_axes=(None, None, eqx.if_array(0), eqx.if_array(0), None, None, None, None),
        out_axes=(eqx.if_array(0), None),
    )
    loss_components, output_state = vmapped_sample_loss(
        model,
        input_state,
        x,
        y,
        rng_key,
        n_samples,
        missing_target_value,
        target_transform,
    )

    batch_size = x.shape[0]
    idx_missing = y.squeeze() == missing_target_value
    idx_not_missing = y.squeeze() != missing_target_value

    (
        target_log_prob,
        target_log_prior,
        latent_log_prior,
        latent_log_prob,
        reconstruction_log_prob,
    ) = loss_components.T

    vae_loss = vae_factor * (
        beta * (latent_log_prob - latent_log_prior) - reconstruction_log_prob
    )

    unsupervised_losses = vae_loss + predictor_factor * (
        target_log_prob - target_log_prior
    )
    supervised_losses = vae_loss - predictor_factor * target_log_prior

    batch_unsupervised_loss = (
        jnp.sum(unsupervised_losses, where=idx_missing) / batch_size
    )
    batch_unsupervised_target_log_prob = (
        jnp.sum(target_log_prior, where=idx_missing) / batch_size
    )
    batch_unsupervised_target_log_prior = (
        jnp.sum(target_log_prior, where=idx_missing) / batch_size
    )
    batch_unsupervised_latent_log_prior = (
        jnp.sum(latent_log_prior, where=idx_missing) / batch_size
    )
    batch_unsupervised_latent_log_prob = (
        jnp.sum(latent_log_prob, where=idx_missing) / batch_size
    )
    batch_unsupervised_reconstruction_log_prob = (
        jnp.sum(reconstruction_log_prob, where=idx_missing) / batch_size
    )

    batch_supervised_loss = (
        jnp.sum(supervised_losses, where=idx_not_missing) / batch_size
    )
    batch_supervised_target_log_prob_loss = (
        -alpha * jnp.mean(target_log_prob, where=idx_not_missing) / batch_size
    )
    batch_supervised_target_log_prob = (
        jnp.sum(target_log_prob, where=idx_not_missing) / batch_size
    )
    batch_supervised_target_log_prior = (
        jnp.sum(target_log_prior, where=idx_not_missing) / batch_size
    )
    batch_supervised_latent_log_prior = (
        jnp.sum(latent_log_prior, where=idx_not_missing) / batch_size
    )
    batch_supervised_latent_log_prob = (
        jnp.sum(latent_log_prob, where=idx_not_missing) / batch_size
    )
    batch_supervised_reconstruction_log_prob = (
        jnp.sum(reconstruction_log_prob, where=idx_not_missing) / batch_size
    )

    sum_array = jnp.asarray(
        [
            batch_unsupervised_loss,
            batch_supervised_loss,
            predictor_factor * batch_supervised_target_log_prob_loss,
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
