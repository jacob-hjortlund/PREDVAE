import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from jax.typing import ArrayLike
from collections.abc import Callable

from .postprocessing import post_process_batch, resample
from .dataloader import DataLoader
from .datasets import SpectroPhotometricDataset


def make_vectorized_dataloader(
    dataset: SpectroPhotometricDataset,
    batch_size: int,
    rng_key: ArrayLike,
    shuffle: bool = False,
    drop_last: bool = False,
):

    dataloader, state = eqx.filter_vmap(
        eqx.nn.make_with_state(DataLoader),
        in_axes=(eqx.if_array(0), None, None, None, eqx.if_array(0)),
    )(dataset, batch_size, shuffle, drop_last, rng_key)

    return dataloader, state


def dataset_iterator(
    dataloader: eqx.Module,
    dataset_statistics: eqx.Module,
    resample_fn: Callable,
    dataloader_state: eqx.Module,
    rng_key: ArrayLike,
):

    resampling_key, rng_key = jr.split(rng_key, 2)
    batch, dataloader_state, reset_condition = dataloader(dataloader_state)

    x, y = post_process_batch(batch, dataset_statistics, resample_fn, resampling_key)

    return x, y, dataloader_state, reset_condition


def make_dataset_iterator(
    resample_photometry: bool = False,
    vectorize: bool = False,
):

    if resample_photometry:
        resample_fn = resample
    else:
        resample_fn = lambda x, y, z: x

    iterator = lambda dataloader, dataset_statistics, dataloader_state, rng_key: dataset_iterator(
        dataloader, dataset_statistics, resample_fn, dataloader_state, rng_key
    )

    if vectorize:
        iterator = eqx.filter_vmap(iterator)

    return iterator


def make_spectrophotometric_iterator(
    photometric_dataloader: eqx.Module,
    spectroscopic_dataloader: eqx.Module,
    dataset_statistics: eqx.Module,
    resample_photometry: bool = False,
    vectorize: bool = False,
):

    photometric_iterator = make_dataset_iterator(
        resample_photometry=resample_photometry,
        vectorize=False,
    )

    spectroscopic_iterator = make_dataset_iterator(
        resample_photometry=resample_photometry,
        vectorize=False,
    )

    def _photometric_spectroscopic_iterator(
        photometric_dataloader: eqx.Module,
        spectroscopic_dataloader: eqx.Module,
        dataset_statistics: eqx.Module,
        photometric_dataloader_state: eqx.Module,
        spectroscopic_dataloader_state: eqx.Module,
        rng_key: ArrayLike,
    ):

        photometric_key, spectroscopic_key, rng_key = jr.split(rng_key, 3)

        (
            x_photometric,
            y_photometric,
            photometric_dataloader_state,
            photometric_reset_condition,
        ) = photometric_iterator(
            photometric_dataloader,
            dataset_statistics,
            photometric_dataloader_state,
            photometric_key,
        )

        (
            x_spectroscopic,
            y_spectroscopic,
            spectroscopic_dataloader_state,
            spectroscopic_reset_condition,
        ) = spectroscopic_iterator(
            spectroscopic_dataloader,
            dataset_statistics,
            spectroscopic_dataloader_state,
            spectroscopic_key,
        )

        x = jnp.concatenate([x_photometric, x_spectroscopic], axis=0)
        y = jnp.concatenate([y_photometric, y_spectroscopic], axis=0)
        reset_condition = jnp.logical_and(
            photometric_reset_condition, spectroscopic_reset_condition
        )

        return (
            x,
            y,
            photometric_dataloader_state,
            spectroscopic_dataloader_state,
            reset_condition,
        )

    if vectorize:
        _photometric_spectroscopic_iterator = eqx.filter_vmap(
            _photometric_spectroscopic_iterator
        )

    def photometric_spectroscopic_iterator(
        photometric_dataloader_state: eqx.Module,
        spectroscopic_dataloader_state: eqx.Module,
        rng_key: ArrayLike,
    ):

        return _photometric_spectroscopic_iterator(
            photometric_dataloader,
            spectroscopic_dataloader,
            dataset_statistics,
            photometric_dataloader_state,
            spectroscopic_dataloader_state,
            rng_key,
        )

    return photometric_spectroscopic_iterator
