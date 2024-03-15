import pandas as pd
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from jax.typing import ArrayLike
from collections.abc import Callable

from .postprocessing import post_process_batch, resample
from .dataloader import DataLoader
from .datasets import SpectroPhotometricDataset


def create_input_arrays(
    input_df: pd.DataFrame,
    psf_columns: list,
    psf_err_columns: list,
    model_columns: list,
    model_err_columns: list,
    objid_column: list,
    additional_columns: list = None,
    z_column: list = None,
    n_splits: int = 1,
    shuffle: bool = True,
    default_value: float = -9999.0,
):

    if shuffle:
        input_df = input_df.sample(frac=1).reset_index(drop=True)

    psf_photometry = jnp.asarray(input_df[psf_columns].values)
    psf_photometry_err = jnp.asarray(input_df[psf_err_columns].values)
    model_photometry = jnp.asarray(input_df[model_columns].values)
    model_photometry_err = jnp.asarray(input_df[model_err_columns].values)
    objid = jnp.asarray(input_df[objid_column].values, dtype=jnp.int64)

    if additional_columns is not None:
        additional_info = jnp.asarray(input_df[additional_columns].values)
    else:
        additional_info = jnp.zeros((psf_photometry.shape[0], 0))

    if z_column is not None:
        z = jnp.asarray(input_df[z_column].values)
    else:
        z = jnp.ones((psf_photometry.shape[0], 1)) * default_value

    if n_splits > 1:
        psf_photometry = psf_photometry.reshape(n_splits, -1, psf_photometry.shape[-1])
        psf_photometry_err = psf_photometry_err.reshape(
            n_splits, -1, psf_photometry_err.shape[-1]
        )
        model_photometry = model_photometry.reshape(
            n_splits, -1, model_photometry.shape[-1]
        )
        model_photometry_err = model_photometry_err.reshape(
            n_splits, -1, model_photometry_err.shape[-1]
        )
        additional_info = additional_info.reshape(
            n_splits, -1, additional_info.shape[-1]
        )
        z = z.reshape(n_splits, -1, z.shape[-1])
        objid = objid.reshape(n_splits, -1, objid.shape[-1])

    return (
        psf_photometry,
        psf_photometry_err,
        model_photometry,
        model_photometry_err,
        additional_info,
        z,
        objid,
    )


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
        y = jnp.squeeze(y)

        return (
            x,
            y,
            photometric_dataloader_state,
            spectroscopic_dataloader_state,
            photometric_reset_condition,
            spectroscopic_reset_condition,
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
