import jax.numpy as jnp
import jax.random as jr

from jax.typing import ArrayLike
from collections.abc import Callable

from .datasets import PhotometryStatistics


def calculate_colors(photometry: ArrayLike) -> ArrayLike:
    """
    Calculate the colors of the photometry.

    Args:
        photometry (ArrayLike): The photometry

    Returns:
        ArrayLike: The colors
    """

    colors = photometry[..., 1:] - photometry[..., :-1]

    return colors


def normalize(photometry: ArrayLike, mean: ArrayLike, std: ArrayLike) -> ArrayLike:
    """
    Normalize the photometry.

    Args:
        photometry (ArrayLike): The photometry
        mean (ArrayLike): The mean of the photometry
        std (ArrayLike): The standard deviation of the photometry

    Returns:
        ArrayLike: The normalized photometry
    """

    return (photometry - mean) / std


def resample(
    photometry: ArrayLike, uncertainty: ArrayLike, rng_key: ArrayLike
) -> ArrayLike:

    return photometry + jr.normal(rng_key, photometry.shape) * uncertainty


def post_process_batch(
    batch: tuple,
    dataset_statistics: PhotometryStatistics,
    resample_fn: Callable,
    rng_key: ArrayLike,
):

    (
        psf_photomotetry,
        psf_uncertainties,
        model_photometry,
        model_uncertainties,
        additional_features,
        log10_redshifts,
        _,
    ) = batch

    psf_photomotetry_key, model_photometry_key, rng_key = jr.split(rng_key, 3)

    psf_photomotetry = resample_fn(
        psf_photomotetry, psf_uncertainties, psf_photomotetry_key
    )
    psf_colors = calculate_colors(psf_photomotetry)
    psf_photomotetry = normalize(
        psf_photomotetry,
        dataset_statistics.psf_photometry_mean,
        dataset_statistics.psf_photometry_std,
    )
    psf_colors = normalize(
        psf_colors,
        dataset_statistics.psf_colors_mean,
        dataset_statistics.psf_colors_std,
    )

    model_photometry = resample_fn(
        model_photometry, model_uncertainties, model_photometry_key
    )
    model_colors = calculate_colors(model_photometry)
    model_photometry = normalize(
        model_photometry,
        dataset_statistics.model_photometry_mean,
        dataset_statistics.model_photometry_std,
    )
    model_colors = normalize(
        model_colors,
        dataset_statistics.model_colors_mean,
        dataset_statistics.model_colors_std,
    )

    additional_features = normalize(
        additional_features,
        dataset_statistics.additional_features_mean,
        dataset_statistics.additional_features_std,
    )

    log10_redshifts = normalize(
        log10_redshifts,
        dataset_statistics.log10_redshift_mean,
        dataset_statistics.log10_redshift_std,
    )

    x = jnp.column_stack(
        [
            psf_photomotetry,
            psf_colors,
            model_photometry,
            model_colors,
            additional_features,
        ]
    )

    return x, log10_redshifts
