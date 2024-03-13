import h5py
import torch

import numpy as np
import pandas as pd
import equinox as eqx
import jax.numpy as jnp

from jaxtyping import Array
from torch.utils.data import Dataset, Sampler
from sklearn.model_selection import StratifiedKFold


class PhotometryDataset(eqx.Module):

    psf_photometry: Array
    psf_uncertainties: Array
    model_photometry: Array
    model_uncertainties: Array
    additional_features: Array
    log10_redshift: Array
    objid: Array

    def __init__(
        self,
        psf_photometry: Array,
        model_photometry: Array,
        psf_uncertainties: Array = None,
        model_uncertainties: Array = None,
        additional_features: Array = None,
        log10_redshift: Array = None,
        objid: Array = None,
    ):

        self.psf_photometry = psf_photometry
        self.model_photometry = model_photometry
        self.psf_uncertainties = (
            psf_uncertainties
            if psf_uncertainties is not None
            else jnp.zeros_like(psf_photometry)
        )
        self.model_uncertainties = (
            model_uncertainties
            if model_uncertainties is not None
            else jnp.zeros_like(model_photometry)
        )
        self.additional_features = (
            additional_features
            if additional_features is not None
            else jnp.zeros((len(psf_photometry), 0))
        )

        self.log10_redshift = (
            log10_redshift
            if log10_redshift is not None
            else jnp.zeros(len(psf_photometry, 1))
        )
        self.objid = objid if objid is not None else jnp.zeros(len(psf_photometry))

    def __len__(self) -> int:
        return len(self.psf_photometry)

    def __getitem__(self, idx: int) -> Array:
        return self(idx)

    def __call__(self, idx: int) -> Array:

        psf_photometry = self.psf_photometry[idx]
        psf_uncertainties = self.psf_uncertainties[idx]

        model_photometry = self.model_photometry[idx]
        model_uncertainties = self.model_uncertainties[idx]

        additional_features = self.additional_features[idx]
        log10_redshift = self.log10_redshift[idx]
        objid = self.objid[idx]

        return (
            psf_photometry,
            psf_uncertainties,
            model_photometry,
            model_uncertainties,
            additional_features,
            log10_redshift,
            objid,
        )


class PhotometryStatistics(eqx.Module):

    psf_photometry_mean: Array
    psf_photometry_std: Array
    psf_colors_mean: Array
    psf_colors_std: Array
    model_photometry_mean: Array
    model_photometry_std: Array
    model_colors_mean: Array
    model_colors_std: Array
    additional_features_mean: Array
    additional_features_std: Array
    log10_redshift_mean: Array
    log10_redshift_std: Array

    def __init__(
        self,
        photometry_dataset: PhotometryDataset,
    ):
        psf_colors = self._calculate_colors(photometry_dataset.psf_photometry)
        self.psf_photometry_mean = jnp.mean(photometry_dataset.psf_photometry, axis=0)
        self.psf_photometry_std = jnp.std(photometry_dataset.psf_photometry, axis=0)
        self.psf_colors_mean = jnp.mean(psf_colors, axis=0)
        self.psf_colors_std = jnp.std(psf_colors, axis=0)

        model_colors = self._calculate_colors(photometry_dataset.model_photometry)
        self.model_photometry_mean = jnp.mean(
            photometry_dataset.model_photometry, axis=0
        )
        self.model_photometry_std = jnp.std(photometry_dataset.model_photometry, axis=0)
        self.model_colors_mean = jnp.mean(model_colors, axis=0)
        self.model_colors_std = jnp.std(model_colors, axis=0)

        self.additional_features_mean = jnp.mean(
            photometry_dataset.additional_features, axis=0
        )
        self.additional_features_std = jnp.std(
            photometry_dataset.additional_features, axis=0
        )

        self.log10_redshift_mean = jnp.mean(photometry_dataset.log10_redshift, axis=0)
        self.log10_redshift_std = jnp.std(photometry_dataset.log10_redshift, axis=0)

    def _calculate_colors(self, photometry: Array) -> Array:
        return photometry[..., 1:] - photometry[..., :-1]
