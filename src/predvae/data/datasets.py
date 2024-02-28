import os
import h5py
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, Sampler
from sklearn.model_selection import StratifiedKFold


class StratifiedBatchSampler(Sampler):

    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_batches = len(y) // batch_size
        self._X = np.random.randn(len(y))
        self.skf = StratifiedKFold(n_splits=self.n_batches, shuffle=shuffle)

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = np.random.randint(0, 1e8)
        for _, test_index in self.skf.split(self._X, self.y):
            yield test_index

    def __len__(self):
        return self.n_batches


class HDF5Dataset(Dataset):

    def __init__(self, path: str, resample: bool = False) -> None:
        super().__init__()
        self.path = path
        self.resample = resample

        with h5py.File(path, "r") as hdf5:
            self._read_statistics(hdf5)
            self.len = hdf5["psf_mags"].shape[0]

    def __len__(self):
        return self.len

    def _normalize(self, arr, mean, std):
        return (arr - mean) / std

    def _resample(self, photometry: np.ndarray, uncertainty: np.ndarray):
        return photometry + np.random.standard_normal(photometry.shape) * uncertainty

    def _open_hdf5(self):
        self.hdf5 = h5py.File(self.path, "r")

    def _read_statistics(self, hdf5):
        self.psf_mag_mean = hdf5["psf_mag_mean"][:]
        self.psf_mag_std = hdf5["psf_mag_std"][:]
        self.psf_color_mean = hdf5["psf_color_mean"][:]
        self.psf_color_std = hdf5["psf_color_std"][:]
        self.model_mag_mean = hdf5["model_mag_mean"][:]
        self.model_mag_std = hdf5["model_mag_std"][:]
        self.model_color_mean = hdf5["model_color_mean"][:]
        self.model_color_std = hdf5["model_color_std"][:]
        self.additional_mean = np.array(hdf5["additional_features_mean"])
        self.additional_std = np.array(hdf5["additional_features_std"])
        self.log10_redshift_mean = np.nanmean(np.array(hdf5["log10_redshift"]))
        self.log10_redshift_std = np.nanmean(np.array(hdf5["log10_redshift"]))

    def _calculate_colors(self, photometry):
        return photometry[..., 1:] - photometry[..., :-1]

    def get_one_hot_redshifts(self):
        with h5py.File(self.path, "r") as hdf5:
            return np.isfinite(np.array(hdf5["log10_redshift"])).astype(int)

    def __getitem__(self, index):
        if not hasattr(self, "hdf5"):
            self._open_hdf5()

        if torch.is_tensor(index):
            index = index.tolist()

        sample_psf_mag = self.hdf5["psf_mags"][index]
        if self.resample:
            sample_psf_uncertainty = self.hdf5["psf_uncertainties"][index]
            sample_psf_mag = self._resample(sample_psf_mag, sample_psf_uncertainty)
        sample_psf_colors = self._normalize(
            self._calculate_colors(sample_psf_mag),
            self.psf_color_mean,
            self.psf_color_std,
        )
        sample_psf_mag = self._normalize(
            sample_psf_mag, self.psf_mag_mean, self.psf_mag_std
        )

        sample_model_mag = self.hdf5["model_mags"][index]
        sample_model_uncertainty = self.hdf5["model_uncertainties"][index]
        if self.resample:
            sample_model_mag = self._resample(
                sample_model_mag, sample_model_uncertainty
            )
        sample_model_colors = self._normalize(
            self._calculate_colors(sample_model_mag),
            self.model_color_mean,
            self.model_color_std,
        )
        sample_model_mag = self._normalize(
            sample_model_mag, self.model_mag_mean, self.model_mag_std
        )

        sample_objid = np.asarray(self.hdf5["objid"][index])
        sample_remaining_features = np.atleast_1d(
            self._normalize(
                self.hdf5["additional_features"][index],
                self.additional_mean,
                self.additional_std,
            )
        )
        sample_log10redshift = np.asarray(
            self._normalize(
                self.hdf5["log10_redshift"][index],
                self.log10_redshift_mean,
                self.log10_redshift_std,
            )
        )

        sample_inputs = np.column_stack(
            [
                sample_psf_mag,
                sample_model_mag,
                sample_psf_colors,
                sample_model_colors,
                sample_remaining_features,
            ],
        )

        return (
            sample_inputs.squeeze(),
            sample_log10redshift.squeeze(),
            sample_objid.squeeze(),
        )


class CSVDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        psf_mag_columns: list,
        psf_mag_uncertainty_columns: list,
        model_mag_columns: list,
        model_mag_uncertainty_columns: list,
        redshift_column: str,
        additional_columns: list = [],
        info_columns: list = [],
        resample: bool = False,
    ):

        self.resample = resample

        csv_file = pd.read_csv(csv_file)
        self.psf_mag = self._extract_array(csv_file, psf_mag_columns)
        self.psf_uncertainty = self._extract_array(
            csv_file, psf_mag_uncertainty_columns
        )
        self.model_mag = self._extract_array(csv_file, model_mag_columns)
        self.model_uncertainty = self._extract_array(
            csv_file, model_mag_uncertainty_columns
        )
        self.additional = self._extract_array(csv_file, additional_columns)
        self.info = self._extract_array(csv_file, info_columns)
        self.redshift = self._extract_array(csv_file, [redshift_column])

        self.psf_mag_mean = np.mean(self.psf_mag, axis=0)
        self.psf_mag_std = np.std(self.psf_mag, axis=0)
        self.model_mag_mean = np.mean(self.model_mag, axis=0)
        self.model_mag_std = np.std(self.model_mag, axis=0)
        self.psf_color_mean = np.mean(self._vectorized_colors(self.psf_mag), axis=0)
        self.psf_color_std = np.std(self._vectorized_colors(self.psf_mag), axis=0)
        self.model_color_mean = np.mean(self._vectorized_colors(self.model_mag), axis=0)
        self.model_color_std = np.std(self._vectorized_colors(self.model_mag), axis=0)
        self.additional_mean = np.mean(self.additional, axis=0)
        self.additional_std = np.std(self.additional, axis=0)
        self.redshift_mean = np.nanmean(self.redshift)
        self.redshift_std = np.nanstd(self.redshift)

    def __len__(self):
        return len(self.psf_mag)

    def _extract_array(self, df: pd.DataFrame, columns: list):
        if columns is not None:
            arr = df[columns].values
        else:
            arr = None

        return arr

    def _normalize(self, arr, mean, std):
        return (arr - mean) / std

    def _resample(self, photometry: np.ndarray, uncertainty: np.ndarray):
        return photometry + np.random.standard_normal(photometry.shape) * uncertainty

    def _vectorized_colors(self, photometry):
        return photometry[:, 1:] - photometry[:, :-1]

    def _calculate_colors(self, photometry):
        return photometry[1:] - photometry[:-1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_psf_mag = self.psf_mag[idx]
        sample_psf_uncertainty = self.psf_uncertainty[idx]
        sample_model_mag = self.model_mag[idx]
        sample_model_uncertainty = self.model_uncertainty[idx]
        sample_redshift = self._normalize(
            self.redshift[idx], self.redshift_mean, self.redshift_std
        )
        sample_info = self.info[idx]

        if self.resample:
            sample_psf_mag = self._resample(sample_psf_mag, sample_psf_uncertainty)
            sample_model_mag = self._resample(
                sample_model_mag, sample_model_uncertainty
            )
        sample_psf_mag = self._normalize(
            sample_psf_mag, self.psf_mag_mean, self.psf_mag_std
        )
        sample_model_mag = self._normalize(
            sample_model_mag, self.model_mag_mean, self.model_mag_std
        )

        sample_psf_colors = self._calculate_colors(sample_psf_mag)
        sample_psf_colors = self._normalize(
            sample_psf_colors, self.psf_color_mean, self.psf_color_std
        )
        sample_model_colors = self._calculate_colors(sample_model_mag)
        sample_model_colors = self._normalize(
            sample_model_colors, self.model_color_mean, self.model_color_std
        )
        sample_inputs = np.concatenate(
            [sample_psf_mag, sample_model_mag, sample_psf_colors, sample_model_colors],
        )
        if self.additional is not None:
            sample_additional = self._normalize(
                self.additional[idx], self.additional_mean, self.additional_std
            )
            sample_inputs = np.concatenate([sample_inputs, sample_additional])

        return sample_inputs.squeeze(), sample_redshift.squeeze(), sample_info.squeeze()
