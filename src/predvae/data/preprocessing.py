import torch
import numpy as np
import jax.numpy as jnp
import jax.random as jr

from typing import Union
from jaxtyping import Array
from torch.utils.data import sampler, DataLoader, Dataset


def convert_to_semisupervised(
    dataset: Dataset,
    unsupervised_fraction: float,
    unsupervised_target: Union[int, float] = -1,
) -> Dataset:
    """
    Convert a supervised dataset to a semi-supervised dataset by setting
    a fraction of the targets to a given value.

    Args:
        dataset (Dataset): The dataset to convert
        unsupervised_fraction (float): The fraction of the dataset to set to unsupervised
        unsupervised_target (Union[int, float], optional): The target to set the unsupervised samples to. Defaults to -1.

    Returns:
        Dataset: The semi-supervised dataset
    """

    num_samples = len(dataset)
    num_unsupervised = int(unsupervised_fraction * num_samples)
    unsupervised_indices = np.random.choice(
        num_samples, num_unsupervised, replace=False
    )

    dataset.targets[unsupervised_indices] = unsupervised_target

    return dataset
