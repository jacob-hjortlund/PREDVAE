import os
import jax

jax.config.update("jax_enable_x64", True)
import optax
import torch
import torchvision

import numpy as np
import pandas as pd
import equinox as eqx
import jax.numpy as jnp
import src.predvae.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from jax.tree_util import tree_map
from src.predvae.training import train, ssvae_loss
from src.predvae.data import HDF5Dataset, StratifiedBatchSampler

INPUT_SIZE = 27
LATENT_SIZE = 2
PREDICTOR_SIZE = 1
TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 8192
LEARNING_RATE = 3e-4
EPOCHS = 1000
TEST_EPOCHS = 1
PRINT_EVERY = 100
SEED = 5678
MISSING_TARGET_VALUE = -9999.0
DATA_DIR = Path("/home/jacob/Uni/Msc/VAEPhotoZ/Data/SS_Splits")
SPLIT = 0
NUM_WORKERS = 0

rng_key = jax.random.PRNGKey(SEED)

train_dataset = HDF5Dataset(
    path=DATA_DIR / f"train_{SPLIT}.hdf5",
    resample=True,
)
train_one_hot_redshifts = train_dataset.get_one_hot_redshifts()

val_dataset = HDF5Dataset(
    path=DATA_DIR / f"val_{SPLIT}.hdf5",
    resample=False,
)
val_one_hot_redshifts = val_dataset.get_one_hot_redshifts()


def collate_fn(batch):
    batch = list(*batch)
    batch = (torch.Tensor(b) for b in batch)
    return (*batch,)


trainloader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=StratifiedBatchSampler(
        train_one_hot_redshifts,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    ),
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
)

valloader = torch.utils.data.DataLoader(
    val_dataset,
    sampler=StratifiedBatchSampler(
        val_one_hot_redshifts,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
    ),
    collate_fn=collate_fn,
)

# Define alpha

z = train_one_hot_redshifts
N_UNSUP = np.count_nonzero(~z)
N_SUP = np.count_nonzero(z)
ALPHA = N_UNSUP / N_SUP
print(f"Unsupervised: {N_UNSUP}, Supervised: {N_SUP}, alpha: {ALPHA}")

# Define the model

predictor_key, encoder_key, decoder_key, rng_key = jax.random.split(rng_key, 4)

predictor = nn.GaussianCoder(
    input_size=INPUT_SIZE,
    output_size=PREDICTOR_SIZE,
    depth=2,
    width=[1024, 512],
    activation=jax.nn.tanh,
    key=predictor_key,
)

encoder = nn.GaussianCoder(
    input_size=INPUT_SIZE + PREDICTOR_SIZE,
    output_size=LATENT_SIZE,
    depth=2,
    width=[1024, 512],
    activation=jax.nn.tanh,
    key=encoder_key,
)

decoder = nn.GaussianCoder(
    input_size=LATENT_SIZE + PREDICTOR_SIZE,
    output_size=INPUT_SIZE,
    depth=2,
    width=[1024, 512],
    activation=jax.nn.tanh,
    key=decoder_key,
)

latent_prior = nn.Gaussian(
    mu=jnp.zeros(LATENT_SIZE),
    log_sigma=jnp.zeros(LATENT_SIZE),
)

target_prior = nn.Gaussian(
    mu=jnp.zeros(PREDICTOR_SIZE),
    log_sigma=jnp.zeros(PREDICTOR_SIZE),
)


ssvae = nn.SSVAE(
    predictor=predictor,
    encoder=encoder,
    decoder=decoder,
    latent_prior=latent_prior,
    target_prior=target_prior,
)

filter_spec = tree_map(lambda _: True, ssvae)
filter_spec = eqx.tree_at(
    lambda tree: (tree.latent_prior.mu, tree.latent_prior.log_sigma),
    filter_spec,
    replace=(False, False),
)

print(f"LATENT PRIOR MEAN: {ssvae.latent_prior.mu}")
print(f"LATENT PRIOR LOG_SIGMA: {ssvae.latent_prior.log_sigma}\n")

print(f"TARGET PRIOR MEAN: {ssvae.target_prior.mu}")
print(f"TARGET PRIOR LOG_SIGMA: {ssvae.target_prior.log_sigma}\n")

# Train the model

train_key, rng_key = jax.random.split(rng_key, 2)
optim = optax.adamw(LEARNING_RATE)
trained_ssvae, train_losses, test_losses, train_auxes, test_auxes = train(
    rng_key=train_key,
    model=ssvae,
    trainloader=trainloader,
    testloader=valloader,
    optim=optim,
    loss_fn=ssvae_loss,
    epochs=EPOCHS,
    test_epochs=TEST_EPOCHS,
    print_every=PRINT_EVERY,
    filter_spec=filter_spec,
    loss_kwargs={
        "alpha": ALPHA,
        "missing_target_value": MISSING_TARGET_VALUE,
    },
)

print(f"LATENT PRIOR MEAN: {trained_ssvae.latent_prior.mu}")
print(f"LATENT PRIOR LOG_SIGMA: {trained_ssvae.latent_prior.log_sigma}\n")

print(f"TARGET PRIOR MEAN: {trained_ssvae.target_prior.mu}")
print(f"TARGET PRIOR LOG_SIGMA: {trained_ssvae.target_prior.log_sigma}\n")
