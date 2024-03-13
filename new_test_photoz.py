import os
import re

# xla_flags = os.getenv("XLA_FLAGS", "")
# xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
# os.environ["XLA_FLAGS"] = " ".join(
#     ["--xla_force_host_platform_device_count={}".format(4)] + xla_flags
# )

import jax

jax.config.update("jax_enable_x64", True)
import optax
import torch

import numpy as np
import pandas as pd
import equinox as eqx
import seaborn as sns
import jax.numpy as jnp
import jax.random as jr
import src.predvae.nn as nn
import matplotlib.pyplot as plt
import src.predvae.data as data
import src.predvae.training as training

from umap import UMAP
from pathlib import Path
from functools import partial
from jax.tree_util import tree_map

# Model Config

RUN_NAME = "SSVAE_test_early_stopping"
INPUT_SIZE = 27
LATENT_SIZE = 15
PREDICTOR_SIZE = 1
USE_SPEC_NORM = True

# Training Config

SEED = 5678
EPOCHS = 2
EVAL_EVERY_N = 1
LEARNING_RATE = 3e-4
BATCH_SIZE = 1024
TRAIN_BATCHES_PER_EPOCH = 1
VAL_BATCHES_PER_EPOCH = 1

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3

REDUCE_LR_FACTOR = 0.1
REDUCE_LR_PATIENCE = 2
REDUCE_LR_ON_PLATEAU = True

# Data Config

SPLIT = 0
SHUFFLE = True
DROP_LAST = True
MISSING_TARGET_VALUE = -9999.0
SPEC_REDUCTION_FACTOR = 10
DATA_DIR = Path("/home/jacob/Uni/Msc/VAEPhotoZ/Data/Base/")
SAVE_DIR = Path(f"/home/jacob/Uni/Msc/VAEPhotoZ/PREDVAE/{RUN_NAME}")

psf_columns = [f"psfmag_{b}" for b in "ugriz"] + ["w1mag", "w2mag"]
psf_err_columns = [f"psfmagerr_{b}" for b in "ugriz"] + ["w1sigmag", "w2sigmag"]
model_columns = [f"modelmag_{b}" for b in "ugriz"] + ["w1mpro", "w2mpro"]
model_err_columns = [f"modelmagerr_{b}" for b in "ugriz"] + ["w1sigmpro", "w2sigmpro"]
additional_columns = ["extinction_i"]
z_column = ["z"]
objid_column = ["objid"]

SAVE_DIR.mkdir(exist_ok=True)

RNG_KEY = jax.random.PRNGKey(SEED)

print(
    "\n--------------------------------- LOADING DATA ---------------------------------\n"
)

spec_df = pd.read_csv(DATA_DIR / "SDSS_spec_xmatch.csv")

n_spec = spec_df.shape[0]
photo_df = pd.read_csv(DATA_DIR / "SDSS_photo_xmatch.csv", nrows=n_spec)

spec_psf_photometry = jnp.asarray(spec_df[psf_columns].values)
spec_psf_photometry_err = jnp.asarray(spec_df[psf_err_columns].values)
spec_model_photometry = jnp.asarray(spec_df[model_columns].values)
spec_model_photometry_err = jnp.asarray(spec_df[model_err_columns].values)
spec_additional_info = jnp.log10(jnp.asarray(spec_df[additional_columns].values))
spec_z = jnp.log10(jnp.asarray(spec_df[z_column].values))
spec_objid = jnp.asarray(spec_df[objid_column].values, dtype=jnp.int64)

photo_psf_photometry = jnp.asarray(photo_df[psf_columns].values)
photo_psf_photometry_err = jnp.asarray(photo_df[psf_err_columns].values)
photo_model_photometry = jnp.asarray(photo_df[model_columns].values)
photo_model_photometry_err = jnp.asarray(photo_df[model_err_columns].values)
photo_additional_info = jnp.log10(jnp.asarray(photo_df[additional_columns].values))
photo_objid = jnp.asarray(photo_df[objid_column].values, dtype=jnp.int64)

###################################################################################
#################### SPECTROPHOTOMETRIC DATASET AND STATISTICS ####################
###################################################################################

spec_dataset = data.SpectroPhotometricDataset(
    spec_psf_photometry,
    spec_psf_photometry_err,
    spec_model_photometry,
    spec_model_photometry_err,
    spec_additional_info,
    spec_z,
    spec_objid,
)

photo_dataset = data.SpectroPhotometricDataset(
    photo_psf_photometry,
    photo_psf_photometry_err,
    photo_model_photometry,
    photo_model_photometry_err,
    photo_additional_info,
    log10_redshift=None,
    objid=photo_objid,
)

dataset_statistics = data.SpectroPhotometricStatistics(
    photometric_dataset=photo_dataset, spectroscopic_dataset=spec_dataset
)

###################################################################################
################################# DATALOADERS #####################################
###################################################################################

photo_dl_key, spec_dl_key, RNG_KEY = jr.split(RNG_KEY, 3)
(
    train_photometric_dataloader,
    train_photometric_dataloader_state,
) = eqx.nn.make_with_state(data.DataLoader)(
    photo_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST,
    rng_key=photo_dl_key,
)

(
    train_spectroscopic_dataloader,
    train_spectroscopic_dataloader_state,
) = eqx.nn.make_with_state(data.DataLoader)(
    spec_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST,
    rng_key=spec_dl_key,
)

train_iterator = data.make_spectrophotometric_iterator(
    train_photometric_dataloader,
    train_spectroscopic_dataloader,
    dataset_statistics,
    resample_photometry=True,
    vectorize=False,
)

photo_dl_key, spec_dl_key, RNG_KEY = jr.split(RNG_KEY, 3)
val_photometric_dataloader, val_photometric_dataloader_state = eqx.nn.make_with_state(
    data.DataLoader
)(
    photo_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST,
    rng_key=photo_dl_key,
)

(
    val_spectroscopic_dataloader,
    val_spectroscopic_dataloader_state,
) = eqx.nn.make_with_state(data.DataLoader)(
    spec_dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST,
    rng_key=spec_dl_key,
)

eval_iterator = data.make_spectrophotometric_iterator(
    val_photometric_dataloader,
    val_spectroscopic_dataloader,
    dataset_statistics,
    resample_photometry=True,
    vectorize=False,
)

###################################################################################
################################### MODEL #########################################
###################################################################################

predictor_key, encoder_key, decoder_key, RNG_KEY = jax.random.split(RNG_KEY, 4)

predictor = nn.GaussianCoder(
    input_size=INPUT_SIZE,
    output_size=PREDICTOR_SIZE,
    depth=3,
    width=[2048, 1024, 512],
    activation=jax.nn.tanh,
    use_spectral_norm=USE_SPEC_NORM,
    key=predictor_key,
)

encoder = nn.GaussianCoder(
    input_size=INPUT_SIZE + PREDICTOR_SIZE,
    output_size=LATENT_SIZE,
    depth=3,
    width=[2048, 1024, 512],
    activation=jax.nn.tanh,
    use_spectral_norm=USE_SPEC_NORM,
    key=encoder_key,
)

decoder = nn.GaussianCoder(
    input_size=LATENT_SIZE + PREDICTOR_SIZE,
    output_size=INPUT_SIZE,
    depth=3,
    width=[2048, 1024, 512],
    activation=jax.nn.tanh,
    use_spectral_norm=USE_SPEC_NORM,
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

ssvae, input_state = eqx.nn.make_with_state(nn.SSVAE)(
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

###################################################################################
################################# TRAINING ########################################
###################################################################################

optimizer = optax.adam(LEARNING_RATE)

loss_kwargs = {"alpha": 1.0, "missing_target_value": MISSING_TARGET_VALUE}
loss_fn = partial(training.ssvae_loss, **loss_kwargs)

optimizer, optimizer_state, lr_reducer, lr_reducer_state = (
    training.initialize_optimizer(
        model=ssvae,
        optimizer=optimizer,
        reduce_lr_on_plateau=False,
        lr_reduction_kwargs={},
    )
)

train_step = training.make_train_step(
    optimizer=optimizer,
    lr_reducer=lr_reducer,
    loss_fn=loss_fn,
    filter_spec=filter_spec,
    vectorize=False,
)
eval_step = training.make_eval_step(
    loss_fn=loss_fn,
    filter_spec=filter_spec,
    vectorize=False,
)

train_step = eqx.filter_jit(train_step)
eval_step = eqx.filter_jit(eval_step)
train_iterator = eqx.filter_jit(train_iterator)
eval_iterator = eqx.filter_jit(eval_iterator)

train_batches = 0
end_of_train_split = False
end_of_val_split = False
val_loss = jnp.inf

expected_no_batches = len(photo_dataset) - len(photo_dataset) % BATCH_SIZE
print(f"Expected number of batches: {expected_no_batches} per epoch")


for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch} / {EPOCHS}\n")

    epoch_train_key, epoch_val_key, RNG_KEY = jr.split(RNG_KEY, 3)

    while not end_of_train_split:

        resampling_key, step_key, epoch_train_key = jr.split(epoch_train_key, 3)
        (
            x,
            y,
            train_photometric_dataloader_state,
            train_spectroscopic_dataloader_state,
            reset_condition,
        ) = train_iterator(
            train_photometric_dataloader_state,
            train_spectroscopic_dataloader_state,
            resampling_key,
        )

        ssvae, input_state, optimizer_state, loss_value, aux = train_step(
            x,
            y,
            step_key,
            ssvae,
            input_state,
            optimizer_state,
            lr_reducer_state,
            val_loss,
        )

        train_batches += 1
        print(
            f"Train Batch {train_batches/expected_no_batches*100:.2f} - Loss: {loss_value}"
        )

        end_of_train_split = jnp.all(reset_condition)

    print("End of Train Split")
    end_of_train_split = False

    while not end_of_val_split:

        resampling_key, step_key, epoch_val_key = jr.split(epoch_val_key, 3)
        (
            x,
            y,
            eval_photometric_dataloader_state,
            eval_spectroscopic_dataloader_state,
            reset_condition,
        ) = eval_iterator(
            eval_photometric_dataloader_state,
            eval_spectroscopic_dataloader_state,
            resampling_key,
        )

        loss_value, output_state, aux = eval_step(x, y, step_key, ssvae, input_state)

        end_of_val_split = jnp.all(reset_condition)

    end_of_val_split = False
