import os
import re
import time

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
from optax import contrib as optax_contrib

# Model Config

RUN_NAME = "test"
INPUT_SIZE = 27
LATENT_SIZE = 15
PREDICTOR_SIZE = 1
USE_SPEC_NORM = False

# Training Config

SEED = 5678
EPOCHS = 2
EVAL_EVERY_N = 1
LEARNING_RATE = 3e-4
BATCH_SIZE = 1024
TRAIN_BATCHES_PER_EPOCH = 1
VAL_BATCHES_PER_EPOCH = 1

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10

REDUCE_LR_FACTOR = 0.1
REDUCE_LR_PATIENCE = 2
REDUCE_LR_ON_PLATEAU = True

# Data Config

N_SPLITS = 1
SHUFFLE = True
DROP_LAST = True
MISSING_TARGET_VALUE = -9999.0
DATA_DIR = Path("/scratch/project/dd-23-98/Base/")
SAVE_DIR = Path(f"/home/it4i-josman/PREDVAE/{RUN_NAME}")

psf_columns = [f"psfmag_{b}" for b in "ugriz"] + ["w1mag", "w2mag"]
psf_err_columns = [f"psfmagerr_{b}" for b in "ugriz"] + ["w1sigmag", "w2sigmag"]
model_columns = [f"modelmag_{b}" for b in "ugriz"] + ["w1mpro", "w2mpro"]
model_err_columns = [f"modelmagerr_{b}" for b in "ugriz"] + ["w1sigmpro", "w2sigmpro"]
additional_columns = ["extinction_i"]
z_column = ["z"]
objid_column = ["objid"]

SAVE_DIR.mkdir(exist_ok=True)
VECTORIZE = True if N_SPLITS > 1 else False
RNG_KEY = jax.random.PRNGKey(SEED)


print(
    "\n--------------------------------- LOADING DATA ---------------------------------\n"
)

spec_df = pd.read_csv(DATA_DIR / "SDSS_spec_train.csv")
photo_df = pd.read_csv(DATA_DIR / "SDSS_photo_xmatch.csv", skiprows=[1])

(
    spec_psf_photometry,
    spec_psf_photometry_err,
    spec_model_photometry,
    spec_model_photometry_err,
    spec_additional_info,
    spec_z,
    spec_objid,
) = data.create_input_arrays(
    input_df=spec_df,
    psf_columns=psf_columns,
    psf_err_columns=psf_err_columns,
    model_columns=model_columns,
    model_err_columns=model_err_columns,
    additional_columns=additional_columns,
    z_column=z_column,
    objid_column=objid_column,
    n_splits=N_SPLITS,
    shuffle=SHUFFLE,
)
spec_additional_info = jnp.log10(spec_additional_info)
spec_z = jnp.log10(spec_z)

(
    photo_psf_photometry,
    photo_psf_photometry_err,
    photo_model_photometry,
    photo_model_photometry_err,
    photo_additional_info,
    _,
    photo_objid,
) = data.create_input_arrays(
    input_df=photo_df,
    psf_columns=psf_columns,
    psf_err_columns=psf_err_columns,
    model_columns=model_columns,
    model_err_columns=model_err_columns,
    additional_columns=additional_columns,
    z_column=None,
    objid_column=objid_column,
    n_splits=N_SPLITS,
    shuffle=SHUFFLE,
)
photo_additional_info = jnp.log10(photo_additional_info)

n_spec = spec_df.shape[0] / N_SPLITS
n_photo = photo_df.shape[0] / N_SPLITS
ALPHA = n_photo / n_spec
spec_ratio = n_spec / (n_spec + n_photo)

PHOTOMETRIC_BATCH_SIZE = np.round(BATCH_SIZE * (1 - spec_ratio)).astype(int)
SPECTROSCOPIC_BATCH_SIZE = BATCH_SIZE - PHOTOMETRIC_BATCH_SIZE
batch_size_ratio = SPECTROSCOPIC_BATCH_SIZE / (
    SPECTROSCOPIC_BATCH_SIZE + PHOTOMETRIC_BATCH_SIZE
)
expected_no_of_spec_batches = n_spec // SPECTROSCOPIC_BATCH_SIZE
expected_no_of_photo_batches = n_photo // PHOTOMETRIC_BATCH_SIZE
photo_larger = expected_no_of_photo_batches > expected_no_of_spec_batches

print(f"N Spec: {n_spec}")
print(f"N Photo: {n_photo}")
print(f"Spec Ratio: {spec_ratio}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Photometric Batch Size: {PHOTOMETRIC_BATCH_SIZE}")
print(f"Spectroscopic Batch Size: {SPECTROSCOPIC_BATCH_SIZE}")
print(f"Batch Size Ratio: {batch_size_ratio}")
print(f"Expected No of Spec Batches: {expected_no_of_spec_batches}")
print(f"Expected No of Photo Batches: {expected_no_of_photo_batches}")

###################################################################################
#################### SPECTROPHOTOMETRIC DATASET AND STATISTICS ####################
###################################################################################

train_spec_dataset = eqx.filter_vmap(data.SpectroPhotometricDataset)(
    spec_psf_photometry,
    spec_psf_photometry_err,
    spec_model_photometry,
    spec_model_photometry_err,
    spec_additional_info,
    spec_z,
    spec_objid,
)

train_photo_dataset = eqx.filter_vmap(
    data.SpectroPhotometricDataset,
    in_axes=(
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        None,
        eqx.if_array(0),
    ),
)(
    photo_psf_photometry,
    photo_psf_photometry_err,
    photo_model_photometry,
    photo_model_photometry_err,
    photo_additional_info,
    None,
    photo_objid,
)

train_dataset_statistics = eqx.filter_vmap(data.SpectroPhotometricStatistics)(
    train_photo_dataset, train_spec_dataset
)

val_spec_dataset = eqx.filter_vmap(data.SpectroPhotometricDataset)(
    jnp.roll(spec_psf_photometry, 1, axis=0),
    jnp.roll(spec_psf_photometry_err, 1, axis=0),
    jnp.roll(spec_model_photometry, 1, axis=0),
    jnp.roll(spec_model_photometry_err, 1, axis=0),
    jnp.roll(spec_additional_info, 1, axis=0),
    jnp.roll(spec_z, 1, axis=0),
    jnp.roll(spec_objid, 1, axis=0),
)

val_photo_dataset = eqx.filter_vmap(
    data.SpectroPhotometricDataset,
    in_axes=(
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        eqx.if_array(0),
        None,
        eqx.if_array(0),
    ),
)(
    jnp.roll(photo_psf_photometry, 1, axis=0),
    jnp.roll(photo_psf_photometry_err, 1, axis=0),
    jnp.roll(photo_model_photometry, 1, axis=0),
    jnp.roll(photo_model_photometry_err, 1, axis=0),
    jnp.roll(photo_additional_info, 1, axis=0),
    None,
    jnp.roll(photo_objid, 1, axis=0),
)

val_dataset_statistics = eqx.filter_vmap(data.SpectroPhotometricStatistics)(
    val_photo_dataset, val_spec_dataset
)

###################################################################################
################################# DATALOADERS #####################################
###################################################################################

keys = jr.split(RNG_KEY, (N_SPLITS + 1))
train_dataloader_keys, RNG_KEY = keys[:-1], keys[-1]

(
    train_photometric_dataloader,
    train_photometric_dataloader_state,
) = data.make_vectorized_dataloader(
    train_photo_dataset,
    batch_size=PHOTOMETRIC_BATCH_SIZE,
    rng_key=train_dataloader_keys,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST,
)

(
    train_spectroscopic_dataloader,
    train_spectroscopic_dataloader_state,
) = data.make_vectorized_dataloader(
    train_spec_dataset,
    batch_size=SPECTROSCOPIC_BATCH_SIZE,
    rng_key=train_dataloader_keys,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST,
)

train_iterator = data.make_spectrophotometric_iterator(
    train_photometric_dataloader,
    train_spectroscopic_dataloader,
    train_dataset_statistics,
    resample_photometry=True,
    vectorize=VECTORIZE,
)

keys = jr.split(RNG_KEY, (N_SPLITS + 1))
val_dataloader_keys, RNG_KEY = keys[:-1], keys[-1]

(
    val_photometric_dataloader,
    val_photometric_dataloader_state,
) = data.make_vectorized_dataloader(
    val_photo_dataset,
    batch_size=PHOTOMETRIC_BATCH_SIZE,
    rng_key=val_dataloader_keys,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST,
)

(
    val_spectroscopic_dataloader,
    val_spectroscopic_dataloader_state,
) = data.make_vectorized_dataloader(
    val_spec_dataset,
    batch_size=SPECTROSCOPIC_BATCH_SIZE,
    rng_key=val_dataloader_keys,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST,
)

val_iterator = data.make_spectrophotometric_iterator(
    val_photometric_dataloader,
    val_spectroscopic_dataloader,
    val_dataset_statistics,
    resample_photometry=True,
    vectorize=VECTORIZE,
)

###################################################################################
################################### MODEL #########################################
###################################################################################

keys = jr.split(RNG_KEY, (4, N_SPLITS))
predictor_key, encoder_key, decoder_key = keys[:3]
RNG_KEY = keys[3][0]

GaussianCoder = eqx.filter_vmap(
    nn.GaussianCoder, in_axes=(None, None, None, None, None, eqx.if_array(0), None)
)
Gaussian = eqx.filter_vmap(nn.Gaussian)
SSVAE = eqx.filter_vmap(eqx.nn.make_with_state(nn.SSVAE))

predictor = GaussianCoder(
    INPUT_SIZE,
    PREDICTOR_SIZE,
    [2048, 1024, 512],
    3,
    jax.nn.tanh,
    predictor_key,
    USE_SPEC_NORM,
)

encoder = GaussianCoder(
    INPUT_SIZE + PREDICTOR_SIZE,
    LATENT_SIZE,
    [2048, 1024, 512],
    3,
    jax.nn.tanh,
    encoder_key,
    USE_SPEC_NORM,
)

decoder = GaussianCoder(
    LATENT_SIZE + PREDICTOR_SIZE,
    INPUT_SIZE,
    [2048, 1024, 512],
    3,
    jax.nn.tanh,
    decoder_key,
    USE_SPEC_NORM,
)

latent_prior = nn.Gaussian(
    mu=jnp.zeros((N_SPLITS, LATENT_SIZE)),
    log_sigma=jnp.zeros((N_SPLITS, LATENT_SIZE)),
)

target_prior = nn.Gaussian(
    mu=jnp.zeros((N_SPLITS, PREDICTOR_SIZE)),
    log_sigma=jnp.zeros((N_SPLITS, PREDICTOR_SIZE)),
)

ssvae, input_state = SSVAE(encoder, decoder, predictor, latent_prior, target_prior)

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
optimizer_state = eqx.filter_vmap(optimizer.init)(eqx.filter(ssvae, eqx.is_array))

loss_kwargs = {"alpha": ALPHA, "missing_target_value": MISSING_TARGET_VALUE}
loss_fn = partial(training.ssvae_loss, **loss_kwargs)

train_step = training.make_train_step(
    optimizer=optimizer,
    loss_fn=loss_fn,
    filter_spec=filter_spec,
    vectorize=VECTORIZE,
)
val_step = training.make_eval_step(
    loss_fn=loss_fn,
    filter_spec=filter_spec,
    vectorize=VECTORIZE,
)

train_step = eqx.filter_jit(train_step)
val_step = eqx.filter_jit(val_step)
train_iterator = eqx.filter_jit(train_iterator)
val_iterator = eqx.filter_jit(val_iterator)

end_of_train_split = False
end_of_val_split = False

val_step_time = 0
train_step_time = 0
epoch_time = 0

train_loss = []
train_aux = []
val_loss = []
val_aux = []
best_val_epoch = -1

# jax.profiler.start_trace("/scratch/project/dd-23-98/tensorboard")

t0 = time.time()

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch} / {EPOCHS}\n")

    train_batches = 0
    val_batches = 0
    epoch_train_loss = []
    epoch_train_aux = []
    epoch_val_loss = []
    epoch_val_aux = []
    if len(val_loss) < 1:
        lr_val_loss = jnp.ones(N_SPLITS) * 1e50
    else:
        lr_val_loss = val_loss[-1]

    t0_epoch = time.time()

    epoch_train_key, epoch_val_key, RNG_KEY = jr.split(RNG_KEY, 3)

    while not end_of_train_split:

        keys = jr.split(epoch_train_key, (3, N_SPLITS))
        resampling_key, step_key = keys[:2]
        epoch_train_key = keys[2][0]

        (
            x,
            y,
            train_photometric_dataloader_state,
            train_spectroscopic_dataloader_state,
            photometric_reset_condition,
            spectroscopic_reset_condition,
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
            epoch_val_loss,
        )
        epoch_train_loss.append(loss_value)
        epoch_train_aux.append(aux)

        train_batches += 1
        end_of_train_split = jnp.all(spectroscopic_reset_condition)

    t1 = time.time()
    train_step_time += t1 - t0_epoch
    epoch_train_loss = jnp.nansum(jnp.array(epoch_train_loss), axis=0) / train_batches
    epoch_train_aux = jnp.nansum(jnp.array(epoch_train_aux), axis=0) / train_batches
    end_of_train_split = False

    t0_val = time.time()
    inference_ssvae = eqx.nn.inference_mode(ssvae)

    while not end_of_val_split:

        keys = jr.split(epoch_val_key, (3, N_SPLITS))
        resampling_key, step_key = keys[:2]
        epoch_val_key = keys[2][0]

        (
            x,
            y,
            val_photometric_dataloader_state,
            val_spectroscopic_dataloader_state,
            photometric_reset_condition,
            spectroscopic_reset_condition,
        ) = val_iterator(
            val_photometric_dataloader_state,
            val_spectroscopic_dataloader_state,
            resampling_key,
        )

        loss_value, input_state, aux = val_step(
            x,
            y,
            step_key,
            inference_ssvae,
            input_state,
        )
        epoch_val_loss.append(loss_value)
        epoch_val_aux.append(aux)

        val_batches += 1
        end_of_val_split = jnp.all(spectroscopic_reset_condition)

    t1_val = time.time()
    val_step_time += t1_val - t0_val
    epoch_val_loss = jnp.nansum(jnp.array(epoch_val_loss), axis=0) / val_batches
    epoch_val_aux = jnp.nansum(jnp.array(epoch_val_aux), axis=0) / val_batches

    t1 = time.time()
    epoch_time += t1 - t0_epoch

    mean_epoch_train_loss = jnp.mean(epoch_train_loss)
    mean_epoch_val_loss = jnp.mean(epoch_val_loss)
    mean_epoch_train_aux = jnp.mean(epoch_train_aux, axis=0)
    mean_epoch_val_aux = jnp.mean(epoch_val_aux, axis=0)
    print(
        f"Epoch: {epoch} - Time: {t1-t0_epoch:.2f} s - Train Loss: {mean_epoch_train_loss:.3f} - Val Loss: {mean_epoch_val_loss:.3f} - "
        + f"TU Loss: {mean_epoch_train_aux[0]:.3f} - TS Loss: {mean_epoch_train_aux[1]:.3f} - TT Loss: {mean_epoch_train_aux[2]:.3f} - "
        + f"VU Loss: {mean_epoch_val_aux[0]:.3f} - VS Loss: {mean_epoch_val_aux[1]:.3f} - VT Loss: {mean_epoch_val_aux[2]:.3f}"
    )
    end_of_val_split = False
    train_loss.append(epoch_train_loss)
    train_aux.append(epoch_train_aux)
    val_loss.append(epoch_val_loss)
    val_aux.append(epoch_val_aux)

    if len(val_loss) == 1 or mean_epoch_val_loss < jnp.nanmean(val_loss[epoch]):
        best_val_epoch = epoch
        training.save(SAVE_DIR / "best_model.pkl", ssvae)
        train_losses = jnp.asarray(train_loss)
        val_losses = jnp.asarray(val_loss)

        train_auxes = jnp.asarray(train_aux)
        val_auxes = jnp.asarray(val_aux)

        np.save(SAVE_DIR / "train_losses.npy", train_losses)
        np.save(SAVE_DIR / "val_losses.npy", val_losses)
        np.save(SAVE_DIR / "train_auxes.npy", train_auxes)
        np.save(SAVE_DIR / "val_auxes.npy", val_auxes)

    if USE_EARLY_STOPPING and epoch - best_val_epoch > EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping after {epoch} epochs\n")

# jax.profiler.stop_trace()

val_step_time = val_step_time / EPOCHS
train_step_time = train_step_time / EPOCHS
epoch_time = epoch_time / EPOCHS
train_time = time.time() - t0

print(
    f"\nTrain Time: {train_time:.2f} s - Train Step Time: {train_step_time:.2f} s - Val Step Time: {val_step_time:.2f} s - Epoch Time: {epoch_time:.2f} s - Best Epoch: {best_val_epoch}"
)

print(
    "\n--------------------------------- SAVING MODEL ---------------------------------\n"
)

training.save(SAVE_DIR / "final_model.pkl", ssvae)

train_losses = jnp.asarray(train_loss)
val_losses = jnp.asarray(val_loss)

train_auxes = jnp.asarray(train_aux)
val_auxes = jnp.asarray(val_aux)

np.save(SAVE_DIR / "train_losses.npy", train_losses)
np.save(SAVE_DIR / "val_losses.npy", val_losses)
np.save(SAVE_DIR / "train_auxes.npy", train_auxes)
np.save(SAVE_DIR / "val_auxes.npy", val_auxes)
# np.save(SAVE_DIR / "lr_history.npy", lr_history)
