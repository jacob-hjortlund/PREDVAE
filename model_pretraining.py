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

RUN_NAME = "SSVAE_L5_M3"  # "VAE_B1000_L5"
INPUT_SIZE = 27
LATENT_SIZE = 5
PREDICTOR_SIZE = 1
NUM_MIXTURE_COMPONENTS = 3
USE_SPEC_NORM = True
NUM_POWER_ITERATIONS = 5
N_MC_SAMPLES = 100
LAYERS = [2048, 1024, 512]
N_LAYERS = 3
BETA = 1.0
USE_V2 = False

# Training Config

SEED = 5678
EPOCHS = 450
WARMUP_EPOCHS = 1
INIT_LEARNING_RATE = 5e-3
FINAL_LEARNING_RATE = 5e-6
BATCH_SIZE = 1024
LOG_EVERY = 1

PRETRAIN_VAE = False
PRETRAIN_PREDICTOR = False
TRAIN_FULL_MODEL = True

USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 10

# Data Config

N_SPLITS = 1
VAL_FRAC = 0.1
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
RNG_KEY = jax.random.PRNGKey(SEED)

print(
    "\n--------------------------------- LOADING DATA ---------------------------------\n"
)

# ----------------------------- LOAD DATA -----------------------------

spec_df = pd.read_csv(DATA_DIR / "SDSS_spec_train.csv")
photo_df = pd.read_csv(DATA_DIR / "SDSS_photo_xmatch.csv", skiprows=[1])

# ----------------------------- RESET BATCH SIZES AND ALPHA -----------------------------

n_spec = spec_df.shape[0] / N_SPLITS
n_photo = photo_df.shape[0] / N_SPLITS
spec_ratio = n_spec / (n_spec + n_photo)

PHOTOMETRIC_BATCH_SIZE = np.round(BATCH_SIZE * (1 - spec_ratio)).astype(int)
SPECTROSCOPIC_BATCH_SIZE = BATCH_SIZE - PHOTOMETRIC_BATCH_SIZE
ALPHA = (PHOTOMETRIC_BATCH_SIZE + SPECTROSCOPIC_BATCH_SIZE) * SPECTROSCOPIC_BATCH_SIZE
batch_size_ratio = SPECTROSCOPIC_BATCH_SIZE / (
    SPECTROSCOPIC_BATCH_SIZE + PHOTOMETRIC_BATCH_SIZE
)
expected_no_of_spec_batches = n_spec // SPECTROSCOPIC_BATCH_SIZE
expected_no_of_photo_batches = n_photo // PHOTOMETRIC_BATCH_SIZE
photo_larger = expected_no_of_photo_batches > expected_no_of_spec_batches

print(f"\nN Spec: {n_spec}")
print(f"N Photo: {n_photo}")
print(f"Spec Ratio: {spec_ratio}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Photometric Batch Size: {PHOTOMETRIC_BATCH_SIZE}")
print(f"Spectroscopic Batch Size: {SPECTROSCOPIC_BATCH_SIZE}")
print(f"Batch Size Ratio: {batch_size_ratio}")
print(f"Expected No of Spec Batches: {expected_no_of_spec_batches}")
print(f"Expected No of Photo Batches: {expected_no_of_photo_batches}\n")

# ----------------------------- CREATE INPUT ARRAYS -----------------------------

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
spec_psf_photometry = spec_psf_photometry.squeeze(axis=0)
spec_psf_photometry_err = spec_psf_photometry_err.squeeze(axis=0)
spec_model_photometry = spec_model_photometry.squeeze(axis=0)
spec_model_photometry_err = spec_model_photometry_err.squeeze(axis=0)
spec_additional_info = jnp.log10(spec_additional_info).squeeze(axis=0)
spec_z = jnp.log10(spec_z).squeeze(axis=0)
spec_objid = spec_objid.squeeze(axis=0)

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
photo_psf_photometry = photo_psf_photometry.squeeze(axis=0)
photo_psf_photometry_err = photo_psf_photometry_err.squeeze(axis=0)
photo_model_photometry = photo_model_photometry.squeeze(axis=0)
photo_model_photometry_err = photo_model_photometry_err.squeeze(axis=0)
photo_additional_info = jnp.log10(photo_additional_info).squeeze(axis=0)
photo_objid = photo_objid.squeeze(axis=0)

# ----------------------------- SPLIT INTO TRAIN AND VAL -----------------------------

spec_split_key, photo_split_key, RNG_KEY = jr.split(RNG_KEY, 3)
spec_val_mask = jax.random.bernoulli(
    spec_split_key, p=VAL_FRAC, shape=(spec_z.shape[0],)
)
photo_val_mask = jax.random.bernoulli(
    photo_split_key, p=VAL_FRAC, shape=(photo_objid.shape[0],)
)

spec_psf_photometry_train = spec_psf_photometry[~spec_val_mask]
spec_psf_photometry_err_train = spec_psf_photometry_err[~spec_val_mask]
spec_model_photometry_train = spec_model_photometry[~spec_val_mask]
spec_model_photometry_err_train = spec_model_photometry_err[~spec_val_mask]
spec_additional_info_train = spec_additional_info[~spec_val_mask]
spec_z_train = spec_z[~spec_val_mask]
spec_objid_train = spec_objid[~spec_val_mask]

spec_psf_photometry_val = spec_psf_photometry[spec_val_mask]
spec_psf_photometry_err_val = spec_psf_photometry_err[spec_val_mask]
spec_model_photometry_val = spec_model_photometry[spec_val_mask]
spec_model_photometry_err_val = spec_model_photometry_err[spec_val_mask]
spec_additional_info_val = spec_additional_info[spec_val_mask]
spec_z_val = spec_z[spec_val_mask]
spec_objid_val = spec_objid[spec_val_mask]

photo_psf_photometry_train = photo_psf_photometry[~photo_val_mask]
photo_psf_photometry_err_train = photo_psf_photometry_err[~photo_val_mask]
photo_model_photometry_train = photo_model_photometry[~photo_val_mask]
photo_model_photometry_err_train = photo_model_photometry_err[~photo_val_mask]
photo_additional_info_train = photo_additional_info[~photo_val_mask]
photo_objid_train = photo_objid[~photo_val_mask]

photo_psf_photometry_val = photo_psf_photometry[photo_val_mask]
photo_psf_photometry_err_val = photo_psf_photometry_err[photo_val_mask]
photo_model_photometry_val = photo_model_photometry[photo_val_mask]
photo_model_photometry_err_val = photo_model_photometry_err[photo_val_mask]
photo_additional_info_val = photo_additional_info[photo_val_mask]
photo_objid_val = photo_objid[photo_val_mask]

n_train_spec = spec_psf_photometry_train.shape[0]
n_train_photo = photo_psf_photometry_train.shape[0]
n_val_spec = spec_psf_photometry_val.shape[0]
n_val_photo = photo_psf_photometry_val.shape[0]

expected_n_train_spec_batches = n_train_spec // SPECTROSCOPIC_BATCH_SIZE
expected_n_train_photo_batches = n_train_photo // PHOTOMETRIC_BATCH_SIZE
expected_n_val_spec_batches = n_val_spec // SPECTROSCOPIC_BATCH_SIZE
expected_n_val_photo_batches = n_val_photo // PHOTOMETRIC_BATCH_SIZE

print(f"\nTrain Spec: {spec_psf_photometry_train.shape[0]}")
print(f"Train Photo: {photo_psf_photometry_train.shape[0]}")
print(
    f"Expected No of Train Batches: {expected_n_train_spec_batches} / {expected_n_train_photo_batches}\n"
)

print(f"\nVal Spec: {spec_psf_photometry_val.shape[0]}")
print(f"Val Photo: {photo_psf_photometry_val.shape[0]}")
print(
    f"Expected No of Val Batches: {expected_n_val_spec_batches} / {expected_n_val_photo_batches}\n"
)

###################################################################################
############################ CREATE DATASETS ######################################
###################################################################################


train_spec_dataset = data.SpectroPhotometricDataset(
    spec_psf_photometry_train,
    spec_psf_photometry_err_train,
    spec_model_photometry_train,
    spec_model_photometry_err_train,
    spec_additional_info_train,
    spec_z_train,
    spec_objid_train,
)

train_photo_dataset = data.SpectroPhotometricDataset(
    photo_psf_photometry_train,
    photo_psf_photometry_err_train,
    photo_model_photometry_train,
    photo_model_photometry_err_train,
    photo_additional_info_train,
    None,
    photo_objid_train,
)

train_dataset_statistics = data.SpectroPhotometricStatistics(
    train_photo_dataset, train_spec_dataset
)

training.save(SAVE_DIR / "train_dataset_statistics.pkl", train_dataset_statistics)

val_spec_dataset = data.SpectroPhotometricDataset(
    spec_psf_photometry_val,
    spec_psf_photometry_err_val,
    spec_model_photometry_val,
    spec_model_photometry_err_val,
    spec_additional_info_val,
    spec_z_val,
    spec_objid_val,
)

val_photo_dataset = data.SpectroPhotometricDataset(
    photo_psf_photometry_val,
    photo_psf_photometry_err_val,
    photo_model_photometry_val,
    photo_model_photometry_err_val,
    photo_additional_info_val,
    None,
    photo_objid_val,
)

val_dataset_statistics = data.SpectroPhotometricStatistics(
    val_photo_dataset, val_spec_dataset
)

###################################################################################
################################# DATALOADERS #####################################
###################################################################################

train_photo_dataloader_key, train_spec_dataloader_key, RNG_KEY = jr.split(RNG_KEY, 3)

(
    train_photometric_dataloader,
    train_photometric_dataloader_state,
) = data.make_dataloader(
    train_photo_dataset,
    batch_size=PHOTOMETRIC_BATCH_SIZE,
    rng_key=train_photo_dataloader_key,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST,
)

(
    train_spectroscopic_dataloader,
    train_spectroscopic_dataloader_state,
) = data.make_dataloader(
    train_spec_dataset,
    batch_size=SPECTROSCOPIC_BATCH_SIZE,
    rng_key=train_spec_dataloader_key,
    shuffle=SHUFFLE,
    drop_last=DROP_LAST,
)

train_iterator = data.make_spectrophotometric_iterator(
    train_photometric_dataloader,
    train_spectroscopic_dataloader,
    train_dataset_statistics,
    resample_photometry=True,
)

val_photo_dataloader_key, val_spec_dataloader_key, RNG_KEY = jr.split(RNG_KEY, 3)

(
    val_photometric_dataloader,
    val_photometric_dataloader_state,
) = data.make_dataloader(
    val_photo_dataset,
    batch_size=PHOTOMETRIC_BATCH_SIZE,
    rng_key=val_photo_dataloader_key,
    shuffle=False,
    drop_last=DROP_LAST,
)

(
    val_spectroscopic_dataloader,
    val_spectroscopic_dataloader_state,
) = data.make_dataloader(
    val_spec_dataset,
    batch_size=SPECTROSCOPIC_BATCH_SIZE,
    rng_key=val_spec_dataloader_key,
    shuffle=False,
    drop_last=DROP_LAST,
)

val_iterator = data.make_spectrophotometric_iterator(
    val_photometric_dataloader,
    val_spectroscopic_dataloader,
    val_dataset_statistics,
    resample_photometry=True,
)
###################################################################################
################################### MODEL #########################################
###################################################################################

(
    predictor_key,
    encoder_input_key,
    encoder_key,
    decoder_input_key,
    decoder_key,
    RNG_KEY,
) = jr.split(RNG_KEY, 6)

decoder_input_layer = nn.InputLayer(
    x_features=LATENT_SIZE,
    y_features=PREDICTOR_SIZE,
    out_features=LATENT_SIZE + PREDICTOR_SIZE,
    key=decoder_input_key,
)

decoder = nn.GaussianCoder(
    LATENT_SIZE + PREDICTOR_SIZE,
    INPUT_SIZE,
    LAYERS,
    N_LAYERS,
    jax.nn.tanh,
    decoder_key,
    USE_SPEC_NORM,
    NUM_POWER_ITERATIONS,
)

latent_prior = nn.Gaussian(
    mu=jnp.zeros(LATENT_SIZE),
    log_sigma=jnp.zeros(LATENT_SIZE),
)

target_prior = nn.Gaussian(
    mu=jnp.zeros(PREDICTOR_SIZE),
    log_sigma=jnp.zeros(PREDICTOR_SIZE),
)

if USE_V2:

    encoder = nn.GaussianCoder(
        INPUT_SIZE,
        LATENT_SIZE,
        LAYERS,
        N_LAYERS,
        jax.nn.tanh,
        encoder_key,
        USE_SPEC_NORM,
        NUM_POWER_ITERATIONS,
    )

    predictor_input_layer = nn.InputLayer(
        x_features=LATENT_SIZE,
        y_features=INPUT_SIZE,
        out_features=LATENT_SIZE + INPUT_SIZE,
        key=encoder_input_key,
    )

    predictor = nn.GaussianMixtureCoder(
        INPUT_SIZE + LATENT_SIZE,
        PREDICTOR_SIZE,
        LAYERS,
        N_LAYERS,
        NUM_MIXTURE_COMPONENTS,
        jax.nn.tanh,
        predictor_key,
        USE_SPEC_NORM,
        NUM_POWER_ITERATIONS,
    )

    SSVAE = partial(
        nn.SSVAEv2,
        encoder=encoder,
        predictor=predictor,
        predictor_input_layer=predictor_input_layer,
    )

else:

    encoder_input_layer = nn.InputLayer(
        x_features=INPUT_SIZE,
        y_features=PREDICTOR_SIZE,
        out_features=INPUT_SIZE + PREDICTOR_SIZE,
        key=encoder_input_key,
    )

    encoder = nn.GaussianCoder(
        INPUT_SIZE + PREDICTOR_SIZE,
        LATENT_SIZE,
        LAYERS,
        N_LAYERS,
        jax.nn.tanh,
        encoder_key,
        USE_SPEC_NORM,
        NUM_POWER_ITERATIONS,
    )

    predictor = nn.GaussianMixtureCoder(
        INPUT_SIZE,
        PREDICTOR_SIZE,
        LAYERS,
        N_LAYERS,
        NUM_MIXTURE_COMPONENTS,
        jax.nn.tanh,
        predictor_key,
        USE_SPEC_NORM,
        NUM_POWER_ITERATIONS,
    )

    SSVAE = partial(
        nn.SSVAE,
        encoder=encoder,
        predictor=predictor,
        encoder_input_layer=encoder_input_layer,
    )

ssvae, input_state = eqx.nn.make_with_state(SSVAE)(
    decoder=decoder,
    latent_prior=latent_prior,
    target_prior=target_prior,
    decoder_input_layer=decoder_input_layer,
)

filter_spec = tree_map(lambda _: True, ssvae)
filter_spec = nn.freeze_prior(ssvae, space="latent", filter_spec=filter_spec)

train_loss = []
train_aux = []
val_loss = []
val_aux = []
best_val_loss = jnp.inf
best_val_epoch = -1

###################################################################################
############################### PRE-TRAIN VAE #####################################
###################################################################################

if PRETRAIN_VAE:

    val_step_time = 0
    train_step_time = 0
    prediction_step_time = 0
    epoch_time = 0

    filter_spec = nn.freeze_prior(ssvae, space="target", filter_spec=filter_spec)
    filter_spec = nn.freeze_submodule(ssvae, "predictor", filter_spec=filter_spec)
    filter_spec = nn.freeze_submodule_inputs(
        ssvae, "decoder", freeze_x=False, freeze_y=True, filter_spec=filter_spec
    )
    ssvae = nn.init_submodule_inputs(
        ssvae, "decoder", RNG_KEY, init_x=False, init_y=True, init_value=0.0
    )
    ssvae = nn.set_submodule_inference_mode(ssvae, "predictor", True)

    if USE_V2:
        filter_spec = nn.freeze_submodule_inputs(
            ssvae, "predictor", freeze_x=True, freeze_y=True, filter_spec=filter_spec
        )
    else:
        filter_spec = nn.freeze_submodule_inputs(
            ssvae, "encoder", freeze_x=False, freeze_y=True, filter_spec=filter_spec
        )
        ssvae = nn.init_submodule_inputs(
            ssvae, "encoder", RNG_KEY, init_x=False, init_y=True, init_value=0.0
        )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        FINAL_LEARNING_RATE,
        INIT_LEARNING_RATE,
        WARMUP_EPOCHS,
        EPOCHS - WARMUP_EPOCHS,
        FINAL_LEARNING_RATE,
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
    optimizer_state = optimizer.init(eqx.filter(ssvae, eqx.is_array))

    loss_kwargs = {
        "alpha": ALPHA,
        "missing_target_value": MISSING_TARGET_VALUE,
        "vae_factor": 1.0,
        "beta": BETA,
        "predictor_factor": 0.0,
        "n_samples": N_MC_SAMPLES,
    }
    pretrain_vae_loss_fn = partial(training.ssvae_loss, **loss_kwargs)

    _train_step = training.make_train_step(
        optimizer=optimizer,
        loss_fn=pretrain_vae_loss_fn,
        filter_spec=filter_spec,
    )

    _val_step = training.make_eval_step(
        loss_fn=pretrain_vae_loss_fn,
        filter_spec=filter_spec,
    )

    @eqx.filter_jit
    def train_step(
        ssvae, ssvae_state, optimizer_state, photo_dl_state, spec_dl_state, rng_key
    ):

        resampling_key, step_key = jr.split(rng_key)

        (
            x,
            y,
            photo_dl_state,
            spec_dl_state,
            _,
            spectroscopic_reset_condition,
        ) = train_iterator(
            photo_dl_state,
            spec_dl_state,
            resampling_key,
        )

        end_of_split = jnp.all(spectroscopic_reset_condition)

        ssvae, ssvae_state, optimizer_state, loss_value, loss_aux = _train_step(
            x,
            y,
            step_key,
            ssvae,
            ssvae_state,
            optimizer_state,
        )

        return (
            loss_value,
            loss_aux,
            ssvae,
            ssvae_state,
            optimizer_state,
            photo_dl_state,
            spec_dl_state,
            end_of_split,
        )

    @eqx.filter_jit
    def eval_step(
        ssvae,
        ssvae_state,
        train_photo_dl_state,
        train_spec_dl_state,
        val_photo_dl_state,
        val_spec_dl_state,
        rng_key,
    ):

        train_resampling_key, val_resampling_key, rng_key = jr.split(rng_key, 3)
        train_step_key, val_step_key = jr.split(rng_key)

        (
            x_val_split,
            y_val_split,
            val_photo_dl_state,
            val_spec_dl_state,
            photometric_reset_condition,
            spectroscopic_reset_condition,
        ) = val_iterator(
            val_photo_dl_state,
            val_spec_dl_state,
            val_resampling_key,
        )

        (
            x_train_split,
            y_train_split,
            train_photo_dl_state,
            train_spec_dl_state,
            _,
            _,
        ) = train_iterator(
            train_photo_dl_state,
            train_spec_dl_state,
            train_resampling_key,
        )

        (
            train_loss_value,
            ssvae_state,
            train_loss_aux,
        ) = _val_step(
            x_train_split,
            y_train_split,
            train_step_key,
            ssvae,
            ssvae_state,
        )

        val_loss_value, input_state, val_loss_aux = _val_step(
            x_val_split,
            y_val_split,
            val_step_key,
            ssvae,
            ssvae_state,
        )

        end_of_val_split = jnp.all(spectroscopic_reset_condition)

        return (
            train_loss_value,
            train_loss_aux,
            val_loss_value,
            val_loss_aux,
            input_state,
            train_photo_dl_state,
            train_spec_dl_state,
            val_photo_dl_state,
            val_spec_dl_state,
            end_of_val_split,
        )

    t0 = time.time()

    for epoch in range(EPOCHS):

        end_of_train_split = False
        end_of_val_split = False
        end_of_prediction_split = False

        train_batches = 0
        val_batches = 0
        epoch_train_loss = []
        epoch_train_aux = []
        epoch_val_loss = []
        epoch_val_aux = []
        epoch_val_target_means = []
        epoch_val_target_stds = []

        t0_epoch = time.time()

        epoch_train_key, epoch_val_key, RNG_KEY = jr.split(RNG_KEY, 3)

        while not end_of_train_split:

            step_key, epoch_train_key = jr.split(epoch_train_key)

            (
                batch_train_loss,
                batch_train_aux,
                ssvae,
                input_state,
                optimizer_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                end_of_train_split,
            ) = train_step(
                ssvae,
                input_state,
                optimizer_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                step_key,
            )

            if end_of_train_split:
                break

            train_batches += 1

        t1 = time.time()
        train_step_time += t1 - t0_epoch

        inference_ssvae = eqx.nn.inference_mode(ssvae)

        t0_val = time.time()
        while not end_of_val_split:

            t0_single = time.time()

            val_step_key, epoch_val_key = jr.split(epoch_val_key)

            (
                batch_train_loss,
                batch_train_aux,
                batch_val_loss,
                batch_val_aux,
                input_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                val_photometric_dataloader_state,
                val_spectroscopic_dataloader_state,
                end_of_val_split,
            ) = eval_step(
                inference_ssvae,
                input_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                val_photometric_dataloader_state,
                val_spectroscopic_dataloader_state,
                val_step_key,
            )

            if end_of_val_split:
                break

            epoch_train_loss.append(batch_train_loss)
            epoch_train_aux.append(batch_train_aux)
            epoch_val_loss.append(batch_val_loss)
            epoch_val_aux.append(batch_val_aux)

            val_batches += 1
            t1_single = time.time()

        train_photometric_dataloader_state = train_photometric_dataloader_state.set(
            train_photometric_dataloader.reset_index, jnp.array(True)
        )
        train_spectroscopic_dataloader_state = train_spectroscopic_dataloader_state.set(
            train_spectroscopic_dataloader.reset_index, jnp.array(True)
        )

        t1_val = time.time()
        val_step_time += t1_val - t0_val

        epoch_train_loss = jnp.mean(jnp.array(epoch_train_loss), axis=0)
        epoch_train_aux = jnp.mean(jnp.array(epoch_train_aux), axis=0)
        epoch_val_loss = jnp.mean(jnp.array(epoch_val_loss), axis=0)
        epoch_val_aux = jnp.mean(jnp.array(epoch_val_aux), axis=0)

        train_loss.append(epoch_train_loss)
        train_aux.append(epoch_train_aux)
        val_loss.append(epoch_val_loss)
        val_aux.append(epoch_val_aux)

        t1_epoch = time.time()
        epoch_time += t1_epoch - t0_epoch
        epoch_lr = lr_schedule(epoch)

        print(
            f"Epoch: {epoch} - Time: {t1_epoch-t0_epoch:.2f} s - LR: {epoch_lr:.5e} - Train Loss: {epoch_train_loss:.5e} - Val Loss: {epoch_val_loss:.5e} - "
            + f"TU Loss: {epoch_train_aux[0]:.5e} - TS Loss: {epoch_train_aux[6]:.5e} - TT Loss: {epoch_train_aux[7]:.5e} - "
            + f"VU Loss: {epoch_val_aux[0]:.5e} - VS Loss: {epoch_val_aux[6]:.5e} - VT Loss: {epoch_val_aux[7]:.5e}"
        )

        if len(val_loss) == 1 or epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_epoch = epoch
            training.save(SAVE_DIR / "best_pretrained_vae.pkl", ssvae)
            training.save(SAVE_DIR / "best_pretrained_vae_state.pkl", input_state)
            training.save(
                SAVE_DIR / "best_pretrained_vae_optimizer_state", optimizer_state
            )

        if USE_EARLY_STOPPING and epoch - best_val_epoch > EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}, setting model to best epoch")
            ssvae = training.load(SAVE_DIR / "best_pretrained_vae.pkl", ssvae)
            input_state = training.load(
                SAVE_DIR / "best_pretrained_vae_state.pkl", input_state
            )
            optimizer_state = training.load(
                SAVE_DIR / "final_pretrained_vae_optimizer_state", optimizer_state
            )
            break

    val_step_time = val_step_time / EPOCHS
    train_step_time = train_step_time / EPOCHS
    epoch_time = epoch_time / EPOCHS
    train_time = time.time() - t0

    print(
        f"\nTrain Time: {train_time:.2f} s - Train Step Time: {train_step_time:.2f} s - Val Step Time: {val_step_time:.2f} s - Epoch Time: {epoch_time:.2f} s - Best Epoch: {best_val_epoch}"
    )

    print(
        "\n--------------------------------- SAVING PRE-TRAINED VAE ---------------------------------\n"
    )

    training.save(SAVE_DIR / "final_pretrained_vae.pkl", ssvae)
    training.save(SAVE_DIR / "final_pretrained_vae_state.pkl", input_state)
    training.save(SAVE_DIR / "final_pretrained_vae_optimizer_state", optimizer_state)

    pretrained_vae_train_losses = jnp.asarray(train_loss)
    pretrained_vae_val_losses = jnp.asarray(val_loss)

    pretrained_vae_train_auxes = jnp.asarray(train_aux)
    pretrained_vae_val_auxes = jnp.asarray(val_aux)

    np.save(SAVE_DIR / "pretrain_vae_train_losses.npy", pretrained_vae_train_losses)
    np.save(SAVE_DIR / "pretrain_vae_val_losses.npy", pretrained_vae_val_losses)
    np.save(SAVE_DIR / "pretrain_vae_train_auxes.npy", pretrained_vae_train_auxes)
    np.save(SAVE_DIR / "pretrain_vae_val_auxes.npy", pretrained_vae_val_auxes)

###################################################################################
############################ PRE-TRAIN PREDICTOR ##################################
###################################################################################

if PRETRAIN_PREDICTOR:

    filter_spec = nn.freeze_prior(
        ssvae, space="target", filter_spec=filter_spec, inverse=True
    )
    filter_spec = nn.freeze_submodule(
        ssvae, "predictor", filter_spec=filter_spec, inverse=True
    )
    filter_spec = nn.freeze_submodule(ssvae, "encoder", filter_spec=filter_spec)
    filter_spec = nn.freeze_submodule(ssvae, "decoder", filter_spec=filter_spec)
    filter_spec = nn.freeze_submodule_inputs(
        ssvae, "decoder", freeze_x=True, freeze_y=True, filter_spec=filter_spec
    )

    ssvae = nn.init_submodule_inputs(
        ssvae, "decoder", RNG_KEY, init_x=False, init_y=True, init_value=0.0
    )
    ssvae = nn.set_submodule_inference_mode(ssvae, "predictor", False)
    ssvae = nn.set_submodule_inference_mode(ssvae, "encoder", True)
    ssvae = nn.set_submodule_inference_mode(ssvae, "decoder", True)

    if USE_V2:
        filter_spec = nn.freeze_submodule_inputs(
            ssvae,
            "predictor",
            freeze_x=True,
            freeze_y=True,
            filter_spec=filter_spec,
            inverse=True,
        )
    else:
        filter_spec = nn.freeze_submodule_inputs(
            ssvae, "encoder", freeze_x=True, freeze_y=True, filter_spec=filter_spec
        )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        FINAL_LEARNING_RATE,
        INIT_LEARNING_RATE,
        WARMUP_EPOCHS,
        EPOCHS - WARMUP_EPOCHS,
        FINAL_LEARNING_RATE,
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
    optimizer_state = optimizer.init(eqx.filter(ssvae, eqx.is_array))

    loss_kwargs = {
        "alpha": ALPHA,
        "missing_target_value": MISSING_TARGET_VALUE,
        "vae_factor": 0.0,
        "beta": BETA,
        "predictor_factor": 1.0,
        "n_samples": N_MC_SAMPLES,
    }
    pretrain_predictor_loss_fn = partial(training.ssvae_loss, **loss_kwargs)

    _train_step = training.make_train_step(
        optimizer=optimizer,
        loss_fn=pretrain_predictor_loss_fn,
        filter_spec=filter_spec,
    )

    _val_step = training.make_eval_step(
        loss_fn=pretrain_predictor_loss_fn,
        filter_spec=filter_spec,
    )

    val_step_time = 0
    train_step_time = 0
    prediction_step_time = 0
    epoch_time = 0
    best_val_loss = jnp.inf
    best_val_epoch = -1

    @eqx.filter_jit
    def train_step(
        ssvae, ssvae_state, optimizer_state, photo_dl_state, spec_dl_state, rng_key
    ):

        resampling_key, step_key = jr.split(rng_key)

        (
            x,
            y,
            photo_dl_state,
            spec_dl_state,
            _,
            spectroscopic_reset_condition,
        ) = train_iterator(
            photo_dl_state,
            spec_dl_state,
            resampling_key,
        )

        end_of_split = jnp.all(spectroscopic_reset_condition)

        ssvae, ssvae_state, optimizer_state, loss_value, loss_aux = _train_step(
            x,
            y,
            step_key,
            ssvae,
            ssvae_state,
            optimizer_state,
        )

        return (
            loss_value,
            loss_aux,
            ssvae,
            ssvae_state,
            optimizer_state,
            photo_dl_state,
            spec_dl_state,
            end_of_split,
        )

    @eqx.filter_jit
    def eval_step(
        ssvae,
        ssvae_state,
        train_photo_dl_state,
        train_spec_dl_state,
        val_photo_dl_state,
        val_spec_dl_state,
        rng_key,
    ):

        train_resampling_key, val_resampling_key, rng_key = jr.split(rng_key, 3)
        train_step_key, val_step_key = jr.split(rng_key)

        (
            x_val_split,
            y_val_split,
            val_photo_dl_state,
            val_spec_dl_state,
            photometric_reset_condition,
            spectroscopic_reset_condition,
        ) = val_iterator(
            val_photo_dl_state,
            val_spec_dl_state,
            val_resampling_key,
        )

        (
            x_train_split,
            y_train_split,
            train_photo_dl_state,
            train_spec_dl_state,
            _,
            _,
        ) = train_iterator(
            train_photo_dl_state,
            train_spec_dl_state,
            train_resampling_key,
        )

        (
            train_loss_value,
            ssvae_state,
            train_loss_aux,
        ) = _val_step(
            x_train_split,
            y_train_split,
            train_step_key,
            ssvae,
            ssvae_state,
        )

        val_loss_value, input_state, val_loss_aux = _val_step(
            x_val_split,
            y_val_split,
            val_step_key,
            ssvae,
            ssvae_state,
        )

        end_of_val_split = jnp.all(spectroscopic_reset_condition)

        return (
            train_loss_value,
            train_loss_aux,
            val_loss_value,
            val_loss_aux,
            input_state,
            train_photo_dl_state,
            train_spec_dl_state,
            val_photo_dl_state,
            val_spec_dl_state,
            end_of_val_split,
        )

    t0 = time.time()

    for epoch in range(EPOCHS):

        end_of_train_split = False
        end_of_val_split = False
        end_of_prediction_split = False

        train_batches = 0
        val_batches = 0
        epoch_train_loss = []
        epoch_train_aux = []
        epoch_val_loss = []
        epoch_val_aux = []
        epoch_val_target_means = []
        epoch_val_target_stds = []

        t0_epoch = time.time()

        epoch_train_key, epoch_val_key, RNG_KEY = jr.split(RNG_KEY, 3)

        while not end_of_train_split:

            step_key, epoch_train_key = jr.split(epoch_train_key)

            (
                batch_train_loss,
                batch_train_aux,
                ssvae,
                input_state,
                optimizer_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                end_of_train_split,
            ) = train_step(
                ssvae,
                input_state,
                optimizer_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                step_key,
            )

            if end_of_train_split:
                break

            train_batches += 1

        t1 = time.time()
        train_step_time += t1 - t0_epoch

        inference_ssvae = eqx.nn.inference_mode(ssvae)

        t0_val = time.time()
        while not end_of_val_split:

            t0_single = time.time()

            val_step_key, epoch_val_key = jr.split(epoch_val_key)

            (
                batch_train_loss,
                batch_train_aux,
                batch_val_loss,
                batch_val_aux,
                input_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                val_photometric_dataloader_state,
                val_spectroscopic_dataloader_state,
                end_of_val_split,
            ) = eval_step(
                inference_ssvae,
                input_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                val_photometric_dataloader_state,
                val_spectroscopic_dataloader_state,
                val_step_key,
            )

            if end_of_val_split:
                break

            epoch_train_loss.append(batch_train_loss)
            epoch_train_aux.append(batch_train_aux)
            epoch_val_loss.append(batch_val_loss)
            epoch_val_aux.append(batch_val_aux)

            val_batches += 1
            t1_single = time.time()

        train_photometric_dataloader_state = train_photometric_dataloader_state.set(
            train_photometric_dataloader.reset_index, jnp.array(True)
        )
        train_spectroscopic_dataloader_state = train_spectroscopic_dataloader_state.set(
            train_spectroscopic_dataloader.reset_index, jnp.array(True)
        )

        t1_val = time.time()
        val_step_time += t1_val - t0_val

        epoch_train_loss = jnp.mean(jnp.array(epoch_train_loss), axis=0)
        epoch_train_aux = jnp.mean(jnp.array(epoch_train_aux), axis=0)
        epoch_val_loss = jnp.mean(jnp.array(epoch_val_loss), axis=0)
        epoch_val_aux = jnp.mean(jnp.array(epoch_val_aux), axis=0)

        train_loss.append(epoch_train_loss)
        train_aux.append(epoch_train_aux)
        val_loss.append(epoch_val_loss)
        val_aux.append(epoch_val_aux)

        t1_epoch = time.time()
        epoch_time += t1_epoch - t0_epoch
        epoch_lr = lr_schedule(epoch)

        print(
            f"Epoch: {epoch} - Time: {t1_epoch-t0_epoch:.2f} s - LR: {epoch_lr:.5e} - Train Loss: {epoch_train_loss:.5e} - Val Loss: {epoch_val_loss:.5e} - "
            + f"TU Loss: {epoch_train_aux[0]:.5e} - TS Loss: {epoch_train_aux[6]:.5e} - TT Loss: {epoch_train_aux[7]:.5e} - "
            + f"VU Loss: {epoch_val_aux[0]:.5e} - VS Loss: {epoch_val_aux[6]:.5e} - VT Loss: {epoch_val_aux[7]:.5e}"
        )

        if len(val_loss) == 1 or epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_epoch = epoch
            training.save(SAVE_DIR / "best_pretrained_predictor.pkl", ssvae)
            training.save(SAVE_DIR / "best_pretrained_predictor_state.pkl", input_state)
            training.save(
                SAVE_DIR / "best_pretrained_predictor_optimizer_state", optimizer_state
            )

        if USE_EARLY_STOPPING and epoch - best_val_epoch > EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}, setting model to best epoch")
            ssvae = training.load(SAVE_DIR / "best_pretrained_predictor.pkl", ssvae)
            input_state = training.load(
                SAVE_DIR / "best_pretrained_predictor_state.pkl", input_state
            )
            optimizer_state = training.load(
                SAVE_DIR / "final_pretrained_predictor_optimizer_state", optimizer_state
            )
            break

    val_step_time = val_step_time / EPOCHS
    train_step_time = train_step_time / EPOCHS
    epoch_time = epoch_time / EPOCHS
    train_time = time.time() - t0

    print(
        f"\nTrain Time: {train_time:.2f} s - Train Step Time: {train_step_time:.2f} s - Val Step Time: {val_step_time:.2f} s - Epoch Time: {epoch_time:.2f} s - Best Epoch: {best_val_epoch}"
    )

    print(
        "\n--------------------------------- SAVING PRE-TRAINED PREDICTOR ---------------------------------\n"
    )

    training.save(SAVE_DIR / "final_pretrained_predictor.pkl", ssvae)
    training.save(SAVE_DIR / "final_pretrained_predictor_state.pkl", input_state)
    training.save(
        SAVE_DIR / "final_pretrained_predictor_optimizer_state", optimizer_state
    )

    pretrained_predictor_train_losses = jnp.asarray(train_loss)
    pretrained_predictor_val_losses = jnp.asarray(val_loss)

    pretrained_predictor_train_auxes = jnp.asarray(train_aux)
    pretrained_predictor_val_auxes = jnp.asarray(val_aux)

    np.save(
        SAVE_DIR / "pretrain_predictor_train_losses.npy",
        pretrained_predictor_train_losses,
    )
    np.save(
        SAVE_DIR / "pretrain_predictor_val_losses.npy", pretrained_predictor_val_losses
    )
    np.save(
        SAVE_DIR / "pretrain_predictor_train_auxes.npy",
        pretrained_predictor_train_auxes,
    )
    np.save(
        SAVE_DIR / "pretrain_predictor_val_auxes.npy", pretrained_predictor_val_auxes
    )

###################################################################################
############################## TRAIN FULL MODEL ###################################
###################################################################################

if TRAIN_FULL_MODEL:

    filter_spec = nn.freeze_submodule(
        ssvae, "encoder", filter_spec=filter_spec, inverse=True
    )
    filter_spec = nn.freeze_submodule(
        ssvae, "decoder", filter_spec=filter_spec, inverse=True
    )
    filter_spec = nn.freeze_prior(
        ssvae, space="target", filter_spec=filter_spec, inverse=True
    )
    filter_spec = nn.freeze_submodule(
        ssvae, "predictor", filter_spec=filter_spec, inverse=True
    )
    filter_spec = nn.freeze_submodule_inputs(
        ssvae,
        "decoder",
        freeze_x=True,
        freeze_y=True,
        filter_spec=filter_spec,
        inverse=True,
    )

    init_key, RNG_KEY = jr.split(RNG_KEY)
    ssvae = nn.init_submodule_inputs(
        ssvae, "decoder", init_key, init_x=True, init_y=True
    )
    ssvae = nn.set_submodule_inference_mode(ssvae, "predictor", False)
    ssvae = nn.set_submodule_inference_mode(ssvae, "encoder", False)
    ssvae = nn.set_submodule_inference_mode(ssvae, "decoder", False)

    if USE_V2:
        filter_spec = nn.freeze_submodule_inputs(
            ssvae,
            "predictor",
            freeze_x=True,
            freeze_y=True,
            filter_spec=filter_spec,
            inverse=True,
        )
    else:
        init_key, RNG_KEY = jr.split(RNG_KEY)
        filter_spec = nn.freeze_submodule_inputs(
            ssvae,
            "encoder",
            freeze_x=True,
            freeze_y=True,
            filter_spec=filter_spec,
            inverse=True,
        )
        ssvae = nn.init_submodule_inputs(
            ssvae, "encoder", init_key, init_x=True, init_y=True
        )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        FINAL_LEARNING_RATE,
        INIT_LEARNING_RATE,
        WARMUP_EPOCHS,
        EPOCHS - WARMUP_EPOCHS,
        FINAL_LEARNING_RATE,
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
    optimizer_state = optimizer.init(eqx.filter(ssvae, eqx.is_array))

    loss_kwargs = {
        "alpha": ALPHA,
        "missing_target_value": MISSING_TARGET_VALUE,
        "vae_factor": 1.0,
        "beta": BETA,
        "predictor_factor": 1.0,
        "n_samples": N_MC_SAMPLES,
    }
    full_loss_fn = partial(training.ssvae_loss, **loss_kwargs)

    _train_step = training.make_train_step(
        optimizer=optimizer,
        loss_fn=full_loss_fn,
        filter_spec=filter_spec,
    )

    _val_step = training.make_eval_step(
        loss_fn=full_loss_fn,
        filter_spec=filter_spec,
    )

    val_step_time = 0
    train_step_time = 0
    prediction_step_time = 0
    epoch_time = 0
    best_val_loss = jnp.inf
    best_val_epoch = -1

    @eqx.filter_jit
    def train_step(
        ssvae, ssvae_state, optimizer_state, photo_dl_state, spec_dl_state, rng_key
    ):

        resampling_key, step_key = jr.split(rng_key)

        (
            x,
            y,
            photo_dl_state,
            spec_dl_state,
            _,
            spectroscopic_reset_condition,
        ) = train_iterator(
            photo_dl_state,
            spec_dl_state,
            resampling_key,
        )

        end_of_split = jnp.all(spectroscopic_reset_condition)

        ssvae, ssvae_state, optimizer_state, loss_value, loss_aux = _train_step(
            x,
            y,
            step_key,
            ssvae,
            ssvae_state,
            optimizer_state,
        )

        return (
            loss_value,
            loss_aux,
            ssvae,
            ssvae_state,
            optimizer_state,
            photo_dl_state,
            spec_dl_state,
            end_of_split,
        )

    @eqx.filter_jit
    def eval_step(
        ssvae,
        ssvae_state,
        train_photo_dl_state,
        train_spec_dl_state,
        val_photo_dl_state,
        val_spec_dl_state,
        rng_key,
    ):

        train_resampling_key, val_resampling_key, rng_key = jr.split(rng_key, 3)
        train_step_key, val_step_key = jr.split(rng_key)

        (
            x_val_split,
            y_val_split,
            val_photo_dl_state,
            val_spec_dl_state,
            photometric_reset_condition,
            spectroscopic_reset_condition,
        ) = val_iterator(
            val_photo_dl_state,
            val_spec_dl_state,
            val_resampling_key,
        )

        (
            x_train_split,
            y_train_split,
            train_photo_dl_state,
            train_spec_dl_state,
            _,
            _,
        ) = train_iterator(
            train_photo_dl_state,
            train_spec_dl_state,
            train_resampling_key,
        )

        (
            train_loss_value,
            ssvae_state,
            train_loss_aux,
        ) = _val_step(
            x_train_split,
            y_train_split,
            train_step_key,
            ssvae,
            ssvae_state,
        )

        val_loss_value, input_state, val_loss_aux = _val_step(
            x_val_split,
            y_val_split,
            val_step_key,
            ssvae,
            ssvae_state,
        )

        end_of_val_split = jnp.all(spectroscopic_reset_condition)

        return (
            train_loss_value,
            train_loss_aux,
            val_loss_value,
            val_loss_aux,
            input_state,
            train_photo_dl_state,
            train_spec_dl_state,
            val_photo_dl_state,
            val_spec_dl_state,
            end_of_val_split,
        )

    t0 = time.time()

    for epoch in range(EPOCHS):

        end_of_train_split = False
        end_of_val_split = False
        end_of_prediction_split = False

        train_batches = 0
        val_batches = 0
        epoch_train_loss = []
        epoch_train_aux = []
        epoch_val_loss = []
        epoch_val_aux = []
        epoch_val_target_means = []
        epoch_val_target_stds = []

        t0_epoch = time.time()

        epoch_train_key, epoch_val_key, RNG_KEY = jr.split(RNG_KEY, 3)

        while not end_of_train_split:

            step_key, epoch_train_key = jr.split(epoch_train_key)

            (
                batch_train_loss,
                batch_train_aux,
                ssvae,
                input_state,
                optimizer_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                end_of_train_split,
            ) = train_step(
                ssvae,
                input_state,
                optimizer_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                step_key,
            )

            if end_of_train_split:
                break

            train_batches += 1

        t1 = time.time()
        train_step_time += t1 - t0_epoch

        inference_ssvae = eqx.nn.inference_mode(ssvae)

        t0_val = time.time()
        while not end_of_val_split:

            t0_single = time.time()

            val_step_key, epoch_val_key = jr.split(epoch_val_key)

            (
                batch_train_loss,
                batch_train_aux,
                batch_val_loss,
                batch_val_aux,
                input_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                val_photometric_dataloader_state,
                val_spectroscopic_dataloader_state,
                end_of_val_split,
            ) = eval_step(
                inference_ssvae,
                input_state,
                train_photometric_dataloader_state,
                train_spectroscopic_dataloader_state,
                val_photometric_dataloader_state,
                val_spectroscopic_dataloader_state,
                val_step_key,
            )

            if end_of_val_split:
                break

            epoch_train_loss.append(batch_train_loss)
            epoch_train_aux.append(batch_train_aux)
            epoch_val_loss.append(batch_val_loss)
            epoch_val_aux.append(batch_val_aux)

            val_batches += 1
            t1_single = time.time()

        train_photometric_dataloader_state = train_photometric_dataloader_state.set(
            train_photometric_dataloader.reset_index, jnp.array(True)
        )
        train_spectroscopic_dataloader_state = train_spectroscopic_dataloader_state.set(
            train_spectroscopic_dataloader.reset_index, jnp.array(True)
        )

        t1_val = time.time()
        val_step_time += t1_val - t0_val

        epoch_train_loss = jnp.mean(jnp.array(epoch_train_loss), axis=0)
        epoch_train_aux = jnp.mean(jnp.array(epoch_train_aux), axis=0)
        epoch_val_loss = jnp.mean(jnp.array(epoch_val_loss), axis=0)
        epoch_val_aux = jnp.mean(jnp.array(epoch_val_aux), axis=0)

        train_loss.append(epoch_train_loss)
        train_aux.append(epoch_train_aux)
        val_loss.append(epoch_val_loss)
        val_aux.append(epoch_val_aux)

        t1_epoch = time.time()
        epoch_time += t1_epoch - t0_epoch
        epoch_lr = lr_schedule(epoch)

        print(
            f"Epoch: {epoch} - Time: {t1_epoch-t0_epoch:.2f} s - LR: {epoch_lr:.5e} - Train Loss: {epoch_train_loss:.5e} - Val Loss: {epoch_val_loss:.5e} - "
            + f"TU Loss: {epoch_train_aux[0]:.5e} - TS Loss: {epoch_train_aux[6]:.5e} - TT Loss: {epoch_train_aux[7]:.5e} - "
            + f"VU Loss: {epoch_val_aux[0]:.5e} - VS Loss: {epoch_val_aux[6]:.5e} - VT Loss: {epoch_val_aux[7]:.5e}"
        )

        if len(val_loss) == 1 or epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_epoch = epoch
            training.save(SAVE_DIR / "best_model.pkl", ssvae)
            training.save(SAVE_DIR / "best_model_state.pkl", input_state)
            training.save(SAVE_DIR / "best_model_optimizer_state", optimizer_state)

        if USE_EARLY_STOPPING and epoch - best_val_epoch > EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}, setting model to best epoch")
            ssvae = training.load(SAVE_DIR / "best_model.pkl", ssvae)
            input_state = training.load(SAVE_DIR / "best_model_state.pkl", input_state)
            optimizer_state = training.load(
                SAVE_DIR / "final_model_optimizer_state", optimizer_state
            )
            break

    val_step_time = val_step_time / EPOCHS
    train_step_time = train_step_time / EPOCHS
    epoch_time = epoch_time / EPOCHS
    train_time = time.time() - t0

    print(
        f"\nTrain Time: {train_time:.2f} s - Train Step Time: {train_step_time:.2f} s - Val Step Time: {val_step_time:.2f} s - Epoch Time: {epoch_time:.2f} s - Best Epoch: {best_val_epoch}"
    )

    print(
        "\n--------------------------------- SAVING FULL MODEL ---------------------------------\n"
    )

    training.save(SAVE_DIR / "final_model.pkl", ssvae)
    training.save(SAVE_DIR / "final_model_state.pkl", input_state)
    training.save(SAVE_DIR / "final_model_optimizer_state", optimizer_state)

    model_predictor_train_losses = jnp.asarray(train_loss)
    model_predictor_val_losses = jnp.asarray(val_loss)

    model_predictor_train_auxes = jnp.asarray(train_aux)
    model_predictor_val_auxes = jnp.asarray(val_aux)

    np.save(SAVE_DIR / "full_train_losses.npy", model_predictor_train_losses)
    np.save(SAVE_DIR / "full_val_losses.npy", model_predictor_val_losses)
    np.save(SAVE_DIR / "full_train_auxes.npy", model_predictor_train_auxes)
    np.save(SAVE_DIR / "full_val_auxes.npy", model_predictor_val_auxes)
