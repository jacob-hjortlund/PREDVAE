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
import src.predvae.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from jax.tree_util import tree_map
from src.predvae.training import train, ssvae_loss, save
from src.predvae.data import HDF5Dataset, StratifiedBatchSampler

colors = sns.color_palette("colorblind")


def collate_fn(batch):
    batch = list(*batch)
    batch = (torch.Tensor(b) for b in batch)
    return (*batch,)


def closest(n, divisor):
    return n - (n % divisor)


@eqx.filter_jit
def evaluate(model, x):
    # discard state
    y, z, x_hat, y_pars, z_pars, x_pars, _ = jax.vmap(model, in_axes=(0, 0))(x, x)
    return y, z, x_hat, y_pars, z_pars, x_pars


def transform_redshift(log10_z, log10_z_err, mean, std):

    denormed_mean = log10_z * std + mean
    denormed_err = log10_z_err * std

    z_median = 10**denormed_mean
    z_lower = 10 ** (denormed_mean - denormed_err)
    z_upper = 10 ** (denormed_mean + denormed_err)
    z_err = (z_upper - z_lower) / 2

    return z_median, z_err


RUN_NAME = "SSVAE_test_early_stopping"
INPUT_SIZE = 27
LATENT_SIZE = 15
PREDICTOR_SIZE = 1
USE_SPEC_NORM = True
TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
EPOCHS = 10
EVAL_EVERY_N = 1
SEED = 5678
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 0
MISSING_TARGET_VALUE = -9999.0
DATA_DIR = Path("/home/jacob/Uni/Msc/VAEPhotoZ/Data/SS_Splits")
SAVE_DIR = Path(f"/home/jacob/Uni/Msc/VAEPhotoZ/PREDVAE/{RUN_NAME}")
SPLIT = 0
NUM_WORKERS = 0
if NUM_WORKERS > 0:
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
else:
    PIN_MEMORY = False
    PERSISTENT_WORKERS = False

print(f"\nSplit Used: {SPLIT}\n")

print(f"\nNumber of devices: {jax.device_count()}")
print(f"Devices: {jax.devices()}\n")

SAVE_DIR.mkdir(exist_ok=True)

rng_key = jax.random.PRNGKey(SEED)

print(
    "\n--------------------------------- LOADING DATA ---------------------------------\n"
)

train_dataset = HDF5Dataset(
    path=DATA_DIR / f"train_{SPLIT}.hdf5",
    resample=True,
)
train_one_hot_redshifts = train_dataset.get_one_hot_redshifts()
TRAIN_BATCHES_PER_EPOCH = (
    10  # closest(len(train_one_hot_redshifts), TRAIN_BATCH_SIZE) // TRAIN_BATCH_SIZE
)
print(f"\nTrain Batches / Epoch: {TRAIN_BATCHES_PER_EPOCH}")

val_dataset = HDF5Dataset(
    path=DATA_DIR / f"val_{SPLIT}.hdf5",
    resample=False,
)
val_one_hot_redshifts = val_dataset.get_one_hot_redshifts()
VAL_BATCHES_PER_EPOCH = (
    10  # closest(len(val_one_hot_redshifts), TEST_BATCH_SIZE) // TEST_BATCH_SIZE
)
print(f"Val Batches / Epoch: {VAL_BATCHES_PER_EPOCH}\n")


trainloader = torch.utils.data.DataLoader(
    train_dataset,
    sampler=StratifiedBatchSampler(
        train_one_hot_redshifts,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    ),
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
)

valloader = torch.utils.data.DataLoader(
    val_dataset,
    sampler=StratifiedBatchSampler(
        val_one_hot_redshifts,
        batch_size=TEST_BATCH_SIZE,
        shuffle=True,
    ),
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
)

# Define alpha

z = train_one_hot_redshifts.astype(bool)
N_UNSUP = np.count_nonzero(~z)
N_SUP = np.count_nonzero(z)
ALPHA = N_UNSUP / N_SUP
print(f"Unsupervised: {N_UNSUP}, Supervised: {N_SUP}, alpha: {ALPHA}")

# Define the model

print(
    "\n--------------------------------- DEFINING MODEL ---------------------------------\n"
)

predictor_key, encoder_key, decoder_key, rng_key = jax.random.split(rng_key, 4)

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

ssvae, state = eqx.nn.make_with_state(nn.SSVAE)(
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

# Train the model

print(
    "\n--------------------------------- TRAINING MODEL ---------------------------------\n"
)

train_key, rng_key = jax.random.split(rng_key, 2)

optim = optax.adamw(LEARNING_RATE)

trained_ssvae, state, train_losses, val_losses, train_auxes, val_auxes = train(
    rng_key=train_key,
    model=ssvae,
    state=state,
    trainloader=trainloader,
    valloader=valloader,
    optim=optim,
    loss_fn=ssvae_loss,
    epochs=EPOCHS,
    train_batches_per_epoch=TRAIN_BATCHES_PER_EPOCH,
    val_batches_per_epoch=VAL_BATCHES_PER_EPOCH,
    validate_every_n=EVAL_EVERY_N,
    checkpoint_dir=SAVE_DIR,
    early_stopping=USE_EARLY_STOPPING,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    filter_spec=filter_spec,
    loss_kwargs={
        "alpha": ALPHA,
        "missing_target_value": MISSING_TARGET_VALUE,
    },
)

# Save the model

print(
    "\n--------------------------------- SAVING MODEL ---------------------------------\n"
)

save(SAVE_DIR / "svae.pkl", trained_ssvae)

train_losses = jnp.asarray(train_losses)
val_losses = jnp.asarray(val_losses)

train_auxes = jnp.asarray(train_auxes)
val_auxes = jnp.asarray(val_auxes)

np.save(SAVE_DIR / "train_losses.npy", train_losses)
np.save(SAVE_DIR / "val_losses.npy", val_losses)
np.save(SAVE_DIR / "train_auxes.npy", train_auxes)
np.save(SAVE_DIR / "val_auxes.npy", val_auxes)

print(
    "\n--------------------------------- PLOTTING LOSSES ---------------------------------\n"
)

train_epochs = np.arange(1, EPOCHS + 1, 1)
val_epochs = np.arange(1, EPOCHS + 1, EVAL_EVERY_N)

fig, ax = plt.subplots(ncols=2, figsize=(16, 8), sharex=True, sharey=False)

ax[0].plot(train_epochs, train_losses, label="Train Loss", color=colors[0])
ax[0].plot(val_epochs, val_losses, label="Val Loss", color=colors[1])
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()

ax[1].plot(
    train_epochs,
    train_auxes[:, 0],
    label="Train Unsupervised Loss",
    alpha=0.5,
    color=colors[0],
)
ax[1].plot(
    train_epochs,
    train_auxes[:, 1],
    label="Train Supervised Loss",
    alpha=0.5,
    color=colors[1],
)
ax[1].plot(
    train_epochs,
    train_auxes[:, 2],
    label="Train Target Loss",
    alpha=0.5,
    color=colors[2],
)

ax[1].plot(
    val_epochs,
    val_auxes[:, 0],
    label="Test Unsupervised Loss",
    alpha=1,
    color=colors[0],
)
ax[1].plot(
    val_epochs, val_auxes[:, 1], label="Test Supervised Loss", alpha=1, color=colors[1]
)
ax[1].plot(
    val_epochs, val_auxes[:, 2], label="Test Target Loss", alpha=1, color=colors[2]
)

ax[1].legend()
ax[1].set_xlabel("Epoch")

fig.tight_layout()
fig.savefig(SAVE_DIR / "losses.png")

print(
    "\n--------------------------------- TEST SET PREDICTIONS ---------------------------------\n"
)

test_dataset = HDF5Dataset(
    path=DATA_DIR / f"test_{SPLIT}.hdf5",
    resample=False,
)
test_one_hot_redshifts = test_dataset.get_one_hot_redshifts()

is_spec = test_one_hot_redshifts.astype(bool)
indeces = np.arange(len(is_spec))[is_spec][::10]

x_true, y_true, info = test_dataset[indeces]

inference_ssvae = eqx.nn.inference_mode(trained_ssvae)

eval_key, rng_key = jax.random.split(rng_key, 2)
inference_ssvae = eqx.Partial(inference_ssvae, input_state=state, rng_key=eval_key)


y, z, x_hat, y_pars, z_pars, x_pars = evaluate(inference_ssvae, x_true)


y_means = y_pars[0].squeeze()
y_stds = y_pars[1].squeeze()
z_means, z_std = transform_redshift(
    y_means, y_stds, test_dataset.log10_redshift_mean, test_dataset.log10_redshift_std
)

print(z_means.shape)
print(z_std.shape)

z_true, _ = transform_redshift(
    y_true, y_stds, test_dataset.log10_redshift_mean, test_dataset.log10_redshift_std
)

print(z_true.shape)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))

# ax.errorbar(z_true, z_means, yerr=z_std, fmt="o")
ax.scatter(z_true, z_means, s=1, alpha=0.5)
xlim, ylim = ax.get_xlim(), ax.get_ylim()
line = np.linspace(*xlim, 100)
ax.plot(line, line, "--", color="black")
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel("Spec Z", fontsize=16)
ax.set_ylabel("Photo Z", fontsize=16)

fig.tight_layout()

fig.savefig(SAVE_DIR / "z_comparison.png")
