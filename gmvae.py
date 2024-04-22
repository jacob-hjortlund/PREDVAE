import os

# n_cpu = os.cpu_count()
# print(f"Number of CPUs available: {n_cpu}")
# env_flag = f"--xla_force_host_platform_device_count={n_cpu}"
# os.environ["XLA_FLAGS"] = env_flag

import jax

jax.config.update("jax_enable_x64", True)
print(f"JAX backend: {jax.devices()}")

import time
import optax
import hydra

import torch  # https://pytorch.org
import torchvision  # https://pytorch.org

import numpy as np
import pandas as pd
import seaborn as sns
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import src.predvae.nn as nn
import src.predvae.data as data
import matplotlib.pyplot as plt
import src.predvae.training as training

from pathlib import Path
from functools import partial
from jax.typing import ArrayLike
from jax.tree_util import tree_map

COLORS = sns.color_palette("colorblind")
SEED = 5678
RNG_KEY = jax.random.PRNGKey(SEED)

# Training parameters
BATCH_SIZE = 64
PEAK_LR = 5e-3
END_LR = 1e-5
EPOCHS = 10
WARMUP = 1
BATCHES_PER_EPOCH = 10

# Model parameters
INPUT_SHAPE = 784
LATENT_DIM = 64
MIXTURE_COMPONENTS = 10
LAYERS = [512, 512]
USE_SPEC_NORM = False

# Dirs
SAVE_DIR = Path("./gmvae_results")
SAVE_DIR.mkdir(exist_ok=True)


#################################################################################################################
########################################## Data Loading #########################################################
#################################################################################################################

normalise_data = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: np.where(x > 0.5, 1.0, 0.0).squeeze()),
        torchvision.transforms.Lambda(lambda x: x.flatten()),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=True,
    download=True,
    transform=normalise_data,
)
test_dataset = torchvision.datasets.MNIST(
    "MNIST",
    train=False,
    download=True,
    transform=normalise_data,
)
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

#################################################################################################################
########################################## Model Definition #####################################################
#################################################################################################################

(
    classifier_key,
    encoder_input_key,
    encoder_key,
    decoder_input_key,
    decoder_key,
    latent_prior_key,
    RNG_KEY,
) = jr.split(RNG_KEY, 7)

classifier = nn.CategoricalCoder(
    input_size=INPUT_SHAPE,
    output_size=MIXTURE_COMPONENTS,
    width=LAYERS,
    depth=len(LAYERS),
    activation=jax.nn.leaky_relu,
    key=classifier_key,
    use_spectral_norm=True,
    use_final_spectral_norm=False,
    num_power_iterations=5,
)

encoder_input_layer = nn.InputLayer(
    x_features=INPUT_SHAPE,
    y_features=MIXTURE_COMPONENTS,
    out_features=INPUT_SHAPE + MIXTURE_COMPONENTS,
    key=encoder_input_key,
)

encoder = nn.GaussianCoder(
    input_size=INPUT_SHAPE + MIXTURE_COMPONENTS,
    output_size=LATENT_DIM,
    width=LAYERS,
    depth=len(LAYERS),
    activation=jax.nn.leaky_relu,
    key=encoder_key,
    use_spectral_norm=True,
    use_final_spectral_norm=False,
    num_power_iterations=5,
)

decoder_input_layer = nn.InputLayer(
    x_features=LATENT_DIM,
    y_features=MIXTURE_COMPONENTS,
    out_features=LATENT_DIM + MIXTURE_COMPONENTS,
    key=decoder_input_key,
)

decoder = nn.BernoulliCoder(
    input_size=LATENT_DIM + MIXTURE_COMPONENTS,
    output_size=INPUT_SHAPE,
    width=LAYERS,
    depth=len(LAYERS),
    activation=jax.nn.leaky_relu,
    key=decoder_key,
    use_spectral_norm=True,
    use_final_spectral_norm=False,
    num_power_iterations=5,
)

classifier_prior = nn.Categorical(
    jnp.log(jnp.ones(MIXTURE_COMPONENTS) / MIXTURE_COMPONENTS)
)
latent_prior = nn.GaussianMixture(
    input_size=MIXTURE_COMPONENTS, output_size=LATENT_DIM, key=latent_prior_key
)

gmvae, input_state = eqx.nn.make_with_state(nn.GMVAE)(
    classifier=classifier,
    encoder_input_layer=encoder_input_layer,
    encoder=encoder,
    decoder_input_layer=decoder_input_layer,
    decoder=decoder,
    classifier_prior=classifier_prior,
    latent_prior=latent_prior,
)

# Freeze classifier prior
filter_spec = tree_map(lambda _: True, gmvae)
get_prior_params = lambda model: [
    model.classifier_prior.logits,
]
filter_spec = eqx.tree_at(get_prior_params, filter_spec, replace=[False])

#################################################################################################################
########################################## Training #############################################################
#################################################################################################################

lr_schedule = optax.warmup_cosine_decay_schedule(
    END_LR,
    PEAK_LR,
    WARMUP,
    EPOCHS - WARMUP,
    PEAK_LR,
)
optimizer = optax.adam(learning_rate=lr_schedule)
optimizer_state = optimizer.init(eqx.filter(gmvae, eqx.is_array))


@eqx.filter_jit
def train_step(
    x: ArrayLike,
    y: ArrayLike,
    rng_key: ArrayLike,
    model: eqx.Module,
    input_state: eqx.nn.State,
    optimizer_state: optax.OptState,
):

    free_params, frozen_params = eqx.partition(model, filter_spec)
    (loss_value, (loss_aux, y_pred, output_state)), grads = eqx.filter_value_and_grad(
        training.unsupervised_clustering_loss, has_aux=True
    )(free_params, frozen_params, input_state, x, y, rng_key)

    updates, optimizer_state = optimizer.update(grads, optimizer_state, model)
    model = eqx.apply_updates(model, updates)

    return model, output_state, optimizer_state, loss_value, loss_aux, y_pred


@eqx.filter_jit
def val_step(
    x: ArrayLike,
    y: ArrayLike,
    rng_key: ArrayLike,
    model: eqx.Module,
    input_state: eqx.nn.State,
):

    free_params, frozen_params = eqx.partition(model, filter_spec)
    loss_value, (loss_aux, y_pred, output_state) = (
        training.unsupervised_clustering_loss(
            free_params, frozen_params, input_state, x, y, rng_key
        )
    )

    return output_state, loss_value, loss_aux, y_pred


def train_model(
    model: eqx.Module,
    input_state: eqx.nn.State,
    optimizer_state: optax.OptState,
    trainloader: torch.utils.data.DataLoader,
    valloader: torch.utils.data.DataLoader,
    epochs: int,
    rng_key: ArrayLike,
    batch_per_epoch: int = 0,
):

    train_loss = []
    val_loss = []
    train_aux = []
    val_aux = []
    train_acc = []
    val_acc = []

    for epoch in range(epochs):

        epoch_train_loss = []
        epoch_train_aux = []
        epoch_train_acc = []
        epoch_val_loss = []
        epoch_val_aux = []
        epoch_val_acc = []

        t0_train = time.time()
        for i, (x, y) in enumerate(trainloader):

            x = jnp.array(x.numpy())
            y = jnp.array(y.numpy())
            y_onehot = jax.nn.one_hot(y, MIXTURE_COMPONENTS)

            model, input_state, optimizer_state, loss_value, loss_aux, y_pred = (
                train_step(
                    x,
                    y_onehot,
                    jr.split(rng_key, 1)[0],
                    model,
                    input_state,
                    optimizer_state,
                )
            )

            acc = training.cluster_acc(y_pred, y)

            epoch_train_loss.append(loss_value)
            epoch_train_aux.append(loss_aux)
            epoch_train_acc.append(acc)

            if batch_per_epoch != 0 and i >= batch_per_epoch:
                break

        t_train = time.time() - t0_train

        t0_val = time.time()
        inference_model = eqx.nn.inference_mode(model)
        for i, (x, y) in enumerate(valloader):

            x = jnp.array(x.numpy())
            y = jnp.array(y.numpy())
            y_onehot = jax.nn.one_hot(y, MIXTURE_COMPONENTS)

            input_state, loss_value, loss_aux, y_pred = val_step(
                x, y_onehot, jr.split(rng_key, 1)[0], inference_model, input_state
            )

            acc = training.cluster_acc(y_pred, y)

            epoch_val_loss.append(loss_value)
            epoch_val_aux.append(loss_aux)
            epoch_val_acc.append(acc)

            if batch_per_epoch != 0 and i >= batch_per_epoch:
                break

        t_val = time.time() - t0_val

        epoch_train_loss = jnp.mean(jnp.array(epoch_train_loss), axis=0)
        epoch_train_aux = jnp.mean(jnp.array(epoch_train_aux), axis=0)
        epoch_train_acc = jnp.mean(jnp.array(epoch_train_acc), axis=0)
        epoch_val_loss = jnp.mean(jnp.array(epoch_val_loss), axis=0)
        epoch_val_aux = jnp.mean(jnp.array(epoch_val_aux), axis=0)
        epoch_val_acc = jnp.mean(jnp.array(epoch_val_acc), axis=0)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        train_aux.append(epoch_train_aux)
        val_aux.append(epoch_val_aux)
        train_acc.append(epoch_train_acc)
        val_acc.append(epoch_val_acc)

        epoch_string = (
            f"Epoch {epoch} - Train Time: {t_train:.2f} s - Val Time: {t_val:.2f} s - "
            + f"Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - "
            + f"Class KL: {epoch_train_aux[2]:.4f} - Latent KL: {epoch_train_aux[5]:.4f} - "
            + f"Rec. Loss: {-epoch_train_aux[-1]:.4f} - Train Acc.: {epoch_train_acc:.3f} - "
            + f"Val Acc.: {epoch_val_acc:.3f}"
        )

        print(epoch_string)

    train_loss = jnp.array(train_loss)
    val_loss = jnp.array(val_loss)
    train_aux = jnp.array(train_aux)
    val_aux = jnp.array(val_aux)
    train_acc = jnp.array(train_acc)
    val_acc = jnp.array(val_acc)

    return (
        model,
        input_state,
        optimizer_state,
        train_loss,
        val_loss,
        train_aux,
        val_aux,
        train_acc,
        val_acc,
    )


(
    trained_gmvae,
    input_state,
    optimizer_state,
    train_loss,
    val_loss,
    train_aux,
    val_aux,
    train_acc,
    val_acc,
) = train_model(
    gmvae,
    input_state,
    optimizer_state,
    trainloader,
    testloader,
    EPOCHS,
    RNG_KEY,
    BATCHES_PER_EPOCH,
)

# Save model
training.save(SAVE_DIR / "init_gmvae.pkl", gmvae)
training.save(SAVE_DIR / "gmvae.pkl", trained_gmvae)
training.save(SAVE_DIR / "model_state.pkl", input_state)

#################################################################################################################
############################################ Figures ############################################################
#################################################################################################################

epochs = np.arange(1, train_loss.shape[0] + 1)

fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(20, 5), sharex=True, sharey=False)
ax = ax.flatten()

ax[0].plot(epochs, train_loss, label="Train", color=COLORS[0])
ax[0].plot(epochs, val_loss, label="Val", color=COLORS[1])
ax[0].set_ylabel("Total Loss")
ax[0].set_xlabel("Epoch")
ax[0].legend()

ax[1].plot(epochs, train_aux[:, 2], label="Train", color=COLORS[0])
ax[1].plot(epochs, val_aux[:, 2], label="Val", color=COLORS[1])
ax[1].set_ylabel("Classifier KL")
ax[1].set_xlabel("Epoch")
ax[1].legend()

ax[2].plot(epochs, train_aux[:, 5], label="Train", color=COLORS[0])
ax[2].plot(epochs, val_aux[:, 5], label="Val", color=COLORS[1])
ax[2].set_ylabel("Latent KL")
ax[2].set_xlabel("Epoch")
ax[2].legend()

ax[3].plot(epochs, -train_aux[:, -1], label="Train", color=COLORS[0])
ax[3].plot(epochs, -val_aux[:, -1], label="Val", color=COLORS[1])
ax[3].set_ylabel("Reconstruction Loss")
ax[3].set_xlabel("Epoch")
ax[3].legend()

ax[4].plot(epochs, train_acc, label="Train", color=COLORS[0])
ax[4].plot(epochs, val_acc, label="Val", color=COLORS[1])
ax[4].set_ylabel("Accuracy")
ax[4].set_xlabel("Epoch")
ax[4].legend()

fig.tight_layout()
fig.savefig(SAVE_DIR / "training_curves.png", dpi=300)
