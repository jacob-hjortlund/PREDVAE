import time
import optax
import torch

import numpy as np
import equinox as eqx
import jax.random as jr

from tqdm import tqdm
from equinox import Module
from jaxtyping import PyTree
from functools import partial
from jax.typing import ArrayLike
from jax.tree_util import tree_map
from collections.abc import Callable
from progress_table import ProgressTable


def save(filename, model):
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load(filename, model):
    with open(filename, "rb") as f:
        model = eqx.tree_deserialise_leaves(f, model)
    return model


def train(
    rng_key,
    model: Module,
    state: eqx.nn.State,
    trainloader: torch.utils.data.DataLoader,
    valloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    loss_fn: Callable,
    epochs: int,
    train_batches_per_epoch: int,
    val_batches_per_epoch: int,
    validate_every_n: int,
    loss_kwargs: dict = {},
    filter_spec: PyTree = None,
    checkpoint_dir: str = None,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
) -> Module:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    loss_fn = partial(loss_fn, **loss_kwargs)

    if filter_spec is None:
        filter_spec = tree_map(lambda x: True, model)

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_train_step(
        model: Module,
        input_state: eqx.nn.State,
        opt_state: PyTree,
        x: ArrayLike,
        y: ArrayLike,
        rng_key: ArrayLike,
    ):

        free_params, frozen_params = eqx.partition(model, filter_spec)
        (loss_value, (aux, output_state)), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(free_params, frozen_params, input_state, x, y, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, output_state, opt_state, loss_value, aux

    @eqx.filter_jit
    def make_eval_step(
        model: Module,
        input_state: eqx.nn.State,
        x: ArrayLike,
        y: ArrayLike,
        rng_key: ArrayLike,
    ):
        free_params, frozen_params = eqx.partition(model, filter_spec)
        loss_value, (aux, output_state) = loss_fn(
            free_params, frozen_params, input_state, x, y, rng_key
        )
        return loss_value, output_state, aux

    train_losses = []
    train_auxes = []
    val_losses = []
    val_auxes = []
    epoch_rates = []
    best_val_epoch = -1

    for epoch in range(epochs):
        t0 = time.time()

        batch_losses = []
        batch_auxes = []

        iterable_trainloader = iter(trainloader)
        iterable_valloader = iter(valloader)

        for _ in tqdm(
            range(train_batches_per_epoch),
            desc=f"Epoch {epoch + 1}",
            leave=False,
        ):

            batch_key, rng_key = jr.split(rng_key, 2)

            batch = next(iterable_trainloader)
            if len(batch) == 2:
                x, y = batch
            else:
                x, y, _ = batch
            x = x.numpy()
            y = y.numpy()
            y[np.isnan(y)] = -9999.0

            model, state, opt_state, train_loss, aux = make_train_step(
                model, state, opt_state, x, y, batch_key
            )
            batch_losses.append(train_loss)
            batch_auxes.append(aux)

        t1 = time.time()

        epoch_train_loss = np.mean(np.asarray(batch_losses), axis=0)
        train_losses.append(epoch_train_loss)

        epoch_train_aux = np.mean(np.asarray(batch_auxes), axis=0)
        train_auxes.append(epoch_train_aux)

        epoch_rate = 60**2 / (t1 - t0)
        epoch_rates.append(epoch_rate)

        if (epoch + 1) % validate_every_n == 0:
            val_loss = []
            val_aux = []

            for _ in range(val_batches_per_epoch):

                batch_key, rng_key = jr.split(rng_key, 2)

                batch = next(iterable_valloader)
                if len(batch) == 2:
                    x, y = batch
                else:
                    x, y, _ = batch
                x = x.numpy()
                y = y.numpy()
                y[np.isnan(y)] = -9999.0

                loss, state, aux = make_eval_step(model, state, x, y, batch_key)
                val_loss.append(loss)
                val_aux.append(aux)

            val_loss = np.mean(np.asarray(val_loss), axis=0)
            val_losses.append(val_loss)
            val_aux = np.mean(np.asarray(val_aux), axis=0)
            val_auxes.append(val_aux)

            print(
                f"Epoch {epoch + 1} | "
                f"Train Total Loss: {epoch_train_loss:.3f} | "
                f"Val Total Loss: {val_loss:.3f} | "
                f"Train Unsupervised Loss: {epoch_train_aux[0]:.3f} | "
                f"Train Supervised Loss: {epoch_train_aux[1]:.3f} | "
                f"Train Target Loss: {epoch_train_aux[2]:.3f} | "
                f"Val Unsupervised Loss: {val_aux[0]:.3f} | "
                f"Val Supervised Loss: {val_aux[1]:.3f} | "
                f"Val Target Loss: {val_aux[2]:.3f} | "
                f"Epoch Rate: {np.mean(epoch_rates):.2f} epochs/hr"
            )

            if len(val_losses) == 1 or val_loss < val_losses[best_val_epoch]:
                best_val_epoch = epoch
                if checkpoint_dir is not None:
                    save(f"{checkpoint_dir}/best_model.pkl", model)

        if early_stopping and epoch - best_val_epoch > early_stopping_patience:
            print(f"\nEarly stopping after {epoch} epochs")
            print(
                f"Best Epoch: {best_val_epoch + 1} | "
                f"Best Train Loss: {train_losses[best_val_epoch]:.3f} | "
                f"Best Val Loss: {val_losses[best_val_epoch]:.3f} | "
                f"Best Train Unsupervised Loss: {train_auxes[best_val_epoch][0]:.3f} | "
                f"Best Train Supervised Loss: {train_auxes[best_val_epoch][1]:.3f} | "
                f"Best Train Target Loss: {train_auxes[best_val_epoch][2]:.3f} | "
                f"Best Val Unsupervised Loss: {val_auxes[best_val_epoch][0]:.3f} | "
                f"Best Val Supervised Loss: {val_auxes[best_val_epoch][1]:.3f} | "
                f"Best Val Target Loss: {val_auxes[best_val_epoch][2]:.3f}"
            )

            if checkpoint_dir is not None:
                print(f"Saving final model to {checkpoint_dir}/final_model.pkl")
                save(f"{checkpoint_dir}/final_model.pkl", model)
                print(f"Setting model to best model from epoch {best_val_epoch + 1}")
                model = load(f"{checkpoint_dir}/best_model.pkl", model)

            break

    if checkpoint_dir is not None:

        print(
            f"Training complete, saving final model to {checkpoint_dir}/final_model.pkl"
        )
        save(f"{checkpoint_dir}/final_model.pkl", model)

        print(
            f"Best Epoch: {best_val_epoch + 1} | "
            f"Best Train Loss: {train_losses[best_val_epoch]:.3f} | "
            f"Best Val Loss: {val_losses[best_val_epoch]:.3f} | "
            f"Best Train Unsupervised Loss: {train_auxes[best_val_epoch][0]:.3f} | "
            f"Best Train Supervised Loss: {train_auxes[best_val_epoch][1]:.3f} | "
            f"Best Train Target Loss: {train_auxes[best_val_epoch][2]:.3f} | "
            f"Best Val Unsupervised Loss: {val_auxes[best_val_epoch][0]:.3f} | "
            f"Best Val Supervised Loss: {val_auxes[best_val_epoch][1]:.3f} | "
            f"Best Val Target Loss: {val_auxes[best_val_epoch][2]:.3f}"
        )
        print(f"Setting model to best model from epoch {best_val_epoch + 1}")

        model = load(f"{checkpoint_dir}/best_model.pkl", model)

    return model, state, train_losses, val_losses, train_auxes, val_auxes
