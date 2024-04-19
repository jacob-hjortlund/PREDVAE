import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optimistix as optx
from jax.scipy.stats import norm


def model_call(model, input_state, x, y, rng_key):

    (y, z, x_hat, y_pars, z_pars, x_pars), output_state = model(
        x, y, input_state, rng_key
    )

    return y, z, x_hat, y_pars, z_pars, x_pars, output_state


vectorized_model_call = jax.vmap(
    model_call, in_axes=(None, None, 0, 0, None), out_axes=(0, 0, 0, 0, 0, 0, None)
)


def latent_space_evaluator(
    model, dataset_iterator, model_state, dataloader_state, rng_key
):

    @jax.jit
    def _call(
        model_state,
        dataloader_state,
        rng_key,
    ):

        resampling_key, step_key = jr.split(rng_key, 2)

        (x, y, dataloader_state, reset_condition) = dataset_iterator(
            dataloader_state, resampling_key
        )
        end_of_split = jnp.all(reset_condition)
        (_, _, _, _, z_pars, _, model_state) = vectorized_model_call(
            model, model_state, x, y, step_key
        )

        return z_pars, model_state, dataloader_state, end_of_split

    latent_pars = []
    end_of_split = False
    n_batches = 0

    while not end_of_split:

        step_key, rng_key = jr.split(rng_key)
        z_pars, model_state, dataloader_state, end_of_split = _call(
            model_state, dataloader_state, step_key
        )
        if end_of_split:
            break
        latent_pars.append(z_pars)
        n_batches += 1
        print(f"Batch {n_batches} completed")

    latent_means = jnp.concatenate([pars[0] for pars in latent_pars], axis=0).squeeze()
    latent_log_stds = jnp.concatenate(
        [pars[1] for pars in latent_pars], axis=0
    ).squeeze()

    return latent_means, latent_log_stds, model_state, dataloader_state


def gaussian_mixture_cdf(x, means, stds, weights):
    return jnp.sum(weights * norm.cdf(x, means, stds), axis=-1)


def _target_inverse_cdf(x, q, means, stds, weights):
    return q - gaussian_mixture_cdf(x, means, stds, weights)


def target_inverse_cdf(x, args):
    return _target_inverse_cdf(x, *args)


def mixture_inverse_cdf(q, means, stds, weights):

    mean = jnp.sum(weights * means, axis=-1)
    std = jnp.sqrt(
        jnp.sum(weights * stds**2, axis=-1)
        + jnp.sum(weights * means**2, axis=-1)
        - jnp.sum(weights * means, axis=-1) ** 2
    )
    lower_bound = mean - 20 * std
    upper_bound = mean + 20 * std
    options = {
        "lower": lower_bound,
        "upper": upper_bound,
    }

    solver = optx.Bisection(rtol=1e-3, atol=1e-3)
    args = (q, means, stds, weights)
    sol = optx.root_find(
        target_inverse_cdf,
        solver=solver,
        y0=mean,
        options=options,
        args=args,
        max_steps=jnp.iinfo(jnp.int64).max,
    )

    return sol


def model_call_and_ppf(q, x, y, model, model_state, rng_key):

    (y, z, x_hat, y_pars, z_pars, x_pars), output_state = model(
        x, y, model_state, rng_key
    )

    logits, means, log_stds = y_pars
    weights = jax.nn.softmax(logits)
    stds = jnp.exp(log_stds)

    ppf_values = mixture_inverse_cdf(q, means, stds, weights).value
    y_pars = (weights, means, stds)

    return y_pars, z_pars, ppf_values, output_state


input_mapped_model_call_and_ppf = jax.vmap(
    model_call_and_ppf, in_axes=(None, 0, 0, None, None, None), out_axes=(0, 0, 0, None)
)
percentile_mapped_model_call_and_ppf = jax.vmap(
    input_mapped_model_call_and_ppf,
    in_axes=(0, None, None, None, None, None),
    out_axes=(None, None, 0, None),
)


def full_model_evaluator(
    model,
    dataset_iterator,
    model_state,
    dataloader_state,
    rng_key,
    min=0.01,
    max=0.99,
    n=100,
):

    q = jnp.linspace(min, max, n)

    @jax.jit
    def _call(
        model_state,
        dataloader_state,
        rng_key,
    ):

        resampling_key, step_key = jr.split(rng_key, 2)

        (x, y, dataloader_state, reset_condition) = dataset_iterator(
            dataloader_state, resampling_key
        )
        end_of_split = jnp.all(reset_condition)
        y_pars, z_pars, ppf_values, model_state = percentile_mapped_model_call_and_ppf(
            q, x, y, model, model_state, step_key
        )

        return (
            y,
            y_pars,
            z_pars,
            ppf_values,
            model_state,
            dataloader_state,
            end_of_split,
        )

    latent_pars = []
    target_values = []
    target_pars = []
    ppfs = []
    end_of_split = False
    n_batches = 0

    while not end_of_split:

        step_key, rng_key = jr.split(rng_key)
        y, y_pars, z_pars, ppf_values, model_state, dataloader_state, end_of_split = (
            _call(model_state, dataloader_state, step_key)
        )
        if end_of_split:
            break
        latent_pars.append(z_pars)
        target_values.append(y)
        target_pars.append(y_pars)
        ppfs.append(ppf_values)

        n_batches += 1
        print(f"Batch {n_batches} completed")

    latent_means = jnp.concatenate([pars[0] for pars in latent_pars], axis=0).squeeze()
    latent_log_stds = jnp.concatenate(
        [pars[1] for pars in latent_pars], axis=0
    ).squeeze()
    target_weights = jnp.concatenate(
        [pars[0] for pars in target_pars], axis=0
    ).squeeze()
    target_means = jnp.concatenate([pars[1] for pars in target_pars], axis=0).squeeze()
    target_stds = jnp.concatenate([pars[2] for pars in target_pars], axis=0).squeeze()
    ppf_values = jnp.concatenate(ppfs, axis=-1).squeeze()
    target_values = jnp.concatenate(target_values, axis=0).squeeze()
    ppf_fractions = (
        jnp.sum(ppf_values > target_values, axis=-1) / target_values.shape[0]
    )

    return (
        target_values,
        latent_means,
        latent_log_stds,
        target_weights,
        target_means,
        target_stds,
        ppf_values,
        ppf_fractions,
        model_state,
        dataloader_state,
    )
