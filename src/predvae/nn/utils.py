import jax
import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from .vae import InputLayer
from jax.tree_util import tree_map, tree_leaves


def freeze_prior(model, space="latent", filter_spec=None, inverse=False):

    get_prior = lambda model: getattr(model, space + "_prior")
    get_prior_params = lambda model: [get_prior(model).mu, get_prior(model).log_sigma]
    replace = [inverse] * len(get_prior_params(model))

    if filter_spec is None:
        filter_spec = tree_map(lambda _: True, model)

    filter_spec = eqx.tree_at(
        get_prior_params,
        filter_spec,
        replace=replace,
    )

    return filter_spec


def freeze_submodule_inputs(
    model, submodule, freeze_x=True, freeze_y=True, filter_spec=None, inverse=False
):

    get_submodule = lambda model: getattr(model, submodule + "_input_layer")
    is_input_layer = lambda layer: isinstance(layer, InputLayer)
    _get_weights = lambda model, input: [
        getattr(layer, f"{input}_weight")
        for layer in tree_leaves(get_submodule(model), is_leaf=is_input_layer)
        if is_input_layer(layer)
    ]
    get_weights = lambda model: (_get_weights(model, "x") if freeze_x else []) + (
        _get_weights(model, "y") if freeze_y else []
    )

    replace = [inverse] * len(get_weights(model))

    if filter_spec is None:
        filter_spec = tree_map(lambda _: True, model)

    filter_spec = eqx.tree_at(
        get_weights,
        filter_spec,
        replace=replace,
    )

    return filter_spec


def freeze_submodule(model, submodule: str, filter_spec=None, inverse=False):

    get_submodule_mlp_layers = lambda model: getattr(model, submodule).mlp.layers
    get_layer_weights = lambda model: [
        layer.spectral_linear.layer.weight for layer in get_submodule_mlp_layers(model)
    ]
    get_layer_biases = lambda model: [
        layer.spectral_linear.layer.bias for layer in get_submodule_mlp_layers(model)
    ]
    get_submodule_params = lambda model: get_layer_weights(model) + get_layer_biases(
        model
    )

    layer_params = get_submodule_params(model)
    n_params = len(layer_params)
    replace = [inverse] * n_params

    if filter_spec is None:
        filter_spec = tree_map(lambda _: True, model)

    filter_spec = eqx.tree_at(
        get_submodule_params,
        filter_spec,
        replace=replace,
    )

    return filter_spec


def _init_fn(weight, key):
    shape = weight.shape
    lim = 1 / math.sqrt(shape[-1])
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def init_submodule_inputs(
    model, submodule, key, init_x=True, init_y=True, init_value=None
):

    if init_value is None:
        init_fn = _init_fn
    else:
        init_fn = lambda weight, key: jnp.ones_like(weight) * init_value

    get_submodule = lambda model: getattr(model, submodule + "_input_layer")
    is_input_layer = lambda layer: isinstance(layer, InputLayer)
    _get_weights = lambda model, input: [
        getattr(layer, f"{input}_weight")
        for layer in tree_leaves(get_submodule(model), is_leaf=is_input_layer)
        if is_input_layer(layer)
    ]
    get_weights = lambda model: (_get_weights(model, "x") if init_x else []) + (
        _get_weights(model, "y") if init_y else []
    )
    weights = get_weights(model)
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jr.split(key, len(weights)))
    ]

    updated_model = eqx.tree_at(
        get_weights,
        model,
        replace=new_weights,
    )

    return updated_model


def init_submodule(model, submodule: str, key, init_value=None):

    if init_value is None:
        init_fn = _init_fn
    else:
        init_fn = lambda weight, key: jnp.ones_like(weight) * init_value

    get_submodule_mlp_layers = lambda model: getattr(model, submodule).mlp.layers
    get_submodule_weights = lambda model: [
        layer.spectral_linear.layer.weight for layer in get_submodule_mlp_layers(model)
    ]
    get_submodule_biases = lambda model: [
        layer.spectral_linear.layer.bias for layer in get_submodule_mlp_layers(model)
    ]
    get_submodule_params = lambda model: get_submodule_weights(
        model
    ) + get_submodule_biases(model)

    submodule_params = get_submodule_params(model)
    new_params = [
        init_fn(param, subkey)
        for param, subkey in zip(submodule_params, jr.split(key, len(submodule_params)))
    ]

    updated_model = eqx.tree_at(
        get_submodule_params,
        model,
        replace=new_params,
    )

    return updated_model


def set_submodule_inference_mode(model, submodule: str, value: bool):

    get_submodule = lambda model: getattr(model, submodule)
    has_inference = lambda leaf: hasattr(leaf, "inference")
    where = lambda model: [
        x.inference
        for x in tree_leaves(get_submodule(model), is_leaf=has_inference)
        if has_inference(x)
    ]

    updated_model = eqx.tree_at(
        where,
        model,
        replace_fn=lambda _: value,
    )

    return updated_model
