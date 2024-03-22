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


def freeze_target_inputs(model, filter_spec=None, inverse=False):

    is_input_layer = lambda layer: isinstance(layer, InputLayer)
    get_target_weights = lambda model: [
        layer.y_weight
        for layer in tree_leaves(model, is_leaf=is_input_layer)
        if is_input_layer(layer)
    ]
    replace = [inverse] * len(get_target_weights(model))

    if filter_spec is None:
        filter_spec = tree_map(lambda _: True, model)

    filter_spec = eqx.tree_at(
        get_target_weights,
        filter_spec,
        replace=replace,
    )

    return filter_spec


def init_target_inputs(model, key, init_value=None):

    if init_value is None:

        def init_fn(weight, key):
            out, in_ = weight.shape
            lim = 1 / math.sqrt(in_)
            return jax.random.uniform(key, (out, in_), minval=-lim, maxval=lim)

    else:
        init_fn = lambda weight, key: jnp.ones_like(weight) * init_value

    is_input_layer = lambda layer: isinstance(layer, InputLayer)
    get_target_weights = lambda model: [
        layer.y_weight
        for layer in tree_leaves(model, is_leaf=is_input_layer)
        if is_input_layer(layer)
    ]
    weights = get_target_weights(model)
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jr.split(key, len(weights)))
    ]

    updated_model = eqx.tree_at(
        get_target_weights,
        model,
        replace=new_weights,
    )

    return updated_model


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
