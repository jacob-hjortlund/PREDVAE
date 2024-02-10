import equinox as eqx

from equinox import Module
from src.predvae.nn import Frozen
from jax.tree_util import tree_map, tree_leaves


def is_frozen(x):
    return isinstance(x, Frozen)


def get_frozen(model: Module):
    return [x.layer for x in tree_leaves(model, is_leaf=is_frozen) if is_frozen(x)]


def freeze_parameters(model: Module):

    filter_spec = tree_map(lambda x: True, model)
    filter_spec = eqx.tree_at(
        get_frozen(model), filter_spec, replace_fn=lambda x: False
    )

    free_params, frozen_params = eqx.partition(model, filter_spec)

    return free_params, frozen_params
