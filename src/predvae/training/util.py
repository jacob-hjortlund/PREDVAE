import equinox as eqx

from jax.lax import cond
from equinox import Module
from src.predvae.nn import Frozen
from collections.abc import Callable
from jax.tree_util import tree_map, tree_leaves


def is_frozen(x):
    return isinstance(x, Frozen)


def get_frozen(model: Module):
    return [x.layer for x in tree_leaves(model, is_leaf=is_frozen) if is_frozen(x)]


def freeze_parameters(model: Module):

    filter_spec = tree_map(lambda x: True, model)
    filter_spec = eqx.tree_at(get_frozen, filter_spec, replace_fn=lambda x: False)

    free_params, frozen_params = eqx.partition(model, filter_spec)

    return free_params, frozen_params


def filter_cond(pred, true_f: Callable, false_f: Callable, func_args: tuple):
    """Same as lax.cond, but allows to return eqx.Module"""
    dynamic_true, static_true = eqx.partition(true_f(*func_args), eqx.is_array)
    dynamic_false, static_false = eqx.partition(false_f(*func_args), eqx.is_array)

    static_part = eqx.error_if(
        static_true,
        static_true != static_false,
        "Filtered conditional arguments should have the same static part",
    )

    dynamic_part = cond(pred, lambda *_: dynamic_true, lambda *_: dynamic_false)
    return eqx.combine(dynamic_part, static_part)
