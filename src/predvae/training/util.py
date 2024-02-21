import equinox as eqx

from jax.lax import cond
from collections.abc import Callable


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
