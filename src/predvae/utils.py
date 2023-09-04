import jax.numpy as jnp

from jax import Array
from jax.typing import ArrayLike
from typing import Sequence


def merge_cond_shapes(shapes: Sequence):
    """Merges shapes (tuples of ints or None) used in bijections and distributions.
    Returns None if all shapes are None, otherwise checks the shapes match, and returns
    the shape.
    """
    if len(shapes) == 0:
        raise ValueError("No shapes have been provided.")
    if all(s is None for s in shapes):
        return None
    shapes = [s for s in shapes if s is not None]
    if all(s == shapes[0] for s in shapes):
        return shapes[0]
    raise ValueError("The shapes do not match.")


def _get_ufunc_signature(in_shapes, out_shapes):
    """Convert a sequence of in_shapes and out_shapes to a universal function signature.

    Example:
        >>> _get_ufunc_signature([(3,),(2,3)], [()])
        "(3),(2,3)->()"
    """

    def _shapes_to_str(shapes):
        result = [str(s) if len(s) != 1 else str(s).replace(",", "") for s in shapes]
        return ",".join(result).replace(" ", "")

    in_shapes_str = _shapes_to_str(in_shapes)
    out_shapes_str = _shapes_to_str(out_shapes)
    return f"{in_shapes_str}->{out_shapes_str}"


def arraylike_to_array(arr, err_name: str = "input", **kwargs) -> Array:
    """Combines jnp.asarray, with an isinstance(arr, ArrayLike) check. This
    allows inputs to be jax.numpy arrays, numpy arrays, python built in numeric types
    (float, int) etc, but does not allow list or tuple inputs (which are not arraylike
    and can introduce overhead and confusing behaviour in certain cases).

    Args:
        arr: Arraylike input to convert to a jax array.
        err_name (str, optional): Name of the input in the error message. Defaults to
            "input".
        **kwargs: Key word arguments passed to jnp.asarray.
    """
    if not isinstance(arr, ArrayLike):
        raise ValueError(
            f"Expected {err_name} to be arraylike; got {type(arr).__name__}."
        )
    return jnp.asarray(arr, **kwargs)
