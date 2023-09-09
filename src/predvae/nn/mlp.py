from collections.abc import Callable
from typing import (
    Literal,
    Optional,
    Union,
)

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from jax.experimental import checkify
from jax.scipy import stats as jstats
from jaxtyping import Array, PRNGKeyArray

_identity = lambda x: x


class MLP(eqx.Module):
    """
    Standard Multi-Layer Perceptron (MLP)
    """

    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
    width_size: Array = eqx.field(static=True, converter=jnp.asarray)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: Union[int, Literal["scalar"]],
        out_size: Union[int, Literal["scalar"]],
        width_size: Array,
        depth: int,
        key: PRNGKeyArray,
        activation: Callable = jnn.relu,
        final_activation: Callable = _identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        **kwargs,
    ):
        """
        Args:
            in_size (Union[int, str('scalar')]): The input size. The input to
            the module should be a vector of shape (in_features,).
            out_size (Union[int, str('scalar')]): The output from the module
            will be a vector of shape (out_features,).
            width_size (Array): The size of each hidden layer. Should be a
            vector of shape (depth,).
            depth (int): The number of hidden layers, including the output layer.
            key (PRNGKeyArray): A `jax.random.PRNGKey` used to provide randomness
            for parameter initialisation.
            activation (Callable, optional): The activation function after each
            hidden layer.Defaults to jnn.relu.
            final_activation (Callable, optional): The activation function after
            the output layer. Defaults to the identity.
            use_bias (bool, optional): Whether to add a bias to internal layers.
            Defaults to True.
            use_final_bias (bool, optional): Whether to add a bias to the output
            layer.Defaults to True.

        Note that `in_size` also supports the string `"scalar"` as a special value.
        In this case the input to the module should be of shape `()`.

        Likewise `out_size` can also be a string `"scalar"`, in which case the
        output from the module will have shape `()`.
        """

        super().__init__(**kwargs)
        keys = jr.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(
                eqx.nn.Linear(in_size, out_size, use_bias=use_final_bias, keys=keys[0])
            )
        else:
            layers.append(
                eqx.nn.Linear(in_size, width_size[0], use_bias=use_bias, keys=keys[0])
            )
            for i in range(depth - 1):
                layers.append(
                    eqx.nn.Linear(
                        width_size[i],
                        width_size[i + 1],
                        use_bias=use_bias,
                        keys=keys[i + 1],
                    )
                )
            layers.append(
                eqx.nn.Linear(
                    width_size[-1], out_size, use_bias=use_final_bias, keys=keys[-1]
                )
            )
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    def __call__(self, x: Array) -> Array:
        """
        Args:
            x (Array): A JAX array with shape (in_size,) or, in the case of in_size='scalar', ().

        Returns:
            Array: A JAX array with shape (out_size,) or, in the case of out_size='scalar', ().
        """

        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)

        return x
