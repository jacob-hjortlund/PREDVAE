from collections.abc import Callable
from typing import (
    Literal,
    Optional,
    Union,
    Any,
)

import jax
import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from jax.experimental import checkify
from jax.scipy import stats as jstats
from jaxtyping import Array, PRNGKeyArray
from jax.typing import ArrayLike

_identity = lambda x: x


class StatefulIdentity(eqx.Module):
    layer: eqx.Module

    def __init__(self, layer: eqx.Module, *args: Any, **kwargs: Any):
        self.layer = layer

    @jax.named_scope("predvae.nn.StatefulIdentity")
    def __call__(
        self, x: ArrayLike, state: eqx.nn.State, *, key: Optional[PRNGKeyArray] = None
    ) -> ArrayLike:

        out = self.layer(x)
        return out, state


class SpectralNormedLinear(eqx.Module):

    spectral_linear: Union[eqx.nn.SpectralNorm[eqx.nn.Linear], StatefulIdentity]
    use_spectral_norm: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        use_spectral_norm: bool = True,
        num_power_iterations: int = 1,
        *,
        key: PRNGKeyArray,
    ):

        linear_key, spectral_norm_key = jr.split(key, 2)
        self.use_spectral_norm = use_spectral_norm
        if use_spectral_norm:
            self.spectral_linear = eqx.nn.SpectralNorm(
                eqx.nn.Linear(
                    in_features, out_features, use_bias=use_bias, key=linear_key
                ),
                weight_name="weight",
                key=spectral_norm_key,
                num_power_iterations=num_power_iterations,
            )
        else:
            linear = eqx.nn.Linear(
                in_features, out_features, use_bias=use_bias, key=linear_key
            )
            self.spectral_linear = StatefulIdentity(linear)

    def __call__(
        self, x: Array, input_state: eqx.nn.State
    ) -> tuple[Array, eqx.nn.State]:
        x, output_state = self.spectral_linear(x, input_state)

        return x, output_state


class MLP(eqx.Module):
    """
    Standard Multi-Layer Perceptron (MLP)
    """

    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    use_spectral_norm: bool = eqx.field(static=True)
    use_final_spectral_norm: bool = eqx.field(static=True)
    num_power_iterations: int = eqx.field(static=True)
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
        use_spectral_norm: bool = False,
        use_final_spectral_norm: bool = False,
        num_power_iterations: int = 1,
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
                SpectralNormedLinear(
                    in_size,
                    out_size,
                    use_bias=use_final_bias,
                    key=keys[0],
                    use_spectral_norm=use_spectral_norm,
                    num_power_iterations=num_power_iterations,
                )
            )
        else:
            layers.append(
                SpectralNormedLinear(
                    in_size,
                    width_size[0],
                    use_bias=use_bias,
                    key=keys[0],
                    use_spectral_norm=use_spectral_norm,
                    num_power_iterations=num_power_iterations,
                )
            )
            for i in range(depth - 1):
                layers.append(
                    SpectralNormedLinear(
                        width_size[i],
                        width_size[i + 1],
                        use_bias=use_bias,
                        key=keys[i + 1],
                        use_spectral_norm=use_spectral_norm,
                        num_power_iterations=num_power_iterations,
                    )
                )
            layers.append(
                SpectralNormedLinear(
                    width_size[-1],
                    out_size,
                    use_bias=use_final_bias,
                    key=keys[-1],
                    use_spectral_norm=use_final_spectral_norm,
                    num_power_iterations=num_power_iterations,
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
        self.use_spectral_norm = use_spectral_norm
        self.use_final_spectral_norm = use_final_spectral_norm
        self.num_power_iterations = num_power_iterations

    def __call__(self, x: Array, state: eqx.nn.State) -> tuple[Array, eqx.nn.State]:
        """
        Args:
            x (Array): A JAX array with shape (in_size,) or, in the case of in_size='scalar', ().

        Returns:
            Array: A JAX array with shape (out_size,) or, in the case of out_size='scalar', ().
        """

        for layer in self.layers[:-1]:
            x, state = layer(x, state)
            x = self.activation(x)
        x, output_state = self.layers[-1](x, state)
        x = self.final_activation(x)

        return x, output_state
