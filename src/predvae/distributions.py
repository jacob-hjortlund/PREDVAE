from abc import abstractmethod
from math import prod

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from jax import Array
from jax.experimental import checkify
from jax.scipy import stats as jstats
from jax.typing import ArrayLike

from predvae.utils import _get_ufunc_signature, arraylike_to_array, merge_cond_shapes


class Distribution(eqx.Module):
    """Distribution base class. Distributions all have an attribute ``shape``,
    denoting the shape of a single sample from the distribution. This corresponds to the
    ``batch_shape + event_shape`` in torch/numpyro distributions. Similarly, the
    ``cond_shape`` attribute denotes the shape of the conditioning variable.
    This attribute is None for unconditional distributions. For example

    .. doctest::

        >>> import jax.numpy as jnp
        >>> from flowjax.distributions import Normal
        >>> dist = Normal(jnp.zeros(2))
        >>> dist.shape
        (2,)
        >>> dist.cond_shape is None
        True

    Distributions are registered as jax PyTrees (as they are equinox modules), and as
    such they are compatible with normal jax operations.

    **Implementing a distribution**

        (1) Inherit from ``Distribution``.
        (2) Define attributes ``shape`` and ``cond_shape``. ``cond_shape`` should be
            ``None`` for unconditional distributions.
        (3) Define the ``_sample`` method, which samples a point with a shape of
            ``shape``, (given a conditioning variable with shape ``cond_shape`` for
            conditional distributions).
        (4) Define the ``_log_prob`` method, which evaluates the log probability,
            given an input of shape ``shape`` (and a conditioning variable with shape
            ``cond_shape`` for conditional distributions).

        The base class will handle defining more convenient log_prob and sample methods
        that support broadcasting and perform argument checks.

    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    @abstractmethod
    def _log_prob(self, x: Array, condition: Array | None = None) -> Array:
        """Evaluate the log probability of point x."""

    @abstractmethod
    def _sample(self, key: jr.KeyArray, condition: Array | None = None) -> Array:
        """Sample a point from the distribution."""

    def _sample_and_log_prob(
        self, key: jr.KeyArray, condition: Array | None = None
    ) -> tuple[Array, Array]:
        """Sample a point from the distribution, and return its log probability.
        Subclasses can reimplement this method in cases where more efficient methods
        exists (e.g. for :class:`Transformed` distributions).
        """
        x = self._sample(key, condition)
        log_prob = self._log_prob(x, condition)
        return x, log_prob

    def log_prob(self, x: ArrayLike, condition: ArrayLike | None = None) -> Array:
        """Evaluate the log probability. Uses numpy like broadcasting if additional
        leading dimensions are passed.

        Args:
            x (ArrayLike): Points at which to evaluate density.
            condition (ArrayLike | None): Conditioning variables. Defaults to None.

        Returns:
            Array: Jax array of log probabilities.
        """
        x = self._argcheck_and_cast_x(x)
        condition = self._argcheck_and_cast_condition(condition)
        if condition is None:
            sig = _get_ufunc_signature([self.shape], [()])
            exclude = frozenset([1])
        else:
            sig = _get_ufunc_signature([self.shape, self.cond_shape], [()])
            exclude = frozenset()

        lps = jnp.vectorize(self._log_prob, signature=sig, excluded=exclude)(
            x, condition
        )
        return jnp.where(jnp.isnan(lps), -jnp.inf, lps)  # type: ignore

    def sample(
        self,
        key: jr.KeyArray,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> Array:
        """Sample from the distribution. For unconditional distributions, the output
        will be of shape ``sample_shape + dist.shape``. For conditional distributions,
        a batch dimension in the condition is supported, and the output shape will be
        ``sample_shape + condition_batch_shape + dist.shape``. See the example for more
        information.

        Args:
            key (jr.KeyArray): Jax random key.
            condition (ArrayLike | None): Conditioning variables. Defaults to None.
            sample_shape (tuple[int, ...]): Sample shape. Defaults to ().

        Example:
            The below example shows the behaviour of sampling, for an unconditional
            and a conditional distribution.

            .. testsetup::

                from flowjax.distributions import StandardNormal
                import jax.random as jr
                import jax.numpy as jnp
                from flowjax.flows import CouplingFlow
                from flowjax.bijections import Affine
                # For a unconditional distribution:
                key = jr.PRNGKey(0)
                dist = StandardNormal((2,))
                # For a conditional distribution
                cond_dist = CouplingFlow(
                    key, StandardNormal((2,)), cond_dim=3, transformer=Affine()
                    )

            For an unconditional distribution:

            .. doctest::

                >>> dist.shape
                (2,)
                >>> samples = dist.sample(key, (10, ))
                >>> samples.shape
                (10, 2)

            For a conditional distribution:

            .. doctest::

                >>> cond_dist.shape
                (2,)
                >>> cond_dist.cond_shape
                (3,)
                >>> # Sample 10 times for a particular condition
                >>> samples = cond_dist.sample(key, (10,), condition=jnp.ones(3))
                >>> samples.shape
                (10, 2)
                >>> # Sampling, batching over a condition
                >>> samples = cond_dist.sample(key, condition=jnp.ones((5, 3)))
                >>> samples.shape
                (5, 2)
                >>> # Sample 10 times for each of 5 conditioning variables
                >>> samples = cond_dist.sample(key, (10,), condition=jnp.ones((5, 3)))
                >>> samples.shape
                (10, 5, 2)


        """
        condition = self._argcheck_and_cast_condition(condition)
        excluded, signature = self._vectorize_sample_args()
        keys = self._get_sample_keys(key, sample_shape, condition)
        return jnp.vectorize(self._sample, excluded=excluded, signature=signature)(
            keys, condition
        )  # type: ignore

    def sample_and_log_prob(
        self,
        key: jr.KeyArray,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ):
        """Sample the distribution and return the samples and corresponding log
        probabilities. For transformed distributions (especially flows), this will
        generally be more efficient than calling the methods seperately. Refer to the
        :py:meth:`~flowjax.distributions.Distribution.sample` documentation for more
        information.

        Args:
            key (jr.KeyArray): Jax random key.
            condition (ArrayLike | None): Conditioning variables. Defaults to None.
            sample_shape (tuple[int, ...]): Sample shape. Defaults to ().
        """
        condition = self._argcheck_and_cast_condition(condition)

        excluded, signature = self._vectorize_sample_args(sample_and_log_prob=True)
        keys = self._get_sample_keys(key, sample_shape, condition)

        return jnp.vectorize(
            self._sample_and_log_prob, excluded=excluded, signature=signature
        )(keys, condition)

    def _vectorize_sample_args(self, sample_and_log_prob=False):
        """Get the excluded arguments and ufunc signature for sample or
        sample_and_log_prob"""
        out_shapes = [self.shape, ()] if sample_and_log_prob else [self.shape]
        if self.cond_shape is None:
            excluded = frozenset([1])
            in_shapes = [(2,)]
        else:
            excluded = frozenset()
            in_shapes = [(2,), self.cond_shape]
        signature = _get_ufunc_signature(in_shapes, out_shapes)
        return excluded, signature

    def _get_sample_keys(self, key, sample_shape, condition):
        """Splits a key into an arrray of keys with shape
        sample_shape + leading_cond_shape + (2,)."""
        if self.cond_shape is None:
            key_shape = sample_shape
        else:
            leading_cond_shape = (
                condition.shape[: -len(self.cond_shape)]
                if len(self.cond_shape) > 0
                else condition.shape
            )
            key_shape = sample_shape + leading_cond_shape

        key_size = max(1, prod(key_shape))  # Still need 1 key for scalar sample
        keys = jnp.reshape(jr.split(key, key_size), key_shape + (2,))  # type: ignore
        return keys

    def _argcheck_and_cast_x(self, x) -> Array:
        x = arraylike_to_array(x, err_name="x")
        x_trailing = x.shape[-self.ndim :] if self.ndim > 0 else ()
        if x_trailing != self.shape:
            raise ValueError(
                "Expected trailing dimensions in x to match the distribution shape "
                f"{self.shape}; got x shape {x.shape}."
            )
        return x

    def _argcheck_and_cast_condition(self, condition) -> Array | None:
        if self.cond_shape is None:
            if condition is not None:
                raise ValueError(
                    "Expected condition to be None for unconditional distribution; "
                    f"got {condition}."
                )
            return None
        condition = arraylike_to_array(condition, err_name="condition")
        condition_trailing = (
            condition.shape[-len(self.cond_shape) :] if self.cond_ndim > 0 else ()
        )
        if condition_trailing != self.cond_shape:
            raise ValueError(
                "Expected trailing dimensions in condition to match cond_shape "
                f"{self.cond_shape}, but got condition shape {condition.shape}."
            )
        return condition

    @property
    def ndim(self):
        """The number of dimensions in the distribution (the length of the shape)."""
        return len(self.shape)

    @property
    def cond_ndim(self):
        """The number of dimensions of the conditioning variable (length of
        cond_shape)."""
        if self.cond_shape is not None:
            return len(self.cond_shape)
        return None
