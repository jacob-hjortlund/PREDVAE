from equinox import Module
from jax.typing import ArrayLike


class Frozen(Module):
    layer: ArrayLike

    def __init__(self, layer: ArrayLike, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
