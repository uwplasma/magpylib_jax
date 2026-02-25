"""Custom source stub for compatibility with Magpylib collections."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.base import BaseSource


class CustomSource(BaseSource):
    """User-defined source that can optionally supply a callable field function."""

    _source_type = "custom"

    def __init__(
        self,
        field_func: callable | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self.field_func = field_func
        super().__init__(
            position=position,
            orientation=orientation,
            style=style,
            style_label=style_label,
            **kwargs,
        )

    @property
    def volume(self) -> float:
        return 0.0

    def getB(self, *observers: ArrayLike, **_kwargs) -> jnp.ndarray:
        if callable(self.field_func):
            obs = observers[0] if len(observers) == 1 else list(observers)
            return jnp.asarray(self.field_func(obs), dtype=jnp.float64)
        obs = observers[0] if len(observers) == 1 else list(observers)
        obs = jnp.asarray(obs, dtype=jnp.float64)
        return jnp.zeros_like(obs)

    def getH(self, observers: ArrayLike, **kwargs) -> jnp.ndarray:
        return self.getB(observers, **kwargs)

    def getJ(self, observers: ArrayLike, **kwargs) -> jnp.ndarray:
        return self.getB(observers, **kwargs)

    def getM(self, observers: ArrayLike, **kwargs) -> jnp.ndarray:
        return self.getB(observers, **kwargs)
