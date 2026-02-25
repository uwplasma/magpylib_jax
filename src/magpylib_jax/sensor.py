"""Sensor compatibility layer."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class Sensor:
    """Sensor with one or multiple pixel locations."""

    pixel: ArrayLike | None = None
    position: ArrayLike | None = None

    def __post_init__(self) -> None:
        if self.pixel is None and self.position is None:
            raise ValueError("Provide `pixel` or `position`.")

    @property
    def observers(self) -> jnp.ndarray:
        value = self.pixel if self.pixel is not None else self.position
        return jnp.asarray(value, dtype=jnp.float64)

    def getB(self, sources: object) -> jnp.ndarray:
        return getB(sources, self.observers)

    def getH(self, sources: object) -> jnp.ndarray:
        return getH(sources, self.observers)

    def getJ(self, sources: object) -> jnp.ndarray:
        return getJ(sources, self.observers)

    def getM(self, sources: object) -> jnp.ndarray:
        return getM(sources, self.observers)
