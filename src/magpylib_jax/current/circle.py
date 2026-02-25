"""Differentiable circular current loop source object."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class Circle:
    """Circular current loop in the local xy-plane."""

    current: ArrayLike
    diameter: ArrayLike
    position: ArrayLike = (0.0, 0.0, 0.0)
    orientation: ArrayLike | None = None

    def getH(self, observers: ArrayLike) -> jnp.ndarray:
        return getH(
            "circle",
            observers,
            current=self.current,
            diameter=self.diameter,
            position=self.position,
            orientation=self.orientation,
        )

    def getB(self, observers: ArrayLike) -> jnp.ndarray:
        return getB(
            "circle",
            observers,
            current=self.current,
            diameter=self.diameter,
            position=self.position,
            orientation=self.orientation,
        )

    def getJ(self, observers: ArrayLike) -> jnp.ndarray:
        return getJ(
            "circle",
            observers,
            current=self.current,
            diameter=self.diameter,
            position=self.position,
            orientation=self.orientation,
        )

    def getM(self, observers: ArrayLike) -> jnp.ndarray:
        return getM(
            "circle",
            observers,
            current=self.current,
            diameter=self.diameter,
            position=self.position,
            orientation=self.orientation,
        )
