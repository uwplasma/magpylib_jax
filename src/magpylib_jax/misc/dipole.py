"""Differentiable dipole source object."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class Dipole:
    """Magnetic dipole source with optional rigid transform."""

    moment: ArrayLike
    position: ArrayLike = (0.0, 0.0, 0.0)
    orientation: ArrayLike | None = None

    def getH(self, observers: ArrayLike) -> jnp.ndarray:
        return getH(
            "dipole",
            observers,
            moment=self.moment,
            position=self.position,
            orientation=self.orientation,
        )

    def getB(self, observers: ArrayLike) -> jnp.ndarray:
        return getB(
            "dipole",
            observers,
            moment=self.moment,
            position=self.position,
            orientation=self.orientation,
        )

    def getJ(self, observers: ArrayLike) -> jnp.ndarray:
        return getJ(
            "dipole",
            observers,
            moment=self.moment,
            position=self.position,
            orientation=self.orientation,
        )

    def getM(self, observers: ArrayLike) -> jnp.ndarray:
        return getM(
            "dipole",
            observers,
            moment=self.moment,
            position=self.position,
            orientation=self.orientation,
        )
