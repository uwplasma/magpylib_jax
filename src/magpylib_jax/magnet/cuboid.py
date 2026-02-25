"""Differentiable cuboid magnet source."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class Cuboid:
    """Homogeneously polarized cuboid."""

    dimension: ArrayLike
    polarization: ArrayLike | None = None
    magnetization: ArrayLike | None = None
    position: ArrayLike = (0.0, 0.0, 0.0)
    orientation: ArrayLike | None = None

    def __post_init__(self) -> None:
        if self.polarization is None and self.magnetization is None:
            raise ValueError("Provide either `polarization` or `magnetization`.")
        if self.polarization is not None and self.magnetization is not None:
            raise ValueError("Provide only one of `polarization` or `magnetization`.")

    @property
    def _polarization(self) -> jnp.ndarray:
        if self.polarization is not None:
            return jnp.asarray(self.polarization, dtype=jnp.float64)
        return MU0 * jnp.asarray(self.magnetization, dtype=jnp.float64)

    @property
    def volume(self) -> float:
        a, b, c = jnp.asarray(self.dimension, dtype=jnp.float64)
        return float(a * b * c)

    @property
    def centroid(self) -> jnp.ndarray:
        return jnp.asarray(self.position, dtype=jnp.float64)

    def getB(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getB(
            "cuboid",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getH(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getH(
            "cuboid",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getJ(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getJ(
            "cuboid",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getM(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getM(
            "cuboid",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )
