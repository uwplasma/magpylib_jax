"""Differentiable cylinder magnet source."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class Cylinder:
    """Homogeneously polarized cylinder with diameter-height dimensions."""

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
        diameter, height = jnp.asarray(self.dimension, dtype=jnp.float64)
        radius = diameter / 2.0
        return float(jnp.pi * radius * radius * height)

    @property
    def centroid(self) -> jnp.ndarray:
        return jnp.asarray(self.position, dtype=jnp.float64)

    def getB(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getB(
            "cylinder",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getH(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getH(
            "cylinder",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getJ(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getJ(
            "cylinder",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getM(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getM(
            "cylinder",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )
