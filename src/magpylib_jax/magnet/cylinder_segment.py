"""Differentiable cylinder-segment magnet source."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class CylinderSegment:
    """Uniformly polarized cylinder segment (r1, r2, h, phi1_deg, phi2_deg)."""

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

        dim = jnp.asarray(self.dimension, dtype=jnp.float64)
        if dim.shape != (5,):
            raise ValueError(f"CylinderSegment `dimension` must have shape (5,), got {dim.shape}.")
        r1, r2, _, phi1, phi2 = dim
        if not (r1 >= 0 and r2 > r1 and phi2 > phi1):
            raise ValueError(
                "CylinderSegment `dimension` must satisfy r2>r1>=0 and phi2>phi1."
            )

    @property
    def _polarization(self) -> jnp.ndarray:
        if self.polarization is not None:
            return jnp.asarray(self.polarization, dtype=jnp.float64)
        return MU0 * jnp.asarray(self.magnetization, dtype=jnp.float64)

    @property
    def volume(self) -> float:
        r1, r2, h, phi1, phi2 = jnp.asarray(self.dimension, dtype=jnp.float64)
        return float((r2 * r2 - r1 * r1) * jnp.pi * h * (phi2 - phi1) / 360.0)

    @property
    def barycenter(self) -> jnp.ndarray:
        r1, r2, _, phi1, phi2 = jnp.asarray(self.dimension, dtype=jnp.float64)
        alpha = jnp.deg2rad((phi2 - phi1) / 2.0)
        phi = jnp.deg2rad((phi1 + phi2) / 2.0)
        centroid_x = (2.0 / 3.0) * jnp.sin(alpha) / alpha * (r2**3 - r1**3) / (r2**2 - r1**2)
        cent = jnp.array([centroid_x * jnp.cos(phi), centroid_x * jnp.sin(phi), 0.0])
        return cent

    @property
    def centroid(self) -> jnp.ndarray:
        return self.barycenter + jnp.asarray(self.position, dtype=jnp.float64)

    def getB(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getB(
            "cylindersegment",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getH(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getH(
            "cylindersegment",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getJ(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getJ(
            "cylindersegment",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getM(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getM(
            "cylindersegment",
            observers,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )
