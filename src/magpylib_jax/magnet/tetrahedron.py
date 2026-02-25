"""Differentiable tetrahedron magnet source."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class Tetrahedron:
    """Homogeneously polarized tetrahedron defined by four vertices."""

    vertices: ArrayLike
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
    def barycenter(self) -> jnp.ndarray:
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        return jnp.mean(verts, axis=0)

    @property
    def centroid(self) -> jnp.ndarray:
        return self.barycenter + jnp.asarray(self.position, dtype=jnp.float64)

    @property
    def volume(self) -> float:
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        a = verts[1] - verts[0]
        b = verts[2] - verts[0]
        c = verts[3] - verts[0]
        return float(jnp.abs(jnp.dot(a, jnp.cross(b, c))) / 6.0)

    def getB(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getB(
            "tetrahedron",
            observers,
            vertices=self.vertices,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getH(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getH(
            "tetrahedron",
            observers,
            vertices=self.vertices,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getJ(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getJ(
            "tetrahedron",
            observers,
            vertices=self.vertices,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getM(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getM(
            "tetrahedron",
            observers,
            vertices=self.vertices,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )
