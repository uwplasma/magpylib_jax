"""Differentiable triangular-strip current source."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class TriangleStrip:
    """Current flowing through adjacent triangles defined by a vertex strip."""

    vertices: ArrayLike
    current: ArrayLike
    position: ArrayLike = (0.0, 0.0, 0.0)
    orientation: ArrayLike | None = None

    def __post_init__(self) -> None:
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 3:
            raise ValueError("TriangleStrip `vertices` must have shape (n>=3,3).")

    @property
    def centroid(self) -> jnp.ndarray:
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        return jnp.mean(verts, axis=0) + jnp.asarray(self.position, dtype=jnp.float64)

    def getB(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getB(
            "trianglestrip",
            observers,
            vertices=self.vertices,
            current=self.current,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getH(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getH(
            "trianglestrip",
            observers,
            vertices=self.vertices,
            current=self.current,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getJ(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getJ(
            "trianglestrip",
            observers,
            vertices=self.vertices,
            current=self.current,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getM(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getM(
            "trianglestrip",
            observers,
            vertices=self.vertices,
            current=self.current,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )
