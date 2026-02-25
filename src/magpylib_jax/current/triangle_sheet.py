"""Differentiable triangular current-sheet source."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class TriangleSheet:
    """Surface current densities flowing over indexed triangular faces."""

    vertices: ArrayLike
    faces: ArrayLike
    current_densities: ArrayLike
    position: ArrayLike = (0.0, 0.0, 0.0)
    orientation: ArrayLike | None = None

    def __post_init__(self) -> None:
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        facs = jnp.asarray(self.faces, dtype=jnp.int32)
        cds = jnp.asarray(self.current_densities, dtype=jnp.float64)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("TriangleSheet `vertices` must have shape (n,3).")
        if facs.ndim != 2 or facs.shape[1] != 3:
            raise ValueError("TriangleSheet `faces` must have shape (m,3).")
        if cds.ndim != 2 or cds.shape[1] != 3:
            raise ValueError("TriangleSheet `current_densities` must have shape (m,3).")
        if facs.shape[0] != cds.shape[0]:
            raise ValueError("TriangleSheet `faces` and `current_densities` must have same length.")
        if jnp.any(facs < 0) or jnp.any(facs >= verts.shape[0]):
            raise ValueError("TriangleSheet `faces` contain indices outside `vertices`.")

    @property
    def centroid(self) -> jnp.ndarray:
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        return jnp.mean(verts, axis=0) + jnp.asarray(self.position, dtype=jnp.float64)

    def getB(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getB(
            "trianglesheet",
            observers,
            vertices=self.vertices,
            faces=self.faces,
            current_densities=self.current_densities,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getH(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getH(
            "trianglesheet",
            observers,
            vertices=self.vertices,
            faces=self.faces,
            current_densities=self.current_densities,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getJ(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getJ(
            "trianglesheet",
            observers,
            vertices=self.vertices,
            faces=self.faces,
            current_densities=self.current_densities,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )

    def getM(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getM(
            "trianglesheet",
            observers,
            vertices=self.vertices,
            faces=self.faces,
            current_densities=self.current_densities,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
        )
