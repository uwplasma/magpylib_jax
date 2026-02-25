"""Differentiable triangular-mesh magnet source."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class TriangularMesh:
    """Uniformly polarized magnet defined by mesh vertices and triangular faces."""

    vertices: ArrayLike
    faces: ArrayLike
    polarization: ArrayLike | None = None
    magnetization: ArrayLike | None = None
    position: ArrayLike = (0.0, 0.0, 0.0)
    orientation: ArrayLike | None = None
    reorient_faces: bool = True

    def __post_init__(self) -> None:
        if self.polarization is None and self.magnetization is None:
            raise ValueError("Provide either `polarization` or `magnetization`.")
        if self.polarization is not None and self.magnetization is not None:
            raise ValueError("Provide only one of `polarization` or `magnetization`.")

        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        facs = jnp.asarray(self.faces, dtype=jnp.int32)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("TriangularMesh `vertices` must have shape (n,3).")
        if facs.ndim != 2 or facs.shape[1] != 3:
            raise ValueError("TriangularMesh `faces` must have shape (m,3).")
        if jnp.any(facs < 0) or jnp.any(facs >= verts.shape[0]):
            raise ValueError("TriangularMesh `faces` contain indices outside `vertices`.")

    @property
    def _polarization(self) -> jnp.ndarray:
        if self.polarization is not None:
            return jnp.asarray(self.polarization, dtype=jnp.float64)
        return MU0 * jnp.asarray(self.magnetization, dtype=jnp.float64)

    @property
    def _faces_oriented(self) -> jnp.ndarray:
        faces = jnp.asarray(self.faces, dtype=jnp.int32)
        if not self.reorient_faces:
            return faces

        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        tri = verts[faces]
        center = jnp.mean(verts, axis=0)
        ctri = jnp.mean(tri, axis=1)
        n = jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        inward = jnp.sum(n * (ctri - center), axis=1) < 0
        flipped = faces[:, (0, 2, 1)]
        return jnp.where(inward[:, None], flipped, faces)

    @property
    def mesh(self) -> jnp.ndarray:
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        return verts[self._faces_oriented]

    @property
    def barycenter(self) -> jnp.ndarray:
        tri = self.mesh
        ctri = jnp.mean(tri, axis=1)
        area = 0.5 * jnp.linalg.norm(
            jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            axis=1,
        )
        total = jnp.maximum(jnp.sum(area), 1e-30)
        return jnp.sum(ctri * area[:, None], axis=0) / total

    @property
    def centroid(self) -> jnp.ndarray:
        return self.barycenter + jnp.asarray(self.position, dtype=jnp.float64)

    @property
    def volume(self) -> float:
        tri = self.mesh
        signed = jnp.sum(jnp.cross(tri[:, 0], tri[:, 1]) * tri[:, 2], axis=1) / 6.0
        return float(jnp.abs(jnp.sum(signed)))

    def getB(self, observers: ArrayLike) -> jnp.ndarray:
        return getB(
            "triangularmesh",
            observers,
            mesh=self.mesh,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
        )

    def getH(self, observers: ArrayLike) -> jnp.ndarray:
        return getH(
            "triangularmesh",
            observers,
            mesh=self.mesh,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
        )

    def getJ(self, observers: ArrayLike) -> jnp.ndarray:
        return getJ(
            "triangularmesh",
            observers,
            mesh=self.mesh,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
        )

    def getM(self, observers: ArrayLike) -> jnp.ndarray:
        return getM(
            "triangularmesh",
            observers,
            mesh=self.mesh,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
        )
