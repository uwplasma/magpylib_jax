"""Differentiable tetrahedron magnet source."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput


class Tetrahedron(BaseSource):
    """Homogeneously polarized tetrahedron defined by four vertices."""

    _source_type = "tetrahedron"

    def __init__(
        self,
        vertices: ArrayLike | None = None,
        polarization: ArrayLike | None = None,
        magnetization: ArrayLike | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self.vertices = vertices
        self.polarization = polarization
        self.magnetization = magnetization
        super().__init__(
            position=position,
            orientation=orientation,
            style=style,
            style_label=style_label,
            **kwargs,
        )

    @property
    def _polarization(self) -> jnp.ndarray:
        if self.polarization is not None:
            return jnp.asarray(self.polarization, dtype=float)
        if self.magnetization is not None:
            return MU0 * jnp.asarray(self.magnetization, dtype=float)
        raise MagpylibMissingInput("Input polarization of Tetrahedron must be set.")

    @property
    def barycenter(self) -> jnp.ndarray:
        if self.vertices is None:
            return jnp.zeros((3,), dtype=float)
        verts = jnp.asarray(self.vertices, dtype=float)
        return jnp.mean(verts, axis=0)

    @property
    def centroid(self) -> jnp.ndarray:
        return self.barycenter + jnp.asarray(self.position, dtype=float)

    @property
    def volume(self) -> float:
        if self.vertices is None:
            return 0.0
        verts = jnp.asarray(self.vertices, dtype=float)
        a = verts[1] - verts[0]
        b = verts[2] - verts[0]
        c = verts[3] - verts[0]
        return float(jnp.abs(jnp.dot(a, jnp.cross(b, c))) / 6.0)

    def _require_inputs(self) -> None:
        if self.vertices is None:
            raise MagpylibMissingInput("Input vertices of Tetrahedron must be set.")
        if self.polarization is None and self.magnetization is None:
            raise MagpylibMissingInput("Input polarization of Tetrahedron must be set.")

    def _field_kwargs(self) -> dict:
        """Geometry + excitation arguments for the field engine."""
        return {"vertices": self.vertices, "polarization": self._polarization}
