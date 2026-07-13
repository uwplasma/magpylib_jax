"""Differentiable triangular-strip current source."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput


class TriangleStrip(BaseSource):
    """Current flowing through adjacent triangles defined by a vertex strip."""

    _source_type = "trianglestrip"

    def __init__(
        self,
        vertices: ArrayLike | None = None,
        current: ArrayLike | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self.vertices = vertices
        self.current = current
        if self.vertices is not None:
            verts = jnp.asarray(self.vertices, dtype=jnp.float64)
            if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 3:
                raise ValueError("TriangleStrip `vertices` must have shape (n>=3,3).")
        super().__init__(
            position=position,
            orientation=orientation,
            style=style,
            style_label=style_label,
            **kwargs,
        )

    @property
    def centroid(self) -> jnp.ndarray:
        if self.vertices is None:
            return jnp.asarray(self.position, dtype=jnp.float64)
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        return jnp.mean(verts, axis=0) + jnp.asarray(self.position, dtype=jnp.float64)

    @property
    def volume(self) -> float:
        return 0.0

    def _require_inputs(self) -> None:
        if self.vertices is None:
            raise MagpylibMissingInput("Input vertices of TriangleStrip must be set.")
        if self.current is None:
            raise MagpylibMissingInput("Input current of TriangleStrip must be set.")

    def _field_kwargs(self) -> dict:
        """Geometry + excitation arguments for the field engine."""
        return {"vertices": self.vertices, "current": self.current}
