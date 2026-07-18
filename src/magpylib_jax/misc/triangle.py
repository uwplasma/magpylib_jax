"""Differentiable triangular magnetic surface source."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput


class Triangle(BaseSource):
    """Triangular magnetic surface with homogeneous polarization."""

    _source_type = "triangle"

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
        return 0.0

    @property
    def _polarization(self) -> jnp.ndarray:
        """Effective polarization, converting ``magnetization`` when needed."""
        if self.polarization is not None:
            return jnp.asarray(self.polarization, dtype=float)
        if self.magnetization is not None:
            return MU0 * jnp.asarray(self.magnetization, dtype=float)
        raise MagpylibMissingInput("Input polarization of Triangle must be set.")

    def _require_inputs(self) -> None:
        if self.vertices is None:
            raise MagpylibMissingInput("Input vertices of Triangle must be set.")
        if self.polarization is None and self.magnetization is None:
            raise MagpylibMissingInput("Input polarization of Triangle must be set.")

    def _field_kwargs(self) -> dict:
        """Geometry + excitation arguments for the field engine."""
        return {"vertices": self.vertices, "polarization": self._polarization}
