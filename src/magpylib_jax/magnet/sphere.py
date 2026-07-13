"""Differentiable sphere magnet source."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput


class Sphere(BaseSource):
    """Homogeneously polarized sphere with scalar diameter."""

    _source_type = "sphere"

    def __init__(
        self,
        diameter: ArrayLike | None = None,
        polarization: ArrayLike | None = None,
        magnetization: ArrayLike | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self.diameter = diameter
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
            return jnp.asarray(self.polarization, dtype=jnp.float64)
        if self.magnetization is not None:
            return MU0 * jnp.asarray(self.magnetization, dtype=jnp.float64)
        raise MagpylibMissingInput("Input polarization of Sphere must be set.")

    @property
    def volume(self) -> float:
        if self.diameter is None:
            return 0.0
        d = float(jnp.asarray(self.diameter, dtype=jnp.float64))
        return float((4.0 / 3.0) * jnp.pi * (d / 2.0) ** 3)

    @property
    def centroid(self) -> jnp.ndarray:
        return jnp.asarray(self.position, dtype=jnp.float64)

    def _require_inputs(self) -> None:
        if self.diameter is None:
            raise MagpylibMissingInput("Input diameter of Sphere must be set.")
        if self.polarization is None and self.magnetization is None:
            raise MagpylibMissingInput("Input polarization of Sphere must be set.")

    def _field_kwargs(self) -> dict:
        """Geometry + excitation arguments for the field engine."""
        return {"diameter": self.diameter, "polarization": self._polarization}
