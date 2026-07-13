"""Differentiable dipole source object."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput


class Dipole(BaseSource):
    """Magnetic dipole source with optional rigid transform."""

    _source_type = "dipole"

    def __init__(
        self,
        moment: ArrayLike | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self.moment = moment
        super().__init__(
            position=position,
            orientation=orientation,
            style=style,
            style_label=style_label,
            **kwargs,
        )

    @property
    def volume(self) -> float:
        return 0.0

    @property
    def dipole_moment(self) -> jnp.ndarray:
        if self.moment is None:
            return jnp.zeros((3,), dtype=jnp.float64)
        return jnp.asarray(self.moment, dtype=jnp.float64)

    def _require_inputs(self) -> None:
        if self.moment is None:
            raise MagpylibMissingInput("Input moment of Dipole must be set.")

    def _field_kwargs(self) -> dict:
        """Geometry + excitation arguments for the field engine."""
        return {"moment": self.moment}
