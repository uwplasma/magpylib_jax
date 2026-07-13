"""Differentiable circular current loop source object."""

from __future__ import annotations

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput


class Circle(BaseSource):
    """Circular current loop in the local xy-plane."""

    _source_type = "circle"

    def __init__(
        self,
        current: ArrayLike | None = None,
        diameter: ArrayLike | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self.current = current
        self.diameter = diameter
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

    def _require_inputs(self) -> None:
        if self.diameter is None:
            raise MagpylibMissingInput("Input diameter of Circle must be set.")
        if self.current is None:
            raise MagpylibMissingInput("Input current of Circle must be set.")

    def _field_kwargs(self) -> dict:
        """Geometry + excitation arguments for the field engine."""
        return {"diameter": self.diameter, "current": self.current}
