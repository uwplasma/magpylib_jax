"""Differentiable cylinder magnet source."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput
from magpylib_jax.functional import getB, getH, getJ, getM


class Cylinder(BaseSource):
    """Homogeneously polarized cylinder with diameter-height dimensions."""

    _source_type = "cylinder"

    def __init__(
        self,
        dimension: ArrayLike | None = None,
        polarization: ArrayLike | None = None,
        magnetization: ArrayLike | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self.dimension = dimension
        self.polarization = polarization
        self.magnetization = magnetization
        self.meshing = None
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
        raise MagpylibMissingInput("Input polarization of Cylinder must be set.")

    @property
    def volume(self) -> float:
        if self.dimension is None:
            return 0.0
        diameter, height = jnp.asarray(self.dimension, dtype=jnp.float64)
        radius = diameter / 2.0
        return float(jnp.pi * radius * radius * height)

    @property
    def centroid(self) -> jnp.ndarray:
        return jnp.asarray(self.position, dtype=jnp.float64)

    def _require_inputs(self) -> None:
        if self.dimension is None:
            raise MagpylibMissingInput("Input dimension of Cylinder must be set.")
        if self.polarization is None and self.magnetization is None:
            raise MagpylibMissingInput("Input polarization of Cylinder must be set.")

    def getB(
        self,
        *observers: ArrayLike,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
        output: str = "ndarray",
        pixel_agg: str | None = None,
    ) -> jnp.ndarray:
        self._require_inputs()
        obs = observers[0] if len(observers) == 1 else list(observers)
        return getB(
            "cylinder",
            obs,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
            output=output,
            pixel_agg=pixel_agg,
        )

    def getH(
        self,
        *observers: ArrayLike,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
        output: str = "ndarray",
        pixel_agg: str | None = None,
    ) -> jnp.ndarray:
        self._require_inputs()
        obs = observers[0] if len(observers) == 1 else list(observers)
        return getH(
            "cylinder",
            obs,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
            output=output,
            pixel_agg=pixel_agg,
        )

    def getJ(
        self,
        *observers: ArrayLike,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
    ) -> jnp.ndarray:
        self._require_inputs()
        obs = observers[0] if len(observers) == 1 else list(observers)
        return getJ(
            "cylinder",
            obs,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
        )

    def getM(
        self,
        *observers: ArrayLike,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
    ) -> jnp.ndarray:
        self._require_inputs()
        obs = observers[0] if len(observers) == 1 else list(observers)
        return getM(
            "cylinder",
            obs,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
        )

    def copy(self, **kwargs) -> Cylinder:
        return super().copy(**kwargs)
