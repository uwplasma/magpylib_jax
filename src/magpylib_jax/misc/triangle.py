"""Differentiable triangular magnetic surface source."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput
from magpylib_jax.functional import getB, getH, getJ, getM


class Triangle(BaseSource):
    """Triangular magnetic surface with homogeneous polarization."""

    _source_type = "triangle"

    def __init__(
        self,
        vertices: ArrayLike | None = None,
        polarization: ArrayLike | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self.vertices = vertices
        self.polarization = polarization
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
            return jnp.zeros((3,), dtype=jnp.float64)
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        return jnp.mean(verts, axis=0)

    @property
    def centroid(self) -> jnp.ndarray:
        return self.barycenter + jnp.asarray(self.position, dtype=jnp.float64)

    @property
    def volume(self) -> float:
        return 0.0

    def _require_inputs(self) -> None:
        if self.vertices is None:
            raise MagpylibMissingInput("Input vertices of Triangle must be set.")
        if self.polarization is None:
            raise MagpylibMissingInput("Input polarization of Triangle must be set.")

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
            "triangle",
            obs,
            vertices=self.vertices,
            polarization=self.polarization,
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
            "triangle",
            obs,
            vertices=self.vertices,
            polarization=self.polarization,
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
            "triangle",
            obs,
            vertices=self.vertices,
            polarization=self.polarization,
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
            "triangle",
            obs,
            vertices=self.vertices,
            polarization=self.polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
        )

    def copy(self, **kwargs) -> Triangle:
        return super().copy(**kwargs)
