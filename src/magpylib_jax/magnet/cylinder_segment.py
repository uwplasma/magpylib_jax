"""Differentiable cylinder-segment magnet source."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput
from magpylib_jax.functional import getB, getH, getJ, getM


class CylinderSegment(BaseSource):
    """Uniformly polarized cylinder segment (r1, r2, h, phi1_deg, phi2_deg)."""

    _source_type = "cylindersegment"

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
        if self.dimension is not None:
            dim = jnp.asarray(self.dimension, dtype=jnp.float64)
            if dim.shape != (5,):
                raise ValueError(
                    f"CylinderSegment `dimension` must have shape (5,), got {dim.shape}."
                )
            r1, r2, _, phi1, phi2 = dim
            if not (r1 >= 0 and r2 > r1 and phi2 > phi1):
                raise ValueError("CylinderSegment `dimension` must satisfy r2>r1>=0 and phi2>phi1.")
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
        raise MagpylibMissingInput("Input polarization of CylinderSegment must be set.")

    @property
    def volume(self) -> float:
        if self.dimension is None:
            return 0.0
        r1, r2, h, phi1, phi2 = jnp.asarray(self.dimension, dtype=jnp.float64)
        return float((r2 * r2 - r1 * r1) * jnp.pi * h * (phi2 - phi1) / 360.0)

    @property
    def barycenter(self) -> jnp.ndarray:
        if self.dimension is None:
            return jnp.zeros((3,), dtype=jnp.float64)
        r1, r2, _, phi1, phi2 = jnp.asarray(self.dimension, dtype=jnp.float64)
        alpha = jnp.deg2rad((phi2 - phi1) / 2.0)
        phi = jnp.deg2rad((phi1 + phi2) / 2.0)
        centroid_x = (2.0 / 3.0) * jnp.sin(alpha) / alpha * (r2**3 - r1**3) / (r2**2 - r1**2)
        cent = jnp.array([centroid_x * jnp.cos(phi), centroid_x * jnp.sin(phi), 0.0])
        return cent

    @property
    def centroid(self) -> jnp.ndarray:
        return self.barycenter + jnp.asarray(self.position, dtype=jnp.float64)

    def _require_inputs(self) -> None:
        if self.dimension is None:
            raise MagpylibMissingInput("Input dimension of CylinderSegment must be set.")
        if self.polarization is None and self.magnetization is None:
            raise MagpylibMissingInput("Input polarization of CylinderSegment must be set.")

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
            "cylindersegment",
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
            "cylindersegment",
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
            "cylindersegment",
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
            "cylindersegment",
            obs,
            dimension=self.dimension,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
        )

    def copy(self, **kwargs) -> CylinderSegment:
        return super().copy(**kwargs)
