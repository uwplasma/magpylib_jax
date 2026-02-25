"""Differentiable polyline current source."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput
from magpylib_jax.functional import getB, getH, getJ, getM


class Polyline(BaseSource):
    """Piecewise-linear current path through `vertices`."""

    _source_type = "polyline"

    def __init__(
        self,
        current: ArrayLike | None = None,
        vertices: ArrayLike | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self.current = current
        self.vertices = vertices
        super().__init__(
            position=position,
            orientation=orientation,
            style=style,
            style_label=style_label,
            **kwargs,
        )

    @property
    def _segments(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.vertices is None:
            raise MagpylibMissingInput("Input vertices of Polyline must be set.")
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        if verts.ndim != 2 or verts.shape[0] < 2 or verts.shape[1] != 3:
            raise ValueError("`vertices` must have shape (n>=2, 3).")
        return verts[:-1], verts[1:]

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
            raise MagpylibMissingInput("Input vertices of Polyline must be set.")
        if self.current is None:
            raise MagpylibMissingInput("Input current of Polyline must be set.")

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
        seg_start, seg_end = self._segments
        return getB(
            "polyline",
            obs,
            segment_start=seg_start,
            segment_end=seg_end,
            current=self.current,
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
        seg_start, seg_end = self._segments
        return getH(
            "polyline",
            obs,
            segment_start=seg_start,
            segment_end=seg_end,
            current=self.current,
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
        seg_start, seg_end = self._segments
        return getJ(
            "polyline",
            obs,
            segment_start=seg_start,
            segment_end=seg_end,
            current=self.current,
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
        seg_start, seg_end = self._segments
        return getM(
            "polyline",
            obs,
            segment_start=seg_start,
            segment_end=seg_end,
            current=self.current,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
        )

    def copy(self, **kwargs) -> Polyline:
        return super().copy(**kwargs)
