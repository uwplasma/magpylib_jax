"""Differentiable triangular current-sheet source."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput
from magpylib_jax.functional import getB, getH, getJ, getM


class TriangleSheet(BaseSource):
    """Surface current densities flowing over indexed triangular faces."""

    _source_type = "trianglesheet"

    def __init__(
        self,
        vertices: ArrayLike | None = None,
        faces: ArrayLike | None = None,
        current_densities: ArrayLike | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self.vertices = vertices
        self.faces = faces
        self.current_densities = current_densities
        if (
            self.vertices is not None
            and self.faces is not None
            and self.current_densities is not None
        ):
            verts = jnp.asarray(self.vertices, dtype=jnp.float64)
            facs = jnp.asarray(self.faces, dtype=jnp.int32)
            cds = jnp.asarray(self.current_densities, dtype=jnp.float64)
            if verts.ndim != 2 or verts.shape[1] != 3:
                raise ValueError("TriangleSheet `vertices` must have shape (n,3).")
            if facs.ndim != 2 or facs.shape[1] != 3:
                raise ValueError("TriangleSheet `faces` must have shape (m,3).")
            if cds.ndim != 2 or cds.shape[1] != 3:
                raise ValueError("TriangleSheet `current_densities` must have shape (m,3).")
            if facs.shape[0] != cds.shape[0]:
                raise ValueError(
                    "TriangleSheet `faces` and `current_densities` must have same length."
                )
            if jnp.any(facs < 0) or jnp.any(facs >= verts.shape[0]):
                raise ValueError("TriangleSheet `faces` contain indices outside `vertices`.")
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
            raise MagpylibMissingInput("Input vertices of TriangleSheet must be set.")
        if self.faces is None:
            raise MagpylibMissingInput("Input faces of TriangleSheet must be set.")
        if self.current_densities is None:
            raise MagpylibMissingInput("Input current_densities of TriangleSheet must be set.")

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
            "trianglesheet",
            obs,
            vertices=self.vertices,
            faces=self.faces,
            current_densities=self.current_densities,
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
            "trianglesheet",
            obs,
            vertices=self.vertices,
            faces=self.faces,
            current_densities=self.current_densities,
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
            "trianglesheet",
            obs,
            vertices=self.vertices,
            faces=self.faces,
            current_densities=self.current_densities,
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
            "trianglesheet",
            obs,
            vertices=self.vertices,
            faces=self.faces,
            current_densities=self.current_densities,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
        )

    def copy(self, **kwargs) -> TriangleSheet:
        return super().copy(**kwargs)
