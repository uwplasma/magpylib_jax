"""Differentiable triangular-mesh magnet source."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.base import BaseSource, MagpylibMissingInput
from magpylib_jax.functional import getB, getH, getJ, getM


class TriangularMesh(BaseSource):
    """Uniformly polarized magnet defined by mesh vertices and triangular faces."""

    _source_type = "triangularmesh"

    def __init__(
        self,
        vertices: ArrayLike | None = None,
        faces: ArrayLike | None = None,
        polarization: ArrayLike | None = None,
        magnetization: ArrayLike | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        reorient_faces: bool = True,
        check_open: bool | str = "warn",
        in_out: str = "auto",
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self.vertices = vertices
        self.faces = faces
        self.polarization = polarization
        self.magnetization = magnetization
        self.reorient_faces = reorient_faces
        self.check_open = check_open
        self.in_out = self._validate_in_out(in_out)
        self.meshing = None
        self.status_open: bool | None = None
        self.status_open_data: list[tuple[int, int]] | None = None
        self.status_disconnected: bool | None = None
        self.status_disconnected_data: list[object] | None = None
        self.status_reoriented: bool | None = bool(reorient_faces)
        self.status_selfintersecting: bool | None = None
        self.status_selfintersecting_data: object | None = None

        if self.vertices is not None and self.faces is not None:
            verts = jnp.asarray(self.vertices, dtype=jnp.float64)
            facs = jnp.asarray(self.faces, dtype=jnp.int32)
            if verts.ndim != 2 or verts.shape[1] != 3:
                raise ValueError("TriangularMesh `vertices` must have shape (n,3).")
            if facs.ndim != 2 or facs.shape[1] != 3:
                raise ValueError("TriangularMesh `faces` must have shape (m,3).")
            if jnp.any(facs < 0) or jnp.any(facs >= verts.shape[0]):
                raise ValueError("TriangularMesh `faces` contain indices outside `vertices`.")

            mode = self._validate_mode_arg(self.check_open, arg_name="check_open")
            if mode != "skip":
                open_edges = self._get_open_edges(np.asarray(self.vertices), np.asarray(self.faces))
                self.status_open = len(open_edges) > 0
                self.status_open_data = [tuple(e) for e in open_edges.tolist()]
                self.status_disconnected = False
                self.status_disconnected_data = [None]
                if self.status_open:
                    msg = (
                        "Open mesh detected in TriangularMesh. Inside-outside checks and "
                        "reorient_faces may yield unexpected results for open meshes. "
                        "This check can be disabled at initialization with check_open='skip'."
                    )
                    if mode == "warn":
                        warnings.warn(msg, stacklevel=2)
                    elif mode == "raise":
                        raise ValueError(msg)

        super().__init__(
            position=position,
            orientation=orientation,
            style=style,
            style_label=style_label,
            **kwargs,
        )

    @staticmethod
    def _validate_mode_arg(arg: bool | str, *, arg_name: str = "mode") -> str:
        accepted = (True, False, "warn", "raise", "ignore", "skip")
        if arg not in accepted:
            msg = (
                "Input "
                f"{arg_name} must be one of {{'warn', 'raise', 'ignore', 'skip', True, False}}; "
                f"instead received {arg!r}."
            )
            raise ValueError(msg)
        return "warn" if arg is True else "skip" if arg is False else arg

    @staticmethod
    def _validate_in_out(value: str) -> str:
        if value not in ("auto", "inside", "outside"):
            raise ValueError(
                "TriangularMesh `in_out` must be one of {'auto', 'inside', 'outside'}."
            )
        return value

    @staticmethod
    def _get_open_edges(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=1)
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        return uniq[counts == 1]

    @classmethod
    def from_ConvexHull(cls, points: ArrayLike, **kwargs) -> TriangularMesh:
        from scipy.spatial import ConvexHull

        opts = dict(kwargs)
        opts.pop("check_selfintersecting", None)
        opts.pop("check_disconnected", None)
        pts = np.asarray(points, dtype=float)
        hull = ConvexHull(pts)
        return cls(vertices=hull.points, faces=hull.simplices, **opts)

    @property
    def _polarization(self) -> jnp.ndarray:
        if self.polarization is not None:
            return jnp.asarray(self.polarization, dtype=jnp.float64)
        if self.magnetization is not None:
            return MU0 * jnp.asarray(self.magnetization, dtype=jnp.float64)
        raise MagpylibMissingInput("Input polarization of TriangularMesh must be set.")

    @property
    def _faces_oriented(self) -> jnp.ndarray:
        if self.vertices is None or self.faces is None:
            return jnp.zeros((0, 3), dtype=jnp.int32)
        faces = jnp.asarray(self.faces, dtype=jnp.int32)
        if not self.reorient_faces:
            return faces

        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        tri = verts[faces]
        center = jnp.mean(verts, axis=0)
        ctri = jnp.mean(tri, axis=1)
        n = jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        inward = jnp.sum(n * (ctri - center), axis=1) < 0
        flipped = faces[:, (0, 2, 1)]
        return jnp.where(inward[:, None], flipped, faces)

    @property
    def mesh(self) -> jnp.ndarray:
        if self.vertices is None or self.faces is None:
            return jnp.zeros((0, 3, 3), dtype=jnp.float64)
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        return verts[self._faces_oriented]

    @property
    def barycenter(self) -> jnp.ndarray:
        if self.vertices is None or self.faces is None:
            return jnp.zeros((3,), dtype=jnp.float64)
        tri = self.mesh
        ctri = jnp.mean(tri, axis=1)
        area = 0.5 * jnp.linalg.norm(
            jnp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            axis=1,
        )
        total = jnp.maximum(jnp.sum(area), 1e-30)
        return jnp.sum(ctri * area[:, None], axis=0) / total

    @property
    def centroid(self) -> jnp.ndarray:
        return self.barycenter + jnp.asarray(self.position, dtype=jnp.float64)

    @property
    def volume(self) -> float:
        if self.vertices is None or self.faces is None:
            return 0.0
        tri = self.mesh
        signed = jnp.sum(jnp.cross(tri[:, 0], tri[:, 1]) * tri[:, 2], axis=1) / 6.0
        return float(jnp.abs(jnp.sum(signed)))

    def _require_inputs(self) -> None:
        if self.vertices is None:
            raise MagpylibMissingInput("Input vertices of TriangularMesh must be set.")
        if self.faces is None:
            raise MagpylibMissingInput("Input faces of TriangularMesh must be set.")
        if self.polarization is None and self.magnetization is None:
            raise MagpylibMissingInput("Input polarization of TriangularMesh must be set.")

    def getB(
        self,
        *observers: ArrayLike,
        in_out: str | None = None,
        squeeze: bool = True,
        sumup: bool = False,
        output: str = "ndarray",
        pixel_agg: str | None = None,
    ) -> jnp.ndarray:
        self._require_inputs()
        if in_out is None:
            in_out = self.in_out
        in_out = self._validate_in_out(in_out)
        obs = observers[0] if len(observers) == 1 else list(observers)
        return getB(
            "triangularmesh",
            obs,
            mesh=self.mesh,
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
        in_out: str | None = None,
        squeeze: bool = True,
        sumup: bool = False,
        output: str = "ndarray",
        pixel_agg: str | None = None,
    ) -> jnp.ndarray:
        self._require_inputs()
        if in_out is None:
            in_out = self.in_out
        in_out = self._validate_in_out(in_out)
        obs = observers[0] if len(observers) == 1 else list(observers)
        return getH(
            "triangularmesh",
            obs,
            mesh=self.mesh,
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
        in_out: str | None = None,
        squeeze: bool = True,
        sumup: bool = False,
    ) -> jnp.ndarray:
        self._require_inputs()
        if in_out is None:
            in_out = self.in_out
        in_out = self._validate_in_out(in_out)
        obs = observers[0] if len(observers) == 1 else list(observers)
        return getJ(
            "triangularmesh",
            obs,
            mesh=self.mesh,
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
        in_out: str | None = None,
        squeeze: bool = True,
        sumup: bool = False,
    ) -> jnp.ndarray:
        self._require_inputs()
        if in_out is None:
            in_out = self.in_out
        in_out = self._validate_in_out(in_out)
        obs = observers[0] if len(observers) == 1 else list(observers)
        return getM(
            "triangularmesh",
            obs,
            mesh=self.mesh,
            polarization=self._polarization,
            position=self.position,
            orientation=self.orientation,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
        )

    def copy(self, **kwargs) -> TriangularMesh:
        return super().copy(**kwargs)
