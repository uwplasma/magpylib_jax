"""Differentiable polyline current source."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class Polyline:
    """Piecewise-linear current path through `vertices`."""

    current: ArrayLike
    vertices: ArrayLike
    position: ArrayLike = (0.0, 0.0, 0.0)
    orientation: ArrayLike | None = None

    @property
    def _segments(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        if verts.ndim != 2 or verts.shape[0] < 2 or verts.shape[1] != 3:
            raise ValueError("`vertices` must have shape (n>=2, 3).")
        return verts[:-1], verts[1:]

    @property
    def centroid(self) -> jnp.ndarray:
        verts = jnp.asarray(self.vertices, dtype=jnp.float64)
        return jnp.mean(verts, axis=0) + jnp.asarray(self.position, dtype=jnp.float64)

    def _field(self, field_name: str, observers: ArrayLike) -> jnp.ndarray:
        obs = jnp.asarray(observers, dtype=jnp.float64)
        if obs.ndim == 1:
            obs2 = obs[None, :]
        else:
            obs2 = obs.reshape((-1, 3))

        seg_start, seg_end = self._segments
        nseg = seg_start.shape[0]

        obs_rep = jnp.repeat(obs2, nseg, axis=0)
        start_rep = jnp.tile(seg_start, (obs2.shape[0], 1))
        end_rep = jnp.tile(seg_end, (obs2.shape[0], 1))
        cur_rep = jnp.broadcast_to(
            jnp.asarray(self.current, dtype=jnp.float64),
            (obs_rep.shape[0],),
        )

        if field_name == "B":
            out = getB(
                "polyline",
                obs_rep,
                segment_start=start_rep,
                segment_end=end_rep,
                current=cur_rep,
                position=self.position,
                orientation=self.orientation,
            )
        elif field_name == "H":
            out = getH(
                "polyline",
                obs_rep,
                segment_start=start_rep,
                segment_end=end_rep,
                current=cur_rep,
                position=self.position,
                orientation=self.orientation,
            )
        elif field_name == "J":
            out = getJ(
                "polyline",
                obs_rep,
                segment_start=start_rep,
                segment_end=end_rep,
                current=cur_rep,
                position=self.position,
                orientation=self.orientation,
            )
        else:
            out = getM(
                "polyline",
                obs_rep,
                segment_start=start_rep,
                segment_end=end_rep,
                current=cur_rep,
                position=self.position,
                orientation=self.orientation,
            )

        out = out.reshape((obs2.shape[0], nseg, 3)).sum(axis=1)
        if obs.ndim == 1:
            return out[0]
        return out.reshape((*obs.shape[:-1], 3))

    def getB(self, observers: ArrayLike) -> jnp.ndarray:
        return self._field("B", observers)

    def getH(self, observers: ArrayLike) -> jnp.ndarray:
        return self._field("H", observers)

    def getJ(self, observers: ArrayLike) -> jnp.ndarray:
        return self._field("J", observers)

    def getM(self, observers: ArrayLike) -> jnp.ndarray:
        return self._field("M", observers)
