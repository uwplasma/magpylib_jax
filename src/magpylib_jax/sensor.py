"""Sensor compatibility layer."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass(frozen=True)
class Sensor:
    """Sensor with one or multiple pixel locations."""

    pixel: ArrayLike | None = None
    position: ArrayLike | None = None

    def __post_init__(self) -> None:
        if self.pixel is None and self.position is None:
            raise ValueError("Provide `pixel` or `position`.")

    @property
    def observers(self) -> jnp.ndarray:
        pix = None if self.pixel is None else jnp.asarray(self.pixel, dtype=jnp.float64)
        pos = None if self.position is None else jnp.asarray(self.position, dtype=jnp.float64)

        if pix is None:
            assert pos is not None
            if pos.ndim == 1:
                return pos
            if pos.ndim == 2 and pos.shape[1] == 3:
                return pos[:, None, :]
            raise ValueError(
                f"Sensor `position` must have shape (3,) or (p,3), got {pos.shape}."
            )
        if pos is None:
            assert pix is not None
            return pix

        if pos.ndim == 1:
            pos = pos[None, :]
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(
                f"Sensor `position` must have shape (3,) or (p,3), got {pos.shape}."
            )
        if pix.shape[-1] != 3:
            raise ValueError(f"Sensor `pixel` must have trailing dimension 3, got {pix.shape}.")

        # Broadcast sensor path over pixel layout.
        return pos[(slice(None),) + (None,) * (pix.ndim - 1)] + pix[None, ...]

    @property
    def centroid(self) -> jnp.ndarray:
        obs = jnp.asarray(self.observers, dtype=jnp.float64)
        if obs.ndim == 1:
            return obs
        return jnp.mean(obs.reshape((-1, 3)), axis=0)

    def getB(self, sources: object, *, in_out: str = "auto") -> jnp.ndarray:
        return getB(sources, self.observers, in_out=in_out)

    def getH(self, sources: object, *, in_out: str = "auto") -> jnp.ndarray:
        return getH(sources, self.observers, in_out=in_out)

    def getJ(self, sources: object, *, in_out: str = "auto") -> jnp.ndarray:
        return getJ(sources, self.observers, in_out=in_out)

    def getM(self, sources: object, *, in_out: str = "auto") -> jnp.ndarray:
        return getM(sources, self.observers, in_out=in_out)
