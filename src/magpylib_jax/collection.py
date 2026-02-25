"""Collection compatibility layer for summing source fields."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.functional import getB, getH, getJ, getM


@dataclass
class Collection:
    """Container for source objects with additive field behavior."""

    sources: list[object] = field(default_factory=list)

    def add(self, *sources: object) -> Collection:
        self.sources.extend(sources)
        return self

    def getB(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getB(self.sources, observers, sumup=True, in_out=in_out)

    def getH(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getH(self.sources, observers, sumup=True, in_out=in_out)

    def getJ(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getJ(self.sources, observers, sumup=True, in_out=in_out)

    def getM(self, observers: ArrayLike, *, in_out: str = "auto") -> jnp.ndarray:
        return getM(self.sources, observers, sumup=True, in_out=in_out)

    def __iter__(self):
        return iter(self.sources)

    def __add__(self, other: object) -> Collection:
        if isinstance(other, Collection):
            return Collection([*self.sources, *other.sources])
        return Collection([*self.sources, other])
