"""Functional public interface for field computations."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.geometry import ensure_observers, to_global_field, to_local_coordinates
from magpylib_jax.core.kernels import (
    current_circle_bfield,
    current_circle_hfield,
    dipole_bfield,
    dipole_hfield,
)

_SUPPORTED = {"dipole", "circle"}


def _restore_shape(result: jnp.ndarray, original_observers: jnp.ndarray) -> jnp.ndarray:
    if original_observers.ndim == 1:
        return result[0]
    return result


def getH(
    source_type: str,
    observers: ArrayLike,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return H-field in A/m for supported source types."""
    source_type = source_type.lower()
    if source_type not in _SUPPORTED:
        raise ValueError(f"Unsupported source_type {source_type!r}. Expected one of {_SUPPORTED}.")

    obs_input = jnp.asarray(observers, dtype=jnp.float64)
    obs_local, rot = to_local_coordinates(obs_input, position=position, orientation=orientation)

    if source_type == "dipole":
        if "moment" not in kwargs:
            raise ValueError("Dipole computation requires `moment`.")
        field_local = dipole_hfield(obs_local, kwargs["moment"])
    elif source_type == "circle":
        if "diameter" not in kwargs or "current" not in kwargs:
            raise ValueError("Circle computation requires `diameter` and `current`.")
        field_local = current_circle_hfield(
            obs_local,
            diameter=kwargs["diameter"],
            current=kwargs["current"],
        )
    else:
        raise RuntimeError("Unreachable source type dispatch.")

    field_global = to_global_field(field_local, rot)
    return _restore_shape(field_global, obs_input)


def getB(
    source_type: str,
    observers: ArrayLike,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return B-field in Tesla for supported source types."""
    source_type = source_type.lower()
    obs_input = jnp.asarray(observers, dtype=jnp.float64)
    obs_local, rot = to_local_coordinates(obs_input, position=position, orientation=orientation)

    if source_type == "dipole":
        if "moment" not in kwargs:
            raise ValueError("Dipole computation requires `moment`.")
        field_local = dipole_bfield(obs_local, kwargs["moment"])
    elif source_type == "circle":
        if "diameter" not in kwargs or "current" not in kwargs:
            raise ValueError("Circle computation requires `diameter` and `current`.")
        field_local = current_circle_bfield(
            obs_local,
            diameter=kwargs["diameter"],
            current=kwargs["current"],
        )
    else:
        raise ValueError(f"Unsupported source_type {source_type!r}. Expected one of {_SUPPORTED}.")

    field_global = to_global_field(field_local, rot)
    return _restore_shape(field_global, obs_input)


def vgetH(source_type: str, observers: ArrayLike, **kwargs: ArrayLike) -> jnp.ndarray:
    """Vectorized alias for getH (shape-preserving)."""
    return getH(source_type, ensure_observers(observers), **kwargs)


def vgetB(source_type: str, observers: ArrayLike, **kwargs: ArrayLike) -> jnp.ndarray:
    """Vectorized alias for getB (shape-preserving)."""
    return getB(source_type, ensure_observers(observers), **kwargs)
