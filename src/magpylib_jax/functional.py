"""Functional public interface and compatibility dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.geometry import ensure_observers, to_global_field, to_local_coordinates
from magpylib_jax.core.kernels import (
    current_circle_bfield,
    current_circle_hfield,
    dipole_bfield,
    dipole_hfield,
    magnet_cuboid_bfield,
    magnet_cuboid_hfield,
    magnet_cuboid_jfield,
    magnet_cuboid_mfield,
    magnet_cylinder_bfield,
    magnet_cylinder_hfield,
    magnet_cylinder_jfield,
    magnet_cylinder_mfield,
)

_ALIASES = {
    "dipole": "dipole",
    "circle": "circle",
    "cuboid": "cuboid",
    "box": "cuboid",
    "cylinder": "cylinder",
}


def _normalize_source_type(source_type: str) -> str:
    key = source_type.lower()
    if key not in _ALIASES:
        valid = sorted(_ALIASES)
        raise ValueError(f"Unsupported source_type {source_type!r}. Expected one of {valid}.")
    return _ALIASES[key]


def _restore_shape(result: jnp.ndarray, original_observers: jnp.ndarray) -> jnp.ndarray:
    if original_observers.ndim == 1:
        return result[0]
    return result


def _extract_observers(observers: Any) -> Any:
    if hasattr(observers, "observers"):
        return observers.observers
    if hasattr(observers, "pixel"):
        return observers.pixel
    if hasattr(observers, "position") and not isinstance(observers, (list, tuple)):
        return observers.position
    return observers


def _evaluate_core_field(
    source_type: str,
    output_field: str,
    obs_local: jnp.ndarray,
    kwargs: dict[str, ArrayLike],
) -> jnp.ndarray:
    if source_type == "dipole":
        moment = kwargs.get("moment")
        if moment is None:
            raise ValueError("Dipole computation requires `moment`.")
        if output_field == "B":
            return dipole_bfield(obs_local, moment)
        if output_field == "H":
            return dipole_hfield(obs_local, moment)
        return jnp.zeros_like(obs_local, dtype=jnp.float64)

    if source_type == "circle":
        diameter = kwargs.get("diameter")
        current = kwargs.get("current")
        if diameter is None or current is None:
            raise ValueError("Circle computation requires `diameter` and `current`.")
        if output_field == "B":
            return current_circle_bfield(obs_local, diameter=diameter, current=current)
        if output_field == "H":
            return current_circle_hfield(obs_local, diameter=diameter, current=current)
        return jnp.zeros_like(obs_local, dtype=jnp.float64)

    if source_type == "cuboid":
        dimension = kwargs.get("dimension")
        polarization = kwargs.get("polarization")
        if dimension is None or polarization is None:
            raise ValueError("Cuboid computation requires `dimension` and `polarization`.")
        if output_field == "B":
            return magnet_cuboid_bfield(obs_local, dimension, polarization)
        if output_field == "H":
            return magnet_cuboid_hfield(obs_local, dimension, polarization)
        if output_field == "J":
            return magnet_cuboid_jfield(obs_local, dimension, polarization)
        return magnet_cuboid_mfield(obs_local, dimension, polarization)

    if source_type == "cylinder":
        dimension = kwargs.get("dimension")
        polarization = kwargs.get("polarization")
        if dimension is None or polarization is None:
            raise ValueError("Cylinder computation requires `dimension` and `polarization`.")
        if output_field == "B":
            return magnet_cylinder_bfield(obs_local, dimension, polarization)
        if output_field == "H":
            return magnet_cylinder_hfield(obs_local, dimension, polarization)
        if output_field == "J":
            return magnet_cylinder_jfield(obs_local, dimension, polarization)
        return magnet_cylinder_mfield(obs_local, dimension, polarization)

    raise RuntimeError("Unhandled source type.")


def _evaluate_source_field(source: object, observers: ArrayLike, field_name: str) -> jnp.ndarray:
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        if not source:
            obs = jnp.asarray(_extract_observers(observers), dtype=jnp.float64)
            base = ensure_observers(obs)
            return jnp.zeros_like(base)
        terms = [_evaluate_source_field(src, observers, field_name) for src in source]
        out = jnp.asarray(terms[0], dtype=jnp.float64)
        for term in terms[1:]:
            out = out + jnp.asarray(term, dtype=jnp.float64)
        return out

    method = getattr(source, f"get{field_name}", None)
    if method is None:
        raise TypeError(f"Source object {type(source).__name__!r} has no get{field_name} method.")
    return jnp.asarray(method(observers), dtype=jnp.float64)


def _get_field_from_type(
    source_type: str,
    observers: ArrayLike,
    output_field: str,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    norm_type = _normalize_source_type(source_type)
    obs_input = jnp.asarray(observers, dtype=jnp.float64)
    obs_local, rot = to_local_coordinates(obs_input, position=position, orientation=orientation)
    field_local = _evaluate_core_field(norm_type, output_field, obs_local, kwargs)
    field_global = to_global_field(field_local, rot)
    return _restore_shape(field_global, obs_input)


def getB(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return B-field in Tesla from source type strings or source objects."""
    if isinstance(source, str):
        if observers is None:
            raise ValueError("Observers are required when calling getB with source type strings.")
        obs = _extract_observers(observers)
        return _get_field_from_type(
            source,
            obs,
            "B",
            position=position,
            orientation=orientation,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getB with source objects.")
    obs = _extract_observers(observers)
    return _evaluate_source_field(source, obs, "B")


def getH(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return H-field in A/m from source type strings or source objects."""
    if isinstance(source, str):
        if observers is None:
            raise ValueError("Observers are required when calling getH with source type strings.")
        obs = _extract_observers(observers)
        return _get_field_from_type(
            source,
            obs,
            "H",
            position=position,
            orientation=orientation,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getH with source objects.")
    obs = _extract_observers(observers)
    return _evaluate_source_field(source, obs, "H")


def getJ(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return J-field from source type strings or source objects."""
    if isinstance(source, str):
        if observers is None:
            raise ValueError("Observers are required when calling getJ with source type strings.")
        obs = _extract_observers(observers)
        return _get_field_from_type(
            source,
            obs,
            "J",
            position=position,
            orientation=orientation,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getJ with source objects.")
    obs = _extract_observers(observers)
    return _evaluate_source_field(source, obs, "J")


def getM(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return M-field from source type strings or source objects."""
    if isinstance(source, str):
        if observers is None:
            raise ValueError("Observers are required when calling getM with source type strings.")
        obs = _extract_observers(observers)
        return _get_field_from_type(
            source,
            obs,
            "M",
            position=position,
            orientation=orientation,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getM with source objects.")
    obs = _extract_observers(observers)
    return _evaluate_source_field(source, obs, "M")


def vgetH(source_type: str, observers: ArrayLike, **kwargs: ArrayLike) -> jnp.ndarray:
    """Vectorized alias for getH (shape-preserving)."""
    return getH(source_type, ensure_observers(observers), **kwargs)


def vgetB(source_type: str, observers: ArrayLike, **kwargs: ArrayLike) -> jnp.ndarray:
    """Vectorized alias for getB (shape-preserving)."""
    return getB(source_type, ensure_observers(observers), **kwargs)
