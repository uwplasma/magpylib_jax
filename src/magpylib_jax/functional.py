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
from magpylib_jax.core.kernels_extended import (
    current_polyline_bfield,
    current_polyline_hfield,
    magnet_sphere_bfield,
    magnet_sphere_hfield,
    magnet_sphere_jfield,
    magnet_sphere_mfield,
    tetrahedron_bfield,
    tetrahedron_hfield,
    tetrahedron_jfield,
    tetrahedron_mfield,
    triangle_bfield,
    triangle_hfield,
    triangle_jfield,
    triangle_mfield,
)

_ALIASES = {
    "dipole": "dipole",
    "circle": "circle",
    "cuboid": "cuboid",
    "box": "cuboid",
    "cylinder": "cylinder",
    "sphere": "sphere",
    "polyline": "polyline",
    "triangle": "triangle",
    "tetrahedron": "tetrahedron",
}


def _normalize_source_type(source_type: str) -> str:
    key = source_type.lower()
    if key not in _ALIASES:
        valid = sorted(_ALIASES)
        raise ValueError(f"Unsupported source_type {source_type!r}. Expected one of {valid}.")
    return _ALIASES[key]


def _extract_observers(observers: Any) -> Any:
    if hasattr(observers, "observers"):
        return observers.observers
    if hasattr(observers, "pixel"):
        return observers.pixel
    if hasattr(observers, "position") and not isinstance(observers, (list, tuple)):
        return observers.position
    return observers


def _reshape_observer_field(result: jnp.ndarray, observer_input: jnp.ndarray) -> jnp.ndarray:
    if observer_input.ndim == 1:
        return result[0]
    return result.reshape((*observer_input.shape[:-1], 3))


def _apply_squeeze(
    field: jnp.ndarray,
    observer_input: jnp.ndarray,
    *,
    squeeze: bool,
    sumup: bool,
    n_sources: int,
) -> jnp.ndarray:
    if squeeze:
        if sumup:
            return _reshape_observer_field(field, observer_input)
        if observer_input.ndim == 1:
            return field.reshape((n_sources, 3))
        return field.reshape((n_sources, *observer_input.shape[:-1], 3))

    obs_prefix = observer_input.shape[:-1] if observer_input.ndim > 1 else ()
    if sumup:
        arr = field.reshape((1, 1, 1, *obs_prefix, 3))
    else:
        arr = field.reshape((1, n_sources, 1, *obs_prefix, 3))
    return arr


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

    if source_type == "sphere":
        diameter = kwargs.get("diameter")
        polarization = kwargs.get("polarization")
        if diameter is None or polarization is None:
            raise ValueError("Sphere computation requires `diameter` and `polarization`.")
        if output_field == "B":
            return magnet_sphere_bfield(obs_local, diameter, polarization)
        if output_field == "H":
            return magnet_sphere_hfield(obs_local, diameter, polarization)
        if output_field == "J":
            return magnet_sphere_jfield(obs_local, diameter, polarization)
        return magnet_sphere_mfield(obs_local, diameter, polarization)

    if source_type == "triangle":
        vertices = kwargs.get("vertices")
        polarization = kwargs.get("polarization")
        if vertices is None or polarization is None:
            raise ValueError("Triangle computation requires `vertices` and `polarization`.")
        if output_field == "B":
            return triangle_bfield(obs_local, vertices, polarization)
        if output_field == "H":
            return triangle_hfield(obs_local, vertices, polarization)
        if output_field == "J":
            return triangle_jfield(obs_local, vertices, polarization)
        return triangle_mfield(obs_local, vertices, polarization)

    if source_type == "polyline":
        segment_start = kwargs.get("segment_start")
        segment_end = kwargs.get("segment_end")
        current = kwargs.get("current")
        if segment_start is None or segment_end is None or current is None:
            raise ValueError(
                "Polyline computation requires `segment_start`, "
                "`segment_end`, and `current`."
            )
        if output_field == "B":
            return current_polyline_bfield(obs_local, segment_start, segment_end, current)
        if output_field == "H":
            return current_polyline_hfield(obs_local, segment_start, segment_end, current)
        return jnp.zeros_like(obs_local, dtype=jnp.float64)

    if source_type == "tetrahedron":
        vertices = kwargs.get("vertices")
        polarization = kwargs.get("polarization")
        in_out = kwargs.get("in_out", "auto")
        if vertices is None or polarization is None:
            raise ValueError("Tetrahedron computation requires `vertices` and `polarization`.")
        if output_field == "B":
            return tetrahedron_bfield(obs_local, vertices, polarization, in_out=in_out)
        if output_field == "H":
            return tetrahedron_hfield(obs_local, vertices, polarization, in_out=in_out)
        if output_field == "J":
            return tetrahedron_jfield(obs_local, vertices, polarization, in_out=in_out)
        return tetrahedron_mfield(obs_local, vertices, polarization, in_out=in_out)

    raise RuntimeError("Unhandled source type.")


def _evaluate_source_field(
    source: object,
    observers: ArrayLike,
    field_name: str,
    *,
    sumup: bool,
) -> tuple[jnp.ndarray, int]:
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        if not source:
            base = ensure_observers(jnp.asarray(observers, dtype=jnp.float64))
            return jnp.zeros_like(base), 0

        terms: list[jnp.ndarray] = []
        for src in source:
            method = getattr(src, f"get{field_name}", None)
            if method is None:
                raise TypeError(
                    f"Source object {type(src).__name__!r} has no get{field_name} method."
                )
            terms.append(ensure_observers(jnp.asarray(method(observers), dtype=jnp.float64)))

        stacked = jnp.stack(terms, axis=0)
        if sumup:
            return jnp.sum(stacked, axis=0), len(terms)
        return stacked, len(terms)

    method = getattr(source, f"get{field_name}", None)
    if method is None:
        raise TypeError(f"Source object {type(source).__name__!r} has no get{field_name} method.")
    return ensure_observers(jnp.asarray(method(observers), dtype=jnp.float64)), 1


def _get_field_from_type(
    source_type: str,
    observers: ArrayLike,
    output_field: str,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = True,
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    norm_type = _normalize_source_type(source_type)
    obs_input = jnp.asarray(observers, dtype=jnp.float64)
    obs_local, rot = to_local_coordinates(obs_input, position=position, orientation=orientation)
    field_local = _evaluate_core_field(norm_type, output_field, obs_local, kwargs)
    field_global = to_global_field(field_local, rot)
    return _apply_squeeze(field_global, obs_input, squeeze=squeeze, sumup=sumup, n_sources=1)


def getB(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = True,
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
            squeeze=squeeze,
            sumup=sumup,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getB with source objects.")
    obs = _extract_observers(observers)
    obs_input = jnp.asarray(obs, dtype=jnp.float64)
    field, n_sources = _evaluate_source_field(source, obs, "B", sumup=sumup)
    return _apply_squeeze(field, obs_input, squeeze=squeeze, sumup=sumup, n_sources=n_sources)


def getH(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = True,
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
            squeeze=squeeze,
            sumup=sumup,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getH with source objects.")
    obs = _extract_observers(observers)
    obs_input = jnp.asarray(obs, dtype=jnp.float64)
    field, n_sources = _evaluate_source_field(source, obs, "H", sumup=sumup)
    return _apply_squeeze(field, obs_input, squeeze=squeeze, sumup=sumup, n_sources=n_sources)


def getJ(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = True,
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
            squeeze=squeeze,
            sumup=sumup,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getJ with source objects.")
    obs = _extract_observers(observers)
    obs_input = jnp.asarray(obs, dtype=jnp.float64)
    field, n_sources = _evaluate_source_field(source, obs, "J", sumup=sumup)
    return _apply_squeeze(field, obs_input, squeeze=squeeze, sumup=sumup, n_sources=n_sources)


def getM(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = True,
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
            squeeze=squeeze,
            sumup=sumup,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getM with source objects.")
    obs = _extract_observers(observers)
    obs_input = jnp.asarray(obs, dtype=jnp.float64)
    field, n_sources = _evaluate_source_field(source, obs, "M", sumup=sumup)
    return _apply_squeeze(field, obs_input, squeeze=squeeze, sumup=sumup, n_sources=n_sources)


def vgetH(source_type: str, observers: ArrayLike, **kwargs: ArrayLike) -> jnp.ndarray:
    """Vectorized alias for getH (shape-preserving)."""
    return getH(source_type, ensure_observers(observers), **kwargs)


def vgetB(source_type: str, observers: ArrayLike, **kwargs: ArrayLike) -> jnp.ndarray:
    """Vectorized alias for getB (shape-preserving)."""
    return getB(source_type, ensure_observers(observers), **kwargs)
