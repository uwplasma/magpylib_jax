"""Functional public interface and compatibility dispatch."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from math import prod
from typing import Any

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.geometry import (
    broadcast_pose,
    ensure_observers,
    to_global_field,
    to_local_coordinates,
)
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
    current_trisheet_bfield,
    current_trisheet_hfield,
    current_tristrip_bfield,
    current_tristrip_hfield,
    magnet_cylinder_segment_bfield,
    magnet_cylinder_segment_hfield,
    magnet_cylinder_segment_jfield,
    magnet_cylinder_segment_mfield,
    magnet_sphere_bfield,
    magnet_sphere_hfield,
    magnet_sphere_jfield,
    magnet_sphere_mfield,
    magnet_trimesh_bfield,
    magnet_trimesh_hfield,
    magnet_trimesh_jfield,
    magnet_trimesh_mfield,
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
    "cylindersegment": "cylindersegment",
    "cylinder_segment": "cylindersegment",
    "sphere": "sphere",
    "polyline": "polyline",
    "triangle": "triangle",
    "trianglesheet": "trianglesheet",
    "triangle_sheet": "trianglesheet",
    "trianglestrip": "trianglestrip",
    "triangle_strip": "trianglestrip",
    "triangularmesh": "triangularmesh",
    "triangular_mesh": "triangularmesh",
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
    obs_prefix = observer_input.shape[:-1] if observer_input.ndim > 1 else ()
    n_obs = prod(obs_prefix) if obs_prefix else 1

    def _unflatten_obs(arr: jnp.ndarray) -> jnp.ndarray:
        if arr.ndim >= 2 and arr.shape[-1] == 3 and arr.shape[-2] == n_obs:
            return arr.reshape((*arr.shape[:-2], *obs_prefix, 3))
        return arr

    field = _unflatten_obs(field)

    if squeeze:
        axes = tuple(i for i, size in enumerate(field.shape[:-1]) if size == 1)
        return jnp.squeeze(field, axis=axes) if axes else field

    obs_has_path = observer_input.ndim >= 3
    if sumup:
        if field.ndim == len(obs_prefix) + 2:
            path_len = field.shape[0]
            pix = field.shape[1:-1]
        elif obs_has_path:
            path_len = field.shape[0]
            pix = field.shape[1:-1]
        else:
            path_len = 1
            pix = field.shape[:-1]
        arr = field.reshape((1, path_len, 1, *pix, 3))
    else:
        if n_sources == 1 and (field.ndim == observer_input.ndim or field.shape[0] != 1):
            field = field[None, ...]
        rest = field.shape[1:-1]
        if len(rest) > len(obs_prefix):
            path_len = rest[0]
            pix = rest[1:]
        elif obs_has_path and len(rest) >= 1:
            path_len = rest[0]
            pix = rest[1:]
        else:
            path_len = 1
            pix = rest
        arr = field.reshape((n_sources, path_len, 1, *pix, 3))
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

    if source_type == "cylindersegment":
        dimension = kwargs.get("dimension")
        polarization = kwargs.get("polarization")
        in_out = kwargs.get("in_out", "auto")
        if dimension is None or polarization is None:
            raise ValueError("CylinderSegment computation requires `dimension` and `polarization`.")
        if output_field == "B":
            return magnet_cylinder_segment_bfield(obs_local, dimension, polarization, in_out=in_out)
        if output_field == "H":
            return magnet_cylinder_segment_hfield(obs_local, dimension, polarization, in_out=in_out)
        if output_field == "J":
            return magnet_cylinder_segment_jfield(obs_local, dimension, polarization, in_out=in_out)
        return magnet_cylinder_segment_mfield(obs_local, dimension, polarization, in_out=in_out)

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

    if source_type == "trianglesheet":
        vertices = kwargs.get("vertices")
        faces = kwargs.get("faces")
        current_densities = kwargs.get("current_densities")
        if vertices is None or faces is None or current_densities is None:
            raise ValueError(
                "TriangleSheet computation requires `vertices`, `faces`, and `current_densities`."
            )
        if output_field == "B":
            return current_trisheet_bfield(obs_local, vertices, faces, current_densities)
        if output_field == "H":
            return current_trisheet_hfield(obs_local, vertices, faces, current_densities)
        return jnp.zeros_like(obs_local, dtype=jnp.float64)

    if source_type == "trianglestrip":
        vertices = kwargs.get("vertices")
        current = kwargs.get("current")
        if vertices is None or current is None:
            raise ValueError("TriangleStrip computation requires `vertices` and `current`.")
        if output_field == "B":
            return current_tristrip_bfield(obs_local, vertices, current)
        if output_field == "H":
            return current_tristrip_hfield(obs_local, vertices, current)
        return jnp.zeros_like(obs_local, dtype=jnp.float64)

    if source_type == "triangularmesh":
        mesh = kwargs.get("mesh")
        polarization = kwargs.get("polarization")
        in_out = kwargs.get("in_out", "auto")
        if mesh is None or polarization is None:
            raise ValueError("TriangularMesh computation requires `mesh` and `polarization`.")
        if output_field == "B":
            return magnet_trimesh_bfield(obs_local, mesh, polarization, in_out=in_out)
        if output_field == "H":
            return magnet_trimesh_hfield(obs_local, mesh, polarization, in_out=in_out)
        if output_field == "J":
            return magnet_trimesh_jfield(obs_local, mesh, polarization, in_out=in_out)
        return magnet_trimesh_mfield(obs_local, mesh, polarization, in_out=in_out)

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
    observers: Any,
    field_name: str,
    *,
    sumup: bool,
    in_out: str,
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
            if "in_out" in inspect.signature(method).parameters:
                terms.append(jnp.asarray(method(observers, in_out=in_out), dtype=jnp.float64))
            else:
                terms.append(jnp.asarray(method(observers), dtype=jnp.float64))

        broadcasted = jnp.broadcast_arrays(*terms)
        stacked = jnp.stack(broadcasted, axis=0)
        if sumup:
            return jnp.sum(stacked, axis=0), len(terms)
        return stacked, len(terms)

    method = getattr(source, f"get{field_name}", None)
    if method is None:
        raise TypeError(f"Source object {type(source).__name__!r} has no get{field_name} method.")
    if "in_out" in inspect.signature(method).parameters:
        return jnp.asarray(method(observers, in_out=in_out), dtype=jnp.float64), 1
    return jnp.asarray(method(observers), dtype=jnp.float64), 1


def _get_field_from_type(
    source_type: str,
    observers: ArrayLike,
    output_field: str,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    norm_type = _normalize_source_type(source_type)
    obs_input = jnp.asarray(observers, dtype=jnp.float64)
    pos_path, rot_path = broadcast_pose(position=position, orientation=orientation)
    n_path = int(pos_path.shape[0])

    pairwise = obs_input.ndim > 2
    obs_path_len = int(obs_input.shape[0]) if pairwise else 1
    n_eval = max(n_path, obs_path_len) if pairwise else n_path

    fields: list[jnp.ndarray] = []
    for i in range(n_eval):
        pose_i = min(i, n_path - 1)
        if pairwise:
            obs_i = obs_input[min(i, obs_path_len - 1)]
        else:
            obs_i = obs_input
        obs_local, rot = to_local_coordinates(
            obs_i,
            position=pos_path[pose_i],
            orientation=rot_path[pose_i],
        )
        field_local = _evaluate_core_field(norm_type, output_field, obs_local, kwargs)
        fields.append(to_global_field(field_local, rot))

    if n_eval == 1:
        field_global = fields[0]
    else:
        field_global = jnp.stack(fields, axis=0)
    return _apply_squeeze(field_global, obs_input, squeeze=squeeze, sumup=sumup, n_sources=1)


def getB(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    in_out: str = "auto",
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
            in_out=in_out,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getB with source objects.")
    obs = _extract_observers(observers)
    obs_input = jnp.asarray(obs, dtype=jnp.float64)
    field, n_sources = _evaluate_source_field(source, observers, "B", sumup=sumup, in_out=in_out)
    return _apply_squeeze(field, obs_input, squeeze=squeeze, sumup=sumup, n_sources=n_sources)


def getH(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    in_out: str = "auto",
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
            in_out=in_out,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getH with source objects.")
    obs = _extract_observers(observers)
    obs_input = jnp.asarray(obs, dtype=jnp.float64)
    field, n_sources = _evaluate_source_field(source, observers, "H", sumup=sumup, in_out=in_out)
    return _apply_squeeze(field, obs_input, squeeze=squeeze, sumup=sumup, n_sources=n_sources)


def getJ(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    in_out: str = "auto",
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
            in_out=in_out,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getJ with source objects.")
    obs = _extract_observers(observers)
    obs_input = jnp.asarray(obs, dtype=jnp.float64)
    field, n_sources = _evaluate_source_field(source, observers, "J", sumup=sumup, in_out=in_out)
    return _apply_squeeze(field, obs_input, squeeze=squeeze, sumup=sumup, n_sources=n_sources)


def getM(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    in_out: str = "auto",
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
            in_out=in_out,
            **kwargs,
        )

    if observers is None:
        raise ValueError("Observers are required when calling getM with source objects.")
    obs = _extract_observers(observers)
    obs_input = jnp.asarray(obs, dtype=jnp.float64)
    field, n_sources = _evaluate_source_field(source, observers, "M", sumup=sumup, in_out=in_out)
    return _apply_squeeze(field, obs_input, squeeze=squeeze, sumup=sumup, n_sources=n_sources)


def vgetH(source_type: str, observers: ArrayLike, **kwargs: ArrayLike) -> jnp.ndarray:
    """Vectorized alias for getH (shape-preserving)."""
    return getH(source_type, ensure_observers(observers), **kwargs)


def vgetB(source_type: str, observers: ArrayLike, **kwargs: ArrayLike) -> jnp.ndarray:
    """Vectorized alias for getB (shape-preserving)."""
    return getB(source_type, ensure_observers(observers), **kwargs)
