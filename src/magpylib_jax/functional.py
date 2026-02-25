"""Functional public interface and compatibility dispatch."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from itertools import product
from math import prod
from typing import Any

import numpy as np
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.geometry import (
    broadcast_pose,
    ensure_observers,
    to_global_field,
    to_local_coordinates,
)
from magpylib_jax.core.base import BaseSource, MagpylibBadUserInput, MagpylibMissingInput
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


def _check_getbh_output_type(output: str) -> str:
    acceptable = ("ndarray", "dataframe")
    if output not in acceptable:
        msg = f"Input output must be one of {acceptable}; instead received {output!r}."
        raise ValueError(msg)
    if output == "dataframe":
        try:  # pragma: no cover - import check
            import pandas as _  # noqa: F401
        except Exception as err:  # pragma: no cover
            msg = (
                "Input output='dataframe' requires Pandas installation, "
                "see https://pandas.pydata.org/docs/getting_started/install.html"
            )
            raise ModuleNotFoundError(msg) from err
    return output


def _check_pixel_agg(pixel_agg: str | None):
    if pixel_agg is None:
        return None
    if callable(pixel_agg):
        return pixel_agg
    if not isinstance(pixel_agg, str):
        raise AttributeError(
            "Input pixel_agg must be a reference to a NumPy callable that reduces "
            "an array shape like 'mean', 'std', 'median', 'min', ...; "
            f"instead received {pixel_agg!r}."
        )
    if not hasattr(jnp, pixel_agg):
        raise AttributeError(
            "Input pixel_agg must be a reference to a NumPy callable that reduces "
            "an array shape like 'mean', 'std', 'median', 'min', ...; "
            f"instead received {pixel_agg!r}."
        )
    return getattr(jnp, pixel_agg)


def _source_label(obj: object) -> str:
    style = getattr(obj, "style", None)
    label = getattr(style, "label", None) if style is not None else None
    if label is None:
        label = getattr(obj, "style_label", None)
    return label or obj.__class__.__name__


def _format_source_groups(source: object) -> list[dict[str, object]]:
    if isinstance(source, (list, tuple)):
        sources = list(source)
    else:
        sources = [source]
    if not sources:
        raise MagpylibBadUserInput("No sources provided.")

    groups: list[dict[str, object]] = []
    for src in sources:
        if isinstance(src, (list, tuple)) and not isinstance(src, (BaseSource, str, bytes)):
            groups.extend(_format_source_groups(src))
            continue
        if getattr(src, "_is_collection", False):
            child_sources = getattr(src, "sources", [])
            if not child_sources:
                raise MagpylibBadUserInput("No sources provided.")
            groups.append({"label": _source_label(src), "sources": child_sources})
        elif isinstance(src, BaseSource) or getattr(src, "_is_source", False):
            groups.append({"label": _source_label(src), "sources": [src]})
        else:
            raise MagpylibBadUserInput(f"Bad sources provided: {src!r}.")
    return groups


def _format_observers(observers: object, pixel_agg: str | None):
    from magpylib_jax.sensor import Sensor  # local import to avoid cycles

    if observers is None:
        raise MagpylibBadUserInput("No observers provided.")

    if getattr(observers, "_is_collection", False) or getattr(observers, "_is_sensor", False):
        observers = (observers,)

    if not isinstance(observers, (list, tuple, np.ndarray)):
        raise MagpylibBadUserInput("Bad observers provided.")

    if len(observers) == 0:  # type: ignore[arg-type]
        raise MagpylibBadUserInput("Bad observers provided.")

    # attempt to parse as single array-like
    try:
        arr = np.array(observers, dtype=float)
        if arr.shape[-1] != 3:
            raise ValueError
        pix_shapes = [(1, 3) if arr.shape == (3,) else arr.shape]
        return [Sensor(pixel=arr)], pix_shapes
    except Exception:
        pass

    sensors = []
    for obj in observers:  # type: ignore[iteration-over-annotation]
        if getattr(obj, "_is_sensor", False):
            sensors.append(obj)
        elif getattr(obj, "_is_collection", False):
            child_sensors = getattr(obj, "sensors", [])
            if not child_sensors:
                raise MagpylibBadUserInput("Bad observers provided.")
            sensors.extend(child_sensors)
        else:
            try:
                arr = np.array(obj, dtype=float)
                if arr.shape[-1] != 3:
                    raise ValueError
                sensors.append(Sensor(pixel=arr))
            except Exception as err:
                raise MagpylibBadUserInput("Bad observers provided.") from err

    pix_shapes = [
        (1, 3) if (s.pixel is None or np.asarray(s.pixel).shape == (3,)) else np.asarray(s.pixel).shape
        for s in sensors
    ]
    if pixel_agg is None and len(set(pix_shapes)) != 1:
        msg = (
            "Input observers must have similar shapes when pixel_agg is None; "
            f"instead received shapes {pix_shapes}."
        )
        raise MagpylibBadUserInput(msg)
    return sensors, pix_shapes


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
            raise MagpylibMissingInput("Input moment of Dipole must be set.")
        if output_field == "B":
            return dipole_bfield(obs_local, moment)
        if output_field == "H":
            return dipole_hfield(obs_local, moment)
        return jnp.zeros_like(obs_local, dtype=jnp.float64)

    if source_type == "circle":
        diameter = kwargs.get("diameter")
        current = kwargs.get("current")
        if diameter is None or current is None:
            raise MagpylibMissingInput("Input diameter of Circle must be set.")
        if output_field == "B":
            return current_circle_bfield(obs_local, diameter=diameter, current=current)
        if output_field == "H":
            return current_circle_hfield(obs_local, diameter=diameter, current=current)
        return jnp.zeros_like(obs_local, dtype=jnp.float64)

    if source_type == "cuboid":
        dimension = kwargs.get("dimension")
        polarization = kwargs.get("polarization")
        if dimension is None or polarization is None:
            raise MagpylibMissingInput("Input dimension of Cuboid must be set.")
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
            raise MagpylibMissingInput("Input dimension of Cylinder must be set.")
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
            raise MagpylibMissingInput("Input dimension of CylinderSegment must be set.")
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
            raise MagpylibMissingInput("Input diameter of Sphere must be set.")
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
            raise MagpylibMissingInput("Input vertices of Triangle must be set.")
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
            raise MagpylibMissingInput("Input vertices of Polyline must be set.")
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
            raise MagpylibMissingInput("Input vertices of TriangleSheet must be set.")
        if output_field == "B":
            return current_trisheet_bfield(obs_local, vertices, faces, current_densities)
        if output_field == "H":
            return current_trisheet_hfield(obs_local, vertices, faces, current_densities)
        return jnp.zeros_like(obs_local, dtype=jnp.float64)

    if source_type == "trianglestrip":
        vertices = kwargs.get("vertices")
        current = kwargs.get("current")
        if vertices is None or current is None:
            raise MagpylibMissingInput("Input vertices of TriangleStrip must be set.")
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
            raise MagpylibMissingInput("Input vertices of TriangularMesh must be set.")
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
            raise MagpylibMissingInput("Input vertices of Tetrahedron must be set.")
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


def _source_kwargs_from_object(source: object, *, in_out: str) -> tuple[str, dict[str, ArrayLike]]:
    stype = getattr(source, "_source_type", None)
    if stype is None:
        stype = type(source).__name__.lower()

    if stype == "cuboid":
        return stype, {"dimension": source.dimension, "polarization": source._polarization}
    if stype == "cylinder":
        return stype, {"dimension": source.dimension, "polarization": source._polarization}
    if stype == "cylindersegment":
        return stype, {
            "dimension": source.dimension,
            "polarization": source._polarization,
            "in_out": in_out,
        }
    if stype == "sphere":
        return stype, {"diameter": source.diameter, "polarization": source._polarization}
    if stype == "triangularmesh":
        return stype, {"mesh": source.mesh, "polarization": source._polarization, "in_out": in_out}
    if stype == "tetrahedron":
        return stype, {"vertices": source.vertices, "polarization": source._polarization, "in_out": in_out}
    if stype == "triangle":
        return stype, {"vertices": source.vertices, "polarization": source.polarization}
    if stype == "circle":
        return stype, {"diameter": source.diameter, "current": source.current}
    if stype == "polyline":
        verts = jnp.asarray(source.vertices, dtype=jnp.float64)
        return stype, {
            "segment_start": verts[:-1],
            "segment_end": verts[1:],
            "current": source.current,
        }
    if stype == "trianglesheet":
        return stype, {
            "vertices": source.vertices,
            "faces": source.faces,
            "current_densities": source.current_densities,
        }
    if stype == "trianglestrip":
        return stype, {"vertices": source.vertices, "current": source.current}
    if stype == "dipole":
        return stype, {"moment": source.moment}

    raise MagpylibBadUserInput(f"Unsupported source type {stype!r}.")


def _compute_field(
    source: str | object,
    observers: object,
    field: str,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    pixel_agg: str | None = None,
    output: str = "ndarray",
    in_out: str = "auto",
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    output = _check_getbh_output_type(output)
    pixel_agg_func = _check_pixel_agg(pixel_agg)

    if isinstance(source, str):
        src_type = _normalize_source_type(source)
        pos_path, rot_path = broadcast_pose(position=position, orientation=orientation)
        src_specs = [
            {
                "type": src_type,
                "pos": jnp.asarray(pos_path, dtype=jnp.float64),
                "rot": jnp.asarray(rot_path, dtype=jnp.float64),
                "kwargs": {**kwargs, "in_out": in_out},
                "label": src_type,
            }
        ]
        group_specs = [{"label": src_type, "indices": [0]}]
    else:
        groups = _format_source_groups(source)
        src_specs = []
        group_specs = []
        for group in groups:
            idxs: list[int] = []
            for src in group["sources"]:  # type: ignore[index]
                if hasattr(src, "_require_inputs"):
                    src._require_inputs()
                stype, skw = _source_kwargs_from_object(src, in_out=in_out)
                idxs.append(len(src_specs))
                src_specs.append(
                    {
                        "type": stype,
                        "pos": jnp.asarray(src._position, dtype=jnp.float64),
                        "rot": jnp.asarray(src._orientation.as_matrix(), dtype=jnp.float64),
                        "kwargs": skw,
                        "label": _source_label(src),
                    }
                )
            group_specs.append(
                {
                    "label": group["label"],
                    "indices": idxs,
                }
            )

    sensors, pix_shapes = _format_observers(observers, pixel_agg)
    pix_nums = [int(np.prod(ps[:-1])) for ps in pix_shapes]
    pix_inds = np.cumsum([0, *pix_nums])
    pix_all_same = len(set(pix_shapes)) == 1

    # precompute sensor data
    sensor_data = []
    for sens in sensors:
        pix = sens.pixel
        if pix is None:
            pix_arr = jnp.zeros((1, 3), dtype=jnp.float64)
            pix_shape = (1, 3)
        else:
            pix_arr = jnp.asarray(pix, dtype=jnp.float64)
            if pix_arr.shape == (3,):
                pix_arr = pix_arr[None, :]
            pix_shape = pix_arr.shape
        pix_flat = pix_arr.reshape((-1, 3))
        sensor_data.append(
            {
                "pix_flat": pix_flat,
                "pix_shape": pix_shape,
                "pos": jnp.asarray(sens._position, dtype=jnp.float64),
                "rot": jnp.asarray(sens._orientation.as_matrix(), dtype=jnp.float64),
                "handedness": sens.handedness,
            }
        )

    path_lengths = [int(spec["pos"].shape[0]) for spec in src_specs] + [
        int(sd["pos"].shape[0]) for sd in sensor_data
    ]
    max_path_len = max(path_lengths) if path_lengths else 1

    b_sources = []
    for spec in src_specs:
        b_paths = []
        for p in range(max_path_len):
            poso_parts = []
            for sd in sensor_data:
                idx = min(p, int(sd["pos"].shape[0]) - 1)
                rot = sd["rot"][idx]
                pos = sd["pos"][idx]
                obs = sd["pix_flat"] @ rot.T + pos
                poso_parts.append(obs)
            poso = jnp.concatenate(poso_parts, axis=0)

            src_idx = min(p, int(spec["pos"].shape[0]) - 1)
            src_pos = spec["pos"][src_idx]
            src_rot = spec["rot"][src_idx]
            obs_local = (poso - src_pos) @ src_rot
            field_local = _evaluate_core_field(spec["type"], field, obs_local, spec["kwargs"])
            field_global = field_local @ src_rot.T

            slices = []
            offset = 0
            for sd, pix_count in zip(sensor_data, pix_nums, strict=False):
                seg = field_global[offset : offset + pix_count]
                sens_rot = sd["rot"][min(p, int(sd["rot"].shape[0]) - 1)]
                seg = seg @ sens_rot
                if sd["handedness"] == "left":
                    seg = seg * jnp.array([-1.0, 1.0, 1.0], dtype=jnp.float64)
                slices.append(seg)
                offset += pix_count
            b_paths.append(jnp.concatenate(slices, axis=0))
        b_sources.append(jnp.stack(b_paths, axis=0))

    if not b_sources:
        raise MagpylibBadUserInput("No sources provided.")

    B_src = jnp.stack(b_sources, axis=0)
    b_groups = []
    for group in group_specs:
        idxs = group["indices"]
        if len(idxs) == 1:
            b_groups.append(B_src[idxs[0]])
        else:
            b_groups.append(jnp.sum(jnp.take(B_src, jnp.asarray(idxs), axis=0), axis=0))
    B = jnp.stack(b_groups, axis=0)
    n_groups = len(group_specs)

    if pix_all_same:
        B = B.reshape((n_groups, max_path_len, len(sensors), *pix_shapes[0]))
        if pixel_agg is not None:
            axes = tuple(range(3, B.ndim - 1))
            if axes:
                B = pixel_agg_func(B, axis=axes)
            else:
                B = pixel_agg_func(B)
    else:
        # pixel_agg must be provided when shapes differ
        Bsplit = jnp.split(B, pix_inds[1:-1], axis=2)
        Bagg = [jnp.expand_dims(pixel_agg_func(b, axis=2), axis=2) for b in Bsplit]
        B = jnp.concatenate(Bagg, axis=2)

    if sumup:
        B = jnp.sum(B, axis=0, keepdims=True)

    if output == "dataframe":
        import pandas as pd  # type: ignore

        if sumup and len(group_specs) > 1:
            src_ids = [f"sumup ({len(group_specs)})"]
        else:
            src_ids = [spec["label"] for spec in group_specs]
        sens_ids = [
            getattr(sens.style, "label", None) or getattr(sens, "style_label", None) or "Sensor"
            for sens in sensors
        ]
        num_pixels = int(np.prod(pix_shapes[0][:-1])) if pixel_agg is None else 1
        df_field = pd.DataFrame(
            data=product(src_ids, range(max_path_len), sens_ids, range(num_pixels)),
            columns=["source", "path", "sensor", "pixel"],
        )
        df_field[[field + k for k in "xyz"]] = np.asarray(B).reshape(-1, 3)
        return df_field

    if squeeze:
        B = jnp.squeeze(B)
    elif pixel_agg is not None:
        B = jnp.expand_dims(B, axis=-2)
    return B


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
    pixel_agg: str | None = None,
    output: str = "ndarray",
    in_out: str = "auto",
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return B-field in Tesla from source type strings or source objects."""
    return _compute_field(
        source,
        observers,
        "B",
        position=position,
        orientation=orientation,
        squeeze=squeeze,
        sumup=sumup,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


def getH(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    pixel_agg: str | None = None,
    output: str = "ndarray",
    in_out: str = "auto",
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return H-field in A/m from source type strings or source objects."""
    return _compute_field(
        source,
        observers,
        "H",
        position=position,
        orientation=orientation,
        squeeze=squeeze,
        sumup=sumup,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


def getJ(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    pixel_agg: str | None = None,
    output: str = "ndarray",
    in_out: str = "auto",
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return J-field from source type strings or source objects."""
    return _compute_field(
        source,
        observers,
        "J",
        position=position,
        orientation=orientation,
        squeeze=squeeze,
        sumup=sumup,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


def getM(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    pixel_agg: str | None = None,
    output: str = "ndarray",
    in_out: str = "auto",
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return M-field from source type strings or source objects."""
    return _compute_field(
        source,
        observers,
        "M",
        position=position,
        orientation=orientation,
        squeeze=squeeze,
        sumup=sumup,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


def vgetH(source_type: str, observers: ArrayLike, **kwargs: ArrayLike) -> jnp.ndarray:
    """Vectorized alias for getH (shape-preserving)."""
    return getH(source_type, ensure_observers(observers), **kwargs)


def vgetB(source_type: str, observers: ArrayLike, **kwargs: ArrayLike) -> jnp.ndarray:
    """Vectorized alias for getB (shape-preserving)."""
    return getB(source_type, ensure_observers(observers), **kwargs)
