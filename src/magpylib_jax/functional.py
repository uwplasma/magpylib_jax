"""Functional public interface and compatibility dispatch."""

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Sequence
from itertools import product
from math import prod
from typing import Any

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.base import BaseSource, MagpylibBadUserInput, MagpylibMissingInput
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
    _in_out_flag,
    _inside_mask_mesh_masked,
    _strip_current_densities,
    _strip_triangles,
    current_polyline_bfield,
    current_polyline_bfield_masked,
    current_polyline_hfield,
    current_trisheet_bfield,
    current_trisheet_bfield_masked,
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
    magnet_trimesh_bfield_precomp_masked,
    magnet_trimesh_hfield,
    magnet_trimesh_jfield,
    magnet_trimesh_mfield,
    precompute_trimesh_geometry,
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

_SOURCE_TYPE_ORDER = (
    "dipole",
    "circle",
    "cuboid",
    "cylinder",
    "cylindersegment",
    "sphere",
    "triangle",
    "polyline",
    "trianglesheet",
    "trianglestrip",
    "triangularmesh",
    "tetrahedron",
)
_SOURCE_TYPE_IDS = {name: idx for idx, name in enumerate(_SOURCE_TYPE_ORDER)}
_SUPPORTED_PIXEL_AGGS = {"mean", "sum", "min", "max"}
_MAX_SOURCE_CHUNK_SIZE = 256
_SOURCE_PREP_CACHE_MAX = 8
_SOURCE_PREP_CACHE: OrderedDict[
    tuple[object, ...],
    tuple[dict[str, jnp.ndarray], dict[str, object]],
] = OrderedDict()


def _select_source_chunk_size(
    n_sources: int, *, observer_count: int, all_circle: bool
) -> int:
    if n_sources <= 1:
        return 1
    if all_circle:
        candidates = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
        target_bytes = 4 * 1024 * 1024
    else:
        candidates = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256)
        target_bytes = 16 * 1024 * 1024

    bytes_per_source = max(1, observer_count) * 3 * 8
    max_by_memory = max(1, target_bytes // bytes_per_source)
    upper = min(_MAX_SOURCE_CHUNK_SIZE, n_sources, max_by_memory)
    for cand in reversed(candidates):
        if cand <= upper:
            return cand
    return 1


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


def _is_array_like(obj: object) -> bool:
    if isinstance(obj, (jnp.ndarray, jax.Array)):
        return True
    if isinstance(obj, jax.core.Tracer):
        return True
    return hasattr(obj, "shape") and hasattr(obj, "dtype")


def _has_tracer(obj: object) -> bool:
    if isinstance(obj, jax.core.Tracer):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_has_tracer(item) for item in obj)
    if isinstance(obj, dict):
        return any(_has_tracer(val) for val in obj.values())
    return False


def _pad_path(arr: ArrayLike, target_len: int) -> jnp.ndarray:
    arr_jnp = jnp.asarray(arr, dtype=jnp.float64)
    if arr_jnp.shape[0] == target_len:
        return arr_jnp
    if arr_jnp.shape[0] == 1:
        return jnp.broadcast_to(arr_jnp, (target_len,) + arr_jnp.shape[1:])
    pad_len = target_len - arr_jnp.shape[0]
    pad = jnp.broadcast_to(arr_jnp[-1:], (pad_len,) + arr_jnp.shape[1:])
    return jnp.concatenate((arr_jnp, pad), axis=0)


def _pad_axis0(arr: ArrayLike, target_len: int, pad_value: float = 0.0) -> jnp.ndarray:
    arr_jnp = jnp.asarray(arr, dtype=jnp.float64)
    if arr_jnp.shape[0] == target_len:
        return arr_jnp
    pad_shape = (target_len - arr_jnp.shape[0],) + arr_jnp.shape[1:]
    pad = jnp.full(pad_shape, pad_value, dtype=arr_jnp.dtype)
    return jnp.concatenate((arr_jnp, pad), axis=0)


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


def _lru_get(
    cache: OrderedDict[tuple[object, ...], object],
    key: tuple[object, ...],
) -> object | None:
    val = cache.get(key)
    if val is not None:
        cache.move_to_end(key)
    return val


def _lru_put(
    cache: OrderedDict[tuple[object, ...], object],
    key: tuple[object, ...],
    value: object,
    *,
    max_items: int,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def _circle_source_cache_key(source: object, *, in_out: str) -> tuple[object, ...] | None:
    if isinstance(source, str):
        return None
    if _has_tracer(source):
        return None
    try:
        groups = _format_source_groups(source)
    except Exception:
        return None

    key_parts: list[object] = ["circle", in_out]
    for group in groups:
        group_label = group.get("label")
        group_sources = group.get("sources")
        if not isinstance(group_sources, list):
            return None
        key_parts.append(("group", group_label, len(group_sources)))
        for src in group_sources:
            if getattr(src, "_source_type", None) != "circle":
                return None
            current = getattr(src, "current", None)
            diameter = getattr(src, "diameter", None)
            if current is None or diameter is None:
                return None
            key_parts.append(
                (
                    id(src),
                    float(current),
                    float(diameter),
                    id(getattr(src, "_position", None)),
                    id(getattr(src, "_orientation", None)),
                )
            )
    return tuple(key_parts)


def _format_observers(observers: object, pixel_agg: str | None):
    from magpylib_jax.sensor import Sensor  # local import to avoid cycles

    if observers is None:
        raise MagpylibBadUserInput("No observers provided.")

    if getattr(observers, "_is_collection", False) or getattr(observers, "_is_sensor", False):
        observers = (observers,)

    if not isinstance(observers, (list, tuple, jnp.ndarray, jax.Array)) and not _is_array_like(
        observers
    ):
        raise MagpylibBadUserInput("Bad observers provided.")

    if len(observers) == 0:  # type: ignore[arg-type]
        raise MagpylibBadUserInput("Bad observers provided.")

    # attempt to parse as single array-like
    try:
        arr = jnp.asarray(observers, dtype=jnp.float64)
        if arr.shape[-1] != 3:
            raise ValueError
        pix_shapes = [(1, 3) if arr.shape == (3,) else tuple(arr.shape)]
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
                arr = jnp.asarray(obj, dtype=jnp.float64)
                if arr.shape[-1] != 3:
                    raise ValueError
                sensors.append(Sensor(pixel=arr))
            except Exception as err:
                raise MagpylibBadUserInput("Bad observers provided.") from err

    pix_shapes = [
        (1, 3)
        if (s.pixel is None or jnp.asarray(s.pixel).shape == (3,))
        else tuple(jnp.asarray(s.pixel).shape)
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


def _build_source_specs(
    source: str | object,
    *,
    position: ArrayLike,
    orientation: ArrayLike | None,
    in_out: str,
    kwargs: dict[str, ArrayLike],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
        return src_specs, group_specs

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
        group_specs.append({"label": group["label"], "indices": idxs})
    return src_specs, group_specs


def _prepare_sources_jit(
    source: str | object,
    *,
    position: ArrayLike,
    orientation: ArrayLike | None,
    in_out: str,
    kwargs: dict[str, ArrayLike],
) -> tuple[dict[str, jnp.ndarray], dict[str, object]]:
    cache_key = _circle_source_cache_key(source, in_out=in_out)
    if cache_key is not None:
        cached = _lru_get(_SOURCE_PREP_CACHE, cache_key)
        if cached is not None:
            cached_src, cached_meta = cached
            return cached_src, cached_meta

    src_specs, group_specs = _build_source_specs(
        source, position=position, orientation=orientation, in_out=in_out, kwargs=kwargs
    )
    if not src_specs:
        raise MagpylibBadUserInput("No sources provided.")

    in_out_flag = _in_out_flag(in_out)
    max_segments = 1
    max_sheet_faces = 1
    max_mesh_faces = 1
    src_data: list[dict[str, object]] = []

    for spec in src_specs:
        stype = spec["type"]
        skw = spec["kwargs"]  # type: ignore[assignment]
        data: dict[str, object] = {
            "type": stype,
            "pos": spec["pos"],
            "rot": spec["rot"],
            "label": spec["label"],
        }

        if stype == "dipole":
            if skw.get("moment") is None:
                raise MagpylibMissingInput("Input moment of Dipole must be set.")
            data["moment"] = jnp.asarray(skw["moment"], dtype=jnp.float64)
        elif stype == "circle":
            if skw.get("diameter") is None or skw.get("current") is None:
                raise MagpylibMissingInput("Input diameter of Circle must be set.")
            data["diameter"] = jnp.asarray(skw["diameter"], dtype=jnp.float64)
            data["current"] = jnp.asarray(skw["current"], dtype=jnp.float64)
        elif stype == "cuboid":
            if skw.get("dimension") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input dimension of Cuboid must be set.")
            data["cuboid_dim"] = jnp.asarray(skw["dimension"], dtype=jnp.float64)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=jnp.float64)
        elif stype == "cylinder":
            if skw.get("dimension") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input dimension of Cylinder must be set.")
            data["cylinder_dim"] = jnp.asarray(skw["dimension"], dtype=jnp.float64)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=jnp.float64)
        elif stype == "cylindersegment":
            if skw.get("dimension") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input dimension of CylinderSegment must be set.")
            data["cseg_dim"] = jnp.asarray(skw["dimension"], dtype=jnp.float64)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=jnp.float64)
        elif stype == "sphere":
            if skw.get("diameter") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input diameter of Sphere must be set.")
            data["diameter"] = jnp.asarray(skw["diameter"], dtype=jnp.float64)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=jnp.float64)
        elif stype == "triangle":
            if skw.get("vertices") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input vertices of Triangle must be set.")
            data["triangle_vertices"] = jnp.asarray(skw["vertices"], dtype=jnp.float64)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=jnp.float64)
        elif stype == "polyline":
            if (
                skw.get("segment_start") is None
                or skw.get("segment_end") is None
                or skw.get("current") is None
            ):
                raise MagpylibMissingInput("Input vertices of Polyline must be set.")
            seg_start = jnp.asarray(skw["segment_start"], dtype=jnp.float64)
            seg_end = jnp.asarray(skw["segment_end"], dtype=jnp.float64)
            if seg_start.ndim == 1:
                seg_start = seg_start[None, :]
                seg_end = seg_end[None, :]
            data["segment_start"] = seg_start
            data["segment_end"] = seg_end
            data["current"] = jnp.asarray(skw["current"], dtype=jnp.float64)
            max_segments = max(max_segments, int(seg_start.shape[0]))
        elif stype == "trianglesheet":
            if (
                skw.get("vertices") is None
                or skw.get("faces") is None
                or skw.get("current_densities") is None
            ):
                raise MagpylibMissingInput("Input vertices of TriangleSheet must be set.")
            verts = jnp.asarray(skw["vertices"], dtype=jnp.float64)
            faces = jnp.asarray(skw["faces"], dtype=jnp.int32)
            cds = jnp.asarray(skw["current_densities"], dtype=jnp.float64)
            tris = verts[faces]
            data["sheet_tris"] = tris
            data["sheet_cd"] = cds
            max_sheet_faces = max(max_sheet_faces, int(tris.shape[0]))
        elif stype == "trianglestrip":
            if skw.get("vertices") is None or skw.get("current") is None:
                raise MagpylibMissingInput("Input vertices of TriangleStrip must be set.")
            verts = jnp.asarray(skw["vertices"], dtype=jnp.float64)
            curr = jnp.asarray(skw["current"], dtype=jnp.float64)
            tris = _strip_triangles(verts)
            cds = _strip_current_densities(verts, curr)
            data["sheet_tris"] = tris
            data["sheet_cd"] = cds
            data["current"] = curr
            max_sheet_faces = max(max_sheet_faces, int(tris.shape[0]))
        elif stype == "triangularmesh":
            if skw.get("mesh") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input vertices of TriangularMesh must be set.")
            mesh_raw = jnp.asarray(skw["mesh"], dtype=jnp.float64)
            if mesh_raw.ndim == 4:
                raise ValueError("TriangularMesh mesh input must have shape (n_faces,3,3).")
            mesh_arr, nvec, L, l1, l2 = precompute_trimesh_geometry(mesh_raw)
            data["mesh"] = mesh_arr
            data["mesh_nvec"] = nvec
            data["mesh_L"] = L
            data["mesh_l1"] = l1
            data["mesh_l2"] = l2
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=jnp.float64)
            max_mesh_faces = max(max_mesh_faces, int(mesh_arr.shape[0]))
        elif stype == "tetrahedron":
            if skw.get("vertices") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input vertices of Tetrahedron must be set.")
            data["tetra_vertices"] = jnp.asarray(skw["vertices"], dtype=jnp.float64)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=jnp.float64)
        else:
            raise MagpylibBadUserInput(f"Unsupported source type {stype!r}.")

        src_data.append(data)

    type_ids = jnp.asarray(
        [_SOURCE_TYPE_IDS[data["type"]] for data in src_data],
        dtype=jnp.int32,  # type: ignore[index]
    )
    group_index = [0] * len(src_data)
    for gid, group in enumerate(group_specs):
        for idx in group["indices"]:  # type: ignore[index]
            group_index[idx] = gid
    group_index = jnp.asarray(group_index, dtype=jnp.int32)

    moment = []
    diameter = []
    cuboid_dim = []
    cylinder_dim = []
    cseg_dim = []
    polarization = []
    triangle_vertices = []
    tetra_vertices = []
    current = []
    poly_seg_start = []
    poly_seg_end = []
    poly_seg_mask = []
    sheet_tris = []
    sheet_cd = []
    sheet_mask = []
    mesh_faces = []
    mesh_mask = []
    mesh_nvec = []
    mesh_L = []
    mesh_l1 = []
    mesh_l2 = []
    pos_list = []
    rot_list = []

    for data in src_data:
        stype = data["type"]
        pos_list.append(data["pos"])
        rot_list.append(data["rot"])

        moment.append(jnp.asarray(data.get("moment", jnp.zeros(3)), dtype=jnp.float64))
        diameter.append(jnp.asarray(data.get("diameter", 0.0), dtype=jnp.float64))
        cuboid_dim.append(jnp.asarray(data.get("cuboid_dim", jnp.zeros(3)), dtype=jnp.float64))
        cylinder_dim.append(jnp.asarray(data.get("cylinder_dim", jnp.zeros(2)), dtype=jnp.float64))
        cseg_dim.append(jnp.asarray(data.get("cseg_dim", jnp.zeros(5)), dtype=jnp.float64))
        polarization.append(jnp.asarray(data.get("polarization", jnp.zeros(3)), dtype=jnp.float64))
        triangle_vertices.append(
            jnp.asarray(data.get("triangle_vertices", jnp.zeros((3, 3))), dtype=jnp.float64)
        )
        tetra_vertices.append(
            jnp.asarray(data.get("tetra_vertices", jnp.zeros((4, 3))), dtype=jnp.float64)
        )
        current.append(jnp.asarray(data.get("current", 0.0), dtype=jnp.float64))

        if stype == "polyline":
            seg_start = data["segment_start"]
            seg_end = data["segment_end"]
            seg_count = int(seg_start.shape[0])
            seg_mask = jnp.concatenate(
                (
                    jnp.ones((seg_count,), dtype=jnp.float64),
                    jnp.zeros((max_segments - seg_count,), dtype=jnp.float64),
                ),
                axis=0,
            )
            poly_seg_start.append(_pad_axis0(seg_start, max_segments))
            poly_seg_end.append(_pad_axis0(seg_end, max_segments))
            poly_seg_mask.append(seg_mask)
        else:
            poly_seg_start.append(jnp.zeros((max_segments, 3), dtype=jnp.float64))
            poly_seg_end.append(jnp.zeros((max_segments, 3), dtype=jnp.float64))
            poly_seg_mask.append(jnp.zeros((max_segments,), dtype=jnp.float64))

        if stype in ("trianglesheet", "trianglestrip"):
            tris = data["sheet_tris"]
            cds = data["sheet_cd"]
            face_count = int(tris.shape[0])
            mask = jnp.concatenate(
                (
                    jnp.ones((face_count,), dtype=jnp.float64),
                    jnp.zeros((max_sheet_faces - face_count,), dtype=jnp.float64),
                ),
                axis=0,
            )
            sheet_tris.append(_pad_axis0(tris, max_sheet_faces))
            sheet_cd.append(_pad_axis0(cds, max_sheet_faces))
            sheet_mask.append(mask)
        else:
            sheet_tris.append(jnp.zeros((max_sheet_faces, 3, 3), dtype=jnp.float64))
            sheet_cd.append(jnp.zeros((max_sheet_faces, 3), dtype=jnp.float64))
            sheet_mask.append(jnp.zeros((max_sheet_faces,), dtype=jnp.float64))

        if stype == "triangularmesh":
            mesh_arr = data["mesh"]
            nvec = data["mesh_nvec"]
            L = data["mesh_L"]
            l1 = data["mesh_l1"]
            l2 = data["mesh_l2"]
            face_count = int(mesh_arr.shape[0])
            mask = jnp.concatenate(
                (
                    jnp.ones((face_count,), dtype=jnp.float64),
                    jnp.zeros((max_mesh_faces - face_count,), dtype=jnp.float64),
                ),
                axis=0,
            )
            mesh_faces.append(_pad_axis0(mesh_arr, max_mesh_faces))
            mesh_nvec.append(_pad_axis0(nvec, max_mesh_faces))
            mesh_L.append(_pad_axis0(L, max_mesh_faces))
            mesh_l1.append(_pad_axis0(l1, max_mesh_faces))
            mesh_l2.append(_pad_axis0(l2, max_mesh_faces))
            mesh_mask.append(mask)
        else:
            mesh_faces.append(jnp.zeros((max_mesh_faces, 3, 3), dtype=jnp.float64))
            mesh_nvec.append(jnp.zeros((max_mesh_faces, 3), dtype=jnp.float64))
            mesh_L.append(jnp.zeros((max_mesh_faces, 3, 3), dtype=jnp.float64))
            mesh_l1.append(jnp.zeros((max_mesh_faces, 3), dtype=jnp.float64))
            mesh_l2.append(jnp.zeros((max_mesh_faces, 3), dtype=jnp.float64))
            mesh_mask.append(jnp.zeros((max_mesh_faces,), dtype=jnp.float64))

    src_arrays = {
        "type_id": type_ids,
        "pos_list": pos_list,
        "rot_list": rot_list,
        "moment": jnp.stack(moment, axis=0),
        "diameter": jnp.stack(diameter, axis=0),
        "cuboid_dim": jnp.stack(cuboid_dim, axis=0),
        "cylinder_dim": jnp.stack(cylinder_dim, axis=0),
        "cseg_dim": jnp.stack(cseg_dim, axis=0),
        "polarization": jnp.stack(polarization, axis=0),
        "triangle_vertices": jnp.stack(triangle_vertices, axis=0),
        "tetra_vertices": jnp.stack(tetra_vertices, axis=0),
        "current": jnp.stack(current, axis=0),
        "poly_seg_start": jnp.stack(poly_seg_start, axis=0),
        "poly_seg_end": jnp.stack(poly_seg_end, axis=0),
        "poly_seg_mask": jnp.stack(poly_seg_mask, axis=0),
        "sheet_tris": jnp.stack(sheet_tris, axis=0),
        "sheet_cd": jnp.stack(sheet_cd, axis=0),
        "sheet_mask": jnp.stack(sheet_mask, axis=0),
        "mesh_faces": jnp.stack(mesh_faces, axis=0),
        "mesh_mask": jnp.stack(mesh_mask, axis=0),
        "mesh_nvec": jnp.stack(mesh_nvec, axis=0),
        "mesh_L": jnp.stack(mesh_L, axis=0),
        "mesh_l1": jnp.stack(mesh_l1, axis=0),
        "mesh_l2": jnp.stack(mesh_l2, axis=0),
        "group_index": group_index,
        "in_out_flag": jnp.full((len(src_data),), in_out_flag, dtype=jnp.int32),
    }

    meta = {
        "group_labels": [group["label"] for group in group_specs],
        "n_groups": len(group_specs),
    }
    if cache_key is not None:
        _lru_put(
            _SOURCE_PREP_CACHE, cache_key, (src_arrays, meta), max_items=_SOURCE_PREP_CACHE_MAX
        )
    return src_arrays, meta


def _prepare_sensors_jit(
    observers: object,
    *,
    pixel_agg: str | None,
) -> tuple[dict[str, jnp.ndarray], dict[str, object]]:
    if observers is None:
        raise MagpylibBadUserInput("No observers provided.")

    if (
        _is_array_like(observers)
        and not isinstance(observers, (list, tuple))
        and not getattr(observers, "_is_sensor", False)
        and not getattr(observers, "_is_collection", False)
    ):
        pix_arr = jnp.asarray(observers, dtype=jnp.float64)
        if pix_arr.shape[-1] != 3:
            raise MagpylibBadUserInput("Bad observers provided.")
        if pix_arr.shape == (3,):
            pix_arr = pix_arr[None, :]
            pix_shape = (1, 3)
        else:
            pix_shape = tuple(pix_arr.shape)
        pix_flat = pix_arr.reshape((-1, 3))
        sensors = [
            {
                "pix_flat": pix_flat,
                "pix_shape": pix_shape,
                "pos": jnp.zeros((1, 3), dtype=jnp.float64),
                "rot": jnp.eye(3, dtype=jnp.float64)[None, :, :],
                "handedness": "right",
                "label": "Sensor",
            }
        ]
        pix_shapes = [pix_shape]
    else:
        sensors_list, pix_shapes = _format_observers(observers, pixel_agg)
        sensors = []
        for sens in sensors_list:
            pix = sens.pixel
            if pix is None:
                pix_arr = jnp.zeros((1, 3), dtype=jnp.float64)
                pix_shape = (1, 3)
            else:
                pix_arr = jnp.asarray(pix, dtype=jnp.float64)
                if pix_arr.shape == (3,):
                    pix_arr = pix_arr[None, :]
                pix_shape = tuple(pix_arr.shape)
            pix_flat = pix_arr.reshape((-1, 3))
            label = (
                getattr(sens.style, "label", None)
                if getattr(sens, "style", None) is not None
                else None
            )
            label = label or getattr(sens, "style_label", None) or "Sensor"
            sensors.append(
                {
                    "pix_flat": pix_flat,
                    "pix_shape": pix_shape,
                    "pos": jnp.asarray(sens._position, dtype=jnp.float64),
                    "rot": jnp.asarray(sens._orientation.as_matrix(), dtype=jnp.float64),
                    "handedness": sens.handedness,
                    "label": label,
                }
            )

    pix_nums = [int(prod(ps[:-1])) for ps in pix_shapes]
    max_pix = max(pix_nums) if pix_nums else 1
    pix_all_same = len(set(pix_shapes)) == 1
    if pixel_agg is None and not pix_all_same:
        msg = (
            "Input observers must have similar shapes when pixel_agg is None; "
            f"instead received shapes {pix_shapes}."
        )
        raise MagpylibBadUserInput(msg)

    pix_flat_list = []
    pix_mask_list = []
    pos_list = []
    rot_list = []
    handedness_list = []
    labels = []
    for sens in sensors:
        pix_flat = sens["pix_flat"]
        pix_count = pix_flat.shape[0]
        pad_len = max_pix - int(pix_count)
        if pad_len > 0:
            pix_flat = _pad_axis0(pix_flat, max_pix)
        mask = jnp.concatenate(
            (jnp.ones((pix_count,), dtype=jnp.float64), jnp.zeros((pad_len,), dtype=jnp.float64)),
            axis=0,
        )
        pix_flat_list.append(pix_flat)
        pix_mask_list.append(mask)
        pos_list.append(sens["pos"])
        rot_list.append(sens["rot"])
        handedness_list.append(sens["handedness"])
        labels.append(sens["label"])

    hand_vec = [
        jnp.array([-1.0, 1.0, 1.0], dtype=jnp.float64)
        if h == "left"
        else jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)
        for h in handedness_list
    ]

    sens_arrays = {
        "pix_flat": jnp.stack(pix_flat_list, axis=0),
        "pix_mask": jnp.stack(pix_mask_list, axis=0),
        "pos_list": pos_list,
        "rot_list": rot_list,
        "handedness": jnp.stack(hand_vec, axis=0),
    }
    pix_inds = [0]
    for pix_num in pix_nums:
        pix_inds.append(pix_inds[-1] + int(pix_num))

    meta = {
        "pix_shapes": pix_shapes,
        "pix_nums": pix_nums,
        "pix_all_same": pix_all_same,
        "pix_inds": tuple(pix_inds),
        "sensor_labels": labels,
    }
    return sens_arrays, meta


def _segment_sum(data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int) -> jnp.ndarray:
    try:
        return jax.lax.segment_sum(data, segment_ids, num_segments)
    except AttributeError:  # pragma: no cover
        return jax.ops.segment_sum(data, segment_ids, num_segments)


def _is_identity_rotation_stack(rot: jnp.ndarray, *, atol: float = 1e-12) -> bool:
    eye = jnp.eye(3, dtype=rot.dtype)
    return _safe_static_bool(jnp.all(jnp.abs(rot - eye) <= atol), default=False)


def _is_all_right_handed(handedness: jnp.ndarray) -> bool:
    right = jnp.array([1.0, 1.0, 1.0], dtype=handedness.dtype)
    return _safe_static_bool(jnp.all(handedness == right), default=False)


def _safe_static_bool(value: jnp.ndarray, *, default: bool) -> bool:
    try:
        return bool(jax.device_get(value))
    except Exception:
        return default


def _pad_sources_for_chunking(
    src_arrays: dict[str, jnp.ndarray], *, chunk_size: int
) -> dict[str, jnp.ndarray]:
    n_src = int(src_arrays["type_id"].shape[0])
    pad = (-n_src) % chunk_size
    source_mask = jnp.concatenate(
        (
            jnp.ones((n_src,), dtype=jnp.float64),
            jnp.zeros((pad,), dtype=jnp.float64),
        ),
        axis=0,
    )
    if pad == 0:
        return {**src_arrays, "source_mask": source_mask}

    out: dict[str, jnp.ndarray] = {}
    for key, arr in src_arrays.items():
        if arr.shape[0] != n_src:
            out[key] = arr
            continue
        pad_cfg = [(0, pad), *[(0, 0)] * (arr.ndim - 1)]
        out[key] = jnp.pad(arr, pad_cfg)
    out["source_mask"] = source_mask
    return out


def _compute_field_jit_core(
    src: dict[str, jnp.ndarray],
    sens: dict[str, jnp.ndarray],
    *,
    field: str,
    in_out: str,
    n_groups: int,
    chunk_size: int,
    all_circle: bool,
    source_rot_identity: bool,
    sensor_rot_identity: bool,
    right_handed: bool,
) -> jnp.ndarray:
    type_id = src["type_id"]
    pos = src["pos"]
    rot = src["rot"]
    group_index = src["group_index"]
    source_mask = src["source_mask"]

    pix_flat = sens["pix_flat"]
    pix_mask = sens["pix_mask"]
    sens_pos = sens["pos"]
    sens_rot = sens["rot"]
    hand_vec = sens["handedness"]

    n_sensors = pix_flat.shape[0]
    max_pix = pix_flat.shape[1]
    n_path = pos.shape[1]
    n_sources = type_id.shape[0]
    n_chunks = n_sources // chunk_size
    pix_valid_mask = pix_mask[None, :, :, None] > 0

    if (
        all_circle
        and n_groups == 1
        and n_sensors == 1
        and source_rot_identity
        and sensor_rot_identity
        and right_handed
    ):

        def _slice_chunk(arr: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
            return jax.lax.dynamic_slice_in_dim(arr, start, chunk_size, axis=0)

        def _step_fast(_, t):
            obs_flat = pix_flat[0] + sens_pos[0, t][None, :]
            pos_t = pos[:, t]

            def _chunk_step(carry: jnp.ndarray, chunk_idx: jnp.ndarray) -> tuple[jnp.ndarray, None]:
                start = chunk_idx * chunk_size
                pos_chunk = _slice_chunk(pos_t, start)
                obs_local = obs_flat[None, :, :] - pos_chunk[:, None, :]
                if field == "B":
                    fields = jax.vmap(current_circle_bfield, in_axes=(0, 0, 0))(
                        obs_local,
                        _slice_chunk(src["diameter"], start),
                        _slice_chunk(src["current"], start),
                    )
                elif field == "H":
                    fields = jax.vmap(current_circle_hfield, in_axes=(0, 0, 0))(
                        obs_local,
                        _slice_chunk(src["diameter"], start),
                        _slice_chunk(src["current"], start),
                    )
                else:
                    fields = jnp.zeros(
                        (chunk_size, obs_flat.shape[0], 3),
                        dtype=jnp.float64,
                    )
                fields = fields * _slice_chunk(source_mask, start)[:, None, None]
                return carry + jnp.sum(fields, axis=0), None

            init = jnp.zeros((obs_flat.shape[0], 3), dtype=jnp.float64)
            out, _ = jax.lax.scan(_chunk_step, init, jnp.arange(n_chunks))
            return None, out

        _, b_path_fast = jax.lax.scan(_step_fast, None, jnp.arange(n_path))
        return b_path_fast[None, :, None, :, :]

    def _mesh_inside(
        obs_local: jnp.ndarray, mesh_faces: jnp.ndarray, mesh_mask: jnp.ndarray, flag: jnp.ndarray
    ) -> jnp.ndarray:
        return jax.lax.switch(
            flag,
            (
                lambda: _inside_mask_mesh_masked(obs_local, mesh_faces, mesh_mask),
                lambda: jnp.ones((obs_local.shape[0],), dtype=bool),
                lambda: jnp.zeros((obs_local.shape[0],), dtype=bool),
            ),
        )

    def per_source(
        stype: jnp.ndarray,
        pos_t: jnp.ndarray,
        rot_t: jnp.ndarray,
        moment: jnp.ndarray,
        diameter: jnp.ndarray,
        cub_dim: jnp.ndarray,
        cyl_dim: jnp.ndarray,
        cseg_dim: jnp.ndarray,
        pol: jnp.ndarray,
        tri_vertices: jnp.ndarray,
        tet_vertices: jnp.ndarray,
        current: jnp.ndarray,
        seg_start: jnp.ndarray,
        seg_end: jnp.ndarray,
        seg_mask: jnp.ndarray,
        sheet_tris: jnp.ndarray,
        sheet_cd: jnp.ndarray,
        sheet_mask: jnp.ndarray,
        mesh_faces: jnp.ndarray,
        mesh_mask: jnp.ndarray,
        mesh_nvec: jnp.ndarray,
        mesh_L: jnp.ndarray,
        mesh_l1: jnp.ndarray,
        mesh_l2: jnp.ndarray,
        in_out_flag: jnp.ndarray,
        obs_flat: jnp.ndarray,
        rot_s: jnp.ndarray,
    ) -> jnp.ndarray:
        obs_local = (obs_flat - pos_t) @ rot_t

        def _dipole(_):
            if field == "B":
                return dipole_bfield(obs_local, moment)
            if field == "H":
                return dipole_hfield(obs_local, moment)
            return jnp.zeros_like(obs_local, dtype=jnp.float64)

        def _circle(_):
            if field == "B":
                return current_circle_bfield(obs_local, diameter, current)
            if field == "H":
                return current_circle_hfield(obs_local, diameter, current)
            return jnp.zeros_like(obs_local, dtype=jnp.float64)

        def _cuboid(_):
            if field == "B":
                return magnet_cuboid_bfield(obs_local, cub_dim, pol)
            if field == "H":
                return magnet_cuboid_hfield(obs_local, cub_dim, pol)
            if field == "J":
                return magnet_cuboid_jfield(obs_local, cub_dim, pol)
            return magnet_cuboid_mfield(obs_local, cub_dim, pol)

        def _cylinder(_):
            if field == "B":
                return magnet_cylinder_bfield(obs_local, cyl_dim, pol)
            if field == "H":
                return magnet_cylinder_hfield(obs_local, cyl_dim, pol)
            if field == "J":
                return magnet_cylinder_jfield(obs_local, cyl_dim, pol)
            return magnet_cylinder_mfield(obs_local, cyl_dim, pol)

        def _cylindersegment(_):
            if field == "B":
                return magnet_cylinder_segment_bfield(obs_local, cseg_dim, pol, in_out=in_out)
            if field == "H":
                return magnet_cylinder_segment_hfield(obs_local, cseg_dim, pol, in_out=in_out)
            if field == "J":
                return magnet_cylinder_segment_jfield(obs_local, cseg_dim, pol, in_out=in_out)
            return magnet_cylinder_segment_mfield(obs_local, cseg_dim, pol, in_out=in_out)

        def _sphere(_):
            if field == "B":
                return magnet_sphere_bfield(obs_local, diameter, pol)
            if field == "H":
                return magnet_sphere_hfield(obs_local, diameter, pol)
            if field == "J":
                return magnet_sphere_jfield(obs_local, diameter, pol)
            return magnet_sphere_mfield(obs_local, diameter, pol)

        def _triangle(_):
            if field == "B":
                return triangle_bfield(obs_local, tri_vertices, pol)
            if field == "H":
                return triangle_hfield(obs_local, tri_vertices, pol)
            if field == "J":
                return triangle_jfield(obs_local, tri_vertices, pol)
            return triangle_mfield(obs_local, tri_vertices, pol)

        def _polyline(_):
            if field == "B":
                return current_polyline_bfield_masked(
                    obs_local, seg_start, seg_end, current, seg_mask
                )
            if field == "H":
                return (
                    current_polyline_bfield_masked(obs_local, seg_start, seg_end, current, seg_mask)
                    / MU0
                )
            return jnp.zeros_like(obs_local, dtype=jnp.float64)

        def _trianglesheet(_):
            if field == "B":
                return current_trisheet_bfield_masked(obs_local, sheet_tris, sheet_cd, sheet_mask)
            if field == "H":
                return (
                    current_trisheet_bfield_masked(obs_local, sheet_tris, sheet_cd, sheet_mask)
                    / MU0
                )
            return jnp.zeros_like(obs_local, dtype=jnp.float64)

        def _trianglestrip(_):
            if field == "B":
                return current_trisheet_bfield_masked(obs_local, sheet_tris, sheet_cd, sheet_mask)
            if field == "H":
                return (
                    current_trisheet_bfield_masked(obs_local, sheet_tris, sheet_cd, sheet_mask)
                    / MU0
                )
            return jnp.zeros_like(obs_local, dtype=jnp.float64)

        def _triangularmesh(_):
            if field == "B":
                return magnet_trimesh_bfield_precomp_masked(
                    obs_local,
                    mesh_faces,
                    pol,
                    mesh_nvec,
                    mesh_L,
                    mesh_l1,
                    mesh_l2,
                    mesh_mask,
                    in_out_flag,
                )
            if field == "H":
                b = magnet_trimesh_bfield_precomp_masked(
                    obs_local,
                    mesh_faces,
                    pol,
                    mesh_nvec,
                    mesh_L,
                    mesh_l1,
                    mesh_l2,
                    mesh_mask,
                    in_out_flag,
                )
                jfield = jnp.where(
                    _mesh_inside(obs_local, mesh_faces, mesh_mask, in_out_flag)[:, None],
                    pol,
                    0.0,
                )
                return (b - jfield) / MU0
            if field == "J":
                inside = _mesh_inside(obs_local, mesh_faces, mesh_mask, in_out_flag)
                return jnp.where(inside[:, None], pol, 0.0)
            inside = _mesh_inside(obs_local, mesh_faces, mesh_mask, in_out_flag)
            return jnp.where(inside[:, None], pol, 0.0) / MU0

        def _tetrahedron(_):
            if field == "B":
                return tetrahedron_bfield(obs_local, tet_vertices, pol, in_out=in_out)
            if field == "H":
                return tetrahedron_hfield(obs_local, tet_vertices, pol, in_out=in_out)
            if field == "J":
                return tetrahedron_jfield(obs_local, tet_vertices, pol, in_out=in_out)
            return tetrahedron_mfield(obs_local, tet_vertices, pol, in_out=in_out)

        branches = (
            _dipole,
            _circle,
            _cuboid,
            _cylinder,
            _cylindersegment,
            _sphere,
            _triangle,
            _polyline,
            _trianglesheet,
            _trianglestrip,
            _triangularmesh,
            _tetrahedron,
        )
        field_local = jax.lax.switch(stype, branches, operand=None)
        field_global = field_local @ rot_t.T
        field_global = field_global.reshape((n_sensors, max_pix, 3))
        field_sens = jnp.einsum("spc,sdc->spd", field_global, rot_s)
        field_sens = field_sens * hand_vec[:, None, :]
        return field_sens

    def per_source_circle(
        pos_t: jnp.ndarray,
        rot_t: jnp.ndarray,
        diameter: jnp.ndarray,
        current: jnp.ndarray,
        obs_flat: jnp.ndarray,
        rot_s: jnp.ndarray,
    ) -> jnp.ndarray:
        obs_local = (obs_flat - pos_t) @ rot_t
        if field == "B":
            field_local = current_circle_bfield(obs_local, diameter, current)
        elif field == "H":
            field_local = current_circle_hfield(obs_local, diameter, current)
        else:
            field_local = jnp.zeros_like(obs_local, dtype=jnp.float64)
        field_global = field_local @ rot_t.T
        field_global = field_global.reshape((n_sensors, max_pix, 3))
        field_sens = jnp.einsum("spc,sdc->spd", field_global, rot_s)
        field_sens = field_sens * hand_vec[:, None, :]
        return field_sens

    def step(_, t):
        pos_s = sens_pos[:, t, :]
        rot_s = sens_rot[:, t, :, :]
        pix_rot = jnp.einsum("spc,sdc->spd", pix_flat, rot_s)
        obs = pix_rot + pos_s[:, None, :]
        obs_flat = obs.reshape((n_sensors * max_pix, 3))
        pos_t = pos[:, t]
        rot_t = rot[:, t]

        def _slice_chunk(arr: jnp.ndarray, start: jnp.ndarray) -> jnp.ndarray:
            return jax.lax.dynamic_slice_in_dim(arr, start, chunk_size, axis=0)

        def _chunk_step(carry: jnp.ndarray, chunk_idx: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            start = chunk_idx * chunk_size
            if all_circle:
                fields = jax.vmap(
                    per_source_circle,
                    in_axes=(0, 0, 0, 0, None, None),
                )(
                    _slice_chunk(pos_t, start),
                    _slice_chunk(rot_t, start),
                    _slice_chunk(src["diameter"], start),
                    _slice_chunk(src["current"], start),
                    obs_flat,
                    rot_s,
                )
            else:
                fields = jax.vmap(
                    per_source,
                    in_axes=(
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        None,
                        None,
                    ),
                )(
                    _slice_chunk(type_id, start),
                    _slice_chunk(pos_t, start),
                    _slice_chunk(rot_t, start),
                    _slice_chunk(src["moment"], start),
                    _slice_chunk(src["diameter"], start),
                    _slice_chunk(src["cuboid_dim"], start),
                    _slice_chunk(src["cylinder_dim"], start),
                    _slice_chunk(src["cseg_dim"], start),
                    _slice_chunk(src["polarization"], start),
                    _slice_chunk(src["triangle_vertices"], start),
                    _slice_chunk(src["tetra_vertices"], start),
                    _slice_chunk(src["current"], start),
                    _slice_chunk(src["poly_seg_start"], start),
                    _slice_chunk(src["poly_seg_end"], start),
                    _slice_chunk(src["poly_seg_mask"], start),
                    _slice_chunk(src["sheet_tris"], start),
                    _slice_chunk(src["sheet_cd"], start),
                    _slice_chunk(src["sheet_mask"], start),
                    _slice_chunk(src["mesh_faces"], start),
                    _slice_chunk(src["mesh_mask"], start),
                    _slice_chunk(src["mesh_nvec"], start),
                    _slice_chunk(src["mesh_L"], start),
                    _slice_chunk(src["mesh_l1"], start),
                    _slice_chunk(src["mesh_l2"], start),
                    _slice_chunk(src["in_out_flag"], start),
                    obs_flat,
                    rot_s,
                )
            fields = jnp.where(pix_valid_mask, fields, 0.0)
            fields = fields * _slice_chunk(source_mask, start)[:, None, None, None]
            if n_groups == 1:
                chunk_group_fields = jnp.sum(fields, axis=0, keepdims=True)
            else:
                chunk_group_fields = _segment_sum(
                    fields,
                    _slice_chunk(group_index, start),
                    n_groups,
                )
            return carry + chunk_group_fields, None

        init = jnp.zeros((n_groups, n_sensors, max_pix, 3), dtype=jnp.float64)
        group_fields, _ = jax.lax.scan(_chunk_step, init, jnp.arange(n_chunks))
        return None, group_fields

    _, b_path = jax.lax.scan(step, None, jnp.arange(n_path))
    return jnp.transpose(b_path, (1, 0, 2, 3, 4))


_compute_field_jit_core_compiled = jax.jit(
    _compute_field_jit_core,
    static_argnames=(
        "field",
        "in_out",
        "n_groups",
        "chunk_size",
        "all_circle",
        "source_rot_identity",
        "sensor_rot_identity",
        "right_handed",
    ),
)


def _apply_pixel_agg_masked(
    field: jnp.ndarray,
    pix_mask: jnp.ndarray,
    *,
    pixel_agg: str,
) -> jnp.ndarray:
    mask = pix_mask[None, None, :, :, None]
    count = jnp.sum(mask, axis=3)
    if pixel_agg == "sum":
        masked = jnp.where(mask.astype(bool), field, 0.0)
        return jnp.sum(masked, axis=3)
    if pixel_agg == "mean":
        denom = jnp.where(count > 0, count, 1.0)
        masked = jnp.where(mask.astype(bool), field, 0.0)
        return jnp.sum(masked, axis=3) / denom
    if pixel_agg == "min":
        large = jnp.finfo(field.dtype).max
        masked = jnp.where(mask.astype(bool), field, large)
        out = jnp.min(masked, axis=3)
        return jnp.where(count > 0, out, 0.0)
    if pixel_agg == "max":
        small = jnp.finfo(field.dtype).min
        masked = jnp.where(mask.astype(bool), field, small)
        out = jnp.max(masked, axis=3)
        return jnp.where(count > 0, out, 0.0)
    raise ValueError(f"Unsupported pixel_agg {pixel_agg!r}.")


def _compute_field_jit(
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
    if callable(pixel_agg_func) and pixel_agg not in _SUPPORTED_PIXEL_AGGS:
        raise ValueError("Unsupported pixel_agg for jit path.")

    src_arrays, src_meta = _prepare_sources_jit(
        source,
        position=position,
        orientation=orientation,
        in_out=in_out,
        kwargs=kwargs,
    )
    sens_arrays, sens_meta = _prepare_sensors_jit(observers, pixel_agg=pixel_agg)

    max_path_len = max(
        [int(pos.shape[0]) for pos in src_arrays["pos_list"]]
        + [int(pos.shape[0]) for pos in sens_arrays["pos_list"]]
    )
    src_pos = jnp.stack([_pad_path(pos, max_path_len) for pos in src_arrays["pos_list"]], axis=0)
    src_rot = jnp.stack([_pad_path(rot, max_path_len) for rot in src_arrays["rot_list"]], axis=0)
    sens_pos = jnp.stack([_pad_path(pos, max_path_len) for pos in sens_arrays["pos_list"]], axis=0)
    sens_rot = jnp.stack([_pad_path(rot, max_path_len) for rot in sens_arrays["rot_list"]], axis=0)

    src_arrays_core = {key: val for key, val in src_arrays.items() if not key.endswith("_list")}
    sens_arrays_core = {key: val for key, val in sens_arrays.items() if not key.endswith("_list")}
    src_arrays_core["pos"] = src_pos
    src_arrays_core["rot"] = src_rot
    all_circle = _safe_static_bool(
        jnp.all(src_arrays_core["type_id"] == _SOURCE_TYPE_IDS["circle"]),
        default=False,
    )
    sens_arrays_core["pos"] = sens_pos
    sens_arrays_core["rot"] = sens_rot

    n_groups = int(src_meta["n_groups"])
    n_sources = int(src_arrays_core["type_id"].shape[0])
    n_observers = int(sens_arrays_core["pix_flat"].shape[0] * sens_arrays_core["pix_flat"].shape[1])
    chunk_size = _select_source_chunk_size(
        n_sources,
        observer_count=n_observers,
        all_circle=all_circle,
    )
    source_rot_identity = False
    sensor_rot_identity = False
    right_handed = False
    if all_circle and n_groups == 1 and int(sens_arrays_core["pix_flat"].shape[0]) == 1:
        source_rot_identity = _is_identity_rotation_stack(src_rot)
        sensor_rot_identity = _is_identity_rotation_stack(sens_rot)
        right_handed = _is_all_right_handed(sens_arrays_core["handedness"])

    src_arrays_core = _pad_sources_for_chunking(src_arrays_core, chunk_size=chunk_size)
    B = _compute_field_jit_core_compiled(
        src_arrays_core,
        sens_arrays_core,
        field=field,
        in_out=in_out,
        n_groups=n_groups,
        chunk_size=chunk_size,
        all_circle=all_circle,
        source_rot_identity=source_rot_identity,
        sensor_rot_identity=sensor_rot_identity,
        right_handed=right_handed,
    )

    pix_shapes = sens_meta["pix_shapes"]
    pix_all_same = sens_meta["pix_all_same"]
    if pixel_agg is not None:
        if pixel_agg not in _SUPPORTED_PIXEL_AGGS:
            return _compute_field_legacy(
                source,
                observers,
                field,
                position=position,
                orientation=orientation,
                squeeze=squeeze,
                sumup=sumup,
                pixel_agg=pixel_agg,
                output=output,
                in_out=in_out,
                **kwargs,
            )
        B = _apply_pixel_agg_masked(B, sens_arrays_core["pix_mask"], pixel_agg=pixel_agg)
    else:
        if pix_all_same:
            pix_shape = pix_shapes[0]
            B = B.reshape((B.shape[0], B.shape[1], B.shape[2], *pix_shape[:-1], 3))

    if sumup:
        B = jnp.sum(B, axis=0, keepdims=True)

    if output == "dataframe":
        import pandas as pd  # type: ignore

        group_labels = src_meta["group_labels"]
        if sumup and len(group_labels) > 1:
            src_ids = [f"sumup ({len(group_labels)})"]
        else:
            src_ids = group_labels
        sens_ids = sens_meta["sensor_labels"]
        num_pixels = int(prod(pix_shapes[0][:-1])) if pixel_agg is None else 1
        df_field = pd.DataFrame(
            data=product(src_ids, range(B.shape[1]), sens_ids, range(num_pixels)),
            columns=["source", "path", "sensor", "pixel"],
        )
        df_field[[field + k for k in "xyz"]] = jax.device_get(B).reshape(-1, 3)
        return df_field

    if squeeze:
        B = jnp.squeeze(B)
    elif pixel_agg is not None:
        B = jnp.expand_dims(B, axis=-2)
    return B


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
        return stype, {
            "vertices": source.vertices,
            "polarization": source._polarization,
            "in_out": in_out,
        }
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


def _compute_field_legacy(
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
    pix_nums = [int(prod(ps[:-1])) for ps in pix_shapes]
    pix_inds = [0]
    for pix_num in pix_nums:
        pix_inds.append(pix_inds[-1] + int(pix_num))
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
        num_pixels = int(prod(pix_shapes[0][:-1])) if pixel_agg is None else 1
        df_field = pd.DataFrame(
            data=product(src_ids, range(max_path_len), sens_ids, range(num_pixels)),
            columns=["source", "path", "sensor", "pixel"],
        )
        df_field[[field + k for k in "xyz"]] = jax.device_get(B).reshape(-1, 3)
        return df_field

    if squeeze:
        B = jnp.squeeze(B)
    elif pixel_agg is not None:
        B = jnp.expand_dims(B, axis=-2)
    return B


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
    if isinstance(source, str):
        src_type = _normalize_source_type(source)
        if src_type == "triangularmesh":
            mesh = kwargs.get("mesh")
            if mesh is not None:
                mesh_arr = jnp.asarray(mesh)
                if mesh_arr.ndim == 4:
                    return _compute_field_legacy(
                        source,
                        observers,
                        field,
                        position=position,
                        orientation=orientation,
                        squeeze=squeeze,
                        sumup=sumup,
                        pixel_agg=pixel_agg,
                        output=output,
                        in_out=in_out,
                        **kwargs,
                    )
    if pixel_agg is not None and not isinstance(pixel_agg, str):
        return _compute_field_legacy(
            source,
            observers,
            field,
            position=position,
            orientation=orientation,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
            **kwargs,
        )
    if isinstance(pixel_agg, str) and pixel_agg not in _SUPPORTED_PIXEL_AGGS:
        return _compute_field_legacy(
            source,
            observers,
            field,
            position=position,
            orientation=orientation,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
            **kwargs,
        )

    return _compute_field_jit(
        source,
        observers,
        field,
        position=position,
        orientation=orientation,
        squeeze=squeeze,
        sumup=sumup,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


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
