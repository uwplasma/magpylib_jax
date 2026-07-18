"""Source and sensor preparation for the vectorized JIT field engine.

Builds the padded, stacked array bundles that ``fields.engine`` consumes and
holds the observer/source spec builders shared with the eager reference path.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.base import MagpylibBadUserInput, MagpylibMissingInput
from magpylib_jax.core.geometry import broadcast_pose
from magpylib_jax.core.kernels import (
    _in_out_flag,
    _strip_current_densities,
    _strip_triangles,
    precompute_cylinder_segment_geometry,
    precompute_trimesh_geometry,
)
from magpylib_jax.fields.api import (
    _SOURCE_TYPE_IDS,
    _is_array_like,
    _normalize_source_type,
    _pad_axis0,
    _pad_path,
)

_MAX_SOURCE_CHUNK_SIZE = 256


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


def _source_label(obj: object) -> str:
    style = getattr(obj, "style", None)
    label = getattr(style, "label", None) if style is not None else None
    if label is None:
        label = getattr(obj, "style_label", None)
    return label or obj.__class__.__name__


def _format_source_groups(source: object) -> list[dict[str, object]]:
    from magpylib_jax.core.base import BaseSource  # local import to avoid cycles

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

    if not isinstance(observers, (list, tuple, jnp.ndarray, jax.Array)) and not _is_array_like(
        observers
    ):
        raise MagpylibBadUserInput("Bad observers provided.")

    if len(observers) == 0:  # type: ignore[arg-type]
        raise MagpylibBadUserInput("Bad observers provided.")

    # attempt to parse as single array-like
    try:
        arr = jnp.asarray(observers, dtype=float)
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
                arr = jnp.asarray(obj, dtype=float)
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
        verts = jnp.asarray(source.vertices, dtype=float)
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
                "pos": jnp.asarray(pos_path, dtype=float),
                "rot": jnp.asarray(rot_path, dtype=float),
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
                    "pos": jnp.asarray(src._position, dtype=float),
                    "rot": jnp.asarray(src._orientation_matrix, dtype=float),
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
    src_specs, group_specs = _build_source_specs(
        source, position=position, orientation=orientation, in_out=in_out, kwargs=kwargs
    )
    if not src_specs:
        raise MagpylibBadUserInput("No sources provided.")

    in_out_flag = _in_out_flag(in_out)
    max_segments = 1
    max_sheet_faces = 1
    max_cseg_faces = 1
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
            data["moment"] = jnp.asarray(skw["moment"], dtype=float)
        elif stype == "circle":
            if skw.get("diameter") is None or skw.get("current") is None:
                raise MagpylibMissingInput("Input diameter of Circle must be set.")
            data["diameter"] = jnp.asarray(skw["diameter"], dtype=float)
            data["current"] = jnp.asarray(skw["current"], dtype=float)
        elif stype == "cuboid":
            if skw.get("dimension") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input dimension of Cuboid must be set.")
            data["cuboid_dim"] = jnp.asarray(skw["dimension"], dtype=float)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=float)
        elif stype == "cylinder":
            if skw.get("dimension") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input dimension of Cylinder must be set.")
            data["cylinder_dim"] = jnp.asarray(skw["dimension"], dtype=float)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=float)
        elif stype == "cylindersegment":
            if skw.get("dimension") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input dimension of CylinderSegment must be set.")
            data["cseg_dim"] = jnp.asarray(skw["dimension"], dtype=float)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=float)
            cseg_mesh, cseg_nvec, cseg_L, cseg_l1, cseg_l2 = precompute_cylinder_segment_geometry(
                data["cseg_dim"]
            )
            data["cseg_faces"] = cseg_mesh
            data["cseg_nvec"] = cseg_nvec
            data["cseg_L"] = cseg_L
            data["cseg_l1"] = cseg_l1
            data["cseg_l2"] = cseg_l2
            max_cseg_faces = max(max_cseg_faces, int(cseg_mesh.shape[0]))
        elif stype == "sphere":
            if skw.get("diameter") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input diameter of Sphere must be set.")
            data["diameter"] = jnp.asarray(skw["diameter"], dtype=float)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=float)
        elif stype == "triangle":
            if skw.get("vertices") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input vertices of Triangle must be set.")
            data["triangle_vertices"] = jnp.asarray(skw["vertices"], dtype=float)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=float)
        elif stype == "polyline":
            if (
                skw.get("segment_start") is None
                or skw.get("segment_end") is None
                or skw.get("current") is None
            ):
                raise MagpylibMissingInput("Input vertices of Polyline must be set.")
            seg_start = jnp.asarray(skw["segment_start"], dtype=float)
            seg_end = jnp.asarray(skw["segment_end"], dtype=float)
            if seg_start.ndim == 1:
                seg_start = seg_start[None, :]
                seg_end = seg_end[None, :]
            data["segment_start"] = seg_start
            data["segment_end"] = seg_end
            data["current"] = jnp.asarray(skw["current"], dtype=float)
            max_segments = max(max_segments, int(seg_start.shape[0]))
        elif stype == "trianglesheet":
            if (
                skw.get("vertices") is None
                or skw.get("faces") is None
                or skw.get("current_densities") is None
            ):
                raise MagpylibMissingInput("Input vertices of TriangleSheet must be set.")
            verts = jnp.asarray(skw["vertices"], dtype=float)
            faces = jnp.asarray(skw["faces"], dtype=jnp.int32)
            cds = jnp.asarray(skw["current_densities"], dtype=float)
            tris = verts[faces]
            data["sheet_tris"] = tris
            data["sheet_cd"] = cds
            max_sheet_faces = max(max_sheet_faces, int(tris.shape[0]))
        elif stype == "trianglestrip":
            if skw.get("vertices") is None or skw.get("current") is None:
                raise MagpylibMissingInput("Input vertices of TriangleStrip must be set.")
            verts = jnp.asarray(skw["vertices"], dtype=float)
            curr = jnp.asarray(skw["current"], dtype=float)
            tris = _strip_triangles(verts)
            cds = _strip_current_densities(verts, curr)
            data["sheet_tris"] = tris
            data["sheet_cd"] = cds
            data["current"] = curr
            max_sheet_faces = max(max_sheet_faces, int(tris.shape[0]))
        elif stype == "triangularmesh":
            if skw.get("mesh") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input vertices of TriangularMesh must be set.")
            mesh_raw = jnp.asarray(skw["mesh"], dtype=float)
            if mesh_raw.ndim == 4:
                raise ValueError("TriangularMesh mesh input must have shape (n_faces,3,3).")
            mesh_arr, nvec, L, l1, l2 = precompute_trimesh_geometry(mesh_raw)
            data["mesh"] = mesh_arr
            data["mesh_nvec"] = nvec
            data["mesh_L"] = L
            data["mesh_l1"] = l1
            data["mesh_l2"] = l2
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=float)
            max_mesh_faces = max(max_mesh_faces, int(mesh_arr.shape[0]))
        elif stype == "tetrahedron":
            if skw.get("vertices") is None or skw.get("polarization") is None:
                raise MagpylibMissingInput("Input vertices of Tetrahedron must be set.")
            data["tetra_vertices"] = jnp.asarray(skw["vertices"], dtype=float)
            data["polarization"] = jnp.asarray(skw["polarization"], dtype=float)
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
    cseg_faces = []
    cseg_mask = []
    cseg_nvec = []
    cseg_L = []
    cseg_l1 = []
    cseg_l2 = []
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

        moment.append(jnp.asarray(data.get("moment", jnp.zeros(3)), dtype=float))
        diameter.append(jnp.asarray(data.get("diameter", 0.0), dtype=float))
        cuboid_dim.append(jnp.asarray(data.get("cuboid_dim", jnp.zeros(3)), dtype=float))
        cylinder_dim.append(jnp.asarray(data.get("cylinder_dim", jnp.zeros(2)), dtype=float))
        cseg_dim.append(jnp.asarray(data.get("cseg_dim", jnp.zeros(5)), dtype=float))
        polarization.append(jnp.asarray(data.get("polarization", jnp.zeros(3)), dtype=float))
        triangle_vertices.append(
            jnp.asarray(data.get("triangle_vertices", jnp.zeros((3, 3))), dtype=float)
        )
        tetra_vertices.append(
            jnp.asarray(data.get("tetra_vertices", jnp.zeros((4, 3))), dtype=float)
        )
        current.append(jnp.asarray(data.get("current", 0.0), dtype=float))

        if stype == "polyline":
            seg_start = data["segment_start"]
            seg_end = data["segment_end"]
            seg_count = int(seg_start.shape[0])
            seg_mask = jnp.concatenate(
                (
                    jnp.ones((seg_count,), dtype=float),
                    jnp.zeros((max_segments - seg_count,), dtype=float),
                ),
                axis=0,
            )
            poly_seg_start.append(_pad_axis0(seg_start, max_segments))
            poly_seg_end.append(_pad_axis0(seg_end, max_segments))
            poly_seg_mask.append(seg_mask)
        else:
            poly_seg_start.append(jnp.zeros((max_segments, 3), dtype=float))
            poly_seg_end.append(jnp.zeros((max_segments, 3), dtype=float))
            poly_seg_mask.append(jnp.zeros((max_segments,), dtype=float))

        if stype in ("trianglesheet", "trianglestrip"):
            tris = data["sheet_tris"]
            cds = data["sheet_cd"]
            face_count = int(tris.shape[0])
            mask = jnp.concatenate(
                (
                    jnp.ones((face_count,), dtype=float),
                    jnp.zeros((max_sheet_faces - face_count,), dtype=float),
                ),
                axis=0,
            )
            sheet_tris.append(_pad_axis0(tris, max_sheet_faces))
            sheet_cd.append(_pad_axis0(cds, max_sheet_faces))
            sheet_mask.append(mask)
        else:
            sheet_tris.append(jnp.zeros((max_sheet_faces, 3, 3), dtype=float))
            sheet_cd.append(jnp.zeros((max_sheet_faces, 3), dtype=float))
            sheet_mask.append(jnp.zeros((max_sheet_faces,), dtype=float))

        if stype == "cylindersegment":
            mesh_arr = data["cseg_faces"]
            nvec = data["cseg_nvec"]
            L = data["cseg_L"]
            l1 = data["cseg_l1"]
            l2 = data["cseg_l2"]
            face_count = int(mesh_arr.shape[0])
            mask = jnp.concatenate(
                (
                    jnp.ones((face_count,), dtype=float),
                    jnp.zeros((max_cseg_faces - face_count,), dtype=float),
                ),
                axis=0,
            )
            cseg_faces.append(_pad_axis0(mesh_arr, max_cseg_faces))
            cseg_nvec.append(_pad_axis0(nvec, max_cseg_faces))
            cseg_L.append(_pad_axis0(L, max_cseg_faces))
            cseg_l1.append(_pad_axis0(l1, max_cseg_faces))
            cseg_l2.append(_pad_axis0(l2, max_cseg_faces))
            cseg_mask.append(mask)
        else:
            cseg_faces.append(jnp.zeros((max_cseg_faces, 3, 3), dtype=float))
            cseg_nvec.append(jnp.zeros((max_cseg_faces, 3), dtype=float))
            cseg_L.append(jnp.zeros((max_cseg_faces, 3, 3), dtype=float))
            cseg_l1.append(jnp.zeros((max_cseg_faces, 3), dtype=float))
            cseg_l2.append(jnp.zeros((max_cseg_faces, 3), dtype=float))
            cseg_mask.append(jnp.zeros((max_cseg_faces,), dtype=float))

        if stype == "triangularmesh":
            mesh_arr = data["mesh"]
            nvec = data["mesh_nvec"]
            L = data["mesh_L"]
            l1 = data["mesh_l1"]
            l2 = data["mesh_l2"]
            face_count = int(mesh_arr.shape[0])
            mask = jnp.concatenate(
                (
                    jnp.ones((face_count,), dtype=float),
                    jnp.zeros((max_mesh_faces - face_count,), dtype=float),
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
            mesh_faces.append(jnp.zeros((max_mesh_faces, 3, 3), dtype=float))
            mesh_nvec.append(jnp.zeros((max_mesh_faces, 3), dtype=float))
            mesh_L.append(jnp.zeros((max_mesh_faces, 3, 3), dtype=float))
            mesh_l1.append(jnp.zeros((max_mesh_faces, 3), dtype=float))
            mesh_l2.append(jnp.zeros((max_mesh_faces, 3), dtype=float))
            mesh_mask.append(jnp.zeros((max_mesh_faces,), dtype=float))

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
        "cseg_faces": jnp.stack(cseg_faces, axis=0),
        "cseg_mask": jnp.stack(cseg_mask, axis=0),
        "cseg_nvec": jnp.stack(cseg_nvec, axis=0),
        "cseg_L": jnp.stack(cseg_L, axis=0),
        "cseg_l1": jnp.stack(cseg_l1, axis=0),
        "cseg_l2": jnp.stack(cseg_l2, axis=0),
        "mesh_faces": jnp.stack(mesh_faces, axis=0),
        "mesh_mask": jnp.stack(mesh_mask, axis=0),
        "mesh_nvec": jnp.stack(mesh_nvec, axis=0),
        "mesh_L": jnp.stack(mesh_L, axis=0),
        "mesh_l1": jnp.stack(mesh_l1, axis=0),
        "mesh_l2": jnp.stack(mesh_l2, axis=0),
        "group_index": group_index,
        "in_out_flag": jnp.full((len(src_data),), in_out_flag, dtype=jnp.int32),
    }
    src_singleton_paths = all(int(path.shape[0]) == 1 for path in pos_list + rot_list)
    if src_singleton_paths:
        src_arrays["pos_path1"] = _stack_singleton_paths(pos_list)
        src_arrays["rot_path1"] = _stack_singleton_paths(rot_list)
    else:
        src_arrays["pos_path1"] = jnp.zeros((0, 1, 3), dtype=float)
        src_arrays["rot_path1"] = jnp.zeros((0, 1, 3, 3), dtype=float)

    meta = {
        "group_labels": [group["label"] for group in group_specs],
        "n_groups": len(group_specs),
        "all_path_len_one": src_singleton_paths,
    }
    return src_arrays, meta


def _stack_padded_paths(paths: Sequence[ArrayLike], target_len: int) -> jnp.ndarray:
    first = jnp.asarray(paths[0], dtype=float)
    tail_shape = first.shape[1:]
    if target_len == 1 and all(jnp.asarray(path).shape[0] == 1 for path in paths):
        stacked = [
            jnp.asarray(path, dtype=float).reshape((1,) + tail_shape) for path in paths
        ]
        return jnp.stack(stacked, axis=0)
    return jnp.stack([_pad_path(path, target_len) for path in paths], axis=0)


def _stack_singleton_paths(paths: Sequence[jnp.ndarray]) -> jnp.ndarray:
    return jnp.stack([path[0] for path in paths], axis=0)[:, None, ...]


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
        pix_arr = jnp.asarray(observers, dtype=float)
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
                "pos": jnp.zeros((1, 3), dtype=float),
                "rot": jnp.eye(3, dtype=float)[None, :, :],
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
                pix_arr = jnp.zeros((1, 3), dtype=float)
                pix_shape = (1, 3)
            else:
                pix_arr = jnp.asarray(pix, dtype=float)
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
                    "pos": jnp.asarray(sens._position, dtype=float),
                    "rot": jnp.asarray(sens._orientation_matrix, dtype=float),
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
            (jnp.ones((pix_count,), dtype=float), jnp.zeros((pad_len,), dtype=float)),
            axis=0,
        )
        pix_flat_list.append(pix_flat)
        pix_mask_list.append(mask)
        pos_list.append(sens["pos"])
        rot_list.append(sens["rot"])
        handedness_list.append(sens["handedness"])
        labels.append(sens["label"])

    hand_vec = [
        jnp.array([-1.0, 1.0, 1.0], dtype=float)
        if h == "left"
        else jnp.array([1.0, 1.0, 1.0], dtype=float)
        for h in handedness_list
    ]

    sens_arrays = {
        "pix_flat": jnp.stack(pix_flat_list, axis=0),
        "pix_mask": jnp.stack(pix_mask_list, axis=0),
        "pos_list": pos_list,
        "rot_list": rot_list,
        "handedness": jnp.stack(hand_vec, axis=0),
    }
    sens_singleton_paths = all(int(path.shape[0]) == 1 for path in pos_list + rot_list)
    if sens_singleton_paths:
        sens_arrays["pos_path1"] = _stack_singleton_paths(pos_list)
        sens_arrays["rot_path1"] = _stack_singleton_paths(rot_list)
    else:
        sens_arrays["pos_path1"] = jnp.zeros((0, 1, 3), dtype=float)
        sens_arrays["rot_path1"] = jnp.zeros((0, 1, 3, 3), dtype=float)
    pix_inds = [0]
    for pix_num in pix_nums:
        pix_inds.append(pix_inds[-1] + int(pix_num))

    meta = {
        "pix_shapes": pix_shapes,
        "pix_nums": pix_nums,
        "pix_all_same": pix_all_same,
        "pix_inds": tuple(pix_inds),
        "sensor_labels": labels,
        "all_path_len_one": sens_singleton_paths,
    }
    return sens_arrays, meta


def _pad_sources_for_chunking(
    src_arrays: dict[str, jnp.ndarray], *, chunk_size: int
) -> dict[str, jnp.ndarray]:
    n_src = int(src_arrays["type_id"].shape[0])
    pad = (-n_src) % chunk_size
    source_mask = jnp.concatenate(
        (
            jnp.ones((n_src,), dtype=float),
            jnp.zeros((pad,), dtype=float),
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
