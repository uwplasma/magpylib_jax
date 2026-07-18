"""Eager, per-pose reference evaluator used for output modes the vectorized JIT
engine does not cover (callable/unsupported pixel_agg, non-uniform pixel grids,
dataframe output, 4-D meshes) and for pairwise-observer evaluation."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from itertools import product
from math import prod
from typing import Any

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.base import MagpylibBadUserInput, MagpylibMissingInput
from magpylib_jax.core.geometry import (
    broadcast_pose,
    ensure_observers,
    to_global_field,
    to_local_coordinates,
)
from magpylib_jax.core.kernels import (
    current_circle_bfield,
    current_circle_hfield,
    current_polyline_bfield,
    current_polyline_hfield,
    current_trisheet_bfield,
    current_trisheet_hfield,
    current_tristrip_bfield,
    current_tristrip_hfield,
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
from magpylib_jax.fields.api import (
    _apply_squeeze,
    _check_getbh_output_type,
    _check_pixel_agg,
    _normalize_source_type,
)
from magpylib_jax.fields.prepare import _build_source_specs, _format_observers


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
        return jnp.zeros_like(obs_local, dtype=float)

    if source_type == "circle":
        diameter = kwargs.get("diameter")
        current = kwargs.get("current")
        if diameter is None or current is None:
            raise MagpylibMissingInput("Input diameter of Circle must be set.")
        if output_field == "B":
            return current_circle_bfield(obs_local, diameter=diameter, current=current)
        if output_field == "H":
            return current_circle_hfield(obs_local, diameter=diameter, current=current)
        return jnp.zeros_like(obs_local, dtype=float)

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
        return jnp.zeros_like(obs_local, dtype=float)

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
        return jnp.zeros_like(obs_local, dtype=float)

    if source_type == "trianglestrip":
        vertices = kwargs.get("vertices")
        current = kwargs.get("current")
        if vertices is None or current is None:
            raise MagpylibMissingInput("Input vertices of TriangleStrip must be set.")
        if output_field == "B":
            return current_tristrip_bfield(obs_local, vertices, current)
        if output_field == "H":
            return current_tristrip_hfield(obs_local, vertices, current)
        return jnp.zeros_like(obs_local, dtype=float)

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
            base = ensure_observers(jnp.asarray(observers, dtype=float))
            return jnp.zeros_like(base), 0

        terms: list[jnp.ndarray] = []
        for src in source:
            method = getattr(src, f"get{field_name}", None)
            if method is None:
                raise TypeError(
                    f"Source object {type(src).__name__!r} has no get{field_name} method."
                )
            if "in_out" in inspect.signature(method).parameters:
                terms.append(jnp.asarray(method(observers, in_out=in_out), dtype=float))
            else:
                terms.append(jnp.asarray(method(observers), dtype=float))

        broadcasted = jnp.broadcast_arrays(*terms)
        stacked = jnp.stack(broadcasted, axis=0)
        if sumup:
            return jnp.sum(stacked, axis=0), len(terms)
        return stacked, len(terms)

    method = getattr(source, f"get{field_name}", None)
    if method is None:
        raise TypeError(f"Source object {type(source).__name__!r} has no get{field_name} method.")
    if "in_out" in inspect.signature(method).parameters:
        return jnp.asarray(method(observers, in_out=in_out), dtype=float), 1
    return jnp.asarray(method(observers), dtype=float), 1


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

    src_specs, group_specs = _build_source_specs(
        source, position=position, orientation=orientation, in_out=in_out, kwargs=kwargs
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
            pix_arr = jnp.zeros((1, 3), dtype=float)
            pix_shape = (1, 3)
        else:
            pix_arr = jnp.asarray(pix, dtype=float)
            if pix_arr.shape == (3,):
                pix_arr = pix_arr[None, :]
            pix_shape = pix_arr.shape
        pix_flat = pix_arr.reshape((-1, 3))
        sensor_data.append(
            {
                "pix_flat": pix_flat,
                "pix_shape": pix_shape,
                "pos": jnp.asarray(sens._position, dtype=float),
                "rot": jnp.asarray(sens._orientation_matrix, dtype=float),
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
                    seg = seg * jnp.array([-1.0, 1.0, 1.0], dtype=float)
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
    obs_input = jnp.asarray(observers, dtype=float)
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
