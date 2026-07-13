"""Vectorized JIT field engine.

Holds the compiled per-pose/per-chunk evaluator ``_compute_field_jit_core``,
its orchestration wrapper ``_compute_field_jit`` and the masked pixel
aggregation used by the fast path.
"""

from __future__ import annotations

from itertools import product
from math import prod

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.kernels import (
    _inside_mask_mesh_masked,
    current_circle_bfield,
    current_circle_hfield,
    current_polyline_bfield_masked,
    current_trisheet_bfield_masked,
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
    magnet_cylinder_segment_jfield,
    magnet_cylinder_segment_mfield,
    magnet_sphere_bfield,
    magnet_sphere_hfield,
    magnet_sphere_jfield,
    magnet_sphere_mfield,
    magnet_trimesh_bfield_precomp_masked,
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
    _SOURCE_TYPE_IDS,
    _SUPPORTED_PIXEL_AGGS,
    _check_getbh_output_type,
    _check_pixel_agg,
)
from magpylib_jax.fields.eager import _compute_field_legacy
from magpylib_jax.fields.prepare import (
    _pad_sources_for_chunking,
    _prepare_sensors_jit,
    _prepare_sources_jit,
    _select_source_chunk_size,
    _stack_padded_paths,
)


def _segment_sum(data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int) -> jnp.ndarray:
    try:
        return jax.lax.segment_sum(data, segment_ids, num_segments)
    except AttributeError:  # pragma: no cover
        return jax.ops.segment_sum(data, segment_ids, num_segments)


def _safe_static_bool(value: jnp.ndarray, *, default: bool) -> bool:
    try:
        return bool(jax.device_get(value))
    except Exception:
        return default


def _is_identity_rotation_stack(rot: jnp.ndarray, *, atol: float = 1e-12) -> bool:
    eye = jnp.eye(3, dtype=rot.dtype)
    return _safe_static_bool(jnp.all(jnp.abs(rot - eye) <= atol), default=False)


def _is_all_right_handed(handedness: jnp.ndarray) -> bool:
    right = jnp.array([1.0, 1.0, 1.0], dtype=handedness.dtype)
    return _safe_static_bool(jnp.all(handedness == right), default=False)


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
        cseg_faces: jnp.ndarray,
        cseg_mask: jnp.ndarray,
        cseg_nvec: jnp.ndarray,
        cseg_L: jnp.ndarray,
        cseg_l1: jnp.ndarray,
        cseg_l2: jnp.ndarray,
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
                return magnet_trimesh_bfield_precomp_masked(
                    obs_local,
                    cseg_faces,
                    pol,
                    cseg_nvec,
                    cseg_L,
                    cseg_l1,
                    cseg_l2,
                    cseg_mask,
                    in_out_flag,
                )
            if field == "H":
                b = magnet_trimesh_bfield_precomp_masked(
                    obs_local,
                    cseg_faces,
                    pol,
                    cseg_nvec,
                    cseg_L,
                    cseg_l1,
                    cseg_l2,
                    cseg_mask,
                    in_out_flag,
                )
                j = magnet_cylinder_segment_jfield(obs_local, cseg_dim, pol, in_out=in_out)
                return (b - j) / MU0
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
                    _slice_chunk(src["cseg_faces"], start),
                    _slice_chunk(src["cseg_mask"], start),
                    _slice_chunk(src["cseg_nvec"], start),
                    _slice_chunk(src["cseg_L"], start),
                    _slice_chunk(src["cseg_l1"], start),
                    _slice_chunk(src["cseg_l2"], start),
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
        use_cache=output == "ndarray",
    )
    sens_arrays, sens_meta = _prepare_sensors_jit(
        observers,
        pixel_agg=pixel_agg,
        use_cache=output == "ndarray",
    )

    max_path_len = max(
        [int(pos.shape[0]) for pos in src_arrays["pos_list"]]
        + [int(pos.shape[0]) for pos in sens_arrays["pos_list"]]
    )
    if (
        max_path_len == 1
        and src_meta.get("all_path_len_one", False)
        and sens_meta.get("all_path_len_one", False)
    ):
        src_pos = src_arrays["pos_path1"]
        src_rot = src_arrays["rot_path1"]
        sens_pos = sens_arrays["pos_path1"]
        sens_rot = sens_arrays["rot_path1"]
    else:
        src_pos = _stack_padded_paths(src_arrays["pos_list"], max_path_len)
        src_rot = _stack_padded_paths(src_arrays["rot_list"], max_path_len)
        sens_pos = _stack_padded_paths(sens_arrays["pos_list"], max_path_len)
        sens_rot = _stack_padded_paths(sens_arrays["rot_list"], max_path_len)

    src_arrays_core = {key: val for key, val in src_arrays.items() if not key.endswith("_list")}
    sens_arrays_core = {key: val for key, val in sens_arrays.items() if not key.endswith("_list")}
    src_arrays_core.pop("pos_path1", None)
    src_arrays_core.pop("rot_path1", None)
    sens_arrays_core.pop("pos_path1", None)
    sens_arrays_core.pop("rot_path1", None)
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
