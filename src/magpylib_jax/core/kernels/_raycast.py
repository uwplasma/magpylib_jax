"""Ray-casting inside-mesh tests for triangular meshes."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _v_norm2_jax(a: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(a * a, axis=-1)


def _v_norm_proj_jax(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    ab = jnp.sum(a * b, axis=-1)
    return ab / jnp.sqrt(_v_norm2_jax(a) * _v_norm2_jax(b))


def _v_dot_cross3d_jax(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.cross(a, b) * c, axis=-1)


def _lines_end_in_trimesh_jax(lines: jnp.ndarray, faces: jnp.ndarray) -> jnp.ndarray:
    normals = jnp.cross(faces[:, 0] - faces[:, 2], faces[:, 1] - faces[:, 2])
    normals = jnp.broadcast_to(normals, (lines.shape[0],) + normals.shape)

    l0 = lines[:, 0][:, None, :]
    l1 = lines[:, 1][:, None, :]

    ref_pts = jnp.broadcast_to(faces[:, 2], (lines.shape[0], faces.shape[0], 3))
    eps = 1e-16
    coincide = _v_norm2_jax(l1 - ref_pts) < eps
    ref_pts2 = jnp.broadcast_to(faces[:, 1], ref_pts.shape)
    ref_pts = jnp.where(coincide[..., None], ref_pts2, ref_pts)

    proj0 = _v_norm_proj_jax(l0 - ref_pts, normals)
    proj1 = _v_norm_proj_jax(l1 - ref_pts, normals)

    eps = 1e-7
    plane_touch = jnp.abs(proj1) < eps
    plane_cross = jnp.sign(proj0) != jnp.sign(proj1)

    faces0 = faces[:, 0][None, :, :]
    faces1 = faces[:, 1][None, :, :]
    faces2 = faces[:, 2][None, :, :]
    a = faces0 - l0
    b = faces1 - l0
    c = faces2 - l0
    d = l1 - l0

    area1 = _v_dot_cross3d_jax(a, b, d)
    area2 = _v_dot_cross3d_jax(b, c, d)
    area3 = _v_dot_cross3d_jax(c, a, d)

    eps = 1e-12
    pass_through_boundary = (jnp.abs(area1) < eps) | (jnp.abs(area2) < eps) | (jnp.abs(area3) < eps)
    area1 = jnp.sign(area1)
    area2 = jnp.sign(area2)
    area3 = jnp.sign(area3)
    pass_through_inside = (area1 == area2) & (area2 == area3)
    pass_through = pass_through_boundary | pass_through_inside

    result_cross = pass_through & plane_cross
    result_touch = pass_through & plane_touch

    inside1 = (jnp.sum(result_cross, axis=1) % 2) != 0
    inside2 = jnp.any(result_touch, axis=1)
    return inside1 | inside2


_MASK_FACE_SENTINEL = jnp.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))


def _lines_end_in_trimesh_jax_masked(
    lines: jnp.ndarray,
    faces: jnp.ndarray,
    face_mask: jnp.ndarray,
) -> jnp.ndarray:
    mask = jnp.asarray(face_mask, dtype=bool)
    faces_safe = jnp.where(mask[:, None, None], faces, _MASK_FACE_SENTINEL)

    normals = jnp.cross(faces_safe[:, 0] - faces_safe[:, 2], faces_safe[:, 1] - faces_safe[:, 2])
    normals = jnp.broadcast_to(normals, (lines.shape[0],) + normals.shape)

    l0 = lines[:, 0][:, None, :]
    l1 = lines[:, 1][:, None, :]

    ref_pts = jnp.broadcast_to(faces_safe[:, 2], (lines.shape[0], faces_safe.shape[0], 3))
    eps = 1e-16
    coincide = _v_norm2_jax(l1 - ref_pts) < eps
    ref_pts2 = jnp.broadcast_to(faces_safe[:, 1], ref_pts.shape)
    ref_pts = jnp.where(coincide[..., None], ref_pts2, ref_pts)

    proj0 = _v_norm_proj_jax(l0 - ref_pts, normals)
    proj1 = _v_norm_proj_jax(l1 - ref_pts, normals)

    eps = 1e-7
    plane_touch = jnp.abs(proj1) < eps
    plane_cross = jnp.sign(proj0) != jnp.sign(proj1)

    faces0 = faces_safe[:, 0][None, :, :]
    faces1 = faces_safe[:, 1][None, :, :]
    faces2 = faces_safe[:, 2][None, :, :]
    a = faces0 - l0
    b = faces1 - l0
    c = faces2 - l0
    d = l1 - l0

    area1 = _v_dot_cross3d_jax(a, b, d)
    area2 = _v_dot_cross3d_jax(b, c, d)
    area3 = _v_dot_cross3d_jax(c, a, d)

    eps = 1e-12
    pass_through_boundary = (jnp.abs(area1) < eps) | (jnp.abs(area2) < eps) | (jnp.abs(area3) < eps)
    area1 = jnp.sign(area1)
    area2 = jnp.sign(area2)
    area3 = jnp.sign(area3)
    pass_through_inside = (area1 == area2) & (area2 == area3)
    pass_through = pass_through_boundary | pass_through_inside

    mask_lines = mask[None, :]
    result_cross = pass_through & plane_cross & mask_lines
    result_touch = pass_through & plane_touch & mask_lines

    inside1 = (jnp.sum(result_cross, axis=1) % 2) != 0
    inside2 = jnp.any(result_touch, axis=1)
    return inside1 | inside2


def _mask_inside_trimesh_jax(points: jnp.ndarray, faces: jnp.ndarray) -> jnp.ndarray:
    vertices = faces.reshape((-1, 3))
    xmin, ymin, zmin = jnp.min(vertices, axis=0)
    xmax, ymax, zmax = jnp.max(vertices, axis=0)
    eps = 1e-12
    mx = (points[:, 0] < xmax + eps) & (points[:, 0] > xmin - eps)
    my = (points[:, 1] < ymax + eps) & (points[:, 1] > ymin - eps)
    mz = (points[:, 2] < zmax + eps) & (points[:, 2] > zmin - eps)
    mask_box = mx & my & mz

    start_point_outside = jnp.array([xmin, ymin, zmin], dtype=float) - jnp.array(
        [12.0012345, 5.9923456, 6.9932109], dtype=float
    )
    start_pts = jnp.broadcast_to(start_point_outside, points.shape)
    lines = jnp.stack((start_pts, points), axis=1)
    mask_inside2 = _lines_end_in_trimesh_jax(lines, faces)
    return mask_box & mask_inside2


def _mask_inside_trimesh_jax_masked(
    points: jnp.ndarray,
    faces: jnp.ndarray,
    face_mask: jnp.ndarray,
) -> jnp.ndarray:
    mask = jnp.asarray(face_mask, dtype=bool)
    any_face = jnp.any(mask)

    def _compute() -> jnp.ndarray:
        verts = faces.reshape((-1, 3))
        vert_mask = jnp.repeat(mask, 3)
        big = 1.0e30
        verts_min = jnp.where(vert_mask[:, None], verts, big)
        verts_max = jnp.where(vert_mask[:, None], verts, -big)
        xmin, ymin, zmin = jnp.min(verts_min, axis=0)
        xmax, ymax, zmax = jnp.max(verts_max, axis=0)
        eps = 1e-12
        mx = (points[:, 0] < xmax + eps) & (points[:, 0] > xmin - eps)
        my = (points[:, 1] < ymax + eps) & (points[:, 1] > ymin - eps)
        mz = (points[:, 2] < zmax + eps) & (points[:, 2] > zmin - eps)
        mask_box = mx & my & mz

        start_point_outside = jnp.array([xmin, ymin, zmin], dtype=float) - jnp.array(
            [12.0012345, 5.9923456, 6.9932109], dtype=float
        )
        start_pts = jnp.broadcast_to(start_point_outside, points.shape)
        lines = jnp.stack((start_pts, points), axis=1)
        mask_inside2 = _lines_end_in_trimesh_jax_masked(lines, faces, mask)
        return mask_box & mask_inside2

    def _empty() -> jnp.ndarray:
        return jnp.zeros((points.shape[0],), dtype=bool)

    return jax.lax.cond(any_face, _compute, _empty)


def _inside_mask_mesh(observers: jnp.ndarray, mesh: jnp.ndarray) -> jnp.ndarray:
    if mesh.ndim == 3:
        return _mask_inside_trimesh_jax(observers, mesh)
    return jax.vmap(lambda obs, face: _mask_inside_trimesh_jax(obs[None, :], face)[0])(
        observers, mesh
    )


def _inside_mask_mesh_masked(
    observers: jnp.ndarray,
    mesh: jnp.ndarray,
    face_mask: jnp.ndarray,
) -> jnp.ndarray:
    if mesh.ndim == 3:
        return _mask_inside_trimesh_jax_masked(observers, mesh, face_mask)
    return jax.vmap(
        lambda obs, face, mask: _mask_inside_trimesh_jax_masked(obs[None, :], face, mask)[0]
    )(observers, mesh, face_mask)
