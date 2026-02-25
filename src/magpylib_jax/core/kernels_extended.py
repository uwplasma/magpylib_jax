"""Extended kernels for additional source families."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import ensure_observers

_FOUR_PI = 4.0 * jnp.pi
_TETRA_FACES = jnp.array(
    [
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [0, 3, 2],
    ],
    dtype=jnp.int32,
)


def _broadcast_vec3(arr: jnp.ndarray, n: int) -> jnp.ndarray:
    if arr.ndim == 1:
        return jnp.broadcast_to(arr[None, :], (n, 3))
    return jnp.broadcast_to(arr, (n, 3))


def _safe_norm(v: jnp.ndarray, axis: int = -1, keepdims: bool = False) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(jnp.sum(v * v, axis=axis, keepdims=keepdims), 1e-30))


def magnet_sphere_bfield(
    observers: ArrayLike,
    diameters: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """B-field of homogeneously polarized spheres centered at the origin."""
    obs = ensure_observers(observers)
    n = obs.shape[0]
    dia = jnp.asarray(diameters, dtype=jnp.float64)
    if dia.ndim == 0:
        dia = jnp.broadcast_to(dia, (n,))
    else:
        dia = jnp.broadcast_to(dia.reshape((-1,)), (n,))

    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)

    r = _safe_norm(obs, axis=1)
    rs = jnp.abs(dia) / 2.0
    outside = r > rs

    b = (2.0 / 3.0) * pol

    mdotr = jnp.sum(pol * obs, axis=1)
    out_term = (
        (3.0 * mdotr[:, None] * obs - pol * (r * r)[:, None])
        * (rs**3 / 3.0)[:, None]
        / (r**5)[:, None]
    )
    out_term = jnp.where(outside[:, None], out_term, 0.0)

    return jnp.where(outside[:, None], out_term, b)


def magnet_sphere_hfield(
    observers: ArrayLike,
    diameters: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    n = obs.shape[0]
    dia = jnp.asarray(diameters, dtype=jnp.float64)
    if dia.ndim == 0:
        dia = jnp.broadcast_to(dia, (n,))
    else:
        dia = jnp.broadcast_to(dia.reshape((-1,)), (n,))

    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)
    r = _safe_norm(obs, axis=1)
    rs = jnp.abs(dia) / 2.0
    outside = r > rs

    b = magnet_sphere_bfield(obs, dia, pol)
    h = b - jnp.where(~outside[:, None], pol, 0.0)
    return h / MU0


def magnet_sphere_jfield(
    observers: ArrayLike,
    diameters: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    n = obs.shape[0]
    dia = jnp.asarray(diameters, dtype=jnp.float64)
    if dia.ndim == 0:
        dia = jnp.broadcast_to(dia, (n,))
    else:
        dia = jnp.broadcast_to(dia.reshape((-1,)), (n,))

    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)
    r = _safe_norm(obs, axis=1)
    rs = jnp.abs(dia) / 2.0
    inside = r <= rs
    return jnp.where(inside[:, None], pol, 0.0)


def magnet_sphere_mfield(
    observers: ArrayLike,
    diameters: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    return magnet_sphere_jfield(observers, diameters, polarizations) / MU0


def current_polyline_hfield(
    observers: ArrayLike,
    segments_start: ArrayLike,
    segments_end: ArrayLike,
    currents: ArrayLike,
) -> jnp.ndarray:
    """H-field of straight current segments."""
    obs = ensure_observers(observers)
    p1 = _broadcast_vec3(jnp.asarray(segments_start, dtype=jnp.float64), obs.shape[0])
    p2 = _broadcast_vec3(jnp.asarray(segments_end, dtype=jnp.float64), obs.shape[0])

    cur = jnp.asarray(currents, dtype=jnp.float64)
    if cur.ndim == 0:
        cur = jnp.broadcast_to(cur, (obs.shape[0],))
    else:
        cur = jnp.broadcast_to(cur.reshape((-1,)), (obs.shape[0],))

    seg = p1 - p2
    norm12 = _safe_norm(seg, axis=1)
    valid_seg = norm12 > 1e-15

    p1s = p1 / norm12[:, None]
    p2s = p2 / norm12[:, None]
    pos = obs / norm12[:, None]

    t = jnp.sum((pos - p1s) * (p1s - p2s), axis=1)
    p4 = p1s + t[:, None] * (p1s - p2s)

    o4 = pos - p4
    norm_o4 = _safe_norm(o4, axis=1)
    off_line = norm_o4 >= 1e-15

    cros = jnp.cross(p2s - p1s, o4)
    norm_cros = _safe_norm(cros, axis=1)
    eB = cros / norm_cros[:, None]

    norm_o1 = _safe_norm(pos - p1s, axis=1)
    norm_o2 = _safe_norm(pos - p2s, axis=1)
    norm_41 = _safe_norm(p4 - p1s, axis=1)
    norm_42 = _safe_norm(p4 - p2s, axis=1)
    sin1 = norm_41 / norm_o1
    sin2 = norm_42 / norm_o2

    mask2 = (norm_41 > 1.0) & (norm_41 > norm_42)
    mask3 = (norm_42 > 1.0) & (norm_42 > norm_41)
    delta = jnp.where(mask2, jnp.abs(sin1 - sin2), jnp.abs(sin1 + sin2))
    delta = jnp.where(mask3, jnp.abs(sin2 - sin1), delta)

    h = (delta / norm_o4)[:, None] * eB / norm12[:, None] * cur[:, None] / _FOUR_PI
    valid = (
        valid_seg
        & off_line
        & jnp.all(jnp.isfinite(p1), axis=1)
        & jnp.all(jnp.isfinite(p2), axis=1)
    )
    return jnp.where(valid[:, None], h, 0.0)


def current_polyline_bfield(
    observers: ArrayLike,
    segments_start: ArrayLike,
    segments_end: ArrayLike,
    currents: ArrayLike,
) -> jnp.ndarray:
    return MU0 * current_polyline_hfield(observers, segments_start, segments_end, currents)


def _triangle_norm_vector(vertices: jnp.ndarray) -> jnp.ndarray:
    a = vertices[:, 1] - vertices[:, 0]
    b = vertices[:, 2] - vertices[:, 0]
    n = jnp.cross(a, b)
    n_norm = _safe_norm(n, axis=1)
    return n / n_norm[:, None]


def _solid_angle(R: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
    """Solid angle for vectors R with shape (n,3,3) and norms r with shape (n,3)."""
    N = jnp.sum(R[:, 2] * jnp.cross(R[:, 1], R[:, 0]), axis=1)
    D = (
        r[:, 0] * r[:, 1] * r[:, 2]
        + jnp.sum(R[:, 2] * R[:, 1], axis=1) * r[:, 0]
        + jnp.sum(R[:, 2] * R[:, 0], axis=1) * r[:, 1]
        + jnp.sum(R[:, 1] * R[:, 0], axis=1) * r[:, 2]
    )
    out = 2.0 * jnp.arctan2(N, D)
    return jnp.where(jnp.abs(out) > 6.2831853, 0.0, out)


def triangle_bfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """B-field of magnetically charged triangular surfaces."""
    obs = ensure_observers(observers)
    n = obs.shape[0]

    tri = jnp.asarray(vertices, dtype=jnp.float64)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)

    if tri.ndim == 2:
        tri_const = tri
        nvec_const = _triangle_norm_vector(tri_const[None, :, :])[0]
        sigma = jnp.sum(pol * nvec_const, axis=1)
        R = tri_const[None, :, :] - obs[:, None, :]
        L = jnp.stack(
            (
                tri_const[1] - tri_const[0],
                tri_const[2] - tri_const[1],
                tri_const[0] - tri_const[2],
            ),
            axis=0,
        )
        l2 = jnp.sum(L * L, axis=-1)
        l1 = jnp.sqrt(l2)
        nvec = jnp.broadcast_to(nvec_const[None, :], (n, 3))
    else:
        tri = jnp.broadcast_to(tri, (n, 3, 3))
        nvec = _triangle_norm_vector(tri)
        sigma = jnp.sum(nvec * pol, axis=1)
        R = tri - obs[:, None, :]
        L = tri[:, (1, 2, 0)] - tri[:, (0, 1, 2)]
        l2 = jnp.sum(L * L, axis=-1)
        l1 = jnp.sqrt(l2)
    r2 = jnp.sum(R * R, axis=-1)
    r = jnp.sqrt(r2)

    b = jnp.sum(R * L, axis=-1)
    bl = b / l1
    ind = jnp.abs(r + bl)

    integ1 = 1.0 / l1 * jnp.log((jnp.sqrt(l2 + 2.0 * b + r2) + l1 + bl) / ind)
    integ2 = -(1.0 / l1) * jnp.log(jnp.abs(l1 - r) / r)
    integ = jnp.where(ind > 1e-12, integ1, integ2)

    PQR = jnp.sum(integ[:, :, None] * L, axis=1)
    B = sigma[:, None] * (nvec * _solid_angle(R, r)[:, None] - jnp.cross(nvec, PQR))
    B = B / (_FOUR_PI)
    return jnp.nan_to_num(B, nan=0.0)


def triangle_hfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    return triangle_bfield(observers, vertices, polarizations) / MU0


def triangle_jfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    return jnp.zeros_like(obs)


def triangle_mfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    return jnp.zeros_like(obs)


def _check_tetra_chirality(vertices: jnp.ndarray) -> jnp.ndarray:
    vecs = jnp.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=-1,
    )
    dets = jnp.linalg.det(vecs)
    swap = dets < 0
    v = vertices
    v_swapped = v.at[:, 2:4].set(v[:, 3:1:-1])
    return jnp.where(swap[:, None, None], v_swapped, v)


def _points_inside_tetra(points: jnp.ndarray, vertices: jnp.ndarray) -> jnp.ndarray:
    mat = jnp.transpose(vertices[:, 1:] - vertices[:, 0][:, None, :], (0, 2, 1))
    inv = jnp.linalg.inv(mat)
    delta = (points - vertices[:, 0])[:, :, None]
    newp = jnp.matmul(inv, delta).squeeze(-1)
    return (
        jnp.all(newp >= 0.0, axis=1)
        & jnp.all(newp <= 1.0, axis=1)
        & (jnp.sum(newp, axis=1) <= 1.0)
    )


def _points_inside_tetra_single(points: jnp.ndarray, vertices: jnp.ndarray) -> jnp.ndarray:
    mat = (vertices[1:] - vertices[0]).T
    inv = jnp.linalg.inv(mat)
    delta = points - vertices[0]
    newp = jnp.matmul(delta, inv.T)
    return (
        jnp.all(newp >= 0.0, axis=1)
        & jnp.all(newp <= 1.0, axis=1)
        & (jnp.sum(newp, axis=1) <= 1.0)
    )


def tetrahedron_bfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    n = obs.shape[0]

    tet = jnp.asarray(vertices, dtype=jnp.float64)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)
    if tet.ndim == 2 or (tet.ndim == 3 and tet.shape[0] == 1):
        tet_const = tet if tet.ndim == 2 else tet[0]
        tet_const = _check_tetra_chirality(tet_const[None, :, :])[0]
        faces = tet_const[_TETRA_FACES]
        b_faces = jax.vmap(lambda tri: triangle_bfield(obs, tri, pol))(faces)
        b = jnp.sum(b_faces, axis=0)

        if in_out == "inside":
            inside = jnp.ones((n,), dtype=bool)
        elif in_out == "outside":
            inside = jnp.zeros((n,), dtype=bool)
        else:
            inside = _points_inside_tetra_single(obs, tet_const)
        return b + jnp.where(inside[:, None], pol, 0.0)

    tet = jnp.broadcast_to(tet, (n, 4, 3))
    tet = _check_tetra_chirality(tet)
    faces = tet[:, _TETRA_FACES, :]
    b = jnp.sum(
        jax.vmap(lambda tri: triangle_bfield(obs, tri, pol))(faces.swapaxes(0, 1)),
        axis=0,
    )

    if in_out == "inside":
        inside = jnp.ones((n,), dtype=bool)
    elif in_out == "outside":
        inside = jnp.zeros((n,), dtype=bool)
    else:
        inside = _points_inside_tetra(obs, tet)
    return b + jnp.where(inside[:, None], pol, 0.0)


def tetrahedron_hfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    b = tetrahedron_bfield(observers, vertices, polarizations, in_out=in_out)
    j = tetrahedron_jfield(observers, vertices, polarizations, in_out=in_out)
    return (b - j) / MU0


def tetrahedron_jfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    n = obs.shape[0]

    tet = jnp.asarray(vertices, dtype=jnp.float64)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)
    if tet.ndim == 2 or (tet.ndim == 3 and tet.shape[0] == 1):
        tet_const = tet if tet.ndim == 2 else tet[0]
        tet_const = _check_tetra_chirality(tet_const[None, :, :])[0]
        if in_out == "inside":
            inside = jnp.ones((n,), dtype=bool)
        elif in_out == "outside":
            inside = jnp.zeros((n,), dtype=bool)
        else:
            inside = _points_inside_tetra_single(obs, tet_const)
        return jnp.where(inside[:, None], pol, 0.0)

    tet = jnp.broadcast_to(tet, (n, 4, 3))
    if in_out == "inside":
        inside = jnp.ones((n,), dtype=bool)
    elif in_out == "outside":
        inside = jnp.zeros((n,), dtype=bool)
    else:
        inside = _points_inside_tetra(obs, tet)
    return jnp.where(inside[:, None], pol, 0.0)


def tetrahedron_mfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    return tetrahedron_jfield(observers, vertices, polarizations, in_out=in_out) / MU0


def _moller_trumbore_hits(
    point: jnp.ndarray,
    triangles: jnp.ndarray,
    ray_dir: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Vectorized ray-triangle intersection flags for one point."""
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0

    h = jnp.cross(jnp.broadcast_to(ray_dir[None, :], e2.shape), e2)
    a = jnp.sum(e1 * h, axis=1)
    valid = jnp.abs(a) > eps
    inv_a = jnp.where(valid, 1.0 / a, 0.0)

    s = point[None, :] - v0
    u = inv_a * jnp.sum(s * h, axis=1)
    q = jnp.cross(s, e1)
    v = inv_a * jnp.sum(jnp.broadcast_to(ray_dir[None, :], q.shape) * q, axis=1)
    t = inv_a * jnp.sum(e2 * q, axis=1)

    return valid & (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps) & (t > eps)


def _point_on_triangles(
    point: jnp.ndarray,
    triangles: jnp.ndarray,
    *,
    eps: float = 1e-7,
) -> jnp.ndarray:
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    n = jnp.cross(v1 - v0, v2 - v0)
    n_norm = _safe_norm(n, axis=1)
    dist = jnp.abs(jnp.sum((point[None, :] - v0) * n, axis=1)) / n_norm
    on_plane = dist <= eps

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = point[None, :] - v0
    dot00 = jnp.sum(v0v1 * v0v1, axis=1)
    dot01 = jnp.sum(v0v1 * v0v2, axis=1)
    dot02 = jnp.sum(v0v1 * v0p, axis=1)
    dot11 = jnp.sum(v0v2 * v0v2, axis=1)
    dot12 = jnp.sum(v0v2 * v0p, axis=1)
    denom = dot00 * dot11 - dot01 * dot01
    inv = jnp.where(jnp.abs(denom) > 1e-16, 1.0 / denom, 0.0)
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    inside = (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps)
    return jnp.any(on_plane & inside & (n_norm > 1e-12))


def _point_inside_mesh(point: jnp.ndarray, triangles: jnp.ndarray) -> jnp.ndarray:
    ray = jnp.array([0.737, 0.511, 0.442], dtype=jnp.float64)
    ray = ray / _safe_norm(ray)
    hits = _moller_trumbore_hits(point, triangles, ray)
    count = jnp.sum(hits.astype(jnp.int32))
    inside = (count % 2) == 1
    return inside | _point_on_triangles(point, triangles)


def _inside_mask_mesh(observers: jnp.ndarray, mesh: jnp.ndarray) -> jnp.ndarray:
    if mesh.ndim == 3:
        return jax.vmap(lambda obs: _point_inside_mesh(obs, mesh))(observers)
    return jax.vmap(_point_inside_mesh, in_axes=(0, 0))(observers, mesh)


def _broadcast_mesh(mesh: jnp.ndarray, n: int) -> jnp.ndarray:
    if mesh.ndim == 3:
        return jnp.broadcast_to(mesh[None, :, :, :], (n, *mesh.shape))
    if mesh.ndim == 4:
        return jnp.broadcast_to(mesh, (n, mesh.shape[1], 3, 3))
    raise ValueError(f"Expected mesh shape (t,3,3) or (n,t,3,3), got {mesh.shape}.")


def magnet_trimesh_bfield(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """B-field of uniformly polarized closed triangular meshes."""
    obs = ensure_observers(observers)
    n = obs.shape[0]
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    if mesh_arr.ndim == 4:
        mesh_arr = _broadcast_mesh(mesh_arr, n)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)

    # Evaluate each face as a batched triangle field and reduce over faces.
    # This avoids flatten+repeat expansions and lowers peak memory pressure.
    if mesh_arr.ndim == 3:
        b_faces = jax.vmap(lambda face_vertices: triangle_bfield(obs, face_vertices, pol))(mesh_arr)
        b = jnp.sum(b_faces, axis=0)
    else:
        mesh_by_face = jnp.swapaxes(mesh_arr, 0, 1)  # (n_faces, n_obs, 3, 3)
        b_faces = jax.vmap(lambda face_vertices: triangle_bfield(obs, face_vertices, pol))(
            mesh_by_face
        )
        b = jnp.sum(b_faces, axis=0)

    if in_out == "outside":
        inside = jnp.zeros((n,), dtype=bool)
    elif in_out == "inside":
        inside = jnp.ones((n,), dtype=bool)
    else:
        inside = _inside_mask_mesh(obs, mesh_arr)

    return b + jnp.where(inside[:, None], pol, 0.0)


def magnet_trimesh_hfield(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    b = magnet_trimesh_bfield(observers, mesh, polarizations, in_out=in_out)
    j = magnet_trimesh_jfield(observers, mesh, polarizations, in_out=in_out)
    return (b - j) / MU0


def magnet_trimesh_jfield(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    n = obs.shape[0]
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    if mesh_arr.ndim == 4:
        mesh_arr = _broadcast_mesh(mesh_arr, n)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)

    if in_out == "outside":
        inside = jnp.zeros((n,), dtype=bool)
    elif in_out == "inside":
        inside = jnp.ones((n,), dtype=bool)
    else:
        inside = _inside_mask_mesh(obs, mesh_arr)

    return jnp.where(inside[:, None], pol, 0.0)


def magnet_trimesh_mfield(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    return magnet_trimesh_jfield(observers, mesh, polarizations, in_out=in_out) / MU0


def _grid_to_triangles(grid: jnp.ndarray, *, flip: bool = False) -> jnp.ndarray:
    a = grid[:-1, :-1, :]
    b = grid[1:, :-1, :]
    c = grid[:-1, 1:, :]
    d = grid[1:, 1:, :]
    t1 = jnp.stack((a, b, c), axis=-2).reshape((-1, 3, 3))
    t2 = jnp.stack((b, d, c), axis=-2).reshape((-1, 3, 3))
    tri = jnp.concatenate((t1, t2), axis=0)
    if flip:
        tri = tri[:, (0, 2, 1), :]
    return tri


def _build_cylinder_segment_mesh(
    dimension: jnp.ndarray,
    *,
    n_phi: int = 64,
    n_r: int = 1,
    n_z: int = 1,
) -> jnp.ndarray:
    r1, r2, h, phi1_deg, phi2_deg = dimension
    zmin = -h / 2.0
    zmax = h / 2.0
    phi1 = jnp.deg2rad(phi1_deg)
    phi2 = jnp.deg2rad(phi2_deg)

    phis = jnp.linspace(phi1, phi2, n_phi + 1, dtype=jnp.float64)
    rs = jnp.linspace(r1, r2, n_r + 1, dtype=jnp.float64)
    zs = jnp.linspace(zmin, zmax, n_z + 1, dtype=jnp.float64)

    cos_p = jnp.cos(phis)
    sin_p = jnp.sin(phis)

    phi_grid = phis[:, None]
    z_grid = zs[None, :]
    outer = jnp.stack(
        (
            jnp.broadcast_to(r2 * jnp.cos(phi_grid), (n_phi + 1, n_z + 1)),
            jnp.broadcast_to(r2 * jnp.sin(phi_grid), (n_phi + 1, n_z + 1)),
            jnp.broadcast_to(z_grid, (n_phi + 1, n_z + 1)),
        ),
        axis=-1,
    )
    inner = jnp.stack(
        (
            jnp.broadcast_to(r1 * jnp.cos(phi_grid), (n_phi + 1, n_z + 1)),
            jnp.broadcast_to(r1 * jnp.sin(phi_grid), (n_phi + 1, n_z + 1)),
            jnp.broadcast_to(z_grid, (n_phi + 1, n_z + 1)),
        ),
        axis=-1,
    )

    r_grid = rs[:, None]
    p_grid = phis[None, :]
    top = jnp.stack(
        (
            r_grid * jnp.cos(p_grid),
            r_grid * jnp.sin(p_grid),
            jnp.broadcast_to(jnp.asarray(zmax), (n_r + 1, n_phi + 1)),
        ),
        axis=-1,
    )
    bottom = jnp.stack(
        (
            r_grid * jnp.cos(p_grid),
            r_grid * jnp.sin(p_grid),
            jnp.broadcast_to(jnp.asarray(zmin), (n_r + 1, n_phi + 1)),
        ),
        axis=-1,
    )

    r_cut = rs[:, None]
    z_cut = zs[None, :]
    cut1 = jnp.stack(
        (
            jnp.broadcast_to(r_cut * cos_p[0], (n_r + 1, n_z + 1)),
            jnp.broadcast_to(r_cut * sin_p[0], (n_r + 1, n_z + 1)),
            jnp.broadcast_to(z_cut, (n_r + 1, n_z + 1)),
        ),
        axis=-1,
    )
    cut2 = jnp.stack(
        (
            jnp.broadcast_to(r_cut * cos_p[-1], (n_r + 1, n_z + 1)),
            jnp.broadcast_to(r_cut * sin_p[-1], (n_r + 1, n_z + 1)),
            jnp.broadcast_to(z_cut, (n_r + 1, n_z + 1)),
        ),
        axis=-1,
    )

    parts = (
        _grid_to_triangles(outer, flip=False),
        _grid_to_triangles(inner, flip=True),
        _grid_to_triangles(top, flip=False),
        _grid_to_triangles(bottom, flip=True),
        _grid_to_triangles(cut1, flip=False),
        _grid_to_triangles(cut2, flip=True),
    )
    return jnp.concatenate(parts, axis=0)


def _ensure_dim5(dimensions: ArrayLike, n: int) -> jnp.ndarray:
    dim = jnp.asarray(dimensions, dtype=jnp.float64)
    if dim.ndim == 1:
        if dim.shape[0] != 5:
            raise ValueError(f"CylinderSegment dimension must have shape (5,), got {dim.shape}.")
        return dim
    if dim.ndim == 2 and dim.shape[1] == 5:
        if dim.shape[0] == 1:
            return dim[0]
        if dim.shape[0] == n:
            first = dim[0]
            same = jnp.all(jnp.abs(dim - first[None, :]) < 1e-14)
            if bool(same):
                return first
            raise ValueError(
                "Per-observer varying CylinderSegment dimensions are not supported."
            )
    raise ValueError(f"CylinderSegment dimension must have shape (5,) or (n,5), got {dim.shape}.")


def magnet_cylinder_segment_bfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    dim = _ensure_dim5(dimensions, obs.shape[0])
    mesh = _build_cylinder_segment_mesh(dim)
    return magnet_trimesh_bfield(obs, mesh, polarizations, in_out=in_out)


def magnet_cylinder_segment_hfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    b = magnet_cylinder_segment_bfield(observers, dimensions, polarizations, in_out=in_out)
    j = magnet_cylinder_segment_jfield(observers, dimensions, polarizations, in_out=in_out)
    return (b - j) / MU0


def magnet_cylinder_segment_jfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    dim = _ensure_dim5(dimensions, obs.shape[0])

    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    r1, r2, h, phi1_deg, phi2_deg = dim
    phi1 = jnp.deg2rad(phi1_deg)
    phi2 = jnp.deg2rad(phi2_deg)

    x, y, z = obs.T
    r = jnp.sqrt(x * x + y * y)
    phi = jnp.arctan2(y, x)
    phi = jnp.where(phi < 0, phi + 2.0 * jnp.pi, phi)
    p1 = jnp.where(phi1 < 0, phi1 + 2.0 * jnp.pi, phi1)
    p2 = jnp.where(phi2 < 0, phi2 + 2.0 * jnp.pi, phi2)
    in_phi = jnp.where(p2 >= p1, (phi >= p1) & (phi <= p2), (phi >= p1) | (phi <= p2))
    inside_geom = (r >= r1) & (r <= r2) & (jnp.abs(z) <= h / 2.0) & in_phi

    if in_out == "inside":
        inside = jnp.ones_like(inside_geom)
    elif in_out == "outside":
        inside = jnp.zeros_like(inside_geom)
    else:
        inside = inside_geom
    return jnp.where(inside[:, None], pol, 0.0)


def magnet_cylinder_segment_mfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    return magnet_cylinder_segment_jfield(
        observers, dimensions, polarizations, in_out=in_out
    ) / MU0


_TRI_Q_W = jnp.asarray(
    [
        0.2250000000000000,
        0.1323941527885062,
        0.1323941527885062,
        0.1323941527885062,
        0.1259391805448272,
        0.1259391805448272,
        0.1259391805448272,
    ],
    dtype=jnp.float64,
)
_TRI_Q_L = jnp.asarray(
    [
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        [0.797426985353087, 0.101286507323456, 0.101286507323456],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
    ],
    dtype=jnp.float64,
)


def _triangle_barycentric_mask(
    points: jnp.ndarray,
    tri: jnp.ndarray,
    normal: jnp.ndarray,
) -> jnp.ndarray:
    a, b, c = tri
    v0 = b - a
    v1 = c - a
    v2 = points - a[None, :]
    d00 = jnp.dot(v0, v0)
    d01 = jnp.dot(v0, v1)
    d11 = jnp.dot(v1, v1)
    d20 = jnp.sum(v2 * v0[None, :], axis=1)
    d21 = jnp.sum(v2 * v1[None, :], axis=1)
    denom = jnp.maximum(d00 * d11 - d01 * d01, 1e-30)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    dist = jnp.abs(jnp.sum((points - a[None, :]) * normal[None, :], axis=1))
    return (dist < 1e-10) & (u >= -1e-10) & (v >= -1e-10) & (w >= -1e-10)


def current_triangle_sheet_hfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    current_densities: ArrayLike,
) -> jnp.ndarray:
    """Differentiable quadrature kernel for a single triangular current sheet."""
    obs = ensure_observers(observers)
    tri = jnp.asarray(vertices, dtype=jnp.float64)
    if tri.shape != (3, 3):
        raise ValueError(f"Triangle sheet vertices must have shape (3,3), got {tri.shape}.")
    cd = jnp.asarray(current_densities, dtype=jnp.float64)
    if cd.shape != (3,):
        raise ValueError(
            f"Triangle sheet current density must have shape (3,), got {cd.shape}."
        )

    a, b, c = tri
    e1 = b - a
    e2 = c - a
    nvec = jnp.cross(e1, e2)
    area2 = _safe_norm(nvec)
    nhat = nvec / area2
    area = area2 / 2.0
    degenerate = area < 1e-15

    # Current density projected onto triangle plane.
    cd_proj = cd - nhat * jnp.dot(cd, nhat)
    qpts = (
        _TRI_Q_L[:, 0:1] * a[None, :]
        + _TRI_Q_L[:, 1:2] * b[None, :]
        + _TRI_Q_L[:, 2:3] * c[None, :]
    )

    def _accumulate_one(point: jnp.ndarray) -> jnp.ndarray:
        rv = point[None, :] - qpts
        rn = _safe_norm(rv, axis=1, keepdims=True)
        integrand = jnp.cross(jnp.broadcast_to(cd_proj[None, :], rv.shape), rv) / (rn**3)
        return jnp.sum((_TRI_Q_W[:, None] * integrand), axis=0)

    h = jax.vmap(_accumulate_one)(obs) * (area / _FOUR_PI)
    on_sheet = _triangle_barycentric_mask(obs, tri, nhat)
    h = jnp.where(on_sheet[:, None], 0.0, h)
    h = jnp.where(degenerate, 0.0, h)
    return h


def current_trisheet_hfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    faces: ArrayLike,
    current_densities: ArrayLike,
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    facs = jnp.asarray(faces, dtype=jnp.int32)
    cds = jnp.asarray(current_densities, dtype=jnp.float64)
    tris = verts[facs]
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError(
            "TriangleSheet requires faces indexing into vertices yielding shape (n,3,3)."
        )
    if cds.ndim != 2 or cds.shape[1] != 3:
        raise ValueError("TriangleSheet current_densities must have shape (n,3).")
    if cds.shape[0] != tris.shape[0]:
        raise ValueError("TriangleSheet current_densities and faces length mismatch.")

    h_faces = jax.vmap(lambda tri, cd: current_triangle_sheet_hfield(obs, tri, cd))(tris, cds)
    return jnp.sum(h_faces, axis=0)


def current_trisheet_bfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    faces: ArrayLike,
    current_densities: ArrayLike,
) -> jnp.ndarray:
    return MU0 * current_trisheet_hfield(observers, vertices, faces, current_densities)


def _strip_triangles(vertices: jnp.ndarray) -> jnp.ndarray:
    return jnp.stack((vertices[:-2], vertices[1:-1], vertices[2:]), axis=1)


def _strip_current_densities(vertices: jnp.ndarray, current: jnp.ndarray) -> jnp.ndarray:
    tris = _strip_triangles(vertices)
    v1 = tris[:, 1] - tris[:, 0]
    v2 = tris[:, 2] - tris[:, 0]
    v1v1 = jnp.sum(v1 * v1, axis=1)
    v2v2 = jnp.sum(v2 * v2, axis=1)
    v1v2 = jnp.sum(v1 * v2, axis=1)

    denom = jnp.maximum(v2v2, 1e-30)
    h = jnp.sqrt(jnp.maximum(v1v1 - (v1v2 * v1v2) / denom, 0.0))
    valid = (v2v2 > 1e-15) & (v1v1 > 1e-15) & (h > 1e-15)
    scale = jnp.where(valid, current / (jnp.sqrt(jnp.maximum(v2v2, 1e-30)) * h), 0.0)
    cds = v2 * scale[:, None]
    return jnp.where(valid[:, None], cds, 0.0)


def current_tristrip_hfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    current: ArrayLike,
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 3:
        raise ValueError("TriangleStrip vertices must have shape (n>=3,3).")
    cur = jnp.asarray(current, dtype=jnp.float64).reshape(())
    tris = _strip_triangles(verts)
    cds = _strip_current_densities(verts, cur)
    h_faces = jax.vmap(lambda tri, cd: current_triangle_sheet_hfield(obs, tri, cd))(tris, cds)
    return jnp.sum(h_faces, axis=0)


def current_tristrip_bfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    current: ArrayLike,
) -> jnp.ndarray:
    return MU0 * current_tristrip_hfield(observers, vertices, current)
