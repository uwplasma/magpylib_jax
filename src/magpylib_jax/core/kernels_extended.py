"""Extended kernels: sphere, polyline, triangle, tetrahedron."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import ensure_observers

_FOUR_PI = 4.0 * jnp.pi


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
    N = jnp.einsum("ij,ij->i", R[2], jnp.cross(R[1], R[0]))
    D = (
        r[0] * r[1] * r[2]
        + jnp.einsum("ij,ij->i", R[2], R[1]) * r[0]
        + jnp.einsum("ij,ij->i", R[2], R[0]) * r[1]
        + jnp.einsum("ij,ij->i", R[1], R[0]) * r[2]
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
    if tri.ndim == 2:
        tri = jnp.broadcast_to(tri[None, :, :], (n, 3, 3))
    else:
        tri = jnp.broadcast_to(tri, (n, 3, 3))

    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)

    nvec = _triangle_norm_vector(tri)
    sigma = jnp.einsum("ij,ij->i", nvec, pol)

    R = jnp.swapaxes(tri, 0, 1) - obs
    r2 = jnp.sum(R * R, axis=-1)
    r = jnp.sqrt(r2)

    L = tri[:, (1, 2, 0)] - tri[:, (0, 1, 2)]
    L = jnp.swapaxes(L, 0, 1)
    l2 = jnp.sum(L * L, axis=-1)
    l1 = jnp.sqrt(l2)

    b = jnp.einsum("ijk,ijk->ij", R, L)
    bl = b / l1
    ind = jnp.abs(r + bl)

    integ1 = 1.0 / l1 * jnp.log((jnp.sqrt(l2 + 2.0 * b + r2) + l1 + bl) / ind)
    integ2 = -(1.0 / l1) * jnp.log(jnp.abs(l1 - r) / r)
    integ = jnp.where(ind > 1e-12, integ1, integ2)

    PQR = jnp.einsum("ij,ijk->jk", integ, L)
    B = sigma * (nvec.T * _solid_angle(R, r) - jnp.cross(nvec, PQR).T)
    B = B / (_FOUR_PI)
    B = jnp.nan_to_num(B, nan=0.0)
    return B.T


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


def tetrahedron_bfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    n = obs.shape[0]

    tet = jnp.asarray(vertices, dtype=jnp.float64)
    if tet.ndim == 2:
        tet = jnp.broadcast_to(tet[None, :, :], (n, 4, 3))
    else:
        tet = jnp.broadcast_to(tet, (n, 4, 3))

    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)
    tet = _check_tetra_chirality(tet)

    tri_vertices = jnp.concatenate(
        (
            tet[:, (0, 2, 1), :],
            tet[:, (0, 1, 3), :],
            tet[:, (1, 2, 3), :],
            tet[:, (0, 3, 2), :],
        ),
        axis=0,
    )
    tri_obs = jnp.tile(obs, (4, 1))
    tri_pol = jnp.tile(pol, (4, 1))
    tri_field = triangle_bfield(tri_obs, tri_vertices, tri_pol)
    b = tri_field[:n] + tri_field[n : 2 * n] + tri_field[2 * n : 3 * n] + tri_field[3 * n :]

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
    if tet.ndim == 2:
        tet = jnp.broadcast_to(tet[None, :, :], (n, 4, 3))
    else:
        tet = jnp.broadcast_to(tet, (n, 4, 3))

    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)

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
