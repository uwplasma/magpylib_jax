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
_IN_OUT_FLAGS = {"auto": 0, "inside": 1, "outside": 2}

_JIT_KERNEL_CACHE: dict[tuple[str, int, int], object] = {}
_JIT_SIMPLE_CACHE: dict[tuple[str, int], object] = {}
_JIT_MESH_CACHE: dict[tuple[str, int, int, int], object] = {}
_JIT_SEGMENT_CACHE: dict[tuple[str, int, int], object] = {}


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


def _current_segment_hfield(
    observers: jnp.ndarray,
    segment_start: jnp.ndarray,
    segment_end: jnp.ndarray,
    current: jnp.ndarray,
) -> jnp.ndarray:
    """H-field for a single current segment."""
    obs = ensure_observers(observers)
    p1 = _broadcast_vec3(segment_start, obs.shape[0])
    p2 = _broadcast_vec3(segment_end, obs.shape[0])

    cur = jnp.asarray(current, dtype=jnp.float64)
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


def current_polyline_hfield(
    observers: ArrayLike,
    segments_start: ArrayLike,
    segments_end: ArrayLike,
    currents: ArrayLike,
) -> jnp.ndarray:
    """H-field of straight current segments."""
    obs = ensure_observers(observers)
    p1 = jnp.asarray(segments_start, dtype=jnp.float64)
    p2 = jnp.asarray(segments_end, dtype=jnp.float64)
    if p1.ndim == 1:
        return _current_segment_hfield(obs, p1, p2, currents)
    if p2.shape != p1.shape or p1.shape[-1] != 3:
        raise ValueError("Polyline segments must have shape (n,3).")

    cur = jnp.asarray(currents, dtype=jnp.float64)
    if cur.ndim == 0:
        cur = jnp.broadcast_to(cur, (p1.shape[0],))
    else:
        cur = jnp.broadcast_to(cur.reshape((-1,)), (p1.shape[0],))

    h_segments = jax.vmap(lambda a, b, c: _current_segment_hfield(obs, a, b, c))(p1, p2, cur)
    return jnp.sum(h_segments, axis=0)


def current_polyline_bfield(
    observers: ArrayLike,
    segments_start: ArrayLike,
    segments_end: ArrayLike,
    currents: ArrayLike,
) -> jnp.ndarray:
    return MU0 * current_polyline_hfield(observers, segments_start, segments_end, currents)


def current_polyline_bfield_masked(
    observers: ArrayLike,
    segments_start: ArrayLike,
    segments_end: ArrayLike,
    currents: ArrayLike,
    segment_mask: ArrayLike,
) -> jnp.ndarray:
    """B-field of current segments with segment masking."""
    obs = ensure_observers(observers)
    p1 = jnp.asarray(segments_start, dtype=jnp.float64)
    p2 = jnp.asarray(segments_end, dtype=jnp.float64)
    cur = jnp.asarray(currents, dtype=jnp.float64)
    if cur.ndim == 0:
        cur = jnp.broadcast_to(cur, (p1.shape[0],))
    else:
        cur = jnp.broadcast_to(cur.reshape((-1,)), (p1.shape[0],))

    mask = jnp.asarray(segment_mask, dtype=jnp.float64).reshape((-1,))
    h_segments = jax.vmap(lambda a, b, c: _current_segment_hfield(obs, a, b, c))(p1, p2, cur)
    h_segments = h_segments * mask[:, None, None]
    return MU0 * jnp.sum(h_segments, axis=0)


def _current_polyline_bfield_segments_impl(
    observers: jnp.ndarray,
    segments_start: jnp.ndarray,
    segments_end: jnp.ndarray,
    currents: jnp.ndarray,
    *,
    n_segments: int,
) -> jnp.ndarray:
    return current_polyline_bfield(observers, segments_start, segments_end, currents)


def current_polyline_bfield_jit(
    observers: ArrayLike,
    segments_start: ArrayLike,
    segments_end: ArrayLike,
    currents: ArrayLike,
) -> jnp.ndarray:
    """JIT-specialized polyline B-field for fixed observer + segment counts."""
    obs = ensure_observers(observers)
    seg_start = jnp.asarray(segments_start, dtype=jnp.float64)
    seg_end = jnp.asarray(segments_end, dtype=jnp.float64)
    if seg_start.ndim == 1:
        n_segments = 1
    else:
        n_segments = int(seg_start.shape[0])
    jit_fn = _jit_kernel_segments("polyline_bfield", _current_polyline_bfield_segments_impl, obs.shape[0], n_segments)
    return jit_fn(obs, seg_start, seg_end, jnp.asarray(currents, dtype=jnp.float64), n_segments=n_segments)


def current_circle_bfield_jit(
    observers: ArrayLike,
    diameter: ArrayLike,
    current: ArrayLike,
) -> jnp.ndarray:
    """JIT-specialized circle B-field for fixed observer counts."""
    from magpylib_jax.core.kernels import current_circle_bfield

    obs = ensure_observers(observers)
    dia = jnp.asarray(diameter, dtype=jnp.float64)
    cur = jnp.asarray(current, dtype=jnp.float64)
    jit_fn = _jit_kernel_simple("circle_bfield", current_circle_bfield, obs.shape[0])
    return jit_fn(obs, dia, cur)


def _triangle_norm_vector(vertices: jnp.ndarray) -> jnp.ndarray:
    a = vertices[:, 1] - vertices[:, 0]
    b = vertices[:, 2] - vertices[:, 0]
    n = jnp.cross(a, b)
    n_norm = _safe_norm(n, axis=1)
    return n / n_norm[:, None]


def _triangle_geom_terms(
    tri: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute triangle normals and edge terms for reuse."""
    a = tri[..., 1, :] - tri[..., 0, :]
    b = tri[..., 2, :] - tri[..., 0, :]
    n = jnp.cross(a, b)
    n_norm = _safe_norm(n, axis=-1, keepdims=True)
    nvec = n / n_norm
    # Use roll to avoid advanced-indexing inconsistencies under JIT tracing.
    L = jnp.roll(tri, shift=-1, axis=-2) - tri
    l2 = jnp.sum(L * L, axis=-1)
    l1 = jnp.sqrt(l2)
    return nvec, L, l1, l2


def _triangle_bfield_const_precomp(
    obs: jnp.ndarray,
    tri: jnp.ndarray,
    pol: jnp.ndarray,
    nvec: jnp.ndarray,
    L: jnp.ndarray,
    l1: jnp.ndarray,
    l2: jnp.ndarray,
) -> jnp.ndarray:
    """B-field for constant triangle using precomputed geometry."""
    R = tri[None, :, :] - obs[:, None, :]
    r2 = jnp.sum(R * R, axis=-1)
    r = jnp.sqrt(r2)

    b = jnp.sum(R * L[None, :, :], axis=-1)
    bl = b / l1
    ind = jnp.abs(r + bl)

    integ1 = 1.0 / l1 * jnp.log((jnp.sqrt(l2 + 2.0 * b + r2) + l1 + bl) / ind)
    integ2 = -(1.0 / l1) * jnp.log(jnp.abs(l1 - r) / r)
    integ = jnp.where(ind > 1e-12, integ1, integ2)

    PQR = jnp.sum(integ[:, :, None] * L[None, :, :], axis=1)
    sigma = jnp.sum(pol * nvec[None, :], axis=1)
    B = sigma[:, None] * (nvec[None, :] * _solid_angle(R, r)[:, None] - jnp.cross(nvec, PQR))
    B = B / (_FOUR_PI)
    return jnp.nan_to_num(B, nan=0.0)


def _in_out_flag(in_out: str) -> int:
    if in_out not in _IN_OUT_FLAGS:
        raise ValueError(f"in_out must be one of {sorted(_IN_OUT_FLAGS)}, got {in_out!r}.")
    return _IN_OUT_FLAGS[in_out]


def _jit_kernel(name: str, fn, n_obs: int, in_out_flag: int):
    key = (name, int(n_obs), int(in_out_flag))
    if key not in _JIT_KERNEL_CACHE:
        _JIT_KERNEL_CACHE[key] = jax.jit(fn, static_argnames=("in_out_flag",))
    return _JIT_KERNEL_CACHE[key]


def _jit_kernel_simple(name: str, fn, n_obs: int):
    key = (name, int(n_obs))
    if key not in _JIT_SIMPLE_CACHE:
        _JIT_SIMPLE_CACHE[key] = jax.jit(fn)
    return _JIT_SIMPLE_CACHE[key]


def _jit_kernel_mesh(name: str, fn, n_obs: int, n_faces: int, in_out_flag: int):
    key = (name, int(n_obs), int(n_faces), int(in_out_flag))
    if key not in _JIT_MESH_CACHE:
        _JIT_MESH_CACHE[key] = jax.jit(fn, static_argnames=("in_out_flag", "n_faces"))
    return _JIT_MESH_CACHE[key]


def _jit_kernel_segments(name: str, fn, n_obs: int, n_segments: int):
    key = (name, int(n_obs), int(n_segments))
    if key not in _JIT_SEGMENT_CACHE:
        _JIT_SEGMENT_CACHE[key] = jax.jit(fn, static_argnames=("n_segments",))
    return _JIT_SEGMENT_CACHE[key]


def _triangle_bfield_const_impl(
    obs: jnp.ndarray,
    tri: jnp.ndarray,
    pol: jnp.ndarray,
) -> jnp.ndarray:
    nvec, L, l1, l2 = _triangle_geom_terms(tri[None, :, :])
    return _triangle_bfield_const_precomp(obs, tri, pol, nvec[0], L[0], l1[0], l2[0])


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


def triangle_bfield_jit(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """JIT-specialized triangle B-field for fixed observer counts."""
    obs = ensure_observers(observers)
    tri = jnp.asarray(vertices, dtype=jnp.float64)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    if tri.ndim != 2:
        return triangle_bfield(obs, tri, pol)
    jit_fn = _jit_kernel_simple("triangle_bfield", _triangle_bfield_const_impl, obs.shape[0])
    return jit_fn(obs, tri, pol)


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


def _tetrahedron_bfield_const_impl(
    obs: jnp.ndarray,
    tet_const: jnp.ndarray,
    pol: jnp.ndarray,
    *,
    in_out_flag: int,
) -> jnp.ndarray:
    tet_const = _check_tetra_chirality(tet_const[None, :, :])[0]
    faces = tet_const[_TETRA_FACES]
    nvec, L, l1, l2 = _triangle_geom_terms(faces)
    b_faces = jax.vmap(
        _triangle_bfield_const_precomp,
        in_axes=(None, 0, None, 0, 0, 0, 0),
    )(obs, faces, pol, nvec, L, l1, l2)
    b = jnp.sum(b_faces, axis=0)

    if in_out_flag == _IN_OUT_FLAGS["outside"]:
        inside = jnp.zeros((obs.shape[0],), dtype=bool)
    elif in_out_flag == _IN_OUT_FLAGS["inside"]:
        inside = jnp.ones((obs.shape[0],), dtype=bool)
    else:
        inside = _points_inside_tetra_single(obs, tet_const)
    return b + jnp.where(inside[:, None], pol, 0.0)


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
        nvec, L, l1, l2 = _triangle_geom_terms(faces)
        b_faces = jax.vmap(
            _triangle_bfield_const_precomp,
            in_axes=(None, 0, None, 0, 0, 0, 0),
        )(obs, faces, pol, nvec, L, l1, l2)
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


def tetrahedron_bfield_jit(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """JIT-specialized tetrahedron B-field for fixed observer counts."""
    obs = ensure_observers(observers)
    tet = jnp.asarray(vertices, dtype=jnp.float64)
    if tet.ndim == 3 and tet.shape[0] == 1:
        tet = tet[0]
    if tet.ndim != 2:
        return tetrahedron_bfield(obs, tet, polarizations, in_out=in_out)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    flag = _in_out_flag(in_out)
    jit_fn = _jit_kernel(
        "tetrahedron_bfield",
        _tetrahedron_bfield_const_impl,
        obs.shape[0],
        flag,
    )
    return jit_fn(obs, tet, pol, in_out_flag=flag)


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


_MASK_FACE_SENTINEL = jnp.array(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=jnp.float64
)


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

    start_point_outside = jnp.array(
        [xmin, ymin, zmin], dtype=jnp.float64
    ) - jnp.array([12.0012345, 5.9923456, 6.9932109], dtype=jnp.float64)
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

        start_point_outside = jnp.array(
            [xmin, ymin, zmin], dtype=jnp.float64
        ) - jnp.array([12.0012345, 5.9923456, 6.9932109], dtype=jnp.float64)
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
        flag = _in_out_flag(in_out)
        return _magnet_trimesh_bfield_const_impl(obs, mesh_arr, pol, in_out_flag=flag)

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


def _magnet_trimesh_bfield_const_impl(
    obs: jnp.ndarray,
    mesh_arr: jnp.ndarray,
    pol: jnp.ndarray,
    *,
    in_out_flag: int,
) -> jnp.ndarray:
    nvec, L, l1, l2 = _triangle_geom_terms(mesh_arr)

    def _accumulate_faces() -> jnp.ndarray:
        def body(i: int, acc: jnp.ndarray) -> jnp.ndarray:
            return acc + _triangle_bfield_const_precomp(
                obs, mesh_arr[i], pol, nvec[i], L[i], l1[i], l2[i]
            )

        init = jnp.zeros((obs.shape[0], 3), dtype=jnp.float64)
        return jax.lax.fori_loop(0, mesh_arr.shape[0], body, init)

    if mesh_arr.shape[0] <= 64:
        b_faces = jax.vmap(
            _triangle_bfield_const_precomp,
            in_axes=(None, 0, None, 0, 0, 0, 0),
        )(obs, mesh_arr, pol, nvec, L, l1, l2)
        b = jnp.sum(b_faces, axis=0)
    else:
        b = _accumulate_faces()

    if in_out_flag == _IN_OUT_FLAGS["outside"]:
        inside = jnp.zeros((obs.shape[0],), dtype=bool)
    elif in_out_flag == _IN_OUT_FLAGS["inside"]:
        inside = jnp.ones((obs.shape[0],), dtype=bool)
    else:
        inside = _inside_mask_mesh(obs, mesh_arr)
    return b + jnp.where(inside[:, None], pol, 0.0)


def precompute_trimesh_geometry(
    mesh: ArrayLike,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute triangle mesh geometry terms for reuse."""
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    if mesh_arr.ndim != 3 or mesh_arr.shape[1:] != (3, 3):
        raise ValueError("Mesh must have shape (n_faces,3,3).")
    nvec, L, l1, l2 = _triangle_geom_terms(mesh_arr)
    return mesh_arr, nvec, L, l1, l2


def _magnet_trimesh_bfield_precomp_impl(
    obs: jnp.ndarray,
    mesh_arr: jnp.ndarray,
    pol: jnp.ndarray,
    nvec: jnp.ndarray,
    L: jnp.ndarray,
    l1: jnp.ndarray,
    l2: jnp.ndarray,
    *,
    in_out_flag: int,
    n_faces: int,
) -> jnp.ndarray:
    def _accumulate_faces() -> jnp.ndarray:
        def body(i: int, acc: jnp.ndarray) -> jnp.ndarray:
            return acc + _triangle_bfield_const_precomp(
                obs, mesh_arr[i], pol, nvec[i], L[i], l1[i], l2[i]
            )

        init = jnp.zeros((obs.shape[0], 3), dtype=jnp.float64)
        return jax.lax.fori_loop(0, n_faces, body, init)

    if n_faces <= 64:
        b_faces = jax.vmap(
            _triangle_bfield_const_precomp,
            in_axes=(None, 0, None, 0, 0, 0, 0),
        )(obs, mesh_arr, pol, nvec, L, l1, l2)
        b = jnp.sum(b_faces, axis=0)
    else:
        b = _accumulate_faces()

    if in_out_flag == _IN_OUT_FLAGS["outside"]:
        inside = jnp.zeros((obs.shape[0],), dtype=bool)
    elif in_out_flag == _IN_OUT_FLAGS["inside"]:
        inside = jnp.ones((obs.shape[0],), dtype=bool)
    else:
        inside = _inside_mask_mesh(obs, mesh_arr)
    return b + jnp.where(inside[:, None], pol, 0.0)


def magnet_trimesh_bfield_precomp_masked(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    nvec: ArrayLike,
    L: ArrayLike,
    l1: ArrayLike,
    l2: ArrayLike,
    face_mask: ArrayLike,
    in_out_flag: int,
) -> jnp.ndarray:
    """B-field of triangular mesh using precomputed geometry with face masking."""
    obs = ensure_observers(observers)
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    nvec_arr = jnp.asarray(nvec, dtype=jnp.float64)
    L_arr = jnp.asarray(L, dtype=jnp.float64)
    l1_arr = jnp.asarray(l1, dtype=jnp.float64)
    l2_arr = jnp.asarray(l2, dtype=jnp.float64)
    mask = jnp.asarray(face_mask, dtype=bool).reshape((-1,))
    n_faces = mesh_arr.shape[0]

    def _accumulate_faces() -> jnp.ndarray:
        def body(i: int, acc: jnp.ndarray) -> jnp.ndarray:
            term = _triangle_bfield_const_precomp(
                obs, mesh_arr[i], pol, nvec_arr[i], L_arr[i], l1_arr[i], l2_arr[i]
            )
            term = jnp.where(mask[i], term, 0.0)
            return acc + term

        init = jnp.zeros((obs.shape[0], 3), dtype=jnp.float64)
        return jax.lax.fori_loop(0, n_faces, body, init)

    if n_faces <= 64:
        b_faces = jax.vmap(
            _triangle_bfield_const_precomp,
            in_axes=(None, 0, None, 0, 0, 0, 0),
        )(obs, mesh_arr, pol, nvec_arr, L_arr, l1_arr, l2_arr)
        b_faces = jnp.where(mask[:, None, None], b_faces, 0.0)
        b = jnp.sum(b_faces, axis=0)
    else:
        b = _accumulate_faces()

    inside = jax.lax.switch(
        in_out_flag,
        (
            lambda: _inside_mask_mesh_masked(obs, mesh_arr, mask),
            lambda: jnp.ones((obs.shape[0],), dtype=bool),
            lambda: jnp.zeros((obs.shape[0],), dtype=bool),
        ),
    )
    return b + jnp.where(inside[:, None], pol, 0.0)


def _magnet_trimesh_bfield_faces_impl(
    obs: jnp.ndarray,
    mesh_arr: jnp.ndarray,
    pol: jnp.ndarray,
    *,
    in_out_flag: int,
    n_faces: int,
) -> jnp.ndarray:
    return _magnet_trimesh_bfield_const_impl(obs, mesh_arr, pol, in_out_flag=in_out_flag)


def magnet_trimesh_bfield_jit(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """JIT-specialized triangular mesh B-field for fixed observer counts."""
    obs = ensure_observers(observers)
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    if mesh_arr.ndim == 3:
        return magnet_trimesh_bfield_jit_faces(
            obs, mesh_arr, pol, in_out=in_out
        )
    flag = _in_out_flag(in_out)
    jit_fn = _jit_kernel(
        "triangularmesh_bfield",
        _magnet_trimesh_bfield_const_impl,
        obs.shape[0],
        flag,
    )
    return jit_fn(obs, mesh_arr, pol, in_out_flag=flag)


def magnet_trimesh_bfield_jit_faces(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """JIT-specialized triangular mesh B-field for fixed observer + face counts."""
    obs = ensure_observers(observers)
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    if mesh_arr.ndim != 3:
        raise ValueError("TriangularMesh JIT expects mesh with shape (n_faces,3,3).")
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    flag = _in_out_flag(in_out)
    n_faces = int(mesh_arr.shape[0])
    jit_fn = _jit_kernel_mesh(
        "triangularmesh_bfield_faces",
        _magnet_trimesh_bfield_faces_impl,
        obs.shape[0],
        n_faces,
        flag,
    )
    return jit_fn(obs, mesh_arr, pol, in_out_flag=flag, n_faces=n_faces)


def magnet_trimesh_bfield_jit_faces_precomp(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    nvec: ArrayLike,
    L: ArrayLike,
    l1: ArrayLike,
    l2: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """JIT-specialized triangular mesh B-field using precomputed geometry."""
    obs = ensure_observers(observers)
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    if mesh_arr.ndim != 3:
        raise ValueError("TriangularMesh JIT expects mesh with shape (n_faces,3,3).")
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    n_faces = int(mesh_arr.shape[0])
    flag = _in_out_flag(in_out)
    jit_fn = _jit_kernel_mesh(
        "triangularmesh_bfield_precomp",
        _magnet_trimesh_bfield_precomp_impl,
        obs.shape[0],
        n_faces,
        flag,
    )
    return jit_fn(
        obs,
        mesh_arr,
        pol,
        jnp.asarray(nvec, dtype=jnp.float64),
        jnp.asarray(L, dtype=jnp.float64),
        jnp.asarray(l1, dtype=jnp.float64),
        jnp.asarray(l2, dtype=jnp.float64),
        in_out_flag=flag,
        n_faces=n_faces,
    )


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
    n_phi: int = 96,
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


def precompute_cylinder_segment_geometry(
    dimension: ArrayLike,
    *,
    n_phi: int = 96,
    n_r: int = 1,
    n_z: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute cylinder segment mesh + geometry terms."""
    dim = jnp.asarray(dimension, dtype=jnp.float64)
    mesh = _build_cylinder_segment_mesh(dim, n_phi=n_phi, n_r=n_r, n_z=n_z)
    mesh_arr, nvec, L, l1, l2 = precompute_trimesh_geometry(mesh)
    return mesh_arr, nvec, L, l1, l2


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


def magnet_cylinder_segment_bfield_jit(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """JIT-specialized cylinder-segment B-field for fixed observer counts."""
    return magnet_cylinder_segment_bfield_jit_faces(
        observers, dimensions, polarizations, in_out=in_out
    )


def magnet_cylinder_segment_bfield_jit_faces(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """JIT-specialized cylinder-segment B-field for fixed observer + face counts."""
    obs = ensure_observers(observers)
    dim = _ensure_dim5(dimensions, obs.shape[0])
    mesh, nvec, L, l1, l2 = precompute_cylinder_segment_geometry(dim)
    return magnet_trimesh_bfield_jit_faces_precomp(
        obs, mesh, polarizations, nvec, L, l1, l2, in_out=in_out
    )


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


def _rot_x(theta: jnp.ndarray) -> jnp.ndarray:
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=jnp.float64
    )


def _rot_z(alpha: jnp.ndarray) -> jnp.ndarray:
    c = jnp.cos(alpha)
    s = jnp.sin(alpha)
    return jnp.asarray(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64
    )


def _triangle_coordinate_transform(
    tri: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Transform a triangle to elementar sheet coordinates.

    Returns (u1, u2, v2) coordinates, translation, and rotation matrix.
    """
    a, b, c = tri
    translation = a
    b1 = b - a
    c1 = c - a

    theta = -jnp.arctan2(b1[2], b1[1])
    r21 = _rot_x(theta)
    b2 = r21 @ b1
    c2 = r21 @ c1

    alpha = -jnp.arctan2(b2[1], b2[0])
    r22 = _rot_z(alpha)
    b3 = r22 @ b2
    c3 = r22 @ c2

    psi = -jnp.arctan2(c3[2], c3[1])
    r3 = _rot_x(psi)
    c4 = r3 @ c3

    rotation = r3 @ r22 @ r21
    coords = jnp.asarray([b3[0], c4[0], c4[1]], dtype=jnp.float64)
    return coords, translation, rotation


def _safe_sqrt(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(x, 0.0))


def _safe_atanh(x: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-15
    return jnp.arctanh(jnp.clip(x, -1.0 + eps, 1.0 - eps))


def _safe_logabs(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(jnp.maximum(jnp.abs(x), 1e-30))


def _elementar_current_sheet_hfield(
    observers: jnp.ndarray,
    coordinates: jnp.ndarray,
    current_densities: jnp.ndarray,
) -> jnp.ndarray:
    """H-field for elementar current sheet in local coordinates."""
    num_tol = 1e-10
    x, y, z = observers.T
    u1, u2, v2 = coordinates
    ju, jv = current_densities

    in_plane = jnp.abs(z) < num_tol
    critical_value01 = (x * v2 - y * u2) / (u1 * v2)
    critical_value02 = y / v2
    critical_value1 = jnp.abs(y)
    critical_value2 = jnp.abs(u2 * y - v2 * x)
    critical_value3 = jnp.abs(v2 * (x - u1) + y * (u1 - u2))

    mask0 = (
        in_plane
        & (critical_value01 + critical_value02 <= 1.0 + num_tol)
        & (critical_value01 >= -num_tol)
        & (critical_value02 >= -num_tol)
    )
    mask1 = in_plane & (critical_value1 < num_tol) & (~mask0)
    mask2 = in_plane & (critical_value2 < num_tol) & (~mask0)
    mask3 = in_plane & (critical_value3 < num_tol) & (~mask0)
    mask_plane = ~(mask0 | mask1 | mask2 | mask3) & in_plane
    mask_general = ~in_plane

    sqrt1 = _safe_sqrt(x**2 + y**2 + z**2)
    sqrt2 = _safe_sqrt(u1**2 - 2 * u1 * x + x**2 + y**2 + z**2)
    sqrt3 = _safe_sqrt(u2**2 - 2 * u2 * x + v2**2 - 2 * v2 * y + x**2 + y**2 + z**2)
    sqrt4 = _safe_sqrt(u1**2 - 2 * u1 * u2 + u2**2 + v2**2)
    sqrt5 = _safe_sqrt(u2**2 + v2**2)

    hx_general = (
        jnp.arctan((-u2 * (y**2 + z**2) + v2 * x * y) / (v2 * z * sqrt1))
        + jnp.arctan((v2 * y * (u1 - x) - (u1 - u2) * (y**2 + z**2)) / (v2 * z * sqrt2))
        - jnp.arctan(
            (-u2 * (y**2 + z**2) - v2**2 * x + v2 * y * (u2 + x)) / (v2 * z * sqrt3)
        )
        - jnp.arctan(
            (
                -u1 * (v2**2 - 2 * v2 * y + y**2 + z**2)
                + u2 * (y**2 + z**2)
                + v2**2 * x
                - v2 * y * (u2 + x)
            )
            / (v2 * z * sqrt3)
        )
    ) / (u1 * v2 * z)

    hz_general = -(
        ju * _safe_atanh(x / sqrt1)
        + ju * _safe_atanh((u1 - x) / sqrt2)
        - (ju * (u1 - u2) - jv * v2)
        * _safe_atanh((u1**2 - u1 * (u2 + x) + u2 * x + v2 * y) / (sqrt4 * sqrt2))
        / sqrt4
        + (ju * (u1 - u2) - jv * v2)
        * _safe_atanh(
            (u1 * (u2 - x) - u2**2 + u2 * x + v2 * (-v2 + y)) / (sqrt4 * sqrt3)
        )
        / sqrt4
        + (ju * u2 + jv * v2) * _safe_atanh((-u2 * x - v2 * y) / (sqrt5 * sqrt1)) / sqrt5
        - (ju * u2 + jv * v2)
        * _safe_atanh((u2**2 - u2 * x + v2 * (v2 - y)) / (sqrt5 * sqrt3))
        / sqrt5
    ) / (u1 * v2)

    sqrt_xy = _safe_sqrt(x**2 + y**2)
    sqrt_u1 = _safe_sqrt(u1**2 - 2 * u1 * x + x**2 + y**2)
    sqrt_u2 = _safe_sqrt(u2**2 - 2 * u2 * x + v2**2 - 2 * v2 * y + x**2 + y**2)
    sqrt_u12 = _safe_sqrt(u1**2 - 2 * u1 * u2 + u2**2 + v2**2)
    sqrt_u2v2 = _safe_sqrt(u2**2 + v2**2)

    hz_plane = -(
        ju * _safe_atanh(x / sqrt_xy)
        + ju * _safe_atanh((u1 - x) / sqrt_u1)
        - (ju * (u1 - u2) - jv * v2)
        * _safe_atanh((u1**2 - u1 * (u2 + x) + u2 * x + v2 * y) / (sqrt_u12 * sqrt_u1))
        / sqrt_u12
        + (ju * (u1 - u2) - jv * v2)
        * _safe_atanh(
            (u1 * (u2 - x) - u2**2 + u2 * x + v2 * (-v2 + y)) / (sqrt_u12 * sqrt_u2)
        )
        / sqrt_u12
        + (ju * u2 + jv * v2)
        * _safe_atanh((-u2 * x - v2 * y) / (sqrt_u2v2 * sqrt_xy))
        / sqrt_u2v2
        - (ju * u2 + jv * v2)
        * _safe_atanh((u2**2 - u2 * x + v2 * (v2 - y)) / (sqrt_u2v2 * sqrt_u2))
        / sqrt_u2v2
    ) / (u1 * v2)

    hz_edge1 = (
        -ju * x * _safe_logabs(x) / _safe_sqrt(x**2)
        - ju * (u1 - x) * _safe_logabs(-u1 + x) / _safe_sqrt((u1 - x) ** 2)
        + (ju * (u1 - u2) - jv * v2)
        * _safe_atanh(
            (u1 * (-u2 + x) + u2**2 - u2 * x + v2**2)
            / (sqrt_u12 * _safe_sqrt(u2**2 - 2 * u2 * x + v2**2 + x**2))
        )
        / sqrt_u12
        + (ju * (u1 - u2) - jv * v2)
        * _safe_atanh((u1 - u2) * (u1 - x) / (sqrt_u12 * _safe_sqrt((u1 - x) ** 2)))
        / sqrt_u12
        + (ju * u2 + jv * v2)
        * _safe_atanh(
            (u2**2 - u2 * x + v2**2)
            / (sqrt_u2v2 * _safe_sqrt(u2**2 - 2 * u2 * x + v2**2 + x**2))
        )
        / sqrt_u2v2
        - (ju * u2 + jv * v2)
        * _safe_atanh(u2 * (u1 - x) / (sqrt_u2v2 * _safe_sqrt((u1 - x) ** 2)))
        / sqrt_u2v2
    ) / (u1 * v2)

    hz_edge2 = (
        -ju
        * _safe_atanh(
            (u1 * v2 - u2 * y)
            / (
                v2
                * _safe_sqrt(u1**2 - 2 * u1 * u2 * y / v2 + y**2 * (u2**2 / v2**2 + 1))
            )
        )
        + ju
        * _safe_atanh(
            u2 * (v2 - y)
            / (v2 * _safe_sqrt((u2**2 + v2**2) * (v2 - y) ** 2 / v2**2))
        )
        + (ju * (u1 - u2) - jv * v2)
        * _safe_atanh(
            (u1**2 * v2 - u1 * u2 * (v2 + y) + y * (u2**2 + v2**2))
            / (
                v2
                * _safe_sqrt(u1**2 - 2 * u1 * u2 * y / v2 + y**2 * (u2**2 / v2**2 + 1))
                * sqrt_u12
            )
        )
        / sqrt_u12
        + (ju * (u1 - u2) - jv * v2)
        * _safe_atanh(
            (v2 - y) * (-u1 * u2 + u2**2 + v2**2)
            / (
                v2
                * _safe_sqrt((u2**2 + v2**2) * (v2 - y) ** 2 / v2**2)
                * sqrt_u12
            )
        )
        / sqrt_u12
        + y
        * (ju * u2 + jv * v2)
        * _safe_logabs(y * (-(u2**2) - v2**2))
        / (v2 * _safe_sqrt(y**2 * (u2**2 + v2**2) / v2**2))
        + (v2 - y)
        * (ju * u2 + jv * v2)
        * _safe_logabs((u2**2 + v2**2) * (v2 - y))
        / (v2 * _safe_sqrt((u2**2 + v2**2) * (v2 - y) ** 2 / v2**2))
    ) / (u1 * v2)

    hz_edge3 = (
        ju
        * v2
        * _safe_atanh(
            (u1 * (-v2 + y) - u2 * y)
            / (
                v2
                * _safe_sqrt(
                    (u1**2 * (v2 - y) ** 2 + 2 * u1 * u2 * y * (v2 - y) + y**2 * (u2**2 + v2**2))
                    / v2**2
                )
            )
        )
        + ju
        * v2
        * _safe_atanh(
            (u1 - u2)
            * (v2 - y)
            / (
                v2
                * _safe_sqrt((v2 - y) ** 2 * (u1**2 - 2 * u1 * u2 + u2**2 + v2**2) / v2**2)
            )
        )
        - v2
        * (ju * u2 + jv * v2)
        * _safe_atanh(
            (u1 * u2 * (-v2 + y) + y * (-(u2**2) - v2**2))
            / (
                v2
                * _safe_sqrt(
                    (u1**2 * (v2 - y) ** 2 + 2 * u1 * u2 * y * (v2 - y) + y**2 * (u2**2 + v2**2))
                    / v2**2
                )
                * sqrt_u2v2
            )
        )
        / sqrt_u2v2
        + v2
        * (ju * u2 + jv * v2)
        * _safe_atanh(
            (v2 - y)
            * (-u1 * u2 + u2**2 + v2**2)
            / (
                v2
                * _safe_sqrt((v2 - y) ** 2 * (u1**2 - 2 * u1 * u2 + u2**2 + v2**2) / v2**2)
                * sqrt_u2v2
            )
        )
        / sqrt_u2v2
        - y
        * (ju * (-u1 + u2) + jv * v2)
        * _safe_logabs(y * (-(u1**2) + 2 * u1 * u2 - u2**2 - v2**2))
        / _safe_sqrt(y**2 * (u1**2 - 2 * u1 * u2 + u2**2 + v2**2) / v2**2)
        - (v2 - y)
        * (ju * (-u1 + u2) + jv * v2)
        * _safe_logabs((v2 - y) * (u1**2 - 2 * u1 * u2 + u2**2 + v2**2))
        / _safe_sqrt((v2 - y) ** 2 * (u1**2 - 2 * u1 * u2 + u2**2 + v2**2) / v2**2)
    ) / (u1 * v2**2)

    hx = jnp.where(mask_general, hx_general, 0.0)
    hz = jnp.where(mask_general, hz_general, 0.0)
    hz = jnp.where(mask_plane, hz_plane, hz)
    hz = jnp.where(mask1, hz_edge1, hz)
    hz = jnp.where(mask2, hz_edge2, hz)
    hz = jnp.where(mask3, hz_edge3, hz)

    scale = (u1 * v2) / _FOUR_PI
    hx_scaled = hx * jv * z * scale
    hy_scaled = hx * (-ju) * z * scale
    hz_scaled = hz * scale

    return jnp.stack((hx_scaled, hy_scaled, hz_scaled), axis=1)


def _current_triangle_sheet_hfield_obs(
    obs: jnp.ndarray,
    tri: jnp.ndarray,
    cd: jnp.ndarray,
) -> jnp.ndarray:
    coords, translation, rotation = _triangle_coordinate_transform(tri)
    obs_loc = (obs - translation[None, :]) @ rotation.T
    cd_loc = (rotation @ cd)[:2]

    u1, u2, v2 = coords
    degenerate = (
        jnp.isnan(u1)
        | jnp.isnan(u2)
        | jnp.isnan(v2)
        | (jnp.abs(u1) < 1e-15)
        | (jnp.abs(v2) < 1e-15)
    )
    h_local = _elementar_current_sheet_hfield(obs_loc, coords, cd_loc)
    h_local = jnp.where(degenerate, 0.0, h_local)
    return h_local @ rotation


def current_triangle_sheet_hfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    current_densities: ArrayLike,
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    tri = jnp.asarray(vertices, dtype=jnp.float64)
    if tri.shape != (3, 3):
        raise ValueError(f"Triangle sheet vertices must have shape (3,3), got {tri.shape}.")
    cd = jnp.asarray(current_densities, dtype=jnp.float64)
    if cd.shape != (3,):
        raise ValueError(
            f"Triangle sheet current density must have shape (3,), got {cd.shape}."
        )

    return _current_triangle_sheet_hfield_obs(obs, tri, cd)


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

    h_faces = jax.vmap(lambda tri, cd: _current_triangle_sheet_hfield_obs(obs, tri, cd))(
        tris, cds
    )
    return jnp.sum(h_faces, axis=0)


def current_trisheet_bfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    faces: ArrayLike,
    current_densities: ArrayLike,
) -> jnp.ndarray:
    return MU0 * current_trisheet_hfield(observers, vertices, faces, current_densities)


def current_trisheet_bfield_masked(
    observers: ArrayLike,
    triangles: ArrayLike,
    current_densities: ArrayLike,
    face_mask: ArrayLike,
) -> jnp.ndarray:
    """B-field of triangle sheet with face masking."""
    obs = ensure_observers(observers)
    tris = jnp.asarray(triangles, dtype=jnp.float64)
    cds = jnp.asarray(current_densities, dtype=jnp.float64)
    mask = jnp.asarray(face_mask, dtype=jnp.float64).reshape((-1,))
    h_faces = jax.vmap(lambda tri, cd: _current_triangle_sheet_hfield_obs(obs, tri, cd))(
        tris, cds
    )
    h_faces = h_faces * mask[:, None, None]
    return MU0 * jnp.sum(h_faces, axis=0)


def current_trisheet_bfield_jit(
    observers: ArrayLike,
    vertices: ArrayLike,
    faces: ArrayLike,
    current_densities: ArrayLike,
) -> jnp.ndarray:
    """JIT-specialized triangle sheet B-field for fixed observer counts."""
    obs = ensure_observers(observers)
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    facs = jnp.asarray(faces, dtype=jnp.int32)
    cds = jnp.asarray(current_densities, dtype=jnp.float64)
    jit_fn = _jit_kernel_simple("trianglesheet_bfield", current_trisheet_bfield, obs.shape[0])
    return jit_fn(obs, verts, facs, cds)


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
    h_faces = jax.vmap(lambda tri, cd: _current_triangle_sheet_hfield_obs(obs, tri, cd))(
        tris, cds
    )
    return jnp.sum(h_faces, axis=0)


def current_tristrip_bfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    current: ArrayLike,
) -> jnp.ndarray:
    return MU0 * current_tristrip_hfield(observers, vertices, current)


def current_tristrip_bfield_jit(
    observers: ArrayLike,
    vertices: ArrayLike,
    current: ArrayLike,
) -> jnp.ndarray:
    """JIT-specialized triangle strip B-field for fixed observer counts."""
    obs = ensure_observers(observers)
    verts = jnp.asarray(vertices, dtype=jnp.float64)
    curr = jnp.asarray(current, dtype=jnp.float64)
    jit_fn = _jit_kernel_simple("trianglestrip_bfield", current_tristrip_bfield, obs.shape[0])
    return jit_fn(obs, verts, curr)
