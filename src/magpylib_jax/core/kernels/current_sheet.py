"""Triangle current-sheet field kernels (closed-form elementar sheet)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import ensure_observers
from magpylib_jax.core.kernels._common import _FOUR_PI, _jit_kernel_simple
from magpylib_jax.core.kernels._safe import _safe_atanh, _safe_logabs, _safe_sqrt

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
    return jnp.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=jnp.float64)


def _rot_z(alpha: jnp.ndarray) -> jnp.ndarray:
    c = jnp.cos(alpha)
    s = jnp.sin(alpha)
    return jnp.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64)


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
        - jnp.arctan((-u2 * (y**2 + z**2) - v2**2 * x + v2 * y * (u2 + x)) / (v2 * z * sqrt3))
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
        * _safe_atanh((u1 * (u2 - x) - u2**2 + u2 * x + v2 * (-v2 + y)) / (sqrt4 * sqrt3))
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
        * _safe_atanh((u1 * (u2 - x) - u2**2 + u2 * x + v2 * (-v2 + y)) / (sqrt_u12 * sqrt_u2))
        / sqrt_u12
        + (ju * u2 + jv * v2) * _safe_atanh((-u2 * x - v2 * y) / (sqrt_u2v2 * sqrt_xy)) / sqrt_u2v2
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
            (u2**2 - u2 * x + v2**2) / (sqrt_u2v2 * _safe_sqrt(u2**2 - 2 * u2 * x + v2**2 + x**2))
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
            / (v2 * _safe_sqrt(u1**2 - 2 * u1 * u2 * y / v2 + y**2 * (u2**2 / v2**2 + 1)))
        )
        + ju
        * _safe_atanh(u2 * (v2 - y) / (v2 * _safe_sqrt((u2**2 + v2**2) * (v2 - y) ** 2 / v2**2)))
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
            (v2 - y)
            * (-u1 * u2 + u2**2 + v2**2)
            / (v2 * _safe_sqrt((u2**2 + v2**2) * (v2 - y) ** 2 / v2**2) * sqrt_u12)
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
            / (v2 * _safe_sqrt((v2 - y) ** 2 * (u1**2 - 2 * u1 * u2 + u2**2 + v2**2) / v2**2))
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
        raise ValueError(f"Triangle sheet current density must have shape (3,), got {cd.shape}.")

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

    h_faces = jax.vmap(lambda tri, cd: _current_triangle_sheet_hfield_obs(obs, tri, cd))(tris, cds)
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
    h_faces = jax.vmap(lambda tri, cd: _current_triangle_sheet_hfield_obs(obs, tri, cd))(tris, cds)
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
