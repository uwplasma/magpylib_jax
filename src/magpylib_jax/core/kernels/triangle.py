"""Triangle surface-charge field kernels and shared triangle geometry."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import ensure_observers
from magpylib_jax.core.kernels._common import (
    _FOUR_PI,
    _broadcast_vec3,
    _jit_kernel_simple,
    _safe_norm,
)


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


def _triangle_bfield_const_impl(
    obs: jnp.ndarray,
    tri: jnp.ndarray,
    pol: jnp.ndarray,
) -> jnp.ndarray:
    nvec, L, l1, l2 = _triangle_geom_terms(tri[None, :, :])
    return _triangle_bfield_const_precomp(obs, tri, pol, nvec[0], L[0], l1[0], l2[0])


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
