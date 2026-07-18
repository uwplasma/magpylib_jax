"""Triangle-strip current field kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import ensure_observers
from magpylib_jax.core.kernels._common import _jit_kernel_simple
from magpylib_jax.core.kernels.current_sheet import _current_triangle_sheet_hfield_obs


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
    verts = jnp.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 3:
        raise ValueError("TriangleStrip vertices must have shape (n>=3,3).")
    cur = jnp.asarray(current, dtype=float).reshape(())
    tris = _strip_triangles(verts)
    cds = _strip_current_densities(verts, cur)
    h_faces = jax.vmap(lambda tri, cd: _current_triangle_sheet_hfield_obs(obs, tri, cd))(tris, cds)
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
    verts = jnp.asarray(vertices, dtype=float)
    curr = jnp.asarray(current, dtype=float)
    jit_fn = _jit_kernel_simple("trianglestrip_bfield", current_tristrip_bfield, obs.shape[0])
    return jit_fn(obs, verts, curr)
