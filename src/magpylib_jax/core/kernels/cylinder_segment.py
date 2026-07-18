"""Cylinder-segment magnet field kernels (meshed representation)."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import ensure_observers
from magpylib_jax.core.kernels._common import _broadcast_vec3
from magpylib_jax.core.kernels.trimesh import (
    magnet_trimesh_bfield,
    magnet_trimesh_bfield_jit_faces_precomp,
    precompute_trimesh_geometry,
)


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

    phis = jnp.linspace(phi1, phi2, n_phi + 1, dtype=float)
    rs = jnp.linspace(r1, r2, n_r + 1, dtype=float)
    zs = jnp.linspace(zmin, zmax, n_z + 1, dtype=float)

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
    dim = jnp.asarray(dimension, dtype=float)
    mesh = _build_cylinder_segment_mesh(dim, n_phi=n_phi, n_r=n_r, n_z=n_z)
    mesh_arr, nvec, L, l1, l2 = precompute_trimesh_geometry(mesh)
    return mesh_arr, nvec, L, l1, l2


def _ensure_dim5(dimensions: ArrayLike, n: int) -> jnp.ndarray:
    dim = jnp.asarray(dimensions, dtype=float)
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
            raise ValueError("Per-observer varying CylinderSegment dimensions are not supported.")
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

    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=float), obs.shape[0])
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
    return magnet_cylinder_segment_jfield(observers, dimensions, polarizations, in_out=in_out) / MU0
