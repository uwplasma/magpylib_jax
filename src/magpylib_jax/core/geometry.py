"""Geometry helpers for frame transforms and coordinate conversions."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike


def _as_array3(x: ArrayLike) -> jnp.ndarray:
    arr = jnp.asarray(x, dtype=jnp.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected trailing dimension 3, got shape {arr.shape}.")
    return arr


def ensure_observers(observers: ArrayLike) -> jnp.ndarray:
    """Normalize observers to a rank-2 array of shape (n, 3)."""
    arr = _as_array3(observers)
    if arr.ndim == 1:
        return arr[None, :]
    return arr.reshape((-1, 3))


def normalize_orientation(orientation: ArrayLike | None) -> jnp.ndarray:
    """Return a 3x3 rotation matrix."""
    if orientation is None:
        return jnp.eye(3, dtype=jnp.float64)
    ori = jnp.asarray(orientation, dtype=jnp.float64)
    if ori.shape != (3, 3):
        raise ValueError(f"Expected orientation matrix with shape (3, 3), got {ori.shape}.")
    return ori


def to_local_coordinates(
    observers: ArrayLike,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map global observer coordinates into source-local frame."""
    obs = ensure_observers(observers)
    pos = _as_array3(position)
    if pos.ndim != 1:
        raise ValueError(f"Expected position shape (3,), got {pos.shape}.")
    rot = normalize_orientation(orientation)
    obs_local = (obs - pos) @ rot
    return obs_local, rot


def to_global_field(field_local: jnp.ndarray, rotation_matrix: jnp.ndarray) -> jnp.ndarray:
    """Map local-frame vectors back to global coordinates."""
    return field_local @ rotation_matrix.T


def cart_to_cyl(observers: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert Cartesian coordinates to cylindrical coordinates."""
    x, y, z = observers.T
    r = jnp.sqrt(x * x + y * y)
    phi = jnp.arctan2(y, x)
    return r, phi, z


def cyl_field_to_cart(phi: jnp.ndarray, hr: jnp.ndarray, hz: jnp.ndarray) -> jnp.ndarray:
    """Convert cylindrical field components (Hr, 0, Hz) to Cartesian field vectors."""
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    hx = hr * cos_phi
    hy = hr * sin_phi
    return jnp.stack((hx, hy, hz), axis=-1)
