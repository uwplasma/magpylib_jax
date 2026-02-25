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
    if hasattr(orientation, "as_matrix"):
        mat = jnp.asarray(orientation.as_matrix(), dtype=jnp.float64)
        if mat.ndim == 3:
            if mat.shape[0] != 1:
                raise ValueError(
                    "Expected single orientation for this context, "
                    f"got {mat.shape[0]} orientations."
                )
            return mat[0]
        if mat.shape != (3, 3):
            raise ValueError(f"Expected orientation matrix with shape (3, 3), got {mat.shape}.")
        return mat

    ori = jnp.asarray(orientation, dtype=jnp.float64)
    if ori.ndim == 3:
        if ori.shape[0] != 1 or ori.shape[1:] != (3, 3):
            raise ValueError(
                "Expected single orientation matrix with shape (3, 3), "
                f"got {ori.shape}."
            )
        return ori[0]
    if ori.shape != (3, 3):
        raise ValueError(f"Expected orientation matrix with shape (3, 3), got {ori.shape}.")
    return ori


def normalize_positions(position: ArrayLike = (0.0, 0.0, 0.0)) -> jnp.ndarray:
    """Return positions as shape (p, 3)."""
    pos = _as_array3(position)
    if pos.ndim == 1:
        return pos[None, :]
    if pos.ndim == 2:
        return pos
    raise ValueError(f"Expected position shape (3,) or (p,3), got {pos.shape}.")


def normalize_orientations(orientation: ArrayLike | None = None) -> jnp.ndarray:
    """Return orientations as shape (p, 3, 3)."""
    if orientation is None:
        return jnp.eye(3, dtype=jnp.float64)[None, :, :]
    if hasattr(orientation, "as_matrix"):
        mat = jnp.asarray(orientation.as_matrix(), dtype=jnp.float64)
    else:
        mat = jnp.asarray(orientation, dtype=jnp.float64)
    if mat.ndim == 2:
        if mat.shape != (3, 3):
            raise ValueError(f"Expected orientation matrix with shape (3, 3), got {mat.shape}.")
        return mat[None, :, :]
    if mat.ndim == 3 and mat.shape[1:] == (3, 3):
        return mat
    raise ValueError(
        "Expected orientation shape (3,3), (p,3,3), or scipy Rotation; "
        f"got {mat.shape}."
    )


def broadcast_pose(
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Broadcast position/orientation path lengths with singleton expansion."""
    pos = normalize_positions(position)
    rot = normalize_orientations(orientation)
    n_pos, n_rot = pos.shape[0], rot.shape[0]
    n = max(n_pos, n_rot)
    if n_pos not in (1, n):
        raise ValueError(
            f"Incompatible position path length {n_pos} for broadcast length {n}."
        )
    if n_rot not in (1, n):
        raise ValueError(
            f"Incompatible orientation path length {n_rot} for broadcast length {n}."
        )
    if n_pos == 1 and n > 1:
        pos = jnp.broadcast_to(pos, (n, 3))
    if n_rot == 1 and n > 1:
        rot = jnp.broadcast_to(rot, (n, 3, 3))
    return pos, rot


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


def cyl_field_to_cart(
    phi: jnp.ndarray,
    hr: jnp.ndarray,
    hphi_or_hz: jnp.ndarray,
    hz: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Convert cylindrical field components to Cartesian field vectors.

    Backward-compatible call forms:
    - ``cyl_field_to_cart(phi, hr, hz)`` assumes ``Hphi=0``
    - ``cyl_field_to_cart(phi, hr, hphi, hz)`` uses full cylindrical vector
    """
    if hz is None:
        hphi = jnp.zeros_like(hr)
        hz_arr = hphi_or_hz
    else:
        hphi = hphi_or_hz
        hz_arr = hz
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    hx = hr * cos_phi - hphi * sin_phi
    hy = hr * sin_phi + hphi * cos_phi
    return jnp.stack((hx, hy, hz_arr), axis=-1)
