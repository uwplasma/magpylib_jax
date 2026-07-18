"""Sphere magnet field kernels."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import ensure_observers
from magpylib_jax.core.kernels._common import _broadcast_vec3, _safe_norm


def magnet_sphere_bfield(
    observers: ArrayLike,
    diameters: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """B-field of homogeneously polarized spheres centered at the origin."""
    obs = ensure_observers(observers)
    n = obs.shape[0]
    dia = jnp.asarray(diameters, dtype=float)
    if dia.ndim == 0:
        dia = jnp.broadcast_to(dia, (n,))
    else:
        dia = jnp.broadcast_to(dia.reshape((-1,)), (n,))

    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=float), n)

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
    dia = jnp.asarray(diameters, dtype=float)
    if dia.ndim == 0:
        dia = jnp.broadcast_to(dia, (n,))
    else:
        dia = jnp.broadcast_to(dia.reshape((-1,)), (n,))

    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=float), n)
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
    dia = jnp.asarray(diameters, dtype=float)
    if dia.ndim == 0:
        dia = jnp.broadcast_to(dia, (n,))
    else:
        dia = jnp.broadcast_to(dia.reshape((-1,)), (n,))

    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=float), n)
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
