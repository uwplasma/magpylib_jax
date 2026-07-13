"""Dipole field kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import ensure_observers
from magpylib_jax.core.kernels._common import _FOUR_PI, _broadcast_vector


@jax.jit
def dipole_hfield(observers: ArrayLike, moments: ArrayLike) -> jnp.ndarray:
    """H-field of dipole moments located at the origin."""
    obs = ensure_observers(observers)
    mom = _broadcast_vector(jnp.asarray(moments, dtype=jnp.float64), obs.shape)

    r2 = jnp.sum(obs * obs, axis=-1)
    inv_r3 = jnp.where(r2 > 0.0, r2 ** (-1.5), jnp.inf)
    inv_r5 = jnp.where(r2 > 0.0, r2 ** (-2.5), jnp.inf)
    mdotr = jnp.sum(mom * obs, axis=-1)

    h = (3.0 * mdotr[:, None] * obs * inv_r5[:, None] - mom * inv_r3[:, None]) / _FOUR_PI

    origin_mask = r2 == 0.0
    h_origin = jnp.where(mom == 0.0, 0.0, jnp.sign(mom) * jnp.inf)
    return jnp.where(origin_mask[:, None], h_origin, h)


def dipole_bfield(observers: ArrayLike, moments: ArrayLike) -> jnp.ndarray:
    """B-field of a dipole (Tesla)."""
    return jnp.asarray(MU0 * dipole_hfield(observers, moments), dtype=jnp.float64)
