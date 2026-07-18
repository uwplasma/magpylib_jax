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
    mom = _broadcast_vector(jnp.asarray(moments, dtype=float), obs.shape)

    r2 = jnp.sum(obs * obs, axis=-1)
    origin_mask = r2 == 0.0
    # Keep the r2 fed to the negative powers strictly positive so the general
    # branch stays finite (primal *and* tangent) at r=0; the physical Inf is
    # restored by the ``origin_mask`` overwrite below, so the primal for r>0 and
    # the Inf at r=0 are both unchanged.
    safe_r2 = jnp.where(origin_mask, 1.0, r2)
    inv_r3 = safe_r2 ** (-1.5)
    inv_r5 = safe_r2 ** (-2.5)
    mdotr = jnp.sum(mom * obs, axis=-1)

    h = (3.0 * mdotr[:, None] * obs * inv_r5[:, None] - mom * inv_r3[:, None]) / _FOUR_PI

    # The singular value is a hard Inf; freeze its gradient so grad/jacfwd on the
    # singular set are finite (0) instead of 0 * Inf = NaN. Primal is unchanged.
    h_origin = jax.lax.stop_gradient(jnp.where(mom == 0.0, 0.0, jnp.sign(mom) * jnp.inf))
    return jnp.where(origin_mask[:, None], h_origin, h)


def dipole_bfield(observers: ArrayLike, moments: ArrayLike) -> jnp.ndarray:
    """B-field of a dipole (Tesla)."""
    return jnp.asarray(MU0 * dipole_hfield(observers, moments), dtype=float)
