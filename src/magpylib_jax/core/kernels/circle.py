"""Circular current-loop field kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import cart_to_cyl, cyl_field_to_cart, ensure_observers
from magpylib_jax.core.kernels._common import _FOUR_PI, _jit_kernel_simple


def _cel_iter(
    qc: jnp.ndarray,
    p: jnp.ndarray,
    g: jnp.ndarray,
    cc: jnp.ndarray,
    ss: jnp.ndarray,
    em: jnp.ndarray,
    kk: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorized Bulirsch CEL iteration in JAX."""

    def body_fn(_: int, state: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
        qc_, p_, g_, cc_, ss_, em_, kk_ = state
        mask = jnp.abs(g_ - qc_) >= qc_ * 1e-8

        qc_new = 2.0 * jnp.sqrt(kk_)
        kk_new = qc_new * em_
        f = cc_
        cc_new = cc_ + ss_ / p_
        g_new = kk_new / p_
        ss_new = 2.0 * (ss_ + f * g_new)
        p_new = p_ + g_new
        g_store = em_
        em_new = em_ + qc_new

        qc_out = jnp.where(mask, qc_new, qc_)
        p_out = jnp.where(mask, p_new, p_)
        g_out = jnp.where(mask, g_store, g_)
        cc_out = jnp.where(mask, cc_new, cc_)
        ss_out = jnp.where(mask, ss_new, ss_)
        em_out = jnp.where(mask, em_new, em_)
        kk_out = jnp.where(mask, kk_new, kk_)
        return qc_out, p_out, g_out, cc_out, ss_out, em_out, kk_out

    qc, p, _, cc, ss, em, _ = lax.fori_loop(0, 32, body_fn, (qc, p, g, cc, ss, em, kk))
    return 0.5 * jnp.pi * (ss + cc * em) / (em * (em + p))


@jax.jit
def current_circle_hfield(
    observers: ArrayLike,
    diameter: ArrayLike,
    current: ArrayLike,
    *,
    singular_tol: float = 1e-15,
) -> jnp.ndarray:
    """H-field of circular current loops centered at the origin in the xy plane."""
    obs = ensure_observers(observers)
    r, phi, z = cart_to_cyl(obs)

    radius = jnp.abs(jnp.asarray(diameter, dtype=float) / 2.0)
    cur = jnp.asarray(current, dtype=float)
    radius = jnp.broadcast_to(radius, r.shape)
    cur = jnp.broadcast_to(cur, r.shape)

    mask_zero_radius = radius == 0.0
    mask_singular = jnp.logical_and(jnp.abs(r - radius) < singular_tol * radius, z == 0.0)
    mask_general = jnp.logical_not(jnp.logical_or(mask_zero_radius, mask_singular))

    safe_radius = jnp.where(mask_general, radius, 1.0)
    rr = r / safe_radius
    zz = z / safe_radius

    z2 = zz * zz
    x0 = z2 + (rr + 1.0) ** 2
    k2 = 4.0 * rr / x0
    q2 = (z2 + (rr - 1.0) ** 2) / x0

    q2 = jnp.where(mask_general, q2, 1.0)
    q = jnp.sqrt(q2)
    p = 1.0 + q
    pf = cur / (_FOUR_PI * safe_radius * jnp.sqrt(x0) * q2)

    cc = k2 * 4.0 * zz / x0
    ss = 2.0 * cc * q / p
    hr = pf * _cel_iter(q, p, jnp.ones_like(q), cc, ss, p, q)

    k4 = k2 * k2
    cc = k4 - (q2 + 1.0) * (4.0 / x0)
    ss = 2.0 * q * (k4 / p - (4.0 / x0) * p)
    hz = -pf * _cel_iter(q, p, jnp.ones_like(q), cc, ss, p, q)

    hr = jnp.where(mask_general, hr, 0.0)
    hz = jnp.where(mask_general, hz, 0.0)

    return cyl_field_to_cart(phi, hr, hz)


def current_circle_bfield(
    observers: ArrayLike,
    diameter: ArrayLike,
    current: ArrayLike,
) -> jnp.ndarray:
    """B-field of a current circle (Tesla)."""
    return jnp.asarray(MU0 * current_circle_hfield(observers, diameter, current), dtype=float)


def current_circle_bfield_jit(
    observers: ArrayLike,
    diameter: ArrayLike,
    current: ArrayLike,
) -> jnp.ndarray:
    """JIT-specialized circle B-field for fixed observer counts."""
    obs = ensure_observers(observers)
    dia = jnp.asarray(diameter, dtype=float)
    cur = jnp.asarray(current, dtype=float)
    jit_fn = _jit_kernel_simple("circle_bfield", current_circle_bfield, obs.shape[0])
    return jit_fn(obs, dia, cur)
