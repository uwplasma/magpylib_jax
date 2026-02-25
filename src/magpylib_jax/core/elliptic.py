"""Bulirsch CEL and derived complete elliptic integrals in JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def cel(
    kc: jnp.ndarray,
    p: jnp.ndarray,
    c: jnp.ndarray,
    s: jnp.ndarray,
    *,
    max_iter: int = 40,
    errtol: float = 1e-8,
) -> jnp.ndarray:
    """Vectorized complete elliptic integral in Bulirsch CEL form."""
    tiny = jnp.array(1e-30, dtype=jnp.float64)

    kc = jnp.asarray(kc, dtype=jnp.float64)
    pp = jnp.asarray(p, dtype=jnp.float64)
    cc = jnp.asarray(c, dtype=jnp.float64)
    ss = jnp.asarray(s, dtype=jnp.float64)

    k = jnp.where(jnp.abs(kc) < tiny, tiny, jnp.abs(kc))
    em = jnp.ones_like(k)

    mask_nonpos = pp <= 0.0

    pp_pos = jnp.sqrt(jnp.maximum(pp, tiny))
    ss_pos = ss / pp_pos

    f = kc * kc
    q = 1.0 - f
    g = 1.0 - pp
    f = f - pp
    safe_g = jnp.where(jnp.abs(g) < tiny, tiny, g)
    q = q * (ss - cc * pp)
    pp_neg = jnp.sqrt(jnp.maximum(f / safe_g, tiny))
    cc_neg = (cc - ss) / safe_g
    ss_neg = -q / (safe_g * safe_g * pp_neg) + cc_neg * pp_neg

    pp = jnp.where(mask_nonpos, pp_neg, pp_pos)
    cc = jnp.where(mask_nonpos, cc_neg, cc)
    ss = jnp.where(mask_nonpos, ss_neg, ss_pos)

    f = cc
    cc = cc + ss / pp
    g = k / pp
    ss = 2.0 * (ss + f * g)
    pp = g + pp
    g = em
    em = k + em
    kk = k

    def body_fn(_: int, state: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
        k_, pp_, g_, cc_, ss_, em_, kk_ = state
        active = jnp.abs(g_ - k_) > g_ * errtol

        k_new = 2.0 * jnp.sqrt(kk_)
        kk_new = k_new * em_
        f_new = cc_
        cc_new = cc_ + ss_ / pp_
        g_new = kk_new / pp_
        ss_new = 2.0 * (ss_ + f_new * g_new)
        pp_new = g_new + pp_
        g_store = em_
        em_new = k_new + em_

        k_out = jnp.where(active, k_new, k_)
        pp_out = jnp.where(active, pp_new, pp_)
        g_out = jnp.where(active, g_store, g_)
        cc_out = jnp.where(active, cc_new, cc_)
        ss_out = jnp.where(active, ss_new, ss_)
        em_out = jnp.where(active, em_new, em_)
        kk_out = jnp.where(active, kk_new, kk_)
        return k_out, pp_out, g_out, cc_out, ss_out, em_out, kk_out

    k, pp, g, cc, ss, em, _ = lax.fori_loop(0, max_iter, body_fn, (k, pp, g, cc, ss, em, kk))

    return 0.5 * jnp.pi * (ss + cc * em) / (em * (em + pp))


@jax.jit
def ellipk(m: jnp.ndarray) -> jnp.ndarray:
    """Complete elliptic integral of the first kind K(m)."""
    m = jnp.asarray(m, dtype=jnp.float64)
    one = jnp.ones_like(m)
    return cel(jnp.sqrt(1.0 - m), one, one, one)


@jax.jit
def ellipe(m: jnp.ndarray) -> jnp.ndarray:
    """Complete elliptic integral of the second kind E(m)."""
    m = jnp.asarray(m, dtype=jnp.float64)
    one = jnp.ones_like(m)
    return cel(jnp.sqrt(1.0 - m), one, one, one - m)


@jax.jit
def ellippi(n: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    """Complete elliptic integral of the third kind Π(n, m)."""
    n = jnp.asarray(n, dtype=jnp.float64)
    m = jnp.asarray(m, dtype=jnp.float64)
    one = jnp.ones_like(m)
    return cel(jnp.sqrt(1.0 - m), 1.0 - n, one, one)
