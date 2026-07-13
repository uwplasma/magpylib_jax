"""Numerically safe math helpers for the singular field kernels.

Each helper carries a :func:`jax.custom_jvp` so that its *tangent* stays finite
at the singular argument (where the naive derivative would be ``Inf``/``NaN``),
while the *primal* value is byte-for-byte identical to the plain expression.
This defuses the classic "double where" gradient trap in which masking is only
applied to the primal output, leaving ``jax.grad`` to propagate a ``NaN``
through the unused (singular) branch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.custom_jvp
def _safe_sqrt(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(x, 0.0))


@_safe_sqrt.defjvp
def _safe_sqrt_jvp(primals, tangents):
    (x,) = primals
    (dx,) = tangents
    y = _safe_sqrt(x)
    pos = x > 0.0
    # 0.5 / sqrt(x) in the interior, 0 at/below the singular argument x == 0.
    # The inner ``where`` keeps the sqrt argument strictly positive so the
    # unused branch never materialises an Inf that could poison the tangent.
    dy = jnp.where(pos, 0.5 / jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0) * dx
    return y, dy


@jax.custom_jvp
def _safe_atanh(x: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-15
    return jnp.arctanh(jnp.clip(x, -1.0 + eps, 1.0 - eps))


@_safe_atanh.defjvp
def _safe_atanh_jvp(primals, tangents):
    (x,) = primals
    (dx,) = tangents
    y = _safe_atanh(x)
    inside = jnp.abs(x) < 1.0
    xs = jnp.where(inside, x, 0.0)
    # True derivative 1/(1 - x^2) in the interior, a finite 0 at/outside the
    # clamp boundary instead of Inf.
    dy = jnp.where(inside, 1.0 / (1.0 - xs * xs), 0.0) * dx
    return y, dy


@jax.custom_jvp
def _safe_arctan2(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(y, x)


@_safe_arctan2.defjvp
def _safe_arctan2_jvp(primals, tangents):
    y, x = primals
    dy, dx = tangents
    out = jnp.arctan2(y, x)
    denom = x * x + y * y
    pos = denom > 0.0
    # d atan2 = (x dy - y dx)/(x^2+y^2) in the interior, finite 0 at the origin
    # (where atan2 itself is defined but its gradient would be 0/0 = NaN).
    d = jnp.where(pos, (x * dy - y * dx) / jnp.where(pos, denom, 1.0), 0.0)
    return out, d


@jax.custom_jvp
def _safe_logabs(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(jnp.maximum(jnp.abs(x), 1e-30))


@_safe_logabs.defjvp
def _safe_logabs_jvp(primals, tangents):
    (x,) = primals
    (dx,) = tangents
    y = _safe_logabs(x)
    safe = jnp.abs(x) > 1e-30
    # d/dx log|x| = 1/x in the interior, a finite 0 at the singular argument.
    dy = jnp.where(safe, 1.0 / jnp.where(safe, x, 1.0), 0.0) * dx
    return y, dy
