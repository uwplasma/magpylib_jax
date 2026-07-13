"""Numerically safe math helpers for the current-sheet kernels."""

from __future__ import annotations

import jax.numpy as jnp


def _safe_sqrt(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(x, 0.0))


def _safe_atanh(x: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-15
    return jnp.arctanh(jnp.clip(x, -1.0 + eps, 1.0 - eps))


def _safe_logabs(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(jnp.maximum(jnp.abs(x), 1e-30))
