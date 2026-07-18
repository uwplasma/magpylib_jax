"""Polyline current-segment field kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import ensure_observers
from magpylib_jax.core.kernels._common import (
    _FOUR_PI,
    _broadcast_vec3,
    _jit_kernel_segments,
    _safe_norm,
)


def _current_segment_hfield(
    observers: jnp.ndarray,
    segment_start: jnp.ndarray,
    segment_end: jnp.ndarray,
    current: jnp.ndarray,
) -> jnp.ndarray:
    """H-field for a single current segment."""
    obs = ensure_observers(observers)
    p1 = _broadcast_vec3(segment_start, obs.shape[0])
    p2 = _broadcast_vec3(segment_end, obs.shape[0])

    cur = jnp.asarray(current, dtype=float)
    if cur.ndim == 0:
        cur = jnp.broadcast_to(cur, (obs.shape[0],))
    else:
        cur = jnp.broadcast_to(cur.reshape((-1,)), (obs.shape[0],))

    seg = p1 - p2
    norm12 = _safe_norm(seg, axis=1)
    valid_seg = norm12 > 1e-15

    p1s = p1 / norm12[:, None]
    p2s = p2 / norm12[:, None]
    pos = obs / norm12[:, None]

    t = jnp.sum((pos - p1s) * (p1s - p2s), axis=1)
    p4 = p1s + t[:, None] * (p1s - p2s)

    o4 = pos - p4
    norm_o4 = _safe_norm(o4, axis=1)
    off_line = norm_o4 >= 1e-15

    cros = jnp.cross(p2s - p1s, o4)
    norm_cros = _safe_norm(cros, axis=1)
    eB = cros / norm_cros[:, None]

    norm_o1 = _safe_norm(pos - p1s, axis=1)
    norm_o2 = _safe_norm(pos - p2s, axis=1)
    norm_41 = _safe_norm(p4 - p1s, axis=1)
    norm_42 = _safe_norm(p4 - p2s, axis=1)
    sin1 = norm_41 / norm_o1
    sin2 = norm_42 / norm_o2

    mask2 = (norm_41 > 1.0) & (norm_41 > norm_42)
    mask3 = (norm_42 > 1.0) & (norm_42 > norm_41)
    delta = jnp.where(mask2, jnp.abs(sin1 - sin2), jnp.abs(sin1 + sin2))
    delta = jnp.where(mask3, jnp.abs(sin2 - sin1), delta)

    h = (delta / norm_o4)[:, None] * eB / norm12[:, None] * cur[:, None] / _FOUR_PI
    valid = (
        valid_seg & off_line & jnp.all(jnp.isfinite(p1), axis=1) & jnp.all(jnp.isfinite(p2), axis=1)
    )
    return jnp.where(valid[:, None], h, 0.0)


def current_polyline_hfield(
    observers: ArrayLike,
    segments_start: ArrayLike,
    segments_end: ArrayLike,
    currents: ArrayLike,
) -> jnp.ndarray:
    """H-field of straight current segments."""
    obs = ensure_observers(observers)
    p1 = jnp.asarray(segments_start, dtype=float)
    p2 = jnp.asarray(segments_end, dtype=float)
    if p1.ndim == 1:
        return _current_segment_hfield(obs, p1, p2, currents)
    if p2.shape != p1.shape or p1.shape[-1] != 3:
        raise ValueError("Polyline segments must have shape (n,3).")

    cur = jnp.asarray(currents, dtype=float)
    if cur.ndim == 0:
        cur = jnp.broadcast_to(cur, (p1.shape[0],))
    else:
        cur = jnp.broadcast_to(cur.reshape((-1,)), (p1.shape[0],))

    h_segments = jax.vmap(lambda a, b, c: _current_segment_hfield(obs, a, b, c))(p1, p2, cur)
    return jnp.sum(h_segments, axis=0)


def current_polyline_bfield(
    observers: ArrayLike,
    segments_start: ArrayLike,
    segments_end: ArrayLike,
    currents: ArrayLike,
) -> jnp.ndarray:
    return MU0 * current_polyline_hfield(observers, segments_start, segments_end, currents)


def current_polyline_bfield_masked(
    observers: ArrayLike,
    segments_start: ArrayLike,
    segments_end: ArrayLike,
    currents: ArrayLike,
    segment_mask: ArrayLike,
) -> jnp.ndarray:
    """B-field of current segments with segment masking."""
    obs = ensure_observers(observers)
    p1 = jnp.asarray(segments_start, dtype=float)
    p2 = jnp.asarray(segments_end, dtype=float)
    cur = jnp.asarray(currents, dtype=float)
    if cur.ndim == 0:
        cur = jnp.broadcast_to(cur, (p1.shape[0],))
    else:
        cur = jnp.broadcast_to(cur.reshape((-1,)), (p1.shape[0],))

    mask = jnp.asarray(segment_mask, dtype=float).reshape((-1,))
    h_segments = jax.vmap(lambda a, b, c: _current_segment_hfield(obs, a, b, c))(p1, p2, cur)
    h_segments = h_segments * mask[:, None, None]
    return MU0 * jnp.sum(h_segments, axis=0)


def _current_polyline_bfield_segments_impl(
    observers: jnp.ndarray,
    segments_start: jnp.ndarray,
    segments_end: jnp.ndarray,
    currents: jnp.ndarray,
    *,
    n_segments: int,
) -> jnp.ndarray:
    return current_polyline_bfield(observers, segments_start, segments_end, currents)


def current_polyline_bfield_jit(
    observers: ArrayLike,
    segments_start: ArrayLike,
    segments_end: ArrayLike,
    currents: ArrayLike,
) -> jnp.ndarray:
    """JIT-specialized polyline B-field for fixed observer + segment counts."""
    obs = ensure_observers(observers)
    seg_start = jnp.asarray(segments_start, dtype=float)
    seg_end = jnp.asarray(segments_end, dtype=float)
    if seg_start.ndim == 1:
        n_segments = 1
    else:
        n_segments = int(seg_start.shape[0])
    jit_fn = _jit_kernel_segments(
        "polyline_bfield", _current_polyline_bfield_segments_impl, obs.shape[0], n_segments
    )
    return jit_fn(
        obs, seg_start, seg_end, jnp.asarray(currents, dtype=float), n_segments=n_segments
    )
