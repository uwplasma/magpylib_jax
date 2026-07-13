"""Shared constants and helpers for the kernel package."""

from __future__ import annotations

import jax
import jax.numpy as jnp

_FOUR_PI = 4.0 * jnp.pi

_IN_OUT_FLAGS = {"auto": 0, "inside": 1, "outside": 2}

_JIT_KERNEL_CACHE: dict[tuple[str, int, int], object] = {}
_JIT_SIMPLE_CACHE: dict[tuple[str, int], object] = {}
_JIT_MESH_CACHE: dict[tuple[str, int, int, int], object] = {}
_JIT_SEGMENT_CACHE: dict[tuple[str, int, int], object] = {}


def _broadcast_vector(vector: jnp.ndarray, target_shape: tuple[int, ...]) -> jnp.ndarray:
    if vector.ndim == 1:
        return jnp.broadcast_to(vector[None, :], target_shape)
    return jnp.broadcast_to(vector, target_shape)


def _broadcast_vec3(arr: jnp.ndarray, n: int) -> jnp.ndarray:
    return _broadcast_vector(arr, (n, 3))


@jax.custom_jvp
def _norm_sqrt(sumsq: jnp.ndarray) -> jnp.ndarray:
    """``sqrt(max(sumsq, 1e-30))`` with a singularity-safe tangent at sumsq==0."""
    return jnp.sqrt(jnp.maximum(sumsq, 1e-30))


@_norm_sqrt.defjvp
def _norm_sqrt_jvp(primals, tangents):
    (sumsq,) = primals
    (dsumsq,) = tangents
    y = _norm_sqrt(sumsq)
    pos = sumsq > 0.0
    dy = jnp.where(pos, 0.5 / jnp.sqrt(jnp.where(pos, sumsq, 1.0)), 0.0) * dsumsq
    return y, dy


def _safe_norm(v: jnp.ndarray, axis: int = -1, keepdims: bool = False) -> jnp.ndarray:
    # Primal is bit-identical to ``sqrt(max(sum(v*v), 1e-30))``; the custom_jvp
    # on ``_norm_sqrt`` yields tangent ``(v.dv)/n`` for n>0 and 0 at v==0 instead
    # of an Inf, so gradients stay finite on each source's singular set.
    return _norm_sqrt(jnp.sum(v * v, axis=axis, keepdims=keepdims))


def _in_out_flag(in_out: str) -> int:
    if in_out not in _IN_OUT_FLAGS:
        raise ValueError(f"in_out must be one of {sorted(_IN_OUT_FLAGS)}, got {in_out!r}.")
    return _IN_OUT_FLAGS[in_out]


def _jit_kernel(name: str, fn, n_obs: int, in_out_flag: int):
    key = (name, int(n_obs), int(in_out_flag))
    if key not in _JIT_KERNEL_CACHE:
        _JIT_KERNEL_CACHE[key] = jax.jit(fn, static_argnames=("in_out_flag",))
    return _JIT_KERNEL_CACHE[key]


def _jit_kernel_simple(name: str, fn, n_obs: int):
    key = (name, int(n_obs))
    if key not in _JIT_SIMPLE_CACHE:
        _JIT_SIMPLE_CACHE[key] = jax.jit(fn)
    return _JIT_SIMPLE_CACHE[key]


def _jit_kernel_mesh(name: str, fn, n_obs: int, n_faces: int, in_out_flag: int):
    key = (name, int(n_obs), int(n_faces), int(in_out_flag))
    if key not in _JIT_MESH_CACHE:
        _JIT_MESH_CACHE[key] = jax.jit(fn, static_argnames=("in_out_flag", "n_faces"))
    return _JIT_MESH_CACHE[key]


def _jit_kernel_segments(name: str, fn, n_obs: int, n_segments: int):
    key = (name, int(n_obs), int(n_segments))
    if key not in _JIT_SEGMENT_CACHE:
        _JIT_SEGMENT_CACHE[key] = jax.jit(fn, static_argnames=("n_segments",))
    return _JIT_SEGMENT_CACHE[key]
