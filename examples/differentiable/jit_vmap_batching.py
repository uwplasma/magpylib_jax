"""Batching fields with jax.vmap and jax.jit.

The field core is ``vmap``-friendly and ``jit``-compilable, so you can evaluate one field function
over a whole batch of source parameters or observers without Python loops. This script shows the
two batching axes and the recommended timing pattern: compile once, then measure compute with
``block_until_ready`` (never time the first, tracing call).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

import magpylib_jax as magpy

OBS = jnp.array([[0.2, 0.1, 0.4], [0.5, 0.0, 0.7], [-0.3, 0.2, 0.5]])


def _field_for_polarization(pol: jnp.ndarray) -> jnp.ndarray:
    return magpy.magnet.Cuboid(dimension=(1.0, 0.8, 1.2), polarization=pol).getB(OBS)


def main() -> dict:
    # Batch over 256 source polarizations, each evaluated at all observers.
    key = jax.random.PRNGKey(0)
    pols = jax.random.normal(key, (256, 3)) * 0.5
    batched = jax.jit(jax.vmap(_field_for_polarization))

    # Compile once (do not time this call).
    out = jax.block_until_ready(batched(pols))

    # Time the compiled, batched evaluation.
    t0 = time.perf_counter()
    for _ in range(20):
        out = jax.block_until_ready(batched(pols))
    per_call_ms = (time.perf_counter() - t0) / 20 * 1e3

    # Sanity: batched result matches an eager single evaluation.
    single = _field_for_polarization(pols[0])
    match = float(jnp.max(jnp.abs(out[0] - single)))

    print(f"batched output shape: {out.shape}  (256 sources x 3 observers x 3 comps)")
    print(f"compiled batched call: {per_call_ms:.3f} ms for 256 sources")
    print(f"batched-vs-eager max diff: {match:.2e}")

    return {"shape": tuple(out.shape), "per_call_ms": per_call_ms, "match": match}


if __name__ == "__main__":
    main()
