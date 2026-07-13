"""Geometry optimization: size a magnet to hit a target field with jax.jit(jax.grad).

We optimize a *geometric* parameter -- the height of a cuboid magnet -- so that the axial field
``Bz`` at a fixed probe point matches a target value. The field depends on the dimension through the
analytic model, and magpylib_jax differentiates straight through it, so a compiled
``jax.jit(jax.grad(...))`` drives Newton-free gradient descent to the answer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import magpylib_jax as magpy

PROBE = jnp.array([0.0, 0.0, 0.03])  # 30 mm above the magnet center (m)
TARGET_BZ = 2.0e-2  # tesla
BASE_XY = (0.02, 0.02)  # fixed footprint (m)


def _bz(height: jnp.ndarray) -> jnp.ndarray:
    dim = jnp.stack([jnp.asarray(BASE_XY[0]), jnp.asarray(BASE_XY[1]), height])
    return magpy.magnet.Cuboid(dimension=dim, polarization=(0.0, 0.0, 1.2)).getB(PROBE)[2]


def main() -> dict:
    def loss(height: jnp.ndarray) -> jnp.ndarray:
        return (_bz(height) - TARGET_BZ) ** 2

    grad = jax.jit(jax.grad(loss))

    height = jnp.asarray(0.005)  # 5 mm initial guess
    for _ in range(200):
        height = jnp.clip(height - 5.0 * grad(height), 1e-3, 0.1)

    height = jax.block_until_ready(height)
    achieved = float(_bz(height))

    print(f"target Bz:       {TARGET_BZ:.4e} T")
    print(f"optimized height: {float(height) * 1e3:.3f} mm")
    print(f"achieved Bz:     {achieved:.4e} T")
    print(f"residual:        {abs(achieved - TARGET_BZ):.2e} T")

    return {"height": float(height), "achieved_bz": achieved, "target_bz": TARGET_BZ}


if __name__ == "__main__":
    main()
