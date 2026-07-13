"""Field Jacobians: the gradient tensor dB/dr and the sensitivity dB/d(polarization).

``getB`` is fully differentiable, so its Jacobians come straight from ``jax.jacrev``/``jacfwd``:

1. ``dB_i/dx_j`` -- the 3x3 field-gradient tensor at an observer point (used e.g. in force
   computation and gradiometer design). We check ``jacfwd == jacrev`` and that the tensor is
   trace-free in current-free space (``div B = 0``).
2. ``dB_i/dJ_j`` -- the 3x3 sensitivity of the field to the source polarization (used in inverse
   design and tolerancing).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import magpylib_jax as magpy

DIMENSION = (0.02, 0.02, 0.02)
POLARIZATION = jnp.array([0.0, 0.0, 1.2])
OBSERVER = jnp.array([0.03, 0.02, 0.04])


def _b_of_position(r: jnp.ndarray) -> jnp.ndarray:
    return magpy.magnet.Cuboid(dimension=DIMENSION, polarization=POLARIZATION).getB(r)


def _b_of_polarization(pol: jnp.ndarray) -> jnp.ndarray:
    return magpy.magnet.Cuboid(dimension=DIMENSION, polarization=pol).getB(OBSERVER)


def main() -> dict:
    # 1. Field-gradient tensor dB/dr at the observer.
    grad_tensor_rev = jax.jacrev(_b_of_position)(OBSERVER)
    grad_tensor_fwd = jax.jacfwd(_b_of_position)(OBSERVER)
    mode_diff = float(jnp.max(jnp.abs(grad_tensor_rev - grad_tensor_fwd)))
    divergence = float(jnp.trace(grad_tensor_rev))  # div B = 0 in current-free region

    # 2. Sensitivity dB/d(polarization) at the observer.
    sensitivity = jax.jacfwd(_b_of_polarization)(POLARIZATION)

    print("dB/dr (field-gradient tensor, 1/m):")
    for row in jnp.asarray(grad_tensor_rev):
        print("   " + "  ".join(f"{v: .3e}" for v in row))
    print(f"jacrev vs jacfwd max diff: {mode_diff:.2e}")
    print(f"trace(dB/dr) = div B:      {divergence:.2e}  (expect ~0)")
    print("dB/d(polarization) (sensitivity, dimensionless):")
    for row in jnp.asarray(sensitivity):
        print("   " + "  ".join(f"{v: .3e}" for v in row))

    return {
        "grad_tensor": grad_tensor_rev,
        "divergence": divergence,
        "mode_diff": mode_diff,
        "sensitivity": sensitivity,
    }


if __name__ == "__main__":
    main()
