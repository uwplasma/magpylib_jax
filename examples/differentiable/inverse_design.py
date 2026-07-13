"""Inverse design: recover a magnet's polarization from field samples with jax.grad.

This is the flagship magpylib_jax feature. Because ``getB`` is differentiable, fitting a source
parameter to measured field data is plain gradient descent -- no finite differences, no wrappers.
Here we generate synthetic B-field samples from a cuboid with an unknown polarization vector and
recover it from a wrong initial guess by minimizing the mean-squared field error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import magpylib_jax as magpy

# Fixed observer cloud around the magnet (m).
OBS = jnp.array([
    [0.2, 0.1, 0.4], [0.5, 0.0, 0.7], [-0.3, 0.2, 0.5],
    [0.1, -0.4, 0.6], [0.4, 0.3, 0.3], [-0.2, -0.1, 0.8],
])
DIMENSION = (1.0, 0.8, 1.2)
TRUE_POL = jnp.array([0.35, -0.20, 0.80])


def _field(pol: jnp.ndarray) -> jnp.ndarray:
    return magpy.magnet.Cuboid(dimension=DIMENSION, polarization=pol).getB(OBS)


def main() -> dict:
    target = _field(TRUE_POL)

    def loss(pol: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((_field(pol) - target) ** 2)

    grad = jax.jit(jax.grad(loss))
    value = jax.jit(loss)

    pol = jnp.array([0.05, 0.05, 0.05])  # deliberately wrong start
    history = [float(value(pol))]
    for _ in range(150):
        pol = pol - 2.0 * grad(pol)
        history.append(float(value(pol)))

    recovered = jax.block_until_ready(pol)
    err = float(jnp.linalg.norm(recovered - TRUE_POL))

    print(f"true polarization:      {TRUE_POL}")
    print(f"recovered polarization: {recovered}")
    print(f"loss: {history[0]:.3e} -> {history[-1]:.3e}  ({len(history)} steps)")
    print(f"parameter error:        {err:.3e} T")

    return {"true": TRUE_POL, "recovered": recovered, "loss_history": history, "error": err}


if __name__ == "__main__":
    main()
