import jax
import jax.numpy as jnp
import numpy as np

import magpylib_jax as mpj


def test_circle_grad_wrt_current_is_finite() -> None:
    observer = jnp.array([0.1, 0.2, 0.35])

    def bz(current: jax.Array) -> jax.Array:
        src = mpj.Circle(current=current, diameter=0.9)
        return src.getB(observer)[2]

    g = jax.grad(bz)(jnp.array(2.0))
    assert np.isfinite(np.asarray(g))


def test_circle_grad_wrt_diameter_is_finite() -> None:
    observer = jnp.array([0.15, -0.05, 0.6])

    def bz(diameter: jax.Array) -> jax.Array:
        src = mpj.Circle(current=2.1, diameter=diameter)
        return src.getB(observer)[2]

    g = jax.grad(bz)(jnp.array(0.8))
    assert np.isfinite(np.asarray(g))
