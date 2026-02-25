import jax
import jax.numpy as jnp
import numpy as np

import magpylib_jax as mpj


def test_sphere_grad_wrt_diameter_is_finite() -> None:
    observer = jnp.array([0.2, 0.1, 0.7])

    def bz(diameter: jax.Array) -> jax.Array:
        src = mpj.magnet.Sphere(polarization=jnp.array([0.1, 0.0, 0.3]), diameter=diameter)
        return src.getB(observer)[2]

    g = jax.grad(bz)(jnp.array(1.3))
    assert np.isfinite(np.asarray(g))


def test_triangle_grad_wrt_polarization_is_finite() -> None:
    observer = jnp.array([0.3, 0.4, 0.8])
    vertices = jnp.array([[-0.2, -0.1, 0.0], [0.9, 0.3, 0.2], [0.1, 0.8, -0.2]])

    def bx(polx: jax.Array) -> jax.Array:
        src = mpj.misc.Triangle(vertices=vertices, polarization=jnp.array([polx, -0.1, 0.2]))
        return src.getB(observer)[0]

    g = jax.grad(bx)(jnp.array(0.3))
    assert np.isfinite(np.asarray(g))
