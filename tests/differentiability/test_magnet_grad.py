import jax
import jax.numpy as jnp
import numpy as np

import magpylib_jax as mpj


def test_cuboid_grad_wrt_dimension_is_finite() -> None:
    observer = jnp.array([0.4, 0.3, 0.7])

    def bz(dim_x: jax.Array) -> jax.Array:
        src = mpj.magnet.Cuboid(
            polarization=jnp.array([0.1, -0.2, 0.3]),
            dimension=jnp.array([dim_x, 1.1, 0.9]),
        )
        return src.getB(observer)[2]

    grad = jax.grad(bz)(jnp.array(1.3))
    assert np.isfinite(np.asarray(grad))


def test_cylinder_grad_wrt_polarization_is_finite() -> None:
    observer = jnp.array([0.25, -0.15, 0.4])

    def bx(polx: jax.Array) -> jax.Array:
        src = mpj.magnet.Cylinder(
            polarization=jnp.array([polx, 0.2, 0.1]),
            dimension=jnp.array([1.2, 1.5]),
        )
        return src.getB(observer)[0]

    grad = jax.grad(bx)(jnp.array(0.12))
    assert np.isfinite(np.asarray(grad))
