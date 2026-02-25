import jax
import jax.numpy as jnp
import numpy as np

import magpylib_jax as mpj


def test_dipole_grad_observer_coordinate_is_finite() -> None:
    src = mpj.Dipole(moment=jnp.array([0.0, 0.0, 1.0]))

    def bz(z: jax.Array) -> jax.Array:
        return src.getB(jnp.array([0.2, -0.1, z]))[2]

    grad_fn = jax.grad(bz)
    g = grad_fn(jnp.array(0.7))

    assert np.isfinite(np.asarray(g))


def test_dipole_jacobian_wrt_moment_is_finite() -> None:
    observer = jnp.array([0.3, 0.4, 0.5])

    def b_vec(m: jax.Array) -> jax.Array:
        return mpj.getB("dipole", observer, moment=m)

    jac = jax.jacrev(b_vec)(jnp.array([1.0, -0.2, 0.7]))
    assert np.all(np.isfinite(np.asarray(jac)))
