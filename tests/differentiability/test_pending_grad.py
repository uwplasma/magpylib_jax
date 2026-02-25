import jax
import jax.numpy as jnp
import numpy as np

import magpylib_jax as mpj


def test_cylinder_segment_grad_wrt_outer_radius_is_finite() -> None:
    observer = jnp.array([2.0, 0.5, 0.2])

    def bz(r2: jax.Array) -> jax.Array:
        src = mpj.magnet.CylinderSegment(
            polarization=jnp.array([0.1, -0.2, 0.3]),
            dimension=jnp.array([0.3, r2, 1.1, -30.0, 110.0]),
        )
        return src.getB(observer)[2]

    g = jax.grad(bz)(jnp.array(1.2))
    assert np.isfinite(np.asarray(g))


def test_trianglesheet_grad_wrt_current_density_is_finite() -> None:
    observer = jnp.array([0.4, 0.2, 0.9])
    vertices = jnp.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    faces = jnp.array([[0, 1, 2]])

    def bx(jx: jax.Array) -> jax.Array:
        src = mpj.current.TriangleSheet(
            vertices=vertices,
            faces=faces,
            current_densities=jnp.array([[jx, 0.1, 0.0]]),
        )
        return src.getB(observer)[0]

    g = jax.grad(bx)(jnp.array(0.7))
    assert np.isfinite(np.asarray(g))


def test_trianglestrip_grad_wrt_current_is_finite() -> None:
    observer = jnp.array([0.6, 0.2, 0.8])
    vertices = jnp.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=float)

    def by(current: jax.Array) -> jax.Array:
        src = mpj.current.TriangleStrip(vertices=vertices, current=current)
        return src.getB(observer)[1]

    g = jax.grad(by)(jnp.array(1.2))
    assert np.isfinite(np.asarray(g))


def test_triangular_mesh_grad_wrt_polarization_is_finite() -> None:
    observer = jnp.array([1.2, 0.8, 0.7])
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = jnp.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])

    def bx(polx: jax.Array) -> jax.Array:
        src = mpj.magnet.TriangularMesh(
            vertices=vertices,
            faces=faces,
            polarization=jnp.array([polx, -0.2, 0.3]),
            reorient_faces=False,
        )
        return src.getB(observer)[0]

    g = jax.grad(bx)(jnp.array(0.1))
    assert np.isfinite(np.asarray(g))
