"""Gradient hardening at the singular set of the divergent-field kernels.

For dipole, circle, cuboid and the triangle current sheet we assert that
``jax.grad`` / ``jax.jacfwd`` return only finite values when the observer sits
exactly on the source's singular set (where the closed-form field diverges),
and that away from the singularity the autodiff gradient still matches a central
finite-difference of the primal field.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import magpylib_jax as mpj


def _finite(g) -> bool:
    return bool(np.all(np.isfinite(np.asarray(g))))


def _central_fd(f, x0: float, h: float = 1e-6) -> float:
    return float((f(x0 + h) - f(x0 - h)) / (2.0 * h))


# --------------------------------------------------------------------------- #
# Dipole: singular set is r == 0 (the dipole's own position).
# --------------------------------------------------------------------------- #
def test_dipole_grad_finite_on_singularity() -> None:
    origin = jnp.array([0.0, 0.0, 0.0])

    def b_vec(m: jax.Array) -> jax.Array:
        return mpj.getB("dipole", origin, moment=m)

    jac = jax.jacrev(b_vec)(jnp.array([1.0, -0.2, 0.7]))
    assert _finite(jac)

    def b_obs(o: jax.Array) -> jax.Array:
        return mpj.getB("dipole", o, moment=jnp.array([0.0, 0.0, 1.0]))

    assert _finite(jax.jacfwd(b_obs)(origin))


def test_dipole_grad_correct_off_singularity() -> None:
    observer = jnp.array([0.3, -0.15, 0.42])

    def bz(mz: jax.Array) -> jax.Array:
        return mpj.getB("dipole", observer, moment=jnp.array([0.4, -0.1, mz]))[2]

    g = float(jax.grad(bz)(jnp.array(0.7)))
    fd = _central_fd(lambda mz: float(bz(jnp.array(mz))), 0.7)
    assert g == pytest.approx(fd, rel=1e-4)


# --------------------------------------------------------------------------- #
# Circle: singular set is the ring (r == radius, z == 0).
# --------------------------------------------------------------------------- #
def test_circle_grad_finite_on_ring() -> None:
    on_ring = jnp.array([1.0, 0.0, 0.0])  # radius = diameter/2 = 1.0

    def bz(current: jax.Array) -> jax.Array:
        return mpj.Circle(current=current, diameter=2.0).getB(on_ring)[2]

    assert _finite(jax.grad(bz)(jnp.array(3.0)))

    def b_obs(o: jax.Array) -> jax.Array:
        return mpj.Circle(current=3.0, diameter=2.0).getB(o)

    assert _finite(jax.jacfwd(b_obs)(on_ring))


def test_circle_grad_correct_off_ring() -> None:
    observer = jnp.array([0.35, 0.1, 0.55])

    def bz(current: jax.Array) -> jax.Array:
        return mpj.Circle(current=current, diameter=1.8).getB(observer)[2]

    g = float(jax.grad(bz)(jnp.array(2.4)))
    fd = _central_fd(lambda c: float(bz(jnp.array(c))), 2.4)
    assert g == pytest.approx(fd, rel=1e-4)


# --------------------------------------------------------------------------- #
# Cuboid: singular set is the edges/faces of the magnet.
# --------------------------------------------------------------------------- #
def test_cuboid_grad_finite_on_face_and_edge() -> None:
    pol = jnp.array([0.1, -0.2, 0.3])

    # Observer exactly on the +x face (a = dim_x / 2 = 0.5).
    on_face = jnp.array([0.5, 0.2, 0.3])

    def bz_dim(dim_x: jax.Array) -> jax.Array:
        src = mpj.magnet.Cuboid(polarization=pol, dimension=jnp.array([dim_x, 1.0, 1.0]))
        return src.getB(on_face)[2]

    assert _finite(jax.grad(bz_dim)(jnp.array(1.0)))

    def b_obs(o: jax.Array) -> jax.Array:
        src = mpj.magnet.Cuboid(polarization=pol, dimension=jnp.array([1.0, 1.0, 1.0]))
        return src.getB(o)

    # On a face and on an edge (corner of the +x,+y faces).
    assert _finite(jax.jacfwd(b_obs)(on_face))
    assert _finite(jax.jacfwd(b_obs)(jnp.array([0.5, 0.5, 0.2])))


def test_cuboid_grad_correct_off_singularity() -> None:
    observer = jnp.array([0.9, 0.4, 0.7])

    def bz(dim_x: jax.Array) -> jax.Array:
        src = mpj.magnet.Cuboid(
            polarization=jnp.array([0.1, -0.2, 0.3]),
            dimension=jnp.array([dim_x, 1.1, 0.9]),
        )
        return src.getB(observer)[2]

    g = float(jax.grad(bz)(jnp.array(1.2)))
    fd = _central_fd(lambda d: float(bz(jnp.array(d))), 1.2)
    assert g == pytest.approx(fd, rel=1e-4)


# --------------------------------------------------------------------------- #
# Triangle current sheet: singular set is the sheet plane / edges / vertices.
# --------------------------------------------------------------------------- #
_VERTS = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_FACES = jnp.array([[0, 1, 2]])
_CDS = jnp.array([[0.7, 0.1, 0.0]])


def test_current_sheet_grad_finite_on_sheet() -> None:
    on_sheet = jnp.array([0.3, 0.3, 0.0])  # in-plane, inside the triangle

    def bz(j0: jax.Array) -> jax.Array:
        src = mpj.current.TriangleSheet(
            vertices=_VERTS, faces=_FACES, current_densities=jnp.array([[j0, 0.1, 0.0]])
        )
        return src.getB(on_sheet)[2]

    assert _finite(jax.grad(bz)(jnp.array(0.7)))

    def b_obs(o: jax.Array) -> jax.Array:
        src = mpj.current.TriangleSheet(vertices=_VERTS, faces=_FACES, current_densities=_CDS)
        return src.getB(o)

    assert _finite(jax.jacfwd(b_obs)(on_sheet))
    # On a vertex and on an edge as well.
    assert _finite(jax.jacfwd(b_obs)(jnp.array([0.0, 0.0, 0.0])))
    assert _finite(jax.jacfwd(b_obs)(jnp.array([0.5, 0.0, 0.0])))


def test_current_sheet_grad_correct_off_sheet() -> None:
    observer = jnp.array([0.3, 0.25, 0.4])

    def bx(j0: jax.Array) -> jax.Array:
        src = mpj.current.TriangleSheet(
            vertices=_VERTS, faces=_FACES, current_densities=jnp.array([[j0, 0.1, 0.0]])
        )
        return src.getB(observer)[0]

    g = float(jax.grad(bx)(jnp.array(0.7)))
    fd = _central_fd(lambda j: float(bx(jnp.array(j))), 0.7)
    assert g == pytest.approx(fd, rel=1e-4)
