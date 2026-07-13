"""Tests for the JAX-native autodiff force/torque ``getFT``.

Covers:
1. Exact dipole-dipole analytic parity (rel err < 1e-8).
2. Parity vs ``magpylib.getFT`` for dipole / sphere / single-cell-cuboid /
   circle / polyline targets.
3. Differentiability (``jax.grad`` of ``|F|^2`` is finite) w.r.t. a source
   moment and a target position.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import magpylib_jax as mj

MU0 = 1.25663706127e-6

# Fixed configuration from the implementation spec.
P1 = np.array([-1.248, 7.835, 9.273])
M1 = np.array([976.0, 4304.0, 2055.0])
P2 = np.array([-2.331, 5.835, 0.578])
M2 = np.array([878.0, -1527.0, 2918.0])
PIVOT = np.array([0.727, 5.152, 5.363])


def _dipole_B(moment, r_vec):
    """Field of a dipole ``moment`` at displacement ``r_vec`` (tesla)."""
    rn = np.linalg.norm(r_vec)
    rhat = r_vec / rn
    return MU0 / (4 * np.pi) * (3 * rhat * np.dot(rhat, moment) - moment) / rn**3


def _analytic_dipole_FT():
    r = P2 - P1
    rn = np.linalg.norm(r)
    rhat = r / rn
    F = (
        3
        * MU0
        / (4 * np.pi * rn**4)
        * (
            np.cross(np.cross(rhat, M1), M2)
            + np.cross(np.cross(rhat, M2), M1)
            - 2 * rhat * np.dot(M1, M2)
            + 5 * rhat * np.dot(np.cross(rhat, M1), np.cross(rhat, M2))
        )
    )
    B = _dipole_B(M1, P2 - P1)
    T = np.cross(M2, B) + np.cross(P2 - PIVOT, F)
    return F, T


def test_dipole_dipole_analytic():
    """Exact parity with the closed-form dipole-dipole force and torque."""
    F_an, T_an = _analytic_dipole_FT()

    src = mj.Dipole(moment=M1, position=P1)
    tgt = mj.Dipole(moment=M2, position=P2)
    F, T = mj.getFT(src, tgt, pivot=PIVOT)

    F = np.asarray(F)
    T = np.asarray(T)

    assert np.linalg.norm(F - F_an) / np.linalg.norm(F_an) < 1e-8
    assert np.linalg.norm(T - T_an) / np.linalg.norm(T_an) < 1e-8


# --- magpylib parity -------------------------------------------------------
magpy = pytest.importorskip("magpylib")


def _rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(b)
    if denom == 0:
        return np.linalg.norm(a)
    return np.linalg.norm(a - b) / denom


def _source_pair():
    src = mj.magnet.Cuboid(
        polarization=(0.2, -0.1, 0.3), dimension=(1.0, 1.0, 1.0), position=(0, 0, 0)
    )
    msrc = magpy.magnet.Cuboid(
        polarization=(0.2, -0.1, 0.3), dimension=(1.0, 1.0, 1.0), position=(0, 0, 0)
    )
    return src, msrc


def test_parity_dipole_target():
    src, msrc = _source_pair()
    tgt = mj.Dipole(moment=(500.0, -200.0, 300.0), position=(2.0, 1.0, 0.5))
    mtgt = magpy.misc.Dipole(moment=(500.0, -200.0, 300.0), position=(2.0, 1.0, 0.5))
    F, T = mj.getFT(src, tgt)
    Fm, Tm = magpy.getFT(msrc, mtgt)
    assert _rel(F, Fm) < 1e-6
    assert _rel(T, Tm) < 1e-6


def test_parity_sphere_target():
    src, msrc = _source_pair()
    tgt = mj.magnet.Sphere(
        diameter=1.0, polarization=(0.3, 0.1, -0.2), position=(2, 1, 0.5)
    )
    mtgt = magpy.magnet.Sphere(
        diameter=1.0, polarization=(0.3, 0.1, -0.2), position=(2, 1, 0.5)
    )
    F, T = mj.getFT(src, tgt)
    Fm, Tm = magpy.getFT(msrc, mtgt)
    # magnet gradient: autodiff (exact) vs magpylib finite differences.
    assert _rel(F, Fm) < 1e-4
    assert _rel(T, Tm) < 1e-4


def test_parity_single_cell_cuboid_target():
    src, msrc = _source_pair()
    tgt = mj.magnet.Cuboid(
        dimension=(0.5, 0.5, 0.5), polarization=(0.1, 0.2, -0.1), position=(1.5, 0.5, 1.0)
    )
    tgt.meshing = 1
    mtgt = magpy.magnet.Cuboid(
        dimension=(0.5, 0.5, 0.5),
        polarization=(0.1, 0.2, -0.1),
        position=(1.5, 0.5, 1.0),
        meshing=1,
    )
    F, T = mj.getFT(src, tgt)
    Fm, Tm = magpy.getFT(msrc, mtgt)
    assert _rel(F, Fm) < 1e-4
    assert _rel(T, Tm) < 1e-4


def test_parity_circle_target():
    src, msrc = _source_pair()
    tgt = mj.current.Circle(diameter=1.0, current=100.0, position=(1, 0, 1))
    tgt.meshing = 80
    mtgt = magpy.current.Circle(
        diameter=1.0, current=100.0, position=(1, 0, 1), meshing=80
    )
    F, T = mj.getFT(src, tgt)
    Fm, Tm = magpy.getFT(msrc, mtgt)
    # Identical polygon meshing -> near-exact agreement.
    assert _rel(F, Fm) < 1e-6
    assert _rel(T, Tm) < 1e-6


def test_parity_polyline_target():
    src, msrc = _source_pair()
    verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    tgt = mj.current.Polyline(current=50.0, vertices=verts, position=(0.5, 0.5, 1.0))
    tgt.meshing = 3
    mtgt = magpy.current.Polyline(
        current=50.0, vertices=verts, position=(0.5, 0.5, 1.0), meshing=3
    )
    F, T = mj.getFT(src, tgt)
    Fm, Tm = magpy.getFT(msrc, mtgt)
    assert _rel(F, Fm) < 1e-6
    assert _rel(T, Tm) < 1e-6


# --- differentiability ------------------------------------------------------
def test_grad_wrt_source_moment_finite():
    tgt = mj.Dipole(moment=M2, position=P2)

    def loss(moment):
        src = mj.Dipole(moment=moment, position=P1)
        F, _ = mj.getFT(src, tgt, pivot=PIVOT)
        return jnp.sum(F**2)

    g = jax.grad(loss)(jnp.asarray(M1))
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.linalg.norm(g) > 0


def test_grad_wrt_target_position_finite():
    src = mj.Dipole(moment=M1, position=P1)

    def loss(pos):
        tgt = mj.Dipole(moment=M2, position=pos)
        F, _ = mj.getFT(src, tgt, pivot=None)
        return jnp.sum(F**2)

    g = jax.grad(loss)(jnp.asarray(P2))
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.linalg.norm(g) > 0


def test_unsupported_target_raises():
    src = mj.Dipole(moment=M1, position=P1)
    cyl = mj.magnet.Cylinder(polarization=(0.1, 0.0, 0.2), dimension=(1.0, 1.0))
    cyl.meshing = 1
    with pytest.raises(NotImplementedError):
        mj.getFT(src, cyl)


# --- extra target/pivot/meshing paths --------------------------------------
def test_parity_multicell_cuboid_target():
    """Multi-cell cuboid meshing (integer target elements) parity vs magpylib."""
    src, msrc = _source_pair()
    tgt = mj.magnet.Cuboid(
        dimension=(0.6, 0.5, 0.4), polarization=(0.1, 0.2, -0.1), position=(1.5, 0.5, 1.0)
    )
    tgt.meshing = 8
    mtgt = magpy.magnet.Cuboid(
        dimension=(0.6, 0.5, 0.4),
        polarization=(0.1, 0.2, -0.1),
        position=(1.5, 0.5, 1.0),
        meshing=8,
    )
    F, T = mj.getFT(src, tgt)
    Fm, Tm = magpy.getFT(msrc, mtgt)
    assert _rel(F, Fm) < 1e-4
    assert _rel(T, Tm) < 1e-4


def test_cuboid_tuple_meshing_runs():
    src, _ = _source_pair()
    tgt = mj.magnet.Cuboid(
        dimension=(0.6, 0.5, 0.4), polarization=(0.1, 0.2, -0.1), position=(1.5, 0.5, 1.0)
    )
    tgt.meshing = (2, 2, 2)
    F, T = mj.getFT(src, tgt)
    assert np.asarray(F).shape == (3,)
    assert np.all(np.isfinite(np.asarray(F)))


def test_collection_target_sums_members():
    """A Collection target flattens to its members and sums their F/T."""
    src = mj.Dipole(moment=M1, position=P1)
    t1 = mj.Dipole(moment=(300.0, -100.0, 50.0), position=(2.0, 1.0, 0.5))
    t2 = mj.Dipole(moment=(-120.0, 200.0, 80.0), position=(1.0, -0.5, 1.2))
    col = mj.Collection(t1, t2)

    Fc, Tc = mj.getFT(src, col)
    F1, T1 = mj.getFT(src, t1)
    F2, T2 = mj.getFT(src, t2)
    assert np.allclose(np.asarray(Fc), np.asarray(F1) + np.asarray(F2), atol=1e-12)
    assert np.allclose(np.asarray(Tc), np.asarray(T1) + np.asarray(T2), atol=1e-12)


def test_pivot_none_and_array_forms():
    src = mj.Dipole(moment=M1, position=P1)
    t1 = mj.Dipole(moment=M2, position=P2)
    t2 = mj.Dipole(moment=(100.0, 50.0, -30.0), position=(1.0, -0.5, 1.2))

    # pivot=None omits the (x - pivot) x F torque term.
    _, T_none = mj.getFT(src, t1, pivot=None)
    assert np.asarray(T_none).shape == (3,)

    # Single (3,) pivot broadcast to all targets.
    F_s, T_s = mj.getFT(src, [t1, t2], pivot=np.array([0.5, 0.5, 0.5]))
    assert np.asarray(F_s).shape == (2, 3)

    # Per-target (t,3) pivot.
    piv = np.array([[0.5, 0.5, 0.5], [0.1, 0.2, 0.3]])
    F_t, T_t = mj.getFT(src, [t1, t2], pivot=piv)
    assert np.asarray(T_t).shape == (2, 3)

    with pytest.raises(ValueError):
        mj.getFT(src, [t1, t2], pivot=np.zeros((5, 2)))
    with pytest.raises(ValueError):
        mj.getFT(src, t1, pivot="middle")


def test_circle_meshing_too_small_raises():
    src = mj.Dipole(moment=M1, position=P1)
    tgt = mj.current.Circle(diameter=1.0, current=10.0, position=(1, 0, 1))
    tgt.meshing = 2
    with pytest.raises(ValueError):
        mj.getFT(src, tgt)


def test_multiple_sources_and_squeeze_false():
    s1 = mj.Dipole(moment=M1, position=P1)
    s2 = mj.Dipole(moment=(200.0, 0.0, 100.0), position=(0.0, 0.0, 3.0))
    tgt = mj.Dipole(moment=M2, position=P2)

    F, T = mj.getFT([s1, s2], tgt, squeeze=False)
    # Unsqueezed shape is (n_src, n_path, n_tgt, 3).
    assert np.asarray(F).shape == (2, 1, 1, 3)
    assert np.asarray(T).shape == (2, 1, 1, 3)

    # Summing the per-source forces matches a single combined evaluation.
    F_each, _ = mj.getFT([s1, s2], tgt)
    F_s1, _ = mj.getFT(s1, tgt)
    F_s2, _ = mj.getFT(s2, tgt)
    assert np.allclose(np.asarray(F_each), np.stack([F_s1, F_s2]), atol=1e-12)


def test_source_path_padding():
    """A source with a longer position path pads shorter target paths."""
    src = mj.Dipole(moment=M1, position=np.array([P1, P1 + 1.0, P1 + 2.0]))
    tgt = mj.Dipole(moment=M2, position=P2)
    F, T = mj.getFT(src, tgt)
    assert np.asarray(F).shape == (3, 3)  # (n_path, 3)
