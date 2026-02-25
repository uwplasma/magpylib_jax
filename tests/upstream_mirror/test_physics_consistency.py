import numpy as np

import magpylib_jax as mpj
from magpylib_jax.constants import MU0


def test_dipole_approximation() -> None:
    pol = np.array([0.111, 0.222, 0.333])
    pos = (1234, -234, 345)

    src1 = mpj.magnet.Cuboid(polarization=pol, dimension=(1, 1, 1))
    b1 = np.asarray(src1.getB(pos))

    dia = np.sqrt(4 / np.pi)
    src2 = mpj.magnet.Cylinder(polarization=pol, dimension=(dia, 1))
    b2 = np.asarray(src2.getB(pos))
    np.testing.assert_allclose(b1, b2, rtol=1e-5, atol=1e-8)

    dia = (6 / np.pi) ** (1 / 3)
    src3 = mpj.magnet.Sphere(polarization=pol, diameter=dia)
    b3 = np.asarray(src3.getB(pos))
    np.testing.assert_allclose(b1, b3, rtol=1e-5, atol=1e-8)

    src4 = mpj.misc.Dipole(moment=pol)
    b4 = np.asarray(src4.getB(pos))
    np.testing.assert_allclose(b1, b4, rtol=1e-5, atol=1e-8)

    dia = 2
    i0 = 234
    m0 = dia**2 * np.pi**2 / 10 * i0
    src1 = mpj.current.Circle(current=i0, diameter=dia)
    src2 = mpj.misc.Dipole(moment=(0, 0, m0))
    h1 = np.asarray(src1.getH(pos))
    h2 = np.asarray(src2.getH(pos))
    np.testing.assert_allclose(h1, h2, rtol=1e-5, atol=1e-8)


def test_circle_vs_cylinder_field() -> None:
    rng = np.random.default_rng(0)
    pos_obs = rng.uniform(0.0, 1.0, size=(25, 3))
    pos_obs[:, 2] += 10.0

    r0 = 2
    h0 = 1e-4
    i0 = 1
    pol = (0, 0, i0 / h0 * 4 * np.pi / 10 * 1e-6)
    src1 = mpj.magnet.Cylinder(polarization=pol, dimension=(r0, h0))
    src2 = mpj.current.Circle(current=i0, diameter=r0)

    h1 = np.asarray(src1.getH(pos_obs))
    h2 = np.asarray(src2.getH(pos_obs))

    np.testing.assert_allclose(h1, h2, rtol=1e-5, atol=1e-8)


def test_polyline_vs_circle() -> None:
    ts = np.linspace(0, 2 * np.pi, 2000)
    verts = np.array([(np.cos(t), np.sin(t), 0.0) for t in ts])

    src_line = mpj.current.Polyline(current=1, vertices=verts)
    src_loop = mpj.current.Circle(current=1, diameter=2)

    pos = np.array([(x, y, z) for x in (-3, 0, 3) for y in (-3, 0, 3) for z in (-3, 0, 3)])

    b_line = np.asarray(src_line.getB(pos))
    b_loop = np.asarray(src_loop.getB(pos))

    np.testing.assert_allclose(b_line, b_loop, rtol=1e-5, atol=1e-8)


def test_polyline_vs_infinite_line() -> None:
    pos_obs = np.array([(1.0, 2, 3), (-3, 2, -1), (2, -1, -4)])

    def binf(i0, pos):
        x, y, _ = pos
        r = np.sqrt(x**2 + y**2)
        e_phi = np.array([-y, x, 0.0])
        e_phi = e_phi / np.linalg.norm(e_phi)
        return i0 * MU0 / 2 / np.pi / r * e_phi

    ps = (0, 0, -1_000_000)
    pe = (0, 0, 1_000_000)
    src = mpj.current.Polyline(current=1, vertices=[ps, pe])

    b_line = np.asarray(src.getB(pos_obs))
    b_inf = np.array([binf(1, p) for p in pos_obs])

    np.testing.assert_allclose(b_line, b_inf, rtol=1e-5, atol=1e-8)
