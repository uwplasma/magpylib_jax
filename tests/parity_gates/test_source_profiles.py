import magpylib as magpy
import numpy as np
import pytest

import magpylib_jax as mpj


@pytest.mark.parity_gate
@pytest.mark.parametrize(
    ("name", "src_ref", "src_new", "points", "rtol_bh", "atol_bh"),
    [
        (
            "dipole",
            magpy.misc.Dipole(moment=(0.1, -0.2, 0.3)),
            mpj.misc.Dipole(moment=(0.1, -0.2, 0.3)),
            np.array([[0.2, 0.3, 0.4], [1.2, -0.8, 0.7], [2.0, 2.0, -1.0]]),
            5e-6,
            1e-10,
        ),
        (
            "circle",
            magpy.current.Circle(current=1.2, diameter=1.4),
            mpj.current.Circle(current=1.2, diameter=1.4),
            np.array([[0.1, 0.1, 0.2], [0.3, -0.4, 0.6], [0.0, 0.0, 1.2]]),
            5e-5,
            5e-10,
        ),
        (
            "cuboid",
            magpy.magnet.Cuboid(
                polarization=(0.1, -0.2, 0.3),
                dimension=(1.1, 0.7, 1.3),
            ),
            mpj.magnet.Cuboid(
                polarization=(0.1, -0.2, 0.3),
                dimension=(1.1, 0.7, 1.3),
            ),
            np.array([[0.0, 0.0, 0.0], [0.7, 0.2, -0.1], [2.0, 0.0, 0.0]]),
            2e-5,
            1e-7,
        ),
        (
            "cylinder",
            magpy.magnet.Cylinder(
                polarization=(0.1, -0.2, 0.3),
                dimension=(1.2, 1.5),
            ),
            mpj.magnet.Cylinder(
                polarization=(0.1, -0.2, 0.3),
                dimension=(1.2, 1.5),
            ),
            np.array([[0.0, 0.0, 0.0], [0.4, 0.1, 0.2], [1.5, 0.0, 0.0]]),
            5e-4,
            1e-4,
        ),
        (
            "sphere",
            magpy.magnet.Sphere(polarization=(0.1, -0.2, 0.3), diameter=1.8),
            mpj.magnet.Sphere(polarization=(0.1, -0.2, 0.3), diameter=1.8),
            np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.1], [2.0, 0.0, 0.0]]),
            2e-6,
            1e-10,
        ),
        (
            "polyline",
            magpy.current.Polyline(
                current=1.1,
                vertices=[(0, 0, 0), (0.5, 0.2, 0.3), (1.0, 0.0, 0.2)],
            ),
            mpj.current.Polyline(
                current=1.1,
                vertices=[(0, 0, 0), (0.5, 0.2, 0.3), (1.0, 0.0, 0.2)],
            ),
            np.array([[0.2, 0.4, 0.3], [0.8, -0.1, 0.7], [1.2, 0.3, -0.2]]),
            1e-4,
            1e-9,
        ),
        (
            "triangle",
            magpy.misc.Triangle(
                vertices=[(-0.2, -0.1, 0.0), (0.9, 0.3, 0.2), (0.1, 0.8, -0.2)],
                polarization=(0.2, -0.1, 0.3),
            ),
            mpj.misc.Triangle(
                vertices=[(-0.2, -0.1, 0.0), (0.9, 0.3, 0.2), (0.1, 0.8, -0.2)],
                polarization=(0.2, -0.1, 0.3),
            ),
            np.array([[0.4, 0.5, 0.3], [0.8, 0.6, 0.7], [1.5, -0.2, 0.4]]),
            2e-4,
            1e-7,
        ),
        (
            "tetrahedron",
            magpy.magnet.Tetrahedron(
                vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                polarization=(0.1, -0.2, 0.3),
            ),
            mpj.magnet.Tetrahedron(
                vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                polarization=(0.1, -0.2, 0.3),
            ),
            np.array([[0.1, 0.1, 0.1], [0.8, 0.3, 0.2], [1.4, 1.2, 1.1]]),
            3e-4,
            2e-5,
        ),
    ],
)
def test_bhjm_profile_parity(name, src_ref, src_new, points, rtol_bh, atol_bh) -> None:
    b_ref = src_ref.getB(points)
    h_ref = src_ref.getH(points)
    j_ref = src_ref.getJ(points)
    m_ref = src_ref.getM(points)

    b_new = np.asarray(src_new.getB(points))
    h_new = np.asarray(src_new.getH(points))
    j_new = np.asarray(src_new.getJ(points))
    m_new = np.asarray(src_new.getM(points))

    np.testing.assert_allclose(b_new, b_ref, rtol=rtol_bh, atol=atol_bh)
    np.testing.assert_allclose(h_new, h_ref, rtol=rtol_bh, atol=max(atol_bh, 2e-3))
    np.testing.assert_allclose(j_new, j_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(m_new, m_ref, rtol=0, atol=2e-6)


@pytest.mark.parity_gate
def test_squeeze_and_sumup_shapes() -> None:
    src1 = mpj.magnet.Sphere(polarization=(0.1, 0.2, 0.3), diameter=1.0)
    src2 = mpj.misc.Dipole(moment=(0.1, 0.0, 0.0))
    sens = mpj.Sensor(pixel=[(1, 2, 3), (1, 2, 3)])

    b_default = np.asarray(mpj.getB(src1, sens))
    b_unsqueezed = np.asarray(mpj.getB(src1, sens, squeeze=False))
    b_unsqueezed_split = np.asarray(mpj.getB([src1, src2], sens, squeeze=False, sumup=False))

    assert b_default.shape == (2, 3)
    assert b_unsqueezed.shape == (1, 1, 1, 2, 3)
    assert b_unsqueezed_split.shape == (1, 2, 1, 2, 3)
