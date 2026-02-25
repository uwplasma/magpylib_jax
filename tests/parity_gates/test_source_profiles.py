import magpylib as magpy
import numpy as np
import pytest

import magpylib_jax as mpj


@pytest.mark.parity_gate
@pytest.mark.parametrize(
    ("name", "src_ref", "src_new", "points", "rtol_bh", "atol_bh", "atol_h"),
    [
        (
            "dipole",
            magpy.misc.Dipole(moment=(0.1, -0.2, 0.3)),
            mpj.misc.Dipole(moment=(0.1, -0.2, 0.3)),
            np.array(
                [
                    [0.2, 0.3, 0.4],
                    [1.2, -0.8, 0.7],
                    [2.0, 2.0, -1.0],
                    [0.12, -0.03, 0.09],
                ]
            ),
            5e-6,
            1e-10,
            5e-4,
        ),
        (
            "circle",
            magpy.current.Circle(current=1.2, diameter=1.4),
            mpj.current.Circle(current=1.2, diameter=1.4),
            np.array(
                [
                    [0.0, 0.0, 0.7],
                    [0.7, 0.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [0.001, 0.0, 0.7],
                ]
            ),
            7e-5,
            2e-9,
            2e-3,
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
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.55, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.549, 0.349, 0.649],
                ]
            ),
            2e-5,
            1e-7,
            5e-2,
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
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.6, 0.0, 0.0],
                    [1.4, 0.0, 0.0],
                    [0.599, 0.0, 0.749],
                ]
            ),
            5e-4,
            1e-4,
            1e-1,
        ),
        (
            "cylindersegment",
            magpy.magnet.CylinderSegment(
                polarization=(0.1, -0.2, 0.3),
                dimension=(0.4, 1.2, 1.1, -30.0, 110.0),
            ),
            mpj.magnet.CylinderSegment(
                polarization=(0.1, -0.2, 0.3),
                dimension=(0.4, 1.2, 1.1, -30.0, 110.0),
            ),
            np.array(
                [
                    [0.7, 0.2, 0.0],
                    [1.19 * np.cos(np.deg2rad(20)), 1.19 * np.sin(np.deg2rad(20)), 0.02],
                    [1.4, 0.1, 0.0],
                    [0.41 * np.cos(np.deg2rad(-29)), 0.41 * np.sin(np.deg2rad(-29)), 0.54],
                ]
            ),
            2e-2,
            2e-3,
            3e2,
        ),
        (
            "sphere",
            magpy.magnet.Sphere(polarization=(0.1, -0.2, 0.3), diameter=1.8),
            mpj.magnet.Sphere(polarization=(0.1, -0.2, 0.3), diameter=1.8),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.9, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.9, 1e-3, 0.0],
                ]
            ),
            2e-6,
            1e-10,
            5e-5,
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
            np.array(
                [
                    [0.2, 0.4, 0.3],
                    [0.8, -0.1, 0.7],
                    [1.2, 0.3, -0.2],
                    [0.5, 0.2005, 0.3005],
                ]
            ),
            1e-4,
            1e-9,
            2e-3,
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
            np.array(
                [
                    [0.4, 0.5, 0.3],
                    [0.8, 0.6, 0.7],
                    [1.5, -0.2, 0.4],
                    [0.3, 0.4, 1e-3],
                ]
            ),
            2e-4,
            1e-7,
            1e-1,
        ),
        (
            "trianglesheet",
            magpy.current.TriangleSheet(
                vertices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
                faces=[[0, 1, 2], [1, 2, 3]],
                current_densities=[[0.7, 0.1, 0.0], [0.7, 0.1, 0.0]],
            ),
            mpj.current.TriangleSheet(
                vertices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
                faces=[[0, 1, 2], [1, 2, 3]],
                current_densities=[[0.7, 0.1, 0.0], [0.7, 0.1, 0.0]],
            ),
            np.array(
                [
                    [0.4, 0.5, 0.6],
                    [0.8, 0.6, 0.7],
                    [1.5, -0.2, 0.4],
                    [0.2, 0.2, 1e-3],
                ]
            ),
            8e-2,
            2e-6,
            2.0,
        ),
        (
            "trianglestrip",
            magpy.current.TriangleStrip(
                vertices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]],
                current=1.4,
            ),
            mpj.current.TriangleStrip(
                vertices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]],
                current=1.4,
            ),
            np.array(
                [
                    [0.4, 0.5, 0.6],
                    [0.8, 0.6, 0.7],
                    [1.5, -0.2, 0.4],
                    [0.2, 0.2, 1e-3],
                ]
            ),
            8e-2,
            2e-6,
            2.0,
        ),
        (
            "triangularmesh",
            magpy.magnet.TriangularMesh(
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                faces=[[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
                polarization=(0.1, -0.2, 0.3),
                reorient_faces=False,
                check_open=False,
            ),
            mpj.magnet.TriangularMesh(
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                faces=[[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
                polarization=(0.1, -0.2, 0.3),
                reorient_faces=False,
            ),
            np.array(
                [
                    [0.1, 0.1, 0.1],
                    [1.1, 0.1, 0.1],
                    [1.4, 1.2, 1.1],
                    [0.24, 0.24, 0.48],
                ]
            ),
            4e-5,
            1e-10,
            2e-3,
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
            np.array(
                [
                    [0.1, 0.1, 0.1],
                    [0.8, 0.3, 0.2],
                    [1.4, 1.2, 1.1],
                    [0.2, 0.2, 0.6],
                ]
            ),
            3e-4,
            2e-5,
            4e-1,
        ),
    ],
)
def test_bhjm_profile_parity(
    name,
    src_ref,
    src_new,
    points,
    rtol_bh,
    atol_bh,
    atol_h,
) -> None:
    b_ref = src_ref.getB(points)
    h_ref = src_ref.getH(points)
    j_ref = src_ref.getJ(points)
    m_ref = src_ref.getM(points)

    b_new = np.asarray(src_new.getB(points))
    h_new = np.asarray(src_new.getH(points))
    j_new = np.asarray(src_new.getJ(points))
    m_new = np.asarray(src_new.getM(points))

    np.testing.assert_allclose(b_new, b_ref, rtol=rtol_bh, atol=atol_bh)
    np.testing.assert_allclose(h_new, h_ref, rtol=rtol_bh, atol=atol_h)
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
    assert b_unsqueezed_split.shape == (2, 1, 1, 2, 3)
