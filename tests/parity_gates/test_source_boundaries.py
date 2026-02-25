import magpylib as magpy
import numpy as np
import pytest

import magpylib_jax as mpj

pytestmark = pytest.mark.filterwarnings(
    "ignore:Unchecked open mesh status in TriangularMesh.*"
)


@pytest.mark.parity_gate
@pytest.mark.parametrize(
    (
        "name",
        "src_ref",
        "src_new",
        "points_all",
        "points_bh",
        "rtol_bh",
        "atol_bh",
    ),
    [
        (
            "cuboid",
            magpy.magnet.Cuboid(
                polarization=(0.1, -0.2, 0.3),
                dimension=(1.0, 2.0, 3.0),
            ),
            mpj.magnet.Cuboid(
                polarization=(0.1, -0.2, 0.3),
                dimension=(1.0, 2.0, 3.0),
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5 + 1e-6, 0.0, 0.0],
                    [0.5, 1.0, 0.0],
                    [0.5 + 1e-6, 1.0 + 1e-6, 1.5 + 1e-6],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.5 + 1e-6, 0.0, 0.0],
                    [0.5 + 1e-6, 1.0 + 1e-6, 1.5 + 1e-6],
                ]
            ),
            3e-5,
            2e-7,
        ),
        (
            "cylinder",
            magpy.magnet.Cylinder(
                polarization=(0.05, -0.1, 0.2),
                dimension=(1.2, 1.0),
            ),
            mpj.magnet.Cylinder(
                polarization=(0.05, -0.1, 0.2),
                dimension=(1.2, 1.0),
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.6, 0.0, 0.0],
                    [0.6 + 1e-6, 0.0, 0.0],
                    [0.0, 0.0, 0.5],
                    [0.0, 0.0, 0.5 + 1e-6],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.6 + 1e-6, 0.0, 0.0],
                    [0.0, 0.0, 0.5 + 1e-6],
                ]
            ),
            5e-4,
            2e-6,
        ),
        (
            "cylindersegment",
            magpy.magnet.CylinderSegment(
                polarization=(0.1, -0.2, 0.3),
                dimension=(0.4, 1.0, 1.0, -30.0, 110.0),
            ),
            mpj.magnet.CylinderSegment(
                polarization=(0.1, -0.2, 0.3),
                dimension=(0.4, 1.0, 1.0, -30.0, 110.0),
            ),
            np.array(
                [
                    [0.7, 0.0, 0.0],
                    [0.4, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0 + 1e-6, 0.0, 0.0],
                    [0.7 * np.cos(np.deg2rad(-30.0)), 0.7 * np.sin(np.deg2rad(-30.0)), 0.0],
                    [
                        0.7 * np.cos(np.deg2rad(-30.0 - 1e-3)),
                        0.7 * np.sin(np.deg2rad(-30.0 - 1e-3)),
                        0.0,
                    ],
                ]
            ),
            np.array(
                [
                    [0.7, 0.0, 0.0],
                    [1.0 + 1e-6, 0.0, 0.0],
                    [
                        0.7 * np.cos(np.deg2rad(-30.0 - 1e-3)),
                        0.7 * np.sin(np.deg2rad(-30.0 - 1e-3)),
                        0.0,
                    ],
                ]
            ),
            3e-2,
            2e-3,
        ),
        (
            "sphere",
            magpy.magnet.Sphere(polarization=(0.1, -0.2, 0.3), diameter=1.8),
            mpj.magnet.Sphere(polarization=(0.1, -0.2, 0.3), diameter=1.8),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.9, 0.0, 0.0],
                    [0.9 + 1e-6, 0.0, 0.0],
                    [0.0, 0.9, 0.0],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.9 + 1e-6, 0.0, 0.0],
                ]
            ),
            3e-6,
            1e-9,
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
                    [0.0, 0.3, 0.3],
                    [1.1, 0.1, 0.1],
                    [0.0, 0.5, 0.5],
                ]
            ),
            np.array(
                [
                    [0.1, 0.1, 0.1],
                    [1.1, 0.1, 0.1],
                ]
            ),
            5e-4,
            1e-6,
        ),
        (
            "triangularmesh",
            magpy.magnet.TriangularMesh(
                vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                faces=[(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)],
                polarization=(0.1, -0.2, 0.3),
                reorient_faces=False,
                check_open=False,
            ),
            mpj.magnet.TriangularMesh(
                vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                faces=[(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)],
                polarization=(0.1, -0.2, 0.3),
                reorient_faces=False,
            ),
            np.array(
                [
                    [0.1, 0.1, 0.1],
                    [0.0, 0.25, 0.25],
                    [1.2, 0.1, 0.1],
                    [0.0, 0.5, 0.5],
                ]
            ),
            np.array(
                [
                    [0.1, 0.1, 0.1],
                    [1.2, 0.1, 0.1],
                ]
            ),
            5e-4,
            1e-6,
        ),
    ],
)
def test_boundary_parity(
    name,
    src_ref,
    src_new,
    points_all,
    points_bh,
    rtol_bh,
    atol_bh,
):
    """Check inside/surface/outside parity near boundaries."""
    for field in ("B", "H"):
        ref = getattr(src_ref, f"get{field}")(points_bh, in_out="auto")
        new = getattr(src_new, f"get{field}")(points_bh, in_out="auto")
        np.testing.assert_allclose(new, ref, rtol=rtol_bh, atol=atol_bh)

    for field in ("J", "M"):
        ref = getattr(src_ref, f"get{field}")(points_all, in_out="auto")
        new = getattr(src_new, f"get{field}")(points_all, in_out="auto")
        np.testing.assert_allclose(new, ref, rtol=1e-6, atol=1e-8)
