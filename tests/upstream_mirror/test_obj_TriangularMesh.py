import magpylib as magpy
import numpy as np
import pytest

import magpylib_jax as mpj
from magpylib_jax.constants import MU0


def _tetra_inputs() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    return vertices, faces


def test_triangular_mesh_barycenter_and_volume() -> None:
    vertices, faces = _tetra_inputs()
    src = mpj.magnet.TriangularMesh(vertices=vertices, faces=faces, polarization=(0, 0, 1))

    np.testing.assert_allclose(
        np.asarray(src.barycenter),
        np.array([0.26289171, 0.26289171, 0.26289171]),
    )
    assert abs(src.volume - (1 / 6)) < 1e-12


def test_triangular_mesh_bh_matches_upstream_for_closed_mesh() -> None:
    vertices, faces = _tetra_inputs()
    pol = (0.11, -0.12, 0.23)

    src_ref = magpy.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=pol,
        reorient_faces=False,
        check_open=False,
    )
    src_new = mpj.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=pol,
        reorient_faces=False,
    )

    observers = np.array([[0.3, 0.2, 0.1], [1.3, 1.0, 0.8], [0.1, 0.1, 0.1]])

    np.testing.assert_allclose(
        np.asarray(src_new.getB(observers)),
        src_ref.getB(observers),
        rtol=4e-5,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(src_new.getH(observers)),
        src_ref.getH(observers),
        rtol=4e-5,
        atol=2e-3,
    )


def test_triangular_mesh_check_open_modes() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])

    with pytest.raises(ValueError, match="Open mesh detected"):
        mpj.magnet.TriangularMesh(
            vertices=vertices,
            faces=faces,
            polarization=(0, 0, 1),
            check_open="raise",
        )

    with pytest.warns(UserWarning, match="Open mesh detected"):
        mpj.magnet.TriangularMesh(
            vertices=vertices,
            faces=faces,
            polarization=(0, 0, 1),
            check_open="warn",
        )

    mpj.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=(0, 0, 1),
        check_open="skip",
    )


def test_triangular_mesh_boundary_points_inside_for_jm() -> None:
    vertices, faces = _tetra_inputs()
    pol = (0.1, -0.2, 0.3)
    src = mpj.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=pol,
        reorient_faces=False,
        check_open="skip",
    )

    # Point on the base face (z=0) should be treated as inside for J/M.
    observers = np.array([[0.25, 0.25, 0.0]])
    j = src.getJ(observers, in_out="auto")
    m = src.getM(observers, in_out="auto")

    np.testing.assert_allclose(j, np.array(pol))
    np.testing.assert_allclose(m, np.array(pol) / MU0)
