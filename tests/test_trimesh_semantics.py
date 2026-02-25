import magpylib as magpy
import numpy as np
import pytest

import magpylib_jax as mpj
from magpylib_jax.constants import MU0


def _tetra_mesh():
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    faces = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]
    pol = (0.1, -0.2, 0.3)
    return vertices, faces, pol


def test_trimesh_boundary_jm_matches_magpylib() -> None:
    vertices, faces, pol = _tetra_mesh()
    ref = magpy.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=pol,
        reorient_faces=False,
        check_open=False,
    )
    src = mpj.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=pol,
        reorient_faces=False,
        check_open="skip",
    )
    points = np.array(
        [
            [0.2, 0.2, 0.0],  # face
            [0.5, 0.0, 0.0],  # edge
            [0.0, 0.0, 0.0],  # vertex
        ]
    )
    np.testing.assert_allclose(np.asarray(src.getJ(points)), ref.getJ(points))
    np.testing.assert_allclose(np.asarray(src.getM(points)), ref.getM(points))


def test_trimesh_in_out_overrides() -> None:
    vertices, faces, pol = _tetra_mesh()
    inside_pt = np.array([[0.1, 0.1, 0.1]])
    outside_pt = np.array([[1.2, 0.2, 0.2]])

    src = mpj.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=pol,
        reorient_faces=False,
        check_open="skip",
    )
    np.testing.assert_allclose(np.asarray(src.getJ(inside_pt)), np.asarray(pol))
    np.testing.assert_allclose(np.asarray(src.getJ(outside_pt)), np.zeros((3,)))

    src_inside = mpj.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=pol,
        reorient_faces=False,
        check_open="skip",
        in_out="inside",
    )
    np.testing.assert_allclose(np.asarray(src_inside.getJ(outside_pt)), np.asarray(pol))

    src_outside = mpj.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=pol,
        reorient_faces=False,
        check_open="skip",
        in_out="outside",
    )
    np.testing.assert_allclose(np.asarray(src_outside.getJ(inside_pt)), np.zeros((3,)))

    np.testing.assert_allclose(
        np.asarray(src_inside.getM(outside_pt)), np.asarray(pol) / MU0, rtol=0, atol=1e-10
    )


def test_trimesh_check_open_modes() -> None:
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    faces_open = [(0, 2, 1), (0, 1, 3), (1, 2, 3)]

    with pytest.warns(UserWarning):
        _ = mpj.magnet.TriangularMesh(
            vertices=vertices,
            faces=faces_open,
            polarization=(1.0, 0.0, 0.0),
            reorient_faces=False,
            check_open="warn",
        )

    with pytest.raises(ValueError):
        _ = mpj.magnet.TriangularMesh(
            vertices=vertices,
            faces=faces_open,
            polarization=(1.0, 0.0, 0.0),
            reorient_faces=False,
            check_open="raise",
        )

    _ = mpj.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces_open,
        polarization=(1.0, 0.0, 0.0),
        reorient_faces=False,
        check_open="skip",
    )
