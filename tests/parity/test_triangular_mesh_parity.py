import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_triangular_mesh_bh_parity_against_magpylib() -> None:
    rng = np.random.default_rng(444)
    observers = rng.normal(size=(350, 3))
    observers += np.array([0.3, 0.6, 0.7])

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    polarization = np.array([0.1, -0.2, 0.3])

    src_ref = magpy.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=polarization,
        reorient_faces=False,
        check_open=False,
    )
    src_new = mpj.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=polarization,
        reorient_faces=False,
    )

    b_ref = src_ref.getB(observers)
    h_ref = src_ref.getH(observers)
    b_new = np.asarray(src_new.getB(observers))
    h_new = np.asarray(src_new.getH(observers))

    np.testing.assert_allclose(b_new, b_ref, rtol=4e-5, atol=1e-10)
    np.testing.assert_allclose(h_new, h_ref, rtol=4e-5, atol=2e-3)


def test_triangular_mesh_jm_inside_outside_parity() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])

    src_ref = magpy.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=(0.1, 0.2, 0.3),
        reorient_faces=False,
        check_open=False,
    )
    src_new = mpj.magnet.TriangularMesh(
        vertices=vertices,
        faces=faces,
        polarization=(0.1, 0.2, 0.3),
        reorient_faces=False,
    )

    observers = np.array([[0.1, 0.1, 0.1], [1.2, 1.2, 1.2], [0.2, 0.2, 0.2]])

    np.testing.assert_allclose(
        np.asarray(src_new.getJ(observers)),
        src_ref.getJ(observers),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(src_new.getM(observers)),
        src_ref.getM(observers),
        atol=2e-6,
    )
