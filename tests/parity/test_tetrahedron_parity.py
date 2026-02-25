import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_tetrahedron_bh_parity_against_magpylib() -> None:
    rng = np.random.default_rng(91)
    observers = rng.normal(size=(450, 3))
    observers += np.array([0.45, 0.4, 0.55])

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.1, 0.0],
            [0.2, 0.9, 0.1],
            [0.1, 0.2, 1.0],
        ]
    )
    polarization = np.array([0.12, -0.2, 0.28])

    src_ref = magpy.magnet.Tetrahedron(vertices=vertices, polarization=polarization)
    src_new = mpj.magnet.Tetrahedron(vertices=vertices, polarization=polarization)

    b_ref = src_ref.getB(observers)
    h_ref = src_ref.getH(observers)
    b_new = np.asarray(src_new.getB(observers))
    h_new = np.asarray(src_new.getH(observers))

    np.testing.assert_allclose(b_new, b_ref, rtol=2e-4, atol=3e-8)
    np.testing.assert_allclose(h_new, h_ref, rtol=2e-4, atol=4e-2)


def test_tetrahedron_jm_inside_outside_parity() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    src_ref = magpy.magnet.Tetrahedron(vertices=vertices, polarization=(0.1, 0.2, 0.3))
    src_new = mpj.magnet.Tetrahedron(vertices=vertices, polarization=(0.1, 0.2, 0.3))

    observers = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.2, 0.2, 0.2]])

    np.testing.assert_allclose(
        np.asarray(src_new.getJ(observers)),
        src_ref.getJ(observers),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(src_new.getM(observers)),
        src_ref.getM(observers),
        atol=1e-8,
    )
