import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_triangle_bh_parity_against_magpylib() -> None:
    rng = np.random.default_rng(71)
    observers = rng.normal(size=(350, 3))
    observers += np.array([0.7, 0.4, 0.5])

    vertices = np.array([[-0.2, -0.1, 0.0], [0.9, 0.3, 0.2], [0.1, 0.8, -0.2]])
    polarization = np.array([0.2, -0.15, 0.3])

    src_ref = magpy.misc.Triangle(vertices=vertices, polarization=polarization)
    src_new = mpj.misc.Triangle(vertices=vertices, polarization=polarization)

    b_ref = src_ref.getB(observers)
    h_ref = src_ref.getH(observers)
    b_new = np.asarray(src_new.getB(observers))
    h_new = np.asarray(src_new.getH(observers))

    np.testing.assert_allclose(b_new, b_ref, rtol=7e-5, atol=2e-9)
    np.testing.assert_allclose(h_new, h_ref, rtol=7e-5, atol=2e-3)


def test_triangle_jm_zero() -> None:
    src = mpj.misc.Triangle(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], polarization=(0, 0, 1))
    observers = np.array([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]])
    np.testing.assert_allclose(np.asarray(src.getJ(observers)), 0.0)
    np.testing.assert_allclose(np.asarray(src.getM(observers)), 0.0)
