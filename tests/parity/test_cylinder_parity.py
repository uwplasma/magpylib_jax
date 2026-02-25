import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_cylinder_bh_parity_against_magpylib() -> None:
    rng = np.random.default_rng(33)
    observers = rng.normal(size=(700, 3))
    observers += np.array([0.2, 0.35, -0.45])

    polarization = np.array([0.11, -0.07, 0.13])
    dimension = np.array([1.4, 1.2])
    position = np.array([0.1, -0.2, 0.25])

    src_ref = magpy.magnet.Cylinder(
        polarization=polarization,
        dimension=dimension,
        position=position,
    )
    src_new = mpj.magnet.Cylinder(
        polarization=polarization,
        dimension=dimension,
        position=position,
    )

    b_ref = src_ref.getB(observers)
    h_ref = src_ref.getH(observers)
    b_new = np.asarray(src_new.getB(observers))
    h_new = np.asarray(src_new.getH(observers))

    np.testing.assert_allclose(b_new, b_ref, rtol=3e-4, atol=1e-8)
    np.testing.assert_allclose(h_new, h_ref, rtol=3e-4, atol=2e-2)


def test_cylinder_jm_inside_outside_parity() -> None:
    src_ref = magpy.magnet.Cylinder(polarization=(0.1, 0.2, 0.3), dimension=(2.0, 2.0))
    src_new = mpj.magnet.Cylinder(polarization=(0.1, 0.2, 0.3), dimension=(2.0, 2.0))

    observers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [2.5, 0.0, 0.0],
            [0.0, 0.0, 1.5],
        ]
    )

    j_ref = src_ref.getJ(observers)
    m_ref = src_ref.getM(observers)
    j_new = np.asarray(src_new.getJ(observers))
    m_new = np.asarray(src_new.getM(observers))

    np.testing.assert_allclose(j_new, j_ref, rtol=0, atol=1e-14)
    np.testing.assert_allclose(m_new, m_ref, rtol=0, atol=1e-8)
