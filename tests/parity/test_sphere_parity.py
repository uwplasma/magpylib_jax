import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_sphere_bh_parity_against_magpylib() -> None:
    rng = np.random.default_rng(45)
    observers = rng.normal(size=(600, 3))
    observers += np.array([0.15, -0.25, 0.3])

    polarization = np.array([0.11, -0.07, 0.13])
    diameter = 1.3
    position = np.array([0.2, 0.1, -0.15])

    src_ref = magpy.magnet.Sphere(
        polarization=polarization,
        diameter=diameter,
        position=position,
    )
    src_new = mpj.magnet.Sphere(
        polarization=polarization,
        diameter=diameter,
        position=position,
    )

    b_ref = src_ref.getB(observers)
    h_ref = src_ref.getH(observers)
    b_new = np.asarray(src_new.getB(observers))
    h_new = np.asarray(src_new.getH(observers))

    np.testing.assert_allclose(b_new, b_ref, rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(h_new, h_ref, rtol=2e-6, atol=5e-5)


def test_sphere_jm_inside_outside_parity() -> None:
    src_ref = magpy.magnet.Sphere(polarization=(0.1, 0.2, 0.3), diameter=2.0)
    src_new = mpj.magnet.Sphere(polarization=(0.1, 0.2, 0.3), diameter=2.0)

    observers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [2.5, 0.0, 0.0],
            [0.0, 0.0, 1.5],
        ]
    )

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
