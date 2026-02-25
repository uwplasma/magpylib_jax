import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_cylinder_segment_bh_parity_against_magpylib() -> None:
    rng = np.random.default_rng(123)
    observers = rng.normal(size=(500, 3))
    observers += np.array([1.0, 0.4, -0.3])

    polarization = np.array([0.1, -0.2, 0.3])
    dimension = np.array([0.4, 1.2, 1.1, -30.0, 110.0])

    src_ref = magpy.magnet.CylinderSegment(
        polarization=polarization,
        dimension=dimension,
    )
    src_new = mpj.magnet.CylinderSegment(
        polarization=polarization,
        dimension=dimension,
    )

    b_ref = src_ref.getB(observers)
    h_ref = src_ref.getH(observers)
    b_new = np.asarray(src_new.getB(observers))
    h_new = np.asarray(src_new.getH(observers))

    np.testing.assert_allclose(b_new, b_ref, rtol=2e-2, atol=2e-3)
    np.testing.assert_allclose(h_new, h_ref, rtol=2e-2, atol=2.5e2)


def test_cylinder_segment_jm_inside_outside_parity() -> None:
    src_ref = magpy.magnet.CylinderSegment(
        polarization=(0.1, 0.2, 0.3),
        dimension=(0.3, 1.0, 1.5, -45.0, 135.0),
    )
    src_new = mpj.magnet.CylinderSegment(
        polarization=(0.1, 0.2, 0.3),
        dimension=(0.3, 1.0, 1.5, -45.0, 135.0),
    )

    observers = np.array(
        [
            [0.6, 0.0, 0.0],
            [0.0, 0.6, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        np.asarray(src_new.getJ(observers)),
        src_ref.getJ(observers),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(src_new.getM(observers)),
        src_ref.getM(observers),
        atol=1e-7,
    )
