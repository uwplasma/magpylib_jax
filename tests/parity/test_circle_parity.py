import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_circle_bh_parity_against_magpylib() -> None:
    rng = np.random.default_rng(11)
    observers = rng.normal(size=(600, 3))
    observers[:, :2] += np.array([0.25, -0.15])
    observers[:, 2] += 0.35

    current = 2.5
    diameter = 0.9
    position = np.array([0.1, 0.2, -0.4])

    src_ref = magpy.current.Circle(current=current, diameter=diameter, position=position)
    src_new = mpj.Circle(current=current, diameter=diameter, position=position)

    b_ref = src_ref.getB(observers)
    h_ref = src_ref.getH(observers)
    b_new = np.asarray(src_new.getB(observers))
    h_new = np.asarray(src_new.getH(observers))

    np.testing.assert_allclose(b_new, b_ref, rtol=2e-5, atol=1e-12)
    np.testing.assert_allclose(h_new, h_ref, rtol=2e-5, atol=1e-6)
