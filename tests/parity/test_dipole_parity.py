import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_dipole_bh_parity_against_magpylib() -> None:
    rng = np.random.default_rng(7)
    observers = rng.normal(size=(512, 3))
    observers += np.array([0.4, -0.3, 0.1])

    moment = np.array([1.2, -0.7, 0.3])
    position = np.array([0.2, -0.1, 0.5])

    src_ref = magpy.misc.Dipole(moment=moment, position=position)
    src_new = mpj.Dipole(moment=moment, position=position)

    b_ref = src_ref.getB(observers)
    h_ref = src_ref.getH(observers)
    b_new = np.asarray(src_new.getB(observers))
    h_new = np.asarray(src_new.getH(observers))

    np.testing.assert_allclose(b_new, b_ref, rtol=5e-7, atol=1e-12)
    np.testing.assert_allclose(h_new, h_ref, rtol=5e-7, atol=1e-6)
