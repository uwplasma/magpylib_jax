import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_polyline_bh_parity_against_magpylib() -> None:
    rng = np.random.default_rng(57)
    observers = rng.normal(size=(300, 3))
    observers += np.array([0.4, 0.2, -0.15])

    vertices = np.array(
        [
            [-0.5, -0.1, 0.3],
            [0.2, 0.4, 0.7],
            [0.9, 0.6, -0.2],
            [1.2, -0.3, 0.1],
        ]
    )

    src_ref = magpy.current.Polyline(current=1.7, vertices=vertices)
    src_new = mpj.current.Polyline(current=1.7, vertices=vertices)

    b_ref = src_ref.getB(observers)
    h_ref = src_ref.getH(observers)
    b_new = np.asarray(src_new.getB(observers))
    h_new = np.asarray(src_new.getH(observers))

    np.testing.assert_allclose(b_new, b_ref, rtol=4e-5, atol=2e-10)
    np.testing.assert_allclose(h_new, h_ref, rtol=4e-5, atol=1e-4)


def test_polyline_special_cases() -> None:
    line = mpj.current.Polyline(current=100.0, vertices=[(0, 0, 0), (0, 0, 0)])
    b = np.asarray(line.getB([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(b, np.zeros(3), atol=0)
