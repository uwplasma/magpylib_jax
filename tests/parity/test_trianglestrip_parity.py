import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_trianglestrip_bh_parity_against_magpylib() -> None:
    rng = np.random.default_rng(333)
    observers = rng.normal(size=(280, 3))
    observers += np.array([0.6, 0.1, 0.9])

    vertices = np.array(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]],
        dtype=float,
    )

    src_ref = magpy.current.TriangleStrip(vertices=vertices, current=1.4)
    src_new = mpj.current.TriangleStrip(vertices=vertices, current=1.4)

    b_ref = src_ref.getB(observers)
    h_ref = src_ref.getH(observers)
    b_new = np.asarray(src_new.getB(observers))
    h_new = np.asarray(src_new.getH(observers))

    np.testing.assert_allclose(b_new, b_ref, rtol=8e-2, atol=2e-6)
    np.testing.assert_allclose(h_new, h_ref, rtol=8e-2, atol=2.0)


def test_trianglestrip_zero_area_triangle_is_stable() -> None:
    src = mpj.current.TriangleStrip(
        vertices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]],
        current=2.0,
    )
    obs = np.array([[0.2, 0.4, 0.8]])
    out = np.asarray(src.getH(obs))
    assert np.all(np.isfinite(out))
