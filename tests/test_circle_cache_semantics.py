import numpy as np

import magpylib_jax as mpj


def test_circle_collection_cache_semantics() -> None:
    obs = np.array([[0.2, 0.1, 0.3], [0.5, -0.2, 0.4]], dtype=float)
    src1 = mpj.current.Circle(current=1.0, diameter=1.0, position=(0.0, 0.0, 0.0))
    src2 = mpj.current.Circle(current=1.5, diameter=0.8, position=(0.0, 0.0, 0.5))
    col = mpj.Collection(src1, src2)

    b0 = np.asarray(mpj.getB(col, obs))
    b1 = np.asarray(mpj.getB(col, obs))
    assert np.allclose(b0, b1, rtol=1e-12, atol=1e-12)

    src1.current = 2.0
    b2 = np.asarray(mpj.getB(col, obs))
    assert not np.allclose(b1, b2, rtol=1e-12, atol=1e-12)

    src1.current = 1.0
    b3 = np.asarray(mpj.getB(col, obs))
    assert np.allclose(b0, b3, rtol=1e-12, atol=1e-12)

    col.move((0.0, 0.0, 0.2))
    b4 = np.asarray(mpj.getB(col, obs))
    assert not np.allclose(b3, b4, rtol=1e-12, atol=1e-12)
