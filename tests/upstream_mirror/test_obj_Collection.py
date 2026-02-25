import numpy as np

import magpylib_jax as mpj


def test_collection_add_and_sum() -> None:
    obs = np.array([[0.1, 0.2, 0.3], [0.5, -0.2, 0.1]])
    src1 = mpj.magnet.Cuboid(polarization=(1, 0, 0), dimension=(1, 1, 1))
    src2 = mpj.magnet.Cylinder(polarization=(0, 1, 0), dimension=(1, 1))

    col = mpj.Collection([src1])
    col.add(src2)

    b_sum = np.asarray(src1.getB(obs) + src2.getB(obs))
    b_col = np.asarray(col.getB(obs))
    np.testing.assert_allclose(b_col, b_sum)


def test_collection_add_operator() -> None:
    obs = np.array([0.2, 0.1, 0.3])
    src1 = mpj.magnet.Cuboid(polarization=(1, 0, 0), dimension=(1, 1, 1))
    src2 = mpj.magnet.Cylinder(polarization=(0, 1, 0), dimension=(1, 1))

    col = mpj.Collection([src1]) + src2
    b_sum = np.asarray(src1.getB(obs) + src2.getB(obs))
    np.testing.assert_allclose(np.asarray(col.getB(obs)), b_sum)
