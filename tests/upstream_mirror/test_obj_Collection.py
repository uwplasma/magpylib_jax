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


def test_collection_len_getitem_iter() -> None:
    src1 = mpj.magnet.Cuboid(polarization=(1, 0, 0), dimension=(1, 1, 1))
    src2 = mpj.magnet.Cylinder(polarization=(0, 1, 0), dimension=(1, 1))
    src3 = mpj.magnet.Sphere(polarization=(0, 0, 1), diameter=1)

    col = mpj.Collection([src1, src2, src3])
    assert len(col) == 3
    assert col[1] is src2
    assert list(col) == [src1, src2, src3]


def test_collection_getHJM_sum() -> None:
    obs = np.array([[0.1, 0.2, 0.3], [0.5, -0.2, 0.1]])
    src1 = mpj.magnet.Cuboid(polarization=(1, 0, 0), dimension=(1, 1, 1))
    src2 = mpj.magnet.Cylinder(polarization=(0, 1, 0), dimension=(1, 1))

    col = mpj.Collection([src1, src2])
    np.testing.assert_allclose(np.asarray(col.getH(obs)), src1.getH(obs) + src2.getH(obs))
    np.testing.assert_allclose(np.asarray(col.getJ(obs)), src1.getJ(obs) + src2.getJ(obs))
    np.testing.assert_allclose(np.asarray(col.getM(obs)), src1.getM(obs) + src2.getM(obs))
