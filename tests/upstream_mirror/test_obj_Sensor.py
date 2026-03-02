import numpy as np

import magpylib_jax as mpj


def test_sensor1():
    """self-consistent test of the sensor class"""
    pm = mpj.magnet.Cuboid(polarization=(11, 22, 33), dimension=(1, 2, 3))
    angs = np.linspace(0, 555, 44)
    possis = [(3 * np.cos(t / 180 * np.pi), 3 * np.sin(t / 180 * np.pi), 1) for t in angs]
    sens = mpj.Sensor()
    sens.move((3, 0, 1))
    sens.rotate_from_angax(angs, "z", start=0, anchor=0)
    sens.rotate_from_angax(-angs, "z", start=0)

    b1 = pm.getB(possis)
    b2 = sens.getB(pm)

    np.testing.assert_allclose(b1, b2)


def test_sensor2():
    """self-consistent test of the sensor class"""
    pm = mpj.magnet.Cuboid(polarization=(11, 22, 33), dimension=(1, 2, 3))
    poz = np.linspace(0, 5, 33)
    poss1 = [(t, 0, 2) for t in poz]
    poss2 = [(t, 0, 3) for t in poz]
    poss3 = [(t, 0, 4) for t in poz]
    b1 = np.array([pm.getB(poss) for poss in [poss1, poss2, poss3]])
    b1 = np.swapaxes(b1, 0, 1)

    sens = mpj.Sensor(pixel=[(0, 0, 2), (0, 0, 3), (0, 0, 4)])
    sens.move([(t, 0, 0) for t in poz], start=0)
    b2 = sens.getB(pm)

    np.testing.assert_allclose(b1, b2)


def test_Sensor_getB_specs():
    """test input of sens getB"""
    sens1 = mpj.Sensor(pixel=(4, 4, 4))
    pm1 = mpj.magnet.Cylinder(polarization=(111, 222, 333), dimension=(1, 2))

    b1 = sens1.getB(pm1)
    b2 = mpj.getB(pm1, sens1)
    np.testing.assert_allclose(b1, b2)


def test_Sensor_squeeze():
    """testing squeeze output"""
    src = mpj.magnet.Sphere(polarization=(1, 1, 1), diameter=1)
    sensor = mpj.Sensor(pixel=[(1, 2, 3), (1, 2, 3)])
    b = sensor.getB(src)
    assert b.shape == (2, 3)
    h = sensor.getH(src)
    assert h.shape == (2, 3)

    b = sensor.getB(src, squeeze=False)
    assert b.shape == (1, 1, 1, 2, 3)
    h = sensor.getH(src, squeeze=False)
    assert h.shape == (1, 1, 1, 2, 3)


def test_repr():
    """test __repr__"""
    sens = mpj.Sensor()
    assert repr(sens)[:6] == "Sensor", "Sensor repr failed"


def test_pixel1():
    """
    squeeze=False Bfield minimal shape is (1, 1, 1, 1, 3)
    logic: single sensor, scalar path, single source all generate
    1 for squeeze=False Bshape. Bare pixel should do the same
    """
    src = mpj.misc.Dipole(moment=(1, 2, 3))

    np.testing.assert_allclose(
        src.getB(mpj.Sensor(pixel=(1, 2, 3)), squeeze=False).shape,
        (1, 1, 1, 1, 3),
    )

    src = mpj.misc.Dipole(moment=(1, 2, 3))
    np.testing.assert_allclose(
        src.getB(mpj.Sensor(pixel=[(1, 2, 3)]), squeeze=False).shape,
        (1, 1, 1, 1, 3),
    )

    np.testing.assert_allclose(
        src.getB(mpj.Sensor(pixel=[[(1, 2, 3)]]), squeeze=False).shape,
        (1, 1, 1, 1, 1, 3),
    )


def test_pixel2():
    """
    Sensor(pixel=pos_vec).pixel should always return pos_vec
    """

    p0 = (1, 2, 3)
    p1 = [(1, 2, 3)]
    p2 = [[(1, 2, 3)]]

    for pos_vec in [p0, p1, p2]:
        np.testing.assert_allclose(
            mpj.Sensor(pixel=pos_vec).pixel,
            pos_vec,
        )


def test_pixel3():
    """
    There should be complete equivalence between pos_vec and
    Sensor(pixel=pos_vec) inputs
    """
    src = mpj.misc.Dipole(moment=(1, 2, 3))

    p0 = (1, 2, 3)
    p1 = [(1, 2, 3)]
    p2 = [[(1, 2, 3)]]
    for pos_vec in [p0, p1, p2]:
        np.testing.assert_allclose(
            src.getB(mpj.Sensor(pixel=pos_vec), squeeze=False),
            src.getB(pos_vec, squeeze=False),
        )


def test_Sensor_centroid_no_pixels():
    """Test Sensor centroid without pixels - should return position"""
    expected = (12, 13, 14)
    sensor = mpj.Sensor(position=expected)
    assert np.allclose(sensor.centroid, expected)


def test_Sensor_centroid_with_pixels():
    """Test Sensor centroid with pixels - should return position + mean(pixels)"""
    position = (12, 13, 14)
    pixels = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    expected = (12.333333, 13.333333, 14.0)
    sensor = mpj.Sensor(position=position, pixel=pixels)
    assert np.allclose(sensor.centroid, expected)
