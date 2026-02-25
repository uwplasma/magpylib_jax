import numpy as np

import magpylib_jax as mpj


def test_sensor_getB_specs() -> None:
    sens = mpj.Sensor(pixel=(4, 4, 4))
    src = mpj.magnet.Cylinder(polarization=(111, 222, 333), dimension=(1, 2))

    b1 = sens.getB(src)
    b2 = mpj.getB(src, sens)
    np.testing.assert_allclose(b1, b2)


def test_sensor_squeeze() -> None:
    src = mpj.magnet.Sphere(polarization=(1, 1, 1), diameter=1)
    sensor = mpj.Sensor(pixel=[(1, 2, 3), (1, 2, 3)])

    b = mpj.getB(src, sensor)
    assert b.shape == (2, 3)
    h = mpj.getH(src, sensor)
    assert h.shape == (2, 3)

    b = mpj.getB(src, sensor, squeeze=False)
    assert b.shape == (1, 1, 1, 2, 3)
    h = mpj.getH(src, sensor, squeeze=False)
    assert h.shape == (1, 1, 1, 2, 3)


def test_sensor_pixel_equivalence() -> None:
    src = mpj.misc.Dipole(moment=(1, 2, 3))
    p0 = (1, 2, 3)
    p1 = [(1, 2, 3)]
    p2 = [[(1, 2, 3)]]

    for pos_vec in [p0, p1, p2]:
        np.testing.assert_allclose(
            mpj.getB(src, mpj.Sensor(pixel=pos_vec), squeeze=False),
            mpj.getB(src, pos_vec, squeeze=False),
        )


def test_sensor_centroid() -> None:
    position = (12, 13, 14)
    pixels = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    expected = (12.333333, 13.333333, 14.0)
    sensor = mpj.Sensor(position=position, pixel=pixels)
    assert np.allclose(np.asarray(sensor.centroid), expected)


def test_sensor_repr() -> None:
    sens = mpj.Sensor(pixel=(1, 2, 3))
    assert repr(sens).startswith("Sensor")
