import numpy as np

import magpylib_jax as mpj


def test_sensor_path_equivalence() -> None:
    pm = mpj.magnet.Cuboid(polarization=(11, 22, 33), dimension=(1, 2, 3))
    angs = np.linspace(0, 555, 44)
    possis = [
        (3 * np.cos(t / 180 * np.pi), 3 * np.sin(t / 180 * np.pi), 1) for t in angs
    ]

    sens = mpj.Sensor(position=possis)
    b1 = pm.getB(possis)
    b2 = sens.getB(pm)

    np.testing.assert_allclose(np.squeeze(b2), b1)


def test_sensor_path_with_pixels() -> None:
    pm = mpj.magnet.Cuboid(polarization=(11, 22, 33), dimension=(1, 2, 3))
    poz = np.linspace(0, 5, 33)
    poss1 = [(t, 0, 2) for t in poz]
    poss2 = [(t, 0, 3) for t in poz]
    poss3 = [(t, 0, 4) for t in poz]
    b1 = np.array([pm.getB(poss) for poss in [poss1, poss2, poss3]])
    b1 = np.swapaxes(b1, 0, 1)

    sens = mpj.Sensor(
        pixel=[(0, 0, 2), (0, 0, 3), (0, 0, 4)],
        position=[(t, 0, 0) for t in poz],
    )
    b2 = sens.getB(pm)

    np.testing.assert_allclose(np.asarray(b2), b1)


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


def test_sensor_pixel_shapes() -> None:
    src = mpj.misc.Dipole(moment=(1, 2, 3))
    np.testing.assert_allclose(
        mpj.getB(src, mpj.Sensor(pixel=(1, 2, 3)), squeeze=False).shape,
        (1, 1, 1, 3),
    )
    np.testing.assert_allclose(
        mpj.getB(src, mpj.Sensor(pixel=[(1, 2, 3)]), squeeze=False).shape,
        (1, 1, 1, 3),
    )
    np.testing.assert_allclose(
        mpj.getB(src, mpj.Sensor(pixel=[[(1, 2, 3)]]), squeeze=False).shape,
        (1, 1, 1, 3),
    )


def test_sensor_pixel_passthrough() -> None:
    p0 = (1, 2, 3)
    p1 = [(1, 2, 3)]
    p2 = [[(1, 2, 3)]]
    for pos_vec in [p0, p1, p2]:
        np.testing.assert_allclose(mpj.Sensor(pixel=pos_vec).pixel, pos_vec)


def test_sensor_centroid_no_pixels() -> None:
    position = (12, 13, 14)
    sensor = mpj.Sensor(position=position)
    assert np.allclose(np.asarray(sensor.centroid), position)


def test_sensor_centroid() -> None:
    position = (12, 13, 14)
    pixels = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    expected = (12.333333, 13.333333, 14.0)
    sensor = mpj.Sensor(position=position, pixel=pixels)
    assert np.allclose(np.asarray(sensor.centroid), expected)


def test_sensor_repr() -> None:
    sens = mpj.Sensor(pixel=(1, 2, 3))
    assert repr(sens).startswith("Sensor")
