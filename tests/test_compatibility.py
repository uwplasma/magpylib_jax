import numpy as np

import magpylib_jax as mpj


def test_collection_and_sensor_additive_behavior() -> None:
    observers = np.array([[0.2, -0.1, 0.6], [0.4, 0.3, -0.2]])
    sensor = mpj.Sensor(pixel=observers)

    src1 = mpj.current.Circle(current=1.1, diameter=0.9)
    src2 = mpj.misc.Dipole(moment=(0.0, 0.0, 1.2))

    col = mpj.Collection([src1, src2])
    b_sensor = np.asarray(sensor.getB(col))
    b_sum = np.asarray(src1.getB(observers) + src2.getB(observers))

    np.testing.assert_allclose(b_sensor, b_sum)


def test_functional_dispatch_accepts_object_and_aliases() -> None:
    obs = np.array([0.1, 0.2, 0.3])
    src = mpj.magnet.Cuboid(polarization=(0.1, -0.2, 0.3), dimension=(1.0, 2.0, 3.0))

    b_obj = np.asarray(mpj.getB(src, obs))
    b_alias = np.asarray(
        mpj.getB("box", obs, polarization=(0.1, -0.2, 0.3), dimension=(1.0, 2.0, 3.0))
    )

    np.testing.assert_allclose(b_obj, b_alias)


def test_geometry_properties_present() -> None:
    cub = mpj.magnet.Cuboid(polarization=(0, 0, 0.1), dimension=(2.0, 3.0, 4.0), position=(1, 2, 3))
    cyl = mpj.magnet.Cylinder(
        polarization=(0, 0, 0.2),
        dimension=(2.0, 4.0),
        position=(4, 5, 6),
    )

    assert abs(cub.volume - 24.0) < 1e-12
    assert abs(cyl.volume - (np.pi * 1.0**2 * 4.0)) < 1e-12
    np.testing.assert_allclose(np.asarray(cub.centroid), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(cyl.centroid), np.array([4.0, 5.0, 6.0]))
