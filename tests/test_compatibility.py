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
    assert np.asarray(sensor.getH(col)).shape == (2, 3)
    assert np.asarray(sensor.getJ(col)).shape == (2, 3)
    assert np.asarray(sensor.getM(col)).shape == (2, 3)

    # exercise additive API coverage
    col2 = mpj.Collection()
    col2.add(src1).add(src2)
    _ = list(iter(col2))
    col3 = col + col2
    np.testing.assert_allclose(np.asarray(col3.getH(observers)).shape, (2, 3))
    np.testing.assert_allclose(np.asarray(col3.getJ(observers)).shape, (2, 3))
    np.testing.assert_allclose(np.asarray(col3.getM(observers)).shape, (2, 3))


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


def test_extended_geometry_properties() -> None:
    sph = mpj.magnet.Sphere(polarization=(0, 0, 0.1), diameter=2.0, position=(3, 4, 5))
    tri = mpj.misc.Triangle(
        vertices=[(-1, 0, 0), (1, -1, 0), (0, 1, 0)],
        polarization=(0, 0, 1),
        position=(10, 11, 12),
    )
    line = mpj.current.Polyline(
        vertices=[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
        current=1.0,
        position=(8, 9, 10),
    )
    tet = mpj.magnet.Tetrahedron(
        vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        polarization=(0, 0, 1),
        position=(5, 6, 7),
    )

    assert abs(sph.volume - ((4 / 3) * np.pi)) < 1e-12
    assert abs(tet.volume - (1 / 6)) < 1e-12
    np.testing.assert_allclose(np.asarray(sph.centroid), np.array([3.0, 4.0, 5.0]))
    np.testing.assert_allclose(np.asarray(tri.barycenter), np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(np.asarray(tri.centroid), np.array([10.0, 11.0, 12.0]))
    np.testing.assert_allclose(np.asarray(line.centroid), np.array([9.0, 9.0, 10.0]))
    np.testing.assert_allclose(np.asarray(tet.barycenter), np.array([0.25, 0.25, 0.25]))
    np.testing.assert_allclose(np.asarray(tet.centroid), np.array([5.25, 6.25, 7.25]))
