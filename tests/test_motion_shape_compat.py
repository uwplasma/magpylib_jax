import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R

import magpylib_jax as mpj


def test_source_path_shape_and_values_match_magpylib() -> None:
    observers = np.array([[0.2, 0.4, 0.8], [0.6, -0.2, 1.1]])

    src_ref = magpy.magnet.Sphere(
        polarization=(0.1, -0.2, 0.3),
        diameter=1.2,
        position=[(0, 0, 0), (0.2, -0.1, 0.3)],
    )
    src_new = mpj.magnet.Sphere(
        polarization=(0.1, -0.2, 0.3),
        diameter=1.2,
        position=[(0, 0, 0), (0.2, -0.1, 0.3)],
    )

    b_ref = src_ref.getB(observers)
    b_new = np.asarray(src_new.getB(observers))

    assert b_new.shape == b_ref.shape
    np.testing.assert_allclose(b_new, b_ref, rtol=3e-6, atol=1e-10)


def test_sensor_pixel_path_and_unsqueezed_layout_match_magpylib() -> None:
    sensor_ref = magpy.Sensor(
        pixel=np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]),
        position=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)],
    )
    sensor_new = mpj.Sensor(
        pixel=np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]),
        position=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)],
    )

    src_ref = magpy.misc.Dipole(moment=(0.3, -0.2, 0.1))
    src_new = mpj.misc.Dipole(moment=(0.3, -0.2, 0.1))

    out_ref = magpy.getB(src_ref, sensor_ref, squeeze=False)
    out_new = np.asarray(mpj.getB(src_new, sensor_new, squeeze=False))

    assert out_new.shape == out_ref.shape
    np.testing.assert_allclose(out_new, out_ref, rtol=1e-10, atol=1e-12)


def test_multisource_unsqueezed_layout_matches_magpylib() -> None:
    observers = np.array([[0.2, 0.1, 0.7], [0.3, -0.4, 1.1]])

    src_ref = [magpy.misc.Dipole(moment=(1.0, 0.0, 0.0)), magpy.misc.Dipole(moment=(0.0, 1.0, 0.0))]
    src_new = [mpj.misc.Dipole(moment=(1.0, 0.0, 0.0)), mpj.misc.Dipole(moment=(0.0, 1.0, 0.0))]

    out_ref = magpy.getB(src_ref, observers, squeeze=False, sumup=False)
    out_new = np.asarray(mpj.getB(src_new, observers, squeeze=False, sumup=False))

    assert out_new.shape == out_ref.shape
    np.testing.assert_allclose(out_new, out_ref, rtol=1e-10, atol=1e-12)


def test_path_length_broadcasting_matches_magpylib() -> None:
    src_ref = magpy.misc.Dipole(
        moment=(1.0, 0.0, 0.0),
        position=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)],
    )
    src_new = mpj.misc.Dipole(
        moment=(1.0, 0.0, 0.0),
        position=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)],
    )
    sensor_ref = magpy.Sensor(position=[(0.0, 0.0, 2.0), (0.0, 0.0, 2.5)])
    sensor_new = mpj.Sensor(position=[(0.0, 0.0, 2.0), (0.0, 0.0, 2.5)])

    out_ref = magpy.getB(src_ref, sensor_ref)
    out_new = np.asarray(mpj.getB(src_new, sensor_new))

    assert out_new.shape == out_ref.shape
    np.testing.assert_allclose(out_new, out_ref, rtol=1e-10, atol=1e-12)


def test_orientation_accepts_rotation_objects() -> None:
    rot = R.from_euler("z", 90, degrees=True)
    src_ref = magpy.current.Circle(current=1.0, diameter=1.0, orientation=rot)
    src_new = mpj.current.Circle(current=1.0, diameter=1.0, orientation=rot)

    observers = np.array([[0.3, 0.1, 0.5], [-0.2, 0.4, 0.8]])
    np.testing.assert_allclose(
        np.asarray(src_new.getB(observers)),
        src_ref.getB(observers),
        rtol=2e-5,
        atol=1e-10,
    )
