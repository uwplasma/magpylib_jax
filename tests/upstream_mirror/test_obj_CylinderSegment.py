import numpy as np

import magpylib_jax as mpj


def test_cylinder_segment_barycenter_and_volume() -> None:
    src = mpj.magnet.CylinderSegment(
        polarization=(0.1, 0.2, 0.3),
        dimension=(1.0, 2.0, 3.0, 0.0, 90.0),
    )
    expected_volume = (2.0**2 - 1.0**2) * np.pi * 3.0 * 90.0 / 360.0
    assert abs(src.volume - expected_volume) < 1e-10

    expected_bary = np.array([0.99029742, 0.99029742, 0.0])
    np.testing.assert_allclose(np.asarray(src.barycenter), expected_bary, rtol=1e-7, atol=1e-9)


def test_cylinder_segment_centroid_includes_position() -> None:
    src = mpj.magnet.CylinderSegment(
        polarization=(0.0, 0.0, 1.0),
        dimension=(1.0, 2.0, 3.0, -145.0, 145.0),
        position=(4.0, 5.0, 6.0),
    )
    expected = np.array([4.35255872, 5.0, 6.0])
    np.testing.assert_allclose(np.asarray(src.centroid), expected, rtol=1e-7, atol=1e-9)
