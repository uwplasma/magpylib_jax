import numpy as np

import magpylib_jax as mpj


def test_sphere_inside_field_is_two_thirds_polarization() -> None:
    pol = np.array([0.15, -0.21, 0.33])
    sph = mpj.magnet.Sphere(polarization=pol, diameter=2.0)
    b = np.asarray(sph.getB([0.2, 0.1, 0.0]))
    np.testing.assert_allclose(b, (2.0 / 3.0) * pol, rtol=0, atol=1e-12)


def test_dipole_far_field_scales_as_inverse_cube() -> None:
    src = mpj.misc.Dipole(moment=(0.0, 0.0, 1.0))
    b1 = np.linalg.norm(np.asarray(src.getB([0.0, 0.0, 2.0])))
    b2 = np.linalg.norm(np.asarray(src.getB([0.0, 0.0, 4.0])))
    np.testing.assert_allclose(b1 / b2, 8.0, rtol=2e-3)


def test_polyline_on_segment_returns_zero() -> None:
    src = mpj.current.Polyline(current=1.0, vertices=[(0, 0, 0), (0, 0, 1)])
    b = np.asarray(src.getB([0.0, 0.0, 0.5]))
    np.testing.assert_allclose(b, np.zeros(3), atol=1e-14)
