import numpy as np
import jax
import jax.numpy as jnp

import magpylib_jax as mpj
from magpylib_jax import functional as mpjf


def test_getB_jit_string_source():
    observers = jnp.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=jnp.float64)

    def field_at(pos):
        return mpj.getB(
            "cuboid",
            observers,
            dimension=(1.0, 2.0, 3.0),
            polarization=(0.2, -0.1, 0.3),
            position=pos,
        )

    pos = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
    out = field_at(pos)
    out_jit = jax.jit(field_at)(pos)
    np.testing.assert_allclose(np.asarray(out_jit), np.asarray(out), rtol=1e-8, atol=1e-10)


def test_getB_jit_object_source():
    src = mpj.Cylinder(dimension=(1.0, 2.0), polarization=(0.1, 0.0, 0.2))

    def field_at(obs):
        return src.getB(obs)

    obs = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64)
    out = field_at(obs)
    out_jit = jax.jit(field_at)(obs)
    np.testing.assert_allclose(np.asarray(out_jit), np.asarray(out), rtol=1e-8, atol=1e-10)


def test_getB_parity_with_legacy():
    src1 = mpj.Cuboid(dimension=(1.0, 1.5, 2.0), polarization=(0.1, -0.1, 0.2))
    src2 = mpj.Sphere(diameter=2.0, polarization=(0.2, 0.0, -0.1))
    sens = mpj.Sensor(pixel=[(0.0, 0.0, 0.0), (0.2, -0.1, 0.0)])

    out_new = mpj.getB([src1, src2], sens, squeeze=False)
    out_old = mpjf._compute_field_legacy([src1, src2], sens, "B", squeeze=False)
    np.testing.assert_allclose(np.asarray(out_new), np.asarray(out_old), rtol=1e-8, atol=1e-10)


def test_getB_parity_pixel_agg_mean():
    src = mpj.Dipole(moment=(0.2, 0.0, -0.1))
    sens1 = mpj.Sensor(pixel=[(0.0, 0.0, 0.0), (0.2, -0.1, 0.0)])
    sens2 = mpj.Sensor(pixel=(0.1, 0.1, 0.1))
    sensors = [sens1, sens2]

    out_new = mpj.getB(src, sensors, pixel_agg="mean", squeeze=False)
    out_old = mpjf._compute_field_legacy(src, sensors, "B", pixel_agg="mean", squeeze=False)
    np.testing.assert_allclose(np.asarray(out_new), np.asarray(out_old), rtol=1e-8, atol=1e-10)


def test_getB_parity_path_broadcast():
    src = mpj.Cuboid(dimension=(1.0, 1.0, 1.0), polarization=(0.1, 0.2, -0.1))
    src.position = [(0.0, 0.0, 0.0), (0.1, -0.1, 0.0)]
    sens = mpj.Sensor(pixel=(0.0, 0.0, 0.0))
    sens.position = [(0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.2, 0.0)]

    out_new = mpj.getB(src, sens, squeeze=False)
    out_old = mpjf._compute_field_legacy(src, sens, "B", squeeze=False)
    np.testing.assert_allclose(np.asarray(out_new), np.asarray(out_old), rtol=1e-8, atol=1e-10)
