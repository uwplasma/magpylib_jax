"""Drop-in compatibility: magpylib scripts should run under ``magpylib_jax``.

These tests import ``magpylib_jax as magpy`` and exercise the public API the way a
magpylib user would, asserting the code runs (no Attribute/TypeError) and — where
a magpylib reference is available — that results match. They guard the promise
that changing ``import magpylib as magpy`` to ``import magpylib_jax as magpy`` keeps
common field-computation code working.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import magpylib_jax as magpy  # the drop-in alias

magpylib = pytest.importorskip("magpylib")


# --- top-level surface --------------------------------------------------------
def test_toplevel_surface():
    assert magpy.mu_0 == pytest.approx(magpylib.mu_0)
    assert "matplotlib" in magpy.SUPPORTED_PLOTTING_BACKENDS
    for name in ["getB", "getH", "getJ", "getM", "getFT", "show", "Collection", "Sensor"]:
        assert hasattr(magpy, name), name
    for ns in ["magnet", "current", "misc"]:
        assert hasattr(magpy, ns)
    # defaults / show_context shims exist and don't crash
    assert magpy.defaults.display.backend
    with magpy.show_context():
        pass


# --- source construction (magpylib-identical kwargs) --------------------------
def test_construct_all_sources_magpylib_style():
    srcs = [
        magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1)),
        magpy.magnet.Cylinder(polarization=(0, 0, 1), dimension=(1, 1)),
        magpy.magnet.CylinderSegment(polarization=(0, 0, 1), dimension=(0.5, 1, 1, 0, 90)),
        magpy.magnet.Sphere(polarization=(0, 0, 1), diameter=1),
        magpy.magnet.Tetrahedron(
            polarization=(0, 0, 1), vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ),
        magpy.current.Circle(current=1, diameter=1),
        magpy.current.Polyline(current=1, vertices=[[0, 0, 0], [1, 0, 0]]),
        magpy.misc.Dipole(moment=(0, 0, 1)),
        magpy.misc.Triangle(polarization=(0, 0, 1), vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    ]
    # magnetization alias also works
    magpy.magnet.Cuboid(magnetization=(0, 0, 1e6), dimension=(1, 1, 1))
    for s in srcs:
        assert np.asarray(s.getB((2, 0, 0))).shape == (3,)


def test_getB_matches_magpylib():
    obs = np.random.default_rng(0).normal(size=(50, 3)) + np.array([1.5, 0, 0])
    kw = dict(polarization=(0.2, -0.1, 0.3), dimension=(1.0, 0.8, 1.2), position=(0.1, 0.2, -0.1))
    b_jax = np.asarray(magpy.magnet.Cuboid(**kw).getB(obs))
    b_ref = magpylib.magnet.Cuboid(**kw).getB(obs)
    np.testing.assert_allclose(b_jax, b_ref, rtol=2e-5, atol=5e-10)


# --- collections, sensors, sumup/squeeze/pixel_agg ---------------------------
def test_collection_sensor_workflow():
    c1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    c2 = magpy.misc.Dipole(moment=(0, 0, 1), position=(2, 0, 0))
    coll = magpy.Collection(c1, c2)
    coll.add(magpy.current.Circle(current=1, diameter=1, position=(0, 2, 0)))
    assert len(coll.sources) == 3

    sensor = magpy.Sensor(position=(1, 1, 1), pixel=[[0, 0, 0], [0.1, 0, 0]])
    b = coll.getB(sensor)
    assert np.asarray(b).shape[-1] == 3

    # sumup / squeeze / pixel_agg keywords behave like magpylib
    b_sum = magpy.getB([c1, c2], [[1, 0, 0], [0, 1, 0]], sumup=True)
    assert np.asarray(b_sum).shape[-1] == 3
    b_agg = c1.getB([[1, 0, 0], [1.1, 0, 0]], pixel_agg="mean")
    assert np.asarray(b_agg).shape[-1] == 3


# --- motion / paths -----------------------------------------------------------
def test_motion_api():
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    src.move((1, 0, 0))
    src.rotate_from_angax(45, "z")
    src.rotate_from_angax(np.linspace(0, 90, 5), "y", start=0)
    assert np.asarray(src.position).shape[-1] == 3
    src.reset_path()
    # magpylib-style getB with position/orientation path
    b = src.getB([[2, 0, 0], [0, 2, 0]])
    assert np.asarray(b).shape[-1] == 3


# --- getFT (magpylib_jax parity) ---------------------------------------------
def test_getFT_api_and_parity():
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    tgt = magpy.misc.Dipole(moment=(0, 0, 1), position=(2, 0, 0))
    F, T = magpy.getFT(src, tgt)
    assert np.asarray(F).shape == (3,) and np.asarray(T).shape == (3,)
    F_ref, T_ref = magpylib.getFT(
        magpylib.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1)),
        magpylib.misc.Dipole(moment=(0, 0, 1), position=(2, 0, 0)),
    )
    np.testing.assert_allclose(np.asarray(F), np.asarray(F_ref), rtol=1e-4, atol=1e-12)


# --- show (matplotlib) --------------------------------------------------------
def test_show_api():
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    fig = magpy.show(src, magpy.Sensor(position=(2, 0, 0)), return_fig=True)
    assert fig is not None
