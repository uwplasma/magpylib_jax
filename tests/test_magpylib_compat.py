"""Drop-in compatibility: magpylib scripts should run under ``magpylib_jax``.

These tests import ``magpylib_jax as magpy`` and exercise the public API the way a
magpylib user would, asserting the code runs (no Attribute/TypeError) and — where
a magpylib reference is available — that results match. They guard the promise
that changing ``import magpylib as magpy`` to ``import magpylib_jax as magpy`` keeps
common field-computation code working.
"""

import warnings

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


# --- collection recursive accessors (magpylib parity) ------------------------
def test_collection_recursive_accessors():
    inner = magpy.Collection(
        magpy.misc.Dipole(moment=(0, 0, 1)), magpy.Sensor(position=(1, 0, 0))
    )
    outer = magpy.Collection(
        magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1)), inner
    )
    assert len(outer.sources_all) == 2  # cuboid + inner dipole
    assert len(outer.sensors_all) == 1
    assert inner in outer.collections
    assert inner in outer.collections_all
    assert set(outer.children_all) >= set(inner.children)


def test_getFT_accepts_collection_source():
    coll = magpy.Collection(
        magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1)),
        magpy.misc.Dipole(moment=(0, 0, 1), position=(0, 1, 0)),
    )
    tgt = magpy.misc.Dipole(moment=(0, 0, 1), position=(2, 0, 0))
    F_coll, T_coll = magpy.getFT(coll, tgt)
    F_list, T_list = magpy.getFT(list(coll.sources), tgt)
    # a Collection source sums to the same force as its flattened leaf list
    np.testing.assert_allclose(
        np.asarray(F_coll), np.asarray(F_list).sum(axis=0), rtol=1e-9, atol=1e-14
    )


# --- core capitalized kernel aliases (magpylib parity) -----------------------
def test_core_capitalized_aliases():
    obs = np.array([[0.2, 0.3, 0.4], [1.0, -0.5, 0.2]])
    # importable and identical to the lowercase kernel
    assert magpy.core.magnet_cuboid_Bfield is magpy.core.magnet_cuboid_bfield
    b_cap = np.asarray(magpy.core.magnet_cuboid_Bfield(obs, (1.0, 1.0, 1.0), (0, 0, 1.0)))
    b_low = np.asarray(magpy.core.magnet_cuboid_bfield(obs, (1.0, 1.0, 1.0), (0, 0, 1.0)))
    np.testing.assert_array_equal(b_cap, b_low)
    # the full set of magpylib-compatible names exists
    for name in [
        "magnet_cuboid_Bfield",
        "magnet_sphere_Bfield",
        "dipole_Hfield",
        "current_circle_Hfield",
        "current_polyline_Hfield",
        "triangle_Bfield",
        "magnet_cylinder_axial_Bfield",
        "magnet_cylinder_diametral_Hfield",
        "magnet_cylinder_segment_Hfield",
        "current_sheet_Hfield",
    ]:
        assert callable(getattr(magpy.core, name)), name


# --- Triangle magnetization kwarg (magpylib parity) --------------------------
def test_triangle_magnetization_kwarg():
    verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    tri_pol = magpy.misc.Triangle(polarization=(0, 0, 1.0), vertices=verts)
    tri_mag = magpy.misc.Triangle(magnetization=(0, 0, 1.0 / magpy.mu_0), vertices=verts)
    obs = np.array([2.0, 0.5, 0.3])
    np.testing.assert_allclose(
        np.asarray(tri_mag.getB(obs)), np.asarray(tri_pol.getB(obs)), rtol=1e-9, atol=1e-14
    )


# --- Collection override_parent constructor param ----------------------------
def test_collection_override_parent():
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    first = magpy.Collection(src)
    assert src in first.sources
    # re-adopting into a new collection without override_parent raises
    with pytest.raises(magpy.MagpylibBadUserInput):
        magpy.Collection(src)
    # override_parent=True steals the child from the first collection
    second = magpy.Collection(src, override_parent=True)
    assert src in second.sources
    assert src not in first.sources


# --- getFT return_mesh / meshreport (magpylib parity) ------------------------
def test_getFT_return_mesh_and_meshreport(capsys):
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    tgt = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1), position=(2, 0, 0))
    tgt.meshing = 8
    meshes = magpy.getFT(src, tgt, return_mesh=True)
    assert isinstance(meshes, list) and len(meshes) == 1
    assert "pts" in meshes[0] and "moments" in meshes[0]
    assert np.asarray(meshes[0]["pts"]).shape[1] == 3
    assert np.asarray(meshes[0]["pts"]).shape[0] >= 1

    # current target -> cvecs key
    coil = magpy.current.Circle(current=1.0, diameter=2.0, position=(0, 0, 1))
    coil.meshing = 20
    cmesh = magpy.getFT(src, coil, return_mesh=True)
    assert "cvecs" in cmesh[0]

    # meshreport prints a per-target line; default return stays (F, T)
    F, T = magpy.getFT(src, tgt, meshreport=True)
    out = capsys.readouterr().out
    assert "Cuboid" in out
    assert np.asarray(F).shape == (3,) and np.asarray(T).shape == (3,)


# --- TriangularMesh check_disconnected / check_selfintersecting ---------------
def _two_tetrahedra():
    v1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    v2 = v1 + np.array([5.0, 0.0, 0.0])
    verts = np.vstack([v1, v2])
    faces1 = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    faces = np.vstack([faces1, faces1 + 4])
    return verts, faces


def test_trimesh_check_disconnected_warns():
    verts, faces = _two_tetrahedra()
    with pytest.warns(UserWarning, match="Disconnected"):
        mesh = magpy.magnet.TriangularMesh(
            vertices=verts, faces=faces, polarization=(0, 0, 1), check_open="skip"
        )
    assert mesh.status_disconnected is True
    assert len(mesh.status_disconnected_data) == 2


def test_trimesh_check_disconnected_connected_ok():
    # single closed tetrahedron: connected, no disconnected warning
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mesh = magpy.magnet.TriangularMesh(vertices=verts, faces=faces, polarization=(0, 0, 1))
    assert mesh.status_disconnected is False


def test_trimesh_check_kwargs_accept_modes():
    verts, faces = _two_tetrahedra()
    # 'raise' turns the disconnected detection into an error
    with pytest.raises(ValueError, match="Disconnected"):
        magpy.magnet.TriangularMesh(
            vertices=verts,
            faces=faces,
            polarization=(0, 0, 1),
            check_open="skip",
            check_disconnected="raise",
        )
    # check_selfintersecting kwarg is accepted; invalid mode is rejected
    magpy.magnet.TriangularMesh(
        vertices=verts,
        faces=faces,
        polarization=(0, 0, 1),
        check_open="skip",
        check_disconnected="skip",
        check_selfintersecting="warn",
    )
    with pytest.raises(ValueError, match="check_selfintersecting"):
        magpy.magnet.TriangularMesh(
            vertices=verts,
            faces=faces,
            polarization=(0, 0, 1),
            check_open="skip",
            check_disconnected="skip",
            check_selfintersecting="nonsense",
        )


# --- func subpackage surface -------------------------------------------------
def test_func_subpackage_surface():
    assert hasattr(magpy, "func")
    for name in [
        "circle_field",
        "polyline_field",
        "cuboid_field",
        "cylinder_field",
        "cylinder_segment_field",
        "sphere_field",
        "tetrahedron_field",
        "dipole_field",
        "triangle_charge_field",
        "triangle_current_field",
    ]:
        assert callable(getattr(magpy.func, name)), name


# --- show (matplotlib) --------------------------------------------------------
def test_show_api():
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    fig = magpy.show(src, magpy.Sensor(position=(2, 0, 0)), return_fig=True)
    assert fig is not None
