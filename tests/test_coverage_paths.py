"""Behaviour/physics tests exercising public-API code paths that the core
parity suite leaves uncovered: the eager reference evaluator (callable
``pixel_agg``, dataframe output, left-handed sensors, pairwise observers),
``in_out`` inside/outside branches of the meshed kernels, source-preparation
chunking/cache/error branches, ``TriangularMesh`` geometry + validation, and
``BaseGeo``/``Collection`` motion, styling and describe helpers.

Assertions compare against magpylib where available, or against known physical
invariants (J = polarization inside a magnet and 0 outside, superposition,
handedness sign flips) and exact shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

import magpylib_jax as mj
from magpylib_jax import getB, getH, getJ, getM
from magpylib_jax.core.base import MagpylibBadUserInput, MagpylibMissingInput

MU0 = 1.25663706127e-6

# Two-pixel observer grid so a reducing ``pixel_agg`` has something to reduce.
OBS2 = np.array([[0.4, -0.2, 0.35], [0.15, 0.25, -0.3]])
OBS1 = np.array([0.4, -0.2, 0.35])


# ---------------------------------------------------------------------------
# Eager reference path: callable pixel_agg routes getB/H/J/M through
# _compute_field_legacy -> _evaluate_core_field for every source family.
# ---------------------------------------------------------------------------
def _mean_agg(arr, axis=None):
    return np.mean(np.asarray(arr), axis=axis)


MAGNET_KWARGS = {
    "cuboid": {"dimension": (1.0, 1.0, 1.0), "polarization": (0.0, 0.0, 1.0)},
    "cylinder": {"dimension": (1.0, 1.0), "polarization": (0.0, 0.0, 1.0)},
    "sphere": {"diameter": 1.0, "polarization": (0.0, 0.0, 1.0)},
    "tetrahedron": {
        "vertices": np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        ),
        "polarization": (0.0, 0.0, 1.0),
    },
    "triangularmesh": {
        "mesh": np.array(
            [[(0, 0, 0), (1, 0, 0), (0, 1, 0)]], dtype=float
        ),
        "polarization": (0.0, 0.0, 1.0),
    },
}


@pytest.mark.parametrize("stype", list(MAGNET_KWARGS))
@pytest.mark.parametrize("getter", [getB, getH, getJ, getM])
def test_eager_callable_pixel_agg_magnets(stype, getter):
    """Callable pixel_agg forces the eager path for B/H/J/M of every magnet."""
    out = getter(stype, OBS2, pixel_agg=_mean_agg, **MAGNET_KWARGS[stype])
    assert np.asarray(out).shape[-1] == 3
    assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.parametrize(
    "stype,kw",
    [
        ("dipole", {"moment": (0.0, 0.0, 1.0)}),
        ("circle", {"diameter": 1.0, "current": 2.0}),
        ("polyline", {
            "segment_start": np.array([[0.0, 0.0, 0.0]]),
            "segment_end": np.array([[1.0, 0.0, 0.0]]),
            "current": 3.0,
        }),
        ("trianglesheet", {
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            "faces": np.array([[0, 1, 2]]),
            "current_densities": np.array([[0.0, 0.0, 1.0]]),
        }),
        ("trianglestrip", {
            "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            "current": 1.5,
        }),
    ],
)
@pytest.mark.parametrize("getter", [getB, getH, getJ, getM])
def test_eager_callable_pixel_agg_currents(stype, kw, getter):
    """Currents have no J/M; the eager path returns zeros there, finite B/H."""
    out = np.asarray(getter(stype, OBS2, pixel_agg=_mean_agg, **kw))
    assert out.shape[-1] == 3
    assert np.all(np.isfinite(out))
    if getter in (getJ, getM):
        assert np.allclose(out, 0.0)


@pytest.mark.parametrize(
    "stype",
    ["dipole", "circle", "cuboid", "cylinder", "cylindersegment", "sphere",
     "polyline", "trianglesheet", "triangularmesh", "tetrahedron"],
)
def test_eager_missing_input_raises(stype):
    """Missing excitation on the eager path raises MagpylibMissingInput."""
    with pytest.raises(MagpylibMissingInput):
        getB(stype, OBS2, pixel_agg=_mean_agg)


def test_eager_callable_pixel_agg_cylinder_segment():
    out = np.asarray(
        getB("cylindersegment", OBS2, pixel_agg=_mean_agg,
             dimension=(0.5, 1.0, 1.0, 0.0, 90.0), polarization=(0.0, 0.0, 1.0))
    )
    assert out.shape[-1] == 3 and np.all(np.isfinite(out))
    for getter in (getH, getJ, getM):
        o = np.asarray(getter("cylindersegment", OBS2, pixel_agg=_mean_agg,
                              dimension=(0.5, 1.0, 1.0, 0.0, 90.0),
                              polarization=(0.0, 0.0, 1.0)))
        assert o.shape[-1] == 3


def test_eager_dataframe_and_left_handed_and_nonuniform():
    pd = pytest.importorskip("pandas")
    src = mj.Circle(current=1.0, diameter=1.0)
    # dataframe output via callable pixel_agg (eager path).
    df = getB(src, OBS2, pixel_agg=_mean_agg, output="dataframe")
    assert isinstance(df, pd.DataFrame)
    assert {"Bx", "By", "Bz"}.issubset(df.columns)

    # Non-uniform pixel grids across sensors require pixel_agg (eager path).
    s1 = mj.Sensor(pixel=np.zeros((2, 3)))
    s2 = mj.Sensor(pixel=np.zeros((3, 3)))
    out = np.asarray(getB(src, [s1, s2], pixel_agg=_mean_agg))
    assert out.shape[-1] == 3

    # Left-handed sensor flips the x component of the field vs a right one.
    right = mj.Sensor(pixel=(0.2, 0.1, 0.3))
    left = mj.Sensor(pixel=(0.2, 0.1, 0.3))
    left.handedness = "left"
    br = np.asarray(getB(src, right, pixel_agg=_mean_agg)).reshape(-1, 3)
    bl = np.asarray(getB(src, left, pixel_agg=_mean_agg)).reshape(-1, 3)
    assert np.allclose(bl[:, 0], -br[:, 0])
    assert np.allclose(bl[:, 1:], br[:, 1:])


# ---------------------------------------------------------------------------
# in_out branches of the meshed kernels via object getJ/getH/getM.
# J = polarization inside a uniformly polarized magnet, 0 outside.
# ---------------------------------------------------------------------------
def test_tetrahedron_in_out_jfield():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    pol = np.array([0.0, 0.0, 1.0])
    tet = mj.Tetrahedron(vertices=verts, polarization=pol)
    inside_pt = np.array([0.2, 0.2, 0.2])
    outside_pt = np.array([2.0, 2.0, 2.0])

    assert np.allclose(np.asarray(tet.getJ(inside_pt)), pol)
    assert np.allclose(np.asarray(tet.getJ(outside_pt)), 0.0)
    # Forced modes ignore geometry.
    assert np.allclose(np.asarray(tet.getJ(outside_pt, in_out="inside")), pol)
    assert np.allclose(np.asarray(tet.getJ(inside_pt, in_out="outside")), 0.0)
    # H and M finite in both modes.
    assert np.all(np.isfinite(np.asarray(tet.getH(inside_pt, in_out="inside"))))
    assert np.all(np.isfinite(np.asarray(tet.getM(inside_pt))))


def test_tetrahedron_multi_cell_pairwise():
    """A per-observer stack of tetrahedra evaluates each obs against its cell."""
    from magpylib_jax.core.kernels import tetrahedron_bfield, tetrahedron_jfield

    v1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    v2 = v1 + np.array([5.0, 0.0, 0.0])
    pol = np.array([0.0, 0.0, 1.0])
    obs = np.array([[0.3, 0.3, 0.3], [5.3, 0.3, 0.3]])  # each inside its own cell

    batched = np.asarray(
        tetrahedron_bfield(obs, np.stack([v1, v2]), pol, in_out="auto")
    )
    e0 = np.asarray(tetrahedron_bfield(obs[0][None], v1, pol))[0]
    e1 = np.asarray(tetrahedron_bfield(obs[1][None], v2, pol))[0]
    assert batched.shape == (2, 3)
    assert np.allclose(batched, np.stack([e0, e1]), atol=1e-9)
    # Both observers sit inside their cell, so J = polarization.
    jbatched = np.asarray(tetrahedron_jfield(obs, np.stack([v1, v2]), pol))
    assert np.allclose(jbatched, pol)


def test_cylinder_segment_in_out_jfield():
    dim = (0.5, 1.0, 1.0, 0.0, 90.0)
    pol = np.array([0.0, 0.0, 1.0])
    cs = mj.CylinderSegment(dimension=dim, polarization=pol)
    inside_pt = np.array([0.55, 0.2, 0.0])  # r~0.6 in [0.5,1], phi~20deg in [0,90]
    outside_pt = np.array([3.0, 0.0, 0.0])
    assert np.allclose(np.asarray(cs.getJ(inside_pt)), pol)
    assert np.allclose(np.asarray(cs.getJ(outside_pt)), 0.0)
    assert np.all(np.isfinite(np.asarray(cs.getH(inside_pt, in_out="inside"))))
    assert np.all(np.isfinite(np.asarray(cs.getM(outside_pt, in_out="outside"))))


def test_triangular_mesh_in_out_and_4d_mesh():
    # Closed tetrahedron-surface mesh.
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    pol = np.array([0.0, 0.0, 1.0])
    tm = mj.TriangularMesh(vertices=verts, faces=faces, polarization=pol,
                           check_open="skip")
    inside_pt = np.array([0.2, 0.2, 0.2])
    outside_pt = np.array([2.0, 2.0, 2.0])
    assert np.allclose(np.asarray(tm.getJ(inside_pt)), pol)
    assert np.allclose(np.asarray(tm.getJ(outside_pt)), 0.0)
    assert np.all(np.isfinite(np.asarray(tm.getH(inside_pt, in_out="inside"))))
    assert np.all(np.isfinite(np.asarray(tm.getM(outside_pt, in_out="outside"))))

    # 4-D mesh (n_obs, n_faces, 3, 3) routes through the eager legacy path.
    mesh4 = np.broadcast_to(verts[faces][None], (2, faces.shape[0], 3, 3))
    out = np.asarray(getB("triangularmesh", OBS2, mesh=np.asarray(mesh4),
                          polarization=pol))
    assert out.shape[-1] == 3
    outj = np.asarray(getJ("triangularmesh", OBS2, mesh=np.asarray(mesh4),
                           polarization=pol, in_out="outside"))
    assert np.allclose(outj, 0.0)


# ---------------------------------------------------------------------------
# TriangularMesh geometry, validation and constructors.
# ---------------------------------------------------------------------------
def test_triangular_mesh_from_convexhull_and_geometry():
    pytest.importorskip("scipy")
    # Unit cube corners.
    pts = np.array(list(np.ndindex(2, 2, 2)), dtype=float)
    tm = mj.TriangularMesh.from_ConvexHull(pts, polarization=(0.0, 0.0, 1.0))
    # Cube volume == 1, barycenter at centre.
    assert abs(float(tm.volume) - 1.0) < 1e-9
    assert np.allclose(np.asarray(tm.barycenter), 0.5, atol=1e-9)
    assert np.allclose(np.asarray(tm.centroid), 0.5, atol=1e-9)


def test_triangular_mesh_validation_errors():
    good_faces = np.array([[0, 1, 2]])
    with pytest.raises(ValueError):
        mj.TriangularMesh(vertices=np.zeros((3, 2)), faces=good_faces,
                          polarization=(0, 0, 1))
    with pytest.raises(ValueError):
        mj.TriangularMesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1]]),
                          polarization=(0, 0, 1))
    with pytest.raises(ValueError):
        mj.TriangularMesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 9]]),
                          polarization=(0, 0, 1))
    with pytest.raises(ValueError):
        mj.TriangularMesh(vertices=np.zeros((3, 3)), faces=good_faces,
                          polarization=(0, 0, 1), in_out="sideways")
    with pytest.raises(ValueError):
        mj.TriangularMesh(vertices=np.zeros((3, 3)), faces=good_faces,
                          polarization=(0, 0, 1), check_open="bogus")


def test_triangular_mesh_open_mesh_warn_and_raise():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]])
    with pytest.warns(UserWarning):
        mj.TriangularMesh(vertices=verts, faces=faces, polarization=(0, 0, 1),
                          check_open="warn")
    with pytest.raises(ValueError):
        mj.TriangularMesh(vertices=verts, faces=faces, polarization=(0, 0, 1),
                          check_open="raise")


def test_triangular_mesh_magnetization_and_missing():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    tm = mj.TriangularMesh(vertices=verts, faces=faces,
                           magnetization=(0.0, 0.0, 1.0 / MU0), check_open="skip")
    assert np.allclose(np.asarray(tm._polarization), (0.0, 0.0, 1.0), atol=1e-6)

    tm_bad = mj.TriangularMesh(check_open="skip")
    with pytest.raises(MagpylibMissingInput):
        tm_bad.getB(OBS1)
    empty = mj.TriangularMesh(check_open="skip")
    assert np.allclose(np.asarray(empty.barycenter), 0.0)
    assert float(empty.volume) == 0.0


# ---------------------------------------------------------------------------
# Source preparation: object specs, chunking, sensor errors.
# ---------------------------------------------------------------------------
def test_object_getb_all_source_types_build_specs():
    """One getB per source object exercises _build_source_specs per type."""
    verts_tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    mesh_faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    mesh_verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    sources = [
        mj.Dipole(moment=(0, 0, 1)),
        mj.Circle(current=1.0, diameter=1.0),
        mj.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1)),
        mj.Cylinder(dimension=(1, 1), polarization=(0, 0, 1)),
        mj.CylinderSegment(dimension=(0.5, 1, 1, 0, 90), polarization=(0, 0, 1)),
        mj.Sphere(diameter=1.0, polarization=(0, 0, 1)),
        mj.misc.Triangle(vertices=verts_tri, polarization=(0, 0, 1)),
        mj.Polyline(current=1.0, vertices=[(0, 0, 0), (1, 0, 0)]),
        mj.TriangleSheet(vertices=verts_tri, faces=[(0, 1, 2)],
                         current_densities=[(0, 0, 1)]),
        mj.TriangleStrip(vertices=verts_tri, current=1.0),
        mj.TriangularMesh(vertices=mesh_verts, faces=mesh_faces,
                          polarization=(0, 0, 1), check_open="skip"),
        mj.Tetrahedron(vertices=mesh_verts, polarization=(0, 0, 1)),
    ]
    for src in sources:
        out = np.asarray(src.getB(OBS1))
        assert out.shape == (3,)


def test_magnet_magnetization_branch_and_missing():
    """The magnetization -> polarization branch and the missing-input raise."""
    for cls, kw in [
        (mj.Cuboid, {"dimension": (1, 1, 1)}),
        (mj.Cylinder, {"dimension": (1, 1)}),
        (mj.Sphere, {"diameter": 1.0}),
        (mj.Tetrahedron, {"vertices": np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)}),
    ]:
        src = cls(magnetization=(0.0, 0.0, 1.0 / MU0), **kw)
        assert np.allclose(np.asarray(src._polarization), (0, 0, 1), atol=1e-6)
        assert np.all(np.isfinite(np.asarray(src.getB(OBS1))))
        with pytest.raises(MagpylibMissingInput):
            _ = cls(**kw)._polarization


def test_source_chunking_many_circles():
    """Many sources trigger the source-chunking/padding path; sum is additive."""
    srcs = [mj.Circle(current=1.0, diameter=1.0, position=(0.1 * i, 0, 0))
            for i in range(10)]
    summed = np.asarray(getB(srcs, OBS1, sumup=True))
    each = np.stack([np.asarray(getB(s, OBS1)) for s in srcs])
    assert summed.shape == (3,)
    assert np.allclose(summed, each.sum(axis=0), atol=1e-9)


def test_source_prep_cache_reuse():
    src = mj.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    a = np.asarray(getB(src, OBS2))
    b = np.asarray(getB(src, OBS2))
    assert np.allclose(a, b)


def test_observer_and_source_errors():
    src = mj.Dipole(moment=(0, 0, 1))
    with pytest.raises(MagpylibBadUserInput):
        getB(src, None)
    with pytest.raises(MagpylibBadUserInput):
        getB(src, "not-an-observer")
    with pytest.raises(MagpylibBadUserInput):
        getB(src, mj.Collection())  # collection with no sensors
    with pytest.raises((MagpylibBadUserInput, ValueError, TypeError)):
        getB(object(), OBS1)


def test_mismatched_pixels_without_agg_raises():
    src = mj.Circle(current=1.0, diameter=1.0)
    s1 = mj.Sensor(pixel=np.zeros((2, 3)))
    s2 = mj.Sensor(pixel=np.zeros((3, 3)))
    with pytest.raises(MagpylibBadUserInput):
        getB(src, [s1, s2])


def test_api_invalid_output_and_pixel_agg():
    src = mj.Circle(current=1.0, diameter=1.0)
    with pytest.raises(ValueError):
        getB(src, OBS1, output="pandas")
    with pytest.raises(AttributeError):
        getB(src, OBS1, pixel_agg=123)


# ---------------------------------------------------------------------------
# BaseGeo motion, styling, describe, copy.
# ---------------------------------------------------------------------------
def test_basegeo_motion_variants():
    src = mj.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    src.move((1.0, 0.0, 0.0))
    assert np.allclose(np.asarray(src.position), (1, 0, 0))

    src.rotate_from_angax(90.0, "z")
    src.rotate_from_rotvec([0.0, 0.0, np.pi / 2], degrees=False)
    src.rotate_from_euler(30.0, "x")
    src.rotate_from_matrix(np.eye(3))
    src.rotate_from_mrp([0.0, 0.0, 0.1])
    src.rotate_from_quat([0.0, 0.0, 0.0, 1.0])
    # Rotation about an anchor moves the position.
    src2 = mj.Dipole(moment=(0, 0, 1), position=(1.0, 0.0, 0.0))
    src2.rotate_from_angax(180.0, "z", anchor=(0.0, 0.0, 0.0))
    assert np.allclose(np.asarray(src2.position), (-1, 0, 0), atol=1e-9)

    # Path motion with start index.
    src3 = mj.Dipole(moment=(0, 0, 1))
    src3.move(np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]))
    assert np.asarray(src3._position).shape[0] >= 3
    src3.reset_path()
    assert np.asarray(src3._position).shape[0] == 1


def test_basegeo_motion_input_errors():
    src = mj.Dipole(moment=(0, 0, 1))
    with pytest.raises(MagpylibBadUserInput):
        src.rotate_from_angax(90.0, "bad-axis")
    with pytest.raises(MagpylibBadUserInput):
        src.rotate_from_angax(90.0, [0.0, 0.0, 0.0])
    with pytest.raises(MagpylibBadUserInput):
        src.move((1, 0, 0), start="nope")


def test_basegeo_describe_style_copy_parent():
    src = mj.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1),
                    style_label="magnet")
    desc = src.describe(return_string=True)
    assert "Cuboid" in desc
    assert isinstance(src._repr_html_(), str)

    src.style.color = "red"
    assert src.style.color == "red"
    with pytest.raises(ValueError):
        src.style = 123

    cp = src.copy()
    assert cp is not src
    assert np.allclose(np.asarray(cp.position), np.asarray(src.position))

    col = mj.Collection()
    src.parent = col
    assert src.parent is col
    src.parent = None
    assert src.parent is None
    with pytest.raises(MagpylibBadUserInput):
        src.parent = "bad"


def test_basegeo_position_orientation_setters_with_children():
    col = mj.Collection()
    child = mj.Dipole(moment=(0, 0, 1), position=(1.0, 0.0, 0.0))
    col.add(child)
    col.position = (0.0, 1.0, 0.0)
    # Child moves rigidly with the parent.
    assert np.allclose(np.asarray(child.position), (1.0, 1.0, 0.0), atol=1e-9)
    col.orientation = None
    assert np.all(np.isfinite(np.asarray(child.position)))


# ---------------------------------------------------------------------------
# Collection behaviour.
# ---------------------------------------------------------------------------
def test_collection_getbh_and_describe():
    s1 = mj.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    s2 = mj.Dipole(moment=(0, 0, 1))
    col = mj.Collection(s1, s2)
    sens = mj.Sensor(pixel=(0.0, 0.0, 2.0))

    # Collection field == sum of member fields.
    b_col = np.asarray(col.getB(sens))
    b_sum = np.asarray(getB(s1, sens)) + np.asarray(getB(s2, sens))
    assert np.allclose(b_col, b_sum, atol=1e-9)
    for getter in ("getH", "getJ", "getM"):
        assert np.asarray(getattr(col, getter)(sens)).shape[-1] == 3

    assert len(col) == 2
    assert col[0] is s1
    assert list(iter(col))[1] is s2
    assert isinstance(col.describe(format="type+label", return_string=True), str)
    assert isinstance(
        col.describe(format="label,type,id,properties", return_string=True), str
    )
    assert isinstance(col._repr_html_(), str)


def test_collection_with_sensor_and_source_and_add():
    col = mj.Collection()
    col.add(mj.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1)))
    col.add(mj.Sensor(pixel=(0, 0, 1)))
    # Both sensors and sources present: getB() takes no extra inputs.
    out = np.asarray(col.getB())
    assert out.shape[-1] == 3
    with pytest.raises(MagpylibBadUserInput):
        col.getB(mj.Sensor(pixel=(0, 0, 1)))

    combined = col + mj.Dipole(moment=(0, 0, 1))
    assert isinstance(combined, mj.Collection)


def test_collection_volume_centroid_and_styles():
    a = mj.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1), position=(0, 0, 0))
    b = mj.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1), position=(2, 0, 0))
    col = mj.Collection(a, b)
    assert abs(float(col.volume) - 2.0) < 1e-9
    assert np.allclose(np.asarray(col.centroid), (1, 0, 0), atol=1e-9)
    col.set_children_styles(magnetization_show=False)
    assert a.style.magnetization.show is False
    with pytest.raises(ValueError):
        col.set_children_styles(bogus=True)

    # Empty collection falls back to its own position as centroid.
    empty = mj.Collection(position=(3.0, 0.0, 0.0))
    assert np.allclose(np.asarray(empty.centroid), (3, 0, 0))


def test_collection_add_remove_errors():
    col = mj.Collection()
    with pytest.raises(MagpylibBadUserInput):
        col.add(123)
    with pytest.raises(MagpylibBadUserInput):
        col.add(col)
    src = mj.Dipole(moment=(0, 0, 1))
    col.add(src)
    with pytest.raises(MagpylibBadUserInput):
        col.remove("nope")  # non-BaseGeo, errors="raise"
    with pytest.raises(MagpylibBadUserInput):
        col.remove(123, errors="weird")  # non-BaseGeo, invalid errors value
    col.remove(src)  # present -> removed
    absent = mj.Dipole(moment=(0, 0, 1))
    with pytest.raises(MagpylibBadUserInput):
        col.remove(absent, errors="raise")  # absent -> raise
    with pytest.raises(MagpylibBadUserInput):
        col.remove(absent, errors="weird")  # absent, invalid errors value
    col.remove(absent, errors="ignore")  # absent -> ignored


# ---------------------------------------------------------------------------
# Style helpers.
# ---------------------------------------------------------------------------
def test_functional_compat_squeeze_and_source_field():
    """Cover the squeeze/sumup output shaping and single-source dispatch used
    by the ``magpylib_jax.functional`` compatibility re-exports."""
    from magpylib_jax.functional import (
        _evaluate_source_field,
        _get_field_from_type,
    )

    obs = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]])

    # squeeze=False keeps the full (n_src, path, sensor, pix, 3) rank.
    out = _get_field_from_type(
        "dipole", obs, "B", moment=(0.0, 0.0, 1.0), squeeze=False, sumup=False
    )
    assert np.asarray(out).shape[-1] == 3
    out = _get_field_from_type(
        "dipole", obs, "B", moment=(0.0, 0.0, 1.0), squeeze=False, sumup=True
    )
    assert np.asarray(out).shape[-1] == 3

    # Pairwise observers (path axis) with squeeze=False.
    obs_pw = np.zeros((2, 1, 3))
    out = _get_field_from_type(
        "dipole", obs_pw, "B", moment=(1.0, 0.0, 0.0), squeeze=False
    )
    assert np.asarray(out).shape[-1] == 3

    # Single (non-list) source objects: with and without an ``in_out`` kwarg.
    cs = mj.CylinderSegment(dimension=(0.5, 1, 1, 0, 90), polarization=(0, 0, 1))
    field, n = _evaluate_source_field(cs, obs, "B", sumup=False, in_out="inside")
    assert n == 1 and np.asarray(field).shape[-1] == 3

    custom = mj.CustomSource(field_func=lambda o: np.zeros_like(np.asarray(o)))
    field, n = _evaluate_source_field(custom, obs, "B", sumup=False, in_out="auto")
    assert n == 1 and np.allclose(np.asarray(field), 0.0)
    # And as a one-element list (source without an ``in_out`` parameter).
    field, n = _evaluate_source_field([custom], obs, "B", sumup=True, in_out="auto")
    assert n == 1


def test_style_update_copy_and_invalid():
    from magpylib_jax.core.style import BaseStyle, SensorStyle

    st = BaseStyle(label="a", color="blue")
    st.update({"color": "green"}, magnetization_show=False)
    assert st.color == "green"
    assert st.magnetization.show is False
    cp = st.copy()
    assert cp.color == "green" and cp.label == "a"
    with pytest.raises(ValueError):
        st.update(nonexistent_prop=1)
    assert "SensorStyle(" in repr(SensorStyle(label="s"))
