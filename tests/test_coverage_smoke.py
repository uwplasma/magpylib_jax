import numpy as np
import pytest

import jax
import jax.numpy as jnp

import magpylib_jax as mpj
from magpylib_jax import functional as functional
from magpylib_jax.collection import Collection
from magpylib_jax.core import base as base
from magpylib_jax.core import geometry as geom
from magpylib_jax.core.base import MagpylibBadUserInput, MagpylibMissingInput
from magpylib_jax.core import kernels_extended as kext
from magpylib_jax.functional import (
    _evaluate_source_field,
    _get_field_from_type,
    getB,
    getH,
    getJ,
    getM,
)
from magpylib_jax.misc.custom import CustomSource


def test_geometry_helpers_smoke() -> None:
    obs = np.array([1.0, 2.0, 3.0])
    assert geom.ensure_observers(obs).shape == (1, 3)
    assert geom.ensure_observers(np.stack([obs, obs], axis=0)).shape == (2, 3)

    with pytest.raises(ValueError):
        geom._as_array3(np.ones((2, 2)))

    assert geom.normalize_orientation(None).shape == (3, 3)
    assert geom.normalize_orientation(np.eye(3)).shape == (3, 3)
    assert geom.normalize_orientation(np.eye(3)[None, :, :]).shape == (3, 3)
    with pytest.raises(ValueError):
        geom.normalize_orientation(np.zeros((2, 3, 3)))
    with pytest.raises(ValueError):
        geom.normalize_orientation(np.zeros((2, 2)))
    class _Dummy:
        def __init__(self, mat):
            self._mat = mat
        def as_matrix(self):
            return self._mat
    with pytest.raises(ValueError):
        geom.normalize_orientation(_Dummy(np.zeros((2, 3, 3))))
    with pytest.raises(ValueError):
        geom.normalize_orientation(_Dummy(np.zeros((2, 2))))

    assert geom.normalize_positions(obs).shape == (1, 3)
    assert geom.normalize_positions(np.stack([obs, obs], axis=0)).shape == (2, 3)
    with pytest.raises(ValueError):
        geom.normalize_positions(np.zeros((2, 2, 3)))

    assert geom.normalize_orientations(None).shape == (1, 3, 3)
    assert geom.normalize_orientations(np.eye(3)).shape == (1, 3, 3)
    assert geom.normalize_orientations(np.eye(3)[None, :, :]).shape == (1, 3, 3)
    with pytest.raises(ValueError):
        geom.normalize_orientations(np.zeros((2, 2)))

    pos, rot = geom.broadcast_pose(position=np.array([[0, 0, 0], [1, 0, 0]]))
    assert pos.shape == (2, 3)
    assert rot.shape == (2, 3, 3)
    with pytest.raises(ValueError):
        geom.broadcast_pose(position=np.zeros((2, 3)), orientation=np.zeros((3, 3, 3)))
    with pytest.raises(ValueError):
        geom.broadcast_pose(position=np.zeros((2, 3)), orientation=np.zeros((4, 3, 3)))

    with pytest.raises(ValueError):
        geom.to_local_coordinates(obs, position=np.zeros((2, 3)))

    r, phi, z = geom.cart_to_cyl(jnp.asarray([[1.0, 0.0, 2.0]]))
    field = geom.cyl_field_to_cart(phi, r, z)
    assert field.shape == (1, 3)
    field = geom.cyl_field_to_cart(phi, r, r, z)
    assert field.shape == (1, 3)


def test_collection_and_sensor_errors() -> None:
    src = mpj.current.Circle(current=1.0, diameter=1.0)
    sensor = mpj.Sensor(pixel=(0, 0, 1))
    col = Collection(src, sensor)
    assert len(col.sources) == 1
    assert len(col.sensors) == 1

    with pytest.raises(MagpylibBadUserInput):
        col.add(123)

    col2 = Collection()
    col2.add(src, override_parent=True)
    assert src in col2.children

    with pytest.raises(MagpylibBadUserInput):
        col2.add(col2)

    col2.remove(src)
    with pytest.raises(MagpylibBadUserInput):
        col2.remove(src)
    col2.remove(src, errors="ignore")
    with pytest.raises(MagpylibBadUserInput):
        col2.remove("bad", errors="raise")
    with pytest.raises(MagpylibBadUserInput):
        col2.remove("bad", errors="maybe")


def test_sensor_custom_behavior() -> None:
    sensor = mpj.Sensor(pixel=None)
    assert sensor.observers.shape == (1, 3)
    sensor.pixel = np.array([1.0, 0.0, 0.0])
    assert sensor.observers.shape == (1, 3)
    sensor.pixel = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert sensor.observers.shape == (2, 3)

    with pytest.raises(MagpylibBadUserInput):
        sensor.handedness = "up"

    custom = CustomSource(field_func=lambda obs: np.zeros_like(np.asarray(obs)))
    out = custom.getB(np.array([0.0, 0.0, 1.0]))
    assert out.shape == (3,)
    custom2 = CustomSource()
    out = custom2.getB(np.array([0.0, 0.0, 1.0]))
    assert out.shape == (3,)
    assert custom2.getH(np.array([0.0, 0.0, 1.0])).shape == (3,)
    assert custom2.getJ(np.array([0.0, 0.0, 1.0])).shape == (3,)
    assert custom2.getM(np.array([0.0, 0.0, 1.0])).shape == (3,)


def test_base_helpers_and_repr() -> None:
    with pytest.raises(MagpylibBadUserInput):
        base.check_format_input_vector(None)
    assert base.check_format_input_vector([1.0, 2.0, 3.0]).shape == (3,)
    with pytest.raises(MagpylibBadUserInput):
        base.check_format_input_vector([1.0, 2.0], shape_m1=3)
    with pytest.raises(MagpylibBadUserInput):
        base.check_format_input_vector([[1.0, 2.0]], shape_m1=3)

    rot = base.R.from_rotvec([0.0, 0.0, 0.0])
    assert base.check_format_input_orientation(rot, init_format=True).shape == (1, 4)
    assert base.check_format_input_orientation([0.0, 0.0, 0.0], init_format=True).shape == (1, 4)
    with pytest.raises(MagpylibBadUserInput):
        base.check_format_input_orientation([[1.0, 2.0]])

    assert base.check_format_input_anchor(0).shape == (3,)
    with pytest.raises(MagpylibBadUserInput):
        base.check_format_input_anchor(1)

    assert base.check_format_input_angle(1.0) == 1.0
    assert base.check_format_input_angle([1.0, 2.0]).shape == (2,)
    with pytest.raises(MagpylibBadUserInput):
        base.check_format_input_angle([[1.0]])

    assert np.allclose(base.check_format_input_axis("x"), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(base.check_format_input_axis("y"), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(base.check_format_input_axis("z"), np.array([0.0, 0.0, 1.0]))
    with pytest.raises(MagpylibBadUserInput):
        base.check_format_input_axis("bad")
    with pytest.raises(MagpylibBadUserInput):
        base.check_format_input_axis([0.0, 0.0, 0.0])

    base.check_start_type("auto")
    base.check_start_type(1)
    with pytest.raises(MagpylibBadUserInput):
        base.check_start_type("bad")
    base.check_degree_type(True)
    with pytest.raises(MagpylibBadUserInput):
        base.check_degree_type(1)

    src = mpj.magnet.Cuboid(polarization=(0.0, 0.0, 1.0), dimension=(1.0, 1.0, 1.0))
    src.move((1.0, 0.0, 0.0))
    desc = src.describe(return_string=True)
    assert "Cuboid" in desc
    col = Collection()
    src.parent = col
    assert src.parent is col
    src.parent = None
    with pytest.raises(MagpylibBadUserInput):
        src.parent = "bad"


def test_functional_smoke_and_errors() -> None:
    obs = np.array([0.1, 0.2, 0.3])
    _ = getB("circle", obs, diameter=1.0, current=2.0)
    _ = getH("circle", obs, diameter=1.0, current=2.0)

    with pytest.raises(MagpylibMissingInput):
        getB("circle", obs)

    with pytest.raises(AttributeError):
        getB("circle", obs, diameter=1.0, current=1.0, pixel_agg="nope")
    with pytest.raises(AttributeError):
        getB("circle", obs, diameter=1.0, current=1.0, pixel_agg=123)

    assert not functional._has_tracer([1.0, 2.0])
    assert not functional._has_tracer({"a": 1.0})
    _ = jax.jit(lambda x: functional._is_array_like(x))(jnp.zeros(3))

    mean = getB("circle", obs, diameter=1.0, current=1.0, pixel_agg=np.mean)
    assert mean.shape[-1] == 3

    b_pix = getB("circle", np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 0.0]]), diameter=1.0, current=1.0, pixel_agg="mean")
    assert b_pix.shape[-1] == 3

    b_pix = getB(
        "circle",
        np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 0.0]]),
        diameter=1.0,
        current=1.0,
        pixel_agg="max",
        squeeze=False,
    )
    assert b_pix.shape[-1] == 3

    mesh = np.array(
        [[(0, 0, 0), (1, 0, 0), (0, 1, 0)]],
        dtype=float,
    )
    mesh4 = mesh[None, ...]
    _ = getB(
        "triangularmesh",
        obs,
        mesh=mesh4,
        polarization=(0.0, 0.0, 1.0),
    )

    df = pytest.importorskip("pandas")
    _ = getB(
        "cuboid",
        obs,
        polarization=(0.0, 0.0, 1.0),
        dimension=(1.0, 1.0, 1.0),
        output="dataframe",
    )
    _ = getB(
        "circle",
        obs,
        diameter=1.0,
        current=1.0,
        pixel_agg=np.mean,
        output="dataframe",
    )

    _ = getB(
        [mpj.magnet.Cuboid(polarization=(0.0, 0.0, 1.0), dimension=(1.0, 1.0, 1.0))],
        obs,
        pixel_agg=np.mean,
        output="dataframe",
        sumup=True,
    )

    src1 = mpj.misc.Dipole(moment=(0.0, 0.0, 1.0))
    src2 = mpj.current.Circle(current=1.0, diameter=1.0)
    summed = getB([src1, src2], obs, sumup=True)
    assert summed.shape[-1] == 3

    legacy = getB([src1, src2], obs, pixel_agg=np.mean, sumup=True, squeeze=False)
    assert legacy.shape[-1] == 3

    observers = np.zeros((2, 1, 3))
    pairwise = _get_field_from_type(
        "dipole", observers, "B", position=(0.0, 0.0, 0.0), moment=(1.0, 0.0, 0.0)
    )
    assert pairwise.shape[-1] == 3

    s1 = mpj.Sensor(pixel=None)
    s2 = mpj.Sensor(pixel=np.zeros((2, 3)))
    s2.handedness = "left"
    with pytest.raises(MagpylibBadUserInput):
        getB(src1, [s1, s2])

    empty, nsrc = _evaluate_source_field([], obs, "B", sumup=True, in_out="auto")
    assert nsrc == 0

    with pytest.raises(TypeError):
        _evaluate_source_field([object()], obs, "B", sumup=False, in_out="auto")

    src = mpj.magnet.CylinderSegment(
        polarization=(0.0, 0.0, 1.0),
        dimension=(0.5, 1.0, 1.0, 0.0, 90.0),
    )
    out, nsrc = _evaluate_source_field([src], obs, "B", sumup=False, in_out="inside")
    assert nsrc == 1 and out.shape[-1] == 3
    out, nsrc = _evaluate_source_field([src], obs, "B", sumup=True, in_out="outside")
    assert nsrc == 1 and out.shape[-1] == 3

    with pytest.raises(MagpylibMissingInput):
        _get_field_from_type("triangle", obs, "B")
    with pytest.raises(MagpylibMissingInput):
        _get_field_from_type("trianglesheet", obs, "B")
    with pytest.raises(MagpylibMissingInput):
        _get_field_from_type("trianglestrip", obs, "B")

    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tet = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mesh = np.array(
        [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]], dtype=float
    )
    seg_start = np.array([[0.0, 0.0, 0.0]])
    seg_end = np.array([[1.0, 0.0, 0.0]])
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    cds = np.array([[0.0, 0.0, 1.0]])

    _ = _get_field_from_type("dipole", obs, "B", moment=(1.0, 0.0, 0.0))
    _ = _get_field_from_type("dipole", obs, "H", moment=(1.0, 0.0, 0.0))
    _ = _get_field_from_type("circle", obs, "B", diameter=1.0, current=1.0)
    _ = _get_field_from_type("circle", obs, "H", diameter=1.0, current=1.0)
    for field in ("B", "H", "J", "M"):
        _ = _get_field_from_type(
            "cuboid", obs, field, dimension=(1.0, 1.0, 1.0), polarization=(0.0, 0.0, 1.0)
        )
        _ = _get_field_from_type(
            "cylinder", obs, field, dimension=(1.0, 1.0), polarization=(0.0, 0.0, 1.0)
        )
        _ = _get_field_from_type(
            "sphere", obs, field, diameter=1.0, polarization=(0.0, 0.0, 1.0)
        )
        _ = _get_field_from_type(
            "triangle", obs, field, vertices=tri, polarization=(0.0, 0.0, 1.0)
        )
        _ = _get_field_from_type(
            "tetrahedron", obs, field, vertices=tet, polarization=(0.0, 0.0, 1.0)
        )
        _ = _get_field_from_type(
            "triangularmesh", obs, field, mesh=mesh, polarization=(0.0, 0.0, 1.0)
        )

    _ = _get_field_from_type(
        "cylindersegment",
        obs,
        "B",
        dimension=(0.5, 1.0, 1.0, 0.0, 90.0),
        polarization=(0.0, 0.0, 1.0),
        in_out="outside",
    )
    _ = _get_field_from_type(
        "polyline",
        obs,
        "B",
        segment_start=seg_start,
        segment_end=seg_end,
        current=1.0,
    )
    _ = _get_field_from_type(
        "trianglesheet",
        obs,
        "B",
        vertices=verts,
        faces=faces,
        current_densities=cds,
    )
    _ = _get_field_from_type(
        "trianglestrip",
        obs,
        "B",
        vertices=verts,
        current=1.0,
    )


def test_functional_internal_helpers() -> None:
    obs = np.array([[0.1, 0.2, 0.3]])
    field = jnp.zeros((1, 1, 1, 3))
    out = functional._apply_squeeze(field, jnp.asarray(obs), squeeze=True, sumup=False, n_sources=1)
    assert out.shape[-1] == 3
    out = functional._apply_squeeze(field, jnp.asarray(obs), squeeze=False, sumup=True, n_sources=1)
    assert out.shape[-1] == 3
    out = functional._apply_squeeze(jnp.zeros((2, 1, 3)), jnp.asarray(obs), squeeze=False, sumup=False, n_sources=2)
    assert out.shape[-1] == 3

    pix_mask = jnp.array([[1.0, 0.0]])
    field = jnp.arange(6.0, dtype=jnp.float64).reshape((1, 1, 1, 2, 3))
    _ = functional._apply_pixel_agg_masked(field, pix_mask, pixel_agg="sum")
    _ = functional._apply_pixel_agg_masked(field, pix_mask, pixel_agg="mean")
    _ = functional._apply_pixel_agg_masked(field, pix_mask, pixel_agg="min")
    _ = functional._apply_pixel_agg_masked(field, pix_mask, pixel_agg="max")

    src = mpj.misc.Dipole(moment=(1.0, 0.0, 0.0))
    groups = functional._format_source_groups([[src]])
    assert len(groups) == 1
    with pytest.raises(MagpylibBadUserInput):
        functional._format_source_groups([])
    with pytest.raises(MagpylibBadUserInput):
        functional._format_source_groups([object()])

    col = Collection()
    col.add(mpj.Sensor(pixel=(0.0, 0.0, 0.0)))
    sensors, shapes = functional._format_observers(col, None)
    assert len(sensors) == 1 and len(shapes) == 1
    with pytest.raises(MagpylibBadUserInput):
        functional._format_observers(Collection(), None)
    with pytest.raises(MagpylibBadUserInput):
        functional._format_observers([], None)
    with pytest.raises(MagpylibBadUserInput):
        functional._format_observers(123, None)

    with pytest.raises(ValueError):
        functional._prepare_sources_jit(
            "triangularmesh",
            position=(0.0, 0.0, 0.0),
            orientation=None,
            in_out="auto",
            kwargs={"mesh": np.zeros((1, 1, 3, 3)), "polarization": (0.0, 0.0, 1.0)},
        )


def test_kernels_extended_smoke() -> None:
    jax.config.update("jax_enable_x64", True)
    obs = np.array([[0.2, -0.1, 0.3]], dtype=float)
    pol = np.array([0.0, 0.0, 1.0], dtype=float)

    grid = jnp.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=jnp.float64,
    )
    tris = kext._grid_to_triangles(grid)
    assert tris.shape[-2:] == (3, 3)

    dim = jnp.array([0.5, 1.0, 1.2, -30.0, 60.0], dtype=jnp.float64)
    mesh = kext._build_cylinder_segment_mesh(dim, n_phi=4, n_r=1, n_z=1)
    mesh_arr, nvec, L, l1, l2 = kext.precompute_trimesh_geometry(mesh)

    _ = kext.magnet_trimesh_bfield(obs, mesh_arr, pol, in_out="outside")
    _ = kext.magnet_trimesh_bfield(obs, mesh_arr[None, ...], pol, in_out="outside")
    _ = kext.magnet_trimesh_bfield(obs, mesh_arr, pol, in_out="inside")
    _ = kext.magnet_trimesh_bfield_jit(obs, mesh_arr, pol, in_out="outside")
    _ = kext.magnet_trimesh_bfield_jit_faces(obs, mesh_arr, pol, in_out="outside")
    _ = kext.magnet_trimesh_bfield_jit_faces_precomp(
        obs, mesh_arr, pol, nvec, L, l1, l2, in_out="outside"
    )
    _ = kext.magnet_cylinder_segment_bfield_jit_faces(obs, dim, pol, in_out="outside")

    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    _ = kext.triangle_bfield_jit(obs, tri, pol)

    tetra = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    _ = kext.tetrahedron_bfield_jit(obs, tetra, pol)
    _ = kext.tetrahedron_bfield(obs, tetra, pol, in_out="inside")
    _ = kext.tetrahedron_jfield(obs, tetra, pol, in_out="outside")

    _ = kext.magnet_sphere_hfield(obs, np.array([1.0]), pol)
    _ = kext.magnet_sphere_jfield(obs, np.array([1.0]), pol)

    _ = kext.current_circle_bfield_jit(obs, 1.0, 1.0)
    poly_start = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    poly_end = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    _ = kext.current_polyline_bfield_jit(obs, poly_start, poly_end, 1.0)
    _ = kext.current_polyline_hfield(obs, poly_start[0], poly_end[0], np.array([1.0]))
    with pytest.raises(ValueError):
        kext.current_polyline_hfield(obs, poly_start, poly_end[:1], 1.0)

    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    cds = np.array([[0.0, 0.0, 1.0]])
    _ = kext.current_trisheet_bfield_jit(obs, verts, faces, cds)
    strip = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    _ = kext.current_tristrip_bfield_jit(obs, strip, 1.0)
    with pytest.raises(ValueError):
        kext._broadcast_mesh(jnp.zeros((2, 2)), 1)

    hits = kext._moller_trumbore_hits(
        jnp.array([0.1, 0.1, 1.0]),
        jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        jnp.array([0.0, 0.0, -1.0]),
    )
    assert hits.shape[0] == 1

    inside = kext._mask_inside_trimesh_jax(
        jnp.array([[0.1, 0.1, 0.1]]),
        jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
    )
    assert inside.shape == (1,)

    mesh_mask = jnp.array([1.0])
    _ = kext._inside_mask_mesh_masked(
        jnp.array([[0.1, 0.1, 0.1]]),
        jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        mesh_mask,
    )


def test_triangle_sheet_strip_validation() -> None:
    with pytest.raises(ValueError):
        mpj.current.TriangleSheet(vertices=[(0, 0)], faces=[(0, 1, 2)], current_densities=[(0, 0, 1)])
    with pytest.raises(ValueError):
        mpj.current.TriangleSheet(vertices=[(0, 0, 0)], faces=[(0, 1)], current_densities=[(0, 0, 1)])
    with pytest.raises(ValueError):
        mpj.current.TriangleSheet(vertices=[(0, 0, 0)], faces=[(0, 1, 2)], current_densities=[(0, 0)])
    with pytest.raises(ValueError):
        mpj.current.TriangleSheet(vertices=[(0, 0, 0)], faces=[(0, 1, 2)], current_densities=[(0, 0, 1)])
    with pytest.raises(ValueError):
        mpj.current.TriangleSheet(vertices=[(0, 0, 0), (1, 0, 0)], faces=[(0, 1, 2)], current_densities=[(0, 0, 1)])
    with pytest.raises(ValueError):
        mpj.current.TriangleSheet(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2)], current_densities=[(0, 0, 1), (0, 1, 0)])
    with pytest.raises(ValueError):
        mpj.current.TriangleStrip(vertices=[(0, 0, 0), (1, 0, 0)], current=1.0)
    assert mpj.current.TriangleStrip(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], current=1.0).centroid.shape == (3,)
