import numpy as np

import magpylib_jax as mpj


def test_sensor_prep_cache_invalidates_on_sensor_updates() -> None:
    src = mpj.misc.Dipole(moment=(0.7, -0.4, 1.1), position=(0.0, 0.1, -0.2))
    sens = mpj.Sensor(
        pixel=[(0.0, 0.0, 0.0), (0.05, 0.02, 0.0)],
        position=(0.2, -0.1, 0.4),
    )

    b0 = np.asarray(mpj.getB(src, sens))
    b1 = np.asarray(mpj.getB(src, sens))
    assert np.allclose(b0, b1, rtol=1e-12, atol=1e-12)

    sens.position = (0.25, -0.1, 0.4)
    b2 = np.asarray(mpj.getB(src, sens))
    assert not np.allclose(b1, b2, rtol=1e-12, atol=1e-12)

    sens.pixel = [(0.0, 0.0, 0.0), (0.08, 0.01, 0.0)]
    b3 = np.asarray(mpj.getB(src, sens))
    assert not np.allclose(b2, b3, rtol=1e-12, atol=1e-12)

    sens.handedness = "left"
    b4 = np.asarray(mpj.getB(src, sens))
    assert not np.allclose(b3, b4, rtol=1e-12, atol=1e-12)


def test_source_prep_cache_invalidates_non_circle_source() -> None:
    obs = np.array([[0.3, -0.2, 0.4], [0.1, 0.5, -0.3]], dtype=float)
    src = mpj.magnet.Cuboid(
        dimension=(0.4, 0.5, 0.6),
        polarization=(0.2, -0.1, 0.3),
        position=(0.0, 0.0, 0.1),
    )

    b0 = np.asarray(mpj.getB(src, obs))
    b1 = np.asarray(mpj.getB(src, obs))
    assert np.allclose(b0, b1, rtol=1e-12, atol=1e-12)

    src.polarization = (0.25, -0.1, 0.3)
    b2 = np.asarray(mpj.getB(src, obs))
    assert not np.allclose(b1, b2, rtol=1e-12, atol=1e-12)

    src.rotate_from_angax(25, "z")
    b3 = np.asarray(mpj.getB(src, obs))
    assert not np.allclose(b2, b3, rtol=1e-12, atol=1e-12)


def test_collection_structure_cache_invalidates_on_add_remove() -> None:
    obs = np.array([[0.2, 0.0, 0.4]], dtype=float)
    src1 = mpj.current.Circle(current=1.0, diameter=0.7, position=(0.0, 0.0, 0.0))
    src2 = mpj.current.Circle(current=0.5, diameter=0.4, position=(0.0, 0.0, 0.3))
    col = mpj.Collection(src1)

    b0 = np.asarray(mpj.getB(col, obs))
    col.add(src2)
    b1 = np.asarray(mpj.getB(col, obs))
    assert not np.allclose(b0, b1, rtol=1e-12, atol=1e-12)

    col.remove(src2)
    b2 = np.asarray(mpj.getB(col, obs))
    assert np.allclose(b0, b2, rtol=1e-12, atol=1e-12)


def test_triangularmesh_geometry_cache_invalidates_vertices() -> None:
    src = mpj.magnet.TriangularMesh(
        vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        faces=[(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)],
        polarization=(0.1, -0.2, 0.3),
        reorient_faces=False,
        check_open="skip",
    )
    obs = np.array([[0.1, 0.2, 0.3]], dtype=float)

    mesh0 = np.asarray(src.mesh)
    mesh1 = np.asarray(src.mesh)
    assert np.allclose(mesh0, mesh1, rtol=0.0, atol=0.0)

    b0 = np.asarray(src.getB(obs))
    src.vertices = [(0, 0, 0), (1.2, 0, 0), (0, 1, 0), (0, 0, 1)]
    mesh2 = np.asarray(src.mesh)
    b1 = np.asarray(src.getB(obs))

    assert not np.allclose(mesh0, mesh2, rtol=1e-12, atol=1e-12)
    assert not np.allclose(b0, b1, rtol=1e-12, atol=1e-12)
