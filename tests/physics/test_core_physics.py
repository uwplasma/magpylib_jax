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


def test_triangular_mesh_and_tetrahedron_match_for_same_body() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    pol = np.array([0.1, -0.2, 0.3])

    tet = mpj.magnet.Tetrahedron(vertices=vertices, polarization=pol)
    mesh = mpj.magnet.TriangularMesh(vertices=vertices, faces=faces, polarization=pol)

    observers = np.array([[0.3, 0.2, 0.1], [1.2, 0.8, 0.7]])
    np.testing.assert_allclose(np.asarray(mesh.getB(observers)), np.asarray(tet.getB(observers)))
    np.testing.assert_allclose(np.asarray(mesh.getH(observers)), np.asarray(tet.getH(observers)))


def test_triangle_sheet_and_strip_consistent_for_matching_strip() -> None:
    vertices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    strip = mpj.current.TriangleStrip(vertices=vertices, current=1.0)

    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=int)
    cds = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    sheet = mpj.current.TriangleSheet(vertices=vertices, faces=faces, current_densities=cds)

    observers = np.array([[0.2, 0.3, 0.5], [0.5, 0.6, 0.8]])
    b_strip = np.asarray(strip.getB(observers))
    b_sheet = np.asarray(sheet.getB(observers))
    ratio = np.linalg.norm(b_strip - b_sheet) / np.maximum(
        np.linalg.norm(b_strip) + np.linalg.norm(b_sheet),
        1e-20,
    )
    assert ratio < 0.25
