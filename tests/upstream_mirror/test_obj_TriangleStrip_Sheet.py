import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_ragged_trianglesheet_style_inputs_mirrored_behavior() -> None:
    vertices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=float)
    faces = np.array([(0, 1, 2), (1, 2, 3), (2, 3, 4)], dtype=int)
    cds = np.array([[10.1, 0, 0]] * 3, dtype=float)

    src_ref = magpy.current.TriangleSheet(vertices=vertices, faces=faces, current_densities=cds)
    src_new = mpj.current.TriangleSheet(vertices=vertices, faces=faces, current_densities=cds)

    obs = np.array([[2, 3, 4], [0.1, 0.2, 0.3]])
    np.testing.assert_allclose(
        np.asarray(src_new.getH(obs)),
        src_ref.getH(obs),
        rtol=1e-1,
        atol=2.0,
    )


def test_trianglestrip_zero_surface_is_stable() -> None:
    strip = mpj.current.TriangleStrip(
        vertices=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        current=10.1,
    )
    out = np.asarray(strip.getH([[2, 3, 4]]))
    np.testing.assert_allclose(out, 0.0, atol=1e-15)
