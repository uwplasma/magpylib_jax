import magpylib as magpy
import numpy as np

import magpylib_jax as mpj


def test_trianglesheet_bh_parity_against_magpylib() -> None:
    rng = np.random.default_rng(222)
    observers = rng.normal(size=(300, 3))
    observers += np.array([0.4, 0.3, 0.8])

    vertices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=int)
    current_densities = np.array([[0.7, 0.1, 0.0], [0.7, 0.1, 0.0]], dtype=float)

    src_ref = magpy.current.TriangleSheet(
        vertices=vertices,
        faces=faces,
        current_densities=current_densities,
    )
    src_new = mpj.current.TriangleSheet(
        vertices=vertices,
        faces=faces,
        current_densities=current_densities,
    )

    b_ref = src_ref.getB(observers)
    h_ref = src_ref.getH(observers)
    b_new = np.asarray(src_new.getB(observers))
    h_new = np.asarray(src_new.getH(observers))

    np.testing.assert_allclose(b_new, b_ref, rtol=8e-2, atol=2e-6)
    np.testing.assert_allclose(h_new, h_ref, rtol=8e-2, atol=2.0)


def test_trianglesheet_jm_zero() -> None:
    src = mpj.current.TriangleSheet(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        current_densities=[[1.0, 0.0, 0.0]],
    )
    obs = np.array([[0.1, 0.2, 0.4], [0.3, 0.1, 0.7]])
    np.testing.assert_allclose(np.asarray(src.getJ(obs)), 0.0)
    np.testing.assert_allclose(np.asarray(src.getM(obs)), 0.0)
