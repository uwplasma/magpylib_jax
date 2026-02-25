import numpy as np

import magpylib_jax as mpj


def _rotx(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_basegeo_broadcast_position_orientation() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ]
    )
    orientation_single = _rotx(0.3)
    orientation_path = np.stack([orientation_single] * 3, axis=0)

    src_single = mpj.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=positions,
        orientation=orientation_single,
    )
    src_path = mpj.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=positions,
        orientation=orientation_path,
    )

    obs = np.array([0.1, 0.2, 0.3])
    b_single = src_single.getB(obs)
    b_path = src_path.getB(obs)

    np.testing.assert_allclose(b_single, b_path)


def test_basegeo_pairwise_observer_path() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    src = mpj.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=positions,
        orientation=None,
    )
    observers = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.4]])
    b = src.getB(observers)
    assert b.shape == (2, 2, 3)
