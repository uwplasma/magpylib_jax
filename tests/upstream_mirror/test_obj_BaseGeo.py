import numpy as np

import magpylib_jax as mpj


def _rotz(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_basegeo_position_orientation_paths() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [0.5, 0.0, -0.5],
        ]
    )
    orientations = np.stack([_rotz(0.0), _rotz(0.2), _rotz(-0.1)], axis=0)

    src = mpj.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=(0, 0, 1),
        position=positions,
        orientation=orientations,
    )
    obs = np.array([0.2, 0.1, 0.3])

    b_path = src.getB(obs)
    manual = []
    for pos, rot in zip(positions, orientations, strict=False):
        manual.append(
            mpj.getB(
                "cuboid",
                obs,
                position=pos,
                orientation=rot,
                dimension=(1, 1, 1),
                polarization=(0, 0, 1),
            )
        )
    manual = np.stack(manual, axis=0)

    np.testing.assert_allclose(b_path, manual)


def test_basegeo_scalar_orientation_matrix() -> None:
    src = mpj.magnet.Cuboid(
        dimension=(1, 2, 3),
        polarization=(0, 0, 1),
        position=(0.0, 0.0, 0.0),
        orientation=_rotz(0.3),
    )
    obs = np.array([0.1, -0.2, 0.3])
    b1 = src.getB(obs)
    b2 = mpj.getB(
        "cuboid",
        obs,
        position=(0.0, 0.0, 0.0),
        orientation=_rotz(0.3),
        dimension=(1, 2, 3),
        polarization=(0, 0, 1),
    )
    np.testing.assert_allclose(b1, b2)
