import numpy as np

import magpylib_jax as mpj


def _rotz(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_path_move_equivalence() -> None:
    n = 50
    s_pos = (0.0, 0.0, 0.0)

    positions = np.array([(x, 0.0, 3.0) for x in np.linspace(-5.0, 5.0, n)])
    src = mpj.magnet.Cylinder(
        polarization=(0, 0, 1),
        dimension=(3, 3),
        position=positions,
    )
    b1 = src.getB(s_pos)

    b2 = []
    for pos in positions:
        b2.append(
            mpj.getB(
                "cylinder",
                s_pos,
                position=pos,
                dimension=(3, 3),
                polarization=(0, 0, 1),
            )
        )
    b2 = np.array(b2)

    np.testing.assert_allclose(b1, b2, err_msg="path move problem")


def test_path_rotate_equivalence() -> None:
    n = 25
    s_pos = (0.0, 0.0, 0.0)
    angles = np.linspace(0.0, np.deg2rad(60.0), n)
    orientations = np.stack([_rotz(a) for a in angles], axis=0)

    src = mpj.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(1, 2, 3),
        position=(0, 0, 3),
        orientation=orientations,
    )
    b1 = src.getB(s_pos)

    b2 = []
    for rot in orientations:
        b2.append(
            mpj.getB(
                "cuboid",
                s_pos,
                position=(0, 0, 3),
                orientation=rot,
                dimension=(1, 2, 3),
                polarization=(0, 0, 1),
            )
        )
    b2 = np.array(b2)

    np.testing.assert_allclose(b1, b2, rtol=1e-5, atol=1e-8, err_msg="path rotate problem")
