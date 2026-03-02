import numpy as np
import pytest

import magpylib_jax as mpj


def test_path_old_new_move():
    """test path move and compare to old style computation"""
    n = 100
    s_pos = (0, 0, 0)

    pm1 = mpj.magnet.Cylinder(polarization=(0, 0, 1), dimension=(3, 3), position=(-5, 0, 3))
    pm1.move([(x, 0, 0) for x in np.linspace(0, 10, 100)], start=-1)
    b1 = pm1.getB(s_pos)

    pm2 = mpj.magnet.Cylinder(polarization=(0, 0, 1), dimension=(3, 3), position=(0, 0, 3))
    ts = np.linspace(-5, 5, n)
    possis = np.array([(t, 0, 0) for t in ts])
    b2 = pm2.getB(possis[::-1])

    np.testing.assert_allclose(b1, b2, err_msg="path move problem")


def _assert_path_old_new_rotate(n: int) -> None:
    s_pos = (0, 0, 0)
    ax = (1, 0, 0)
    anch = (0, 0, 10)

    pm1 = mpj.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 2, 3), position=(0, 0, 3))
    pm1.rotate_from_angax(-30, ax, anch)
    pm1.rotate_from_angax(np.linspace(0, 60, n), "x", anch, start=-1)
    b1 = pm1.getB(s_pos)

    pm2 = mpj.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 2, 3), position=(0, 0, 3))
    pm2.rotate_from_angax(-30, ax, anch)
    b2 = []
    for _ in range(n):
        b2 += [pm2.getB(s_pos)]
        pm2.rotate_from_angax(60 / (n - 1), ax, anch)
    b2 = np.array(b2)

    np.testing.assert_allclose(
        b1,
        b2,
        rtol=1e-5,
        atol=1e-8,
        err_msg="path rotate problem",
    )


def test_path_old_new_rotate():
    """test path rotate compare to old style computation (reduced CI sample)."""
    _assert_path_old_new_rotate(n=21)


@pytest.mark.slow
def test_path_old_new_rotate_full():
    """Full upstream mirror coverage for path rotate parity."""
    _assert_path_old_new_rotate(n=111)
