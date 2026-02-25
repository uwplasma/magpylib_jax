import numpy as np

import magpylib_jax as mpj


def _make_source_path(n: int = 6):
    pos = np.linspace((0.1, 0.2, 0.3), (1.0, 2.0, 3.0), n)
    src = mpj.magnet.Cuboid(
        polarization=(1, 2, 3),
        dimension=(1, 2, 3),
        position=pos,
    )
    return src, pos


def test_getB_interface_equivalence() -> None:
    src, _ = _make_source_path()
    poso = [[(-1, -1, -1)] * 2] * 2
    sens = mpj.Sensor(pixel=poso)

    b1 = mpj.getB(src, sens)
    b2 = src.getB(poso)
    b3 = src.getB(sens)
    b4 = sens.getB(src)

    np.testing.assert_allclose(b1, b2)
    np.testing.assert_allclose(b1, b3)
    np.testing.assert_allclose(b1, b4)


def test_getB_interfaces_list_sources() -> None:
    src, _ = _make_source_path()
    poso = [[(-1, -1, -1)] * 2] * 2
    sens = mpj.Sensor(pixel=poso)

    b_stack = mpj.getB([src, src], sens, sumup=False)
    b_sum = mpj.getB([src, src], sens, sumup=True)

    np.testing.assert_allclose(b_stack[0], b_stack[1])
    np.testing.assert_allclose(b_sum, b_stack[0] + b_stack[1])


def test_getB_interfaces_list_observers() -> None:
    src = mpj.magnet.Cuboid(
        polarization=(1, 2, 3),
        dimension=(1, 2, 3),
    )
    poso = [[(-1, -1, -1)] * 2] * 2

    b_stack = src.getB([poso, poso])
    assert b_stack.shape == (2, 4, 3)
    np.testing.assert_allclose(b_stack[0], b_stack[1])


def test_getH_interface_equivalence() -> None:
    src = mpj.magnet.Cuboid(
        polarization=(22, -33, 44),
        dimension=(3, 2, 3),
        position=np.linspace((0.1, 0.2, 0.3), (1.0, 2.0, 3.0), 6),
    )
    poso = [[(-1, -2, -3)] * 2] * 2
    sens = mpj.Sensor(pixel=poso)

    h1 = mpj.getH(src, sens)
    h2 = src.getH(poso)
    h3 = src.getH(sens)
    h4 = sens.getH(src)

    np.testing.assert_allclose(h1, h2)
    np.testing.assert_allclose(h1, h3)
    np.testing.assert_allclose(h1, h4)


def test_getH_interfaces_list_sources() -> None:
    src = mpj.magnet.Cuboid(
        polarization=(22, -33, 44),
        dimension=(3, 2, 3),
        position=np.linspace((0.1, 0.2, 0.3), (1.0, 2.0, 3.0), 6),
    )
    poso = [[(-1, -2, -3)] * 2] * 2
    sens = mpj.Sensor(pixel=poso)

    h_stack = mpj.getH([src, src], sens, sumup=False)
    h_sum = mpj.getH([src, src], sens, sumup=True)

    np.testing.assert_allclose(h_stack[0], h_stack[1])
    np.testing.assert_allclose(h_sum, h_stack[0] + h_stack[1])


def test_getH_interfaces_list_observers() -> None:
    src = mpj.magnet.Cuboid(
        polarization=(22, -33, 44),
        dimension=(3, 2, 3),
    )
    poso = [[(-1, -2, -3)] * 2] * 2

    h_stack = src.getH([poso, poso])
    assert h_stack.shape == (2, 4, 3)
    np.testing.assert_allclose(h_stack[0], h_stack[1])


def test_getHBMJ_self_consistency() -> None:
    sources = [
        mpj.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1)),
        mpj.current.Circle(diameter=1, current=1),
    ]
    sens = mpj.Sensor(position=np.linspace((-1, 0, 0), (1, 0, 0), 10))
    src = sources[0]

    for field in "BHJM":
        f1 = getattr(mpj, f"get{field}")(src, sens)
        f2 = getattr(sens, f"get{field}")(src)
        f3 = getattr(src, f"get{field}")(sens)
        np.testing.assert_allclose(f1, f2)
        np.testing.assert_allclose(f1, f3)
