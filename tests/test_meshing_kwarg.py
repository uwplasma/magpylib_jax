"""Every source accepts a ``meshing`` constructor kwarg (magpylib parity).

Regression test for a 2.0.0 bug where only magnet sources hardcoded
``self.meshing`` and current/misc sources raised ``TypeError`` on
``meshing=...``. getFT reads ``target.meshing``, and magpylib accepts it on all
sources, so it must be settable at construction time everywhere.
"""

import numpy as np
import pytest

import magpylib_jax as mpj


def _sources_with_meshing(value):
    return [
        mpj.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1), meshing=value),
        mpj.magnet.Cylinder(dimension=(1, 1), polarization=(0, 0, 1), meshing=value),
        mpj.magnet.CylinderSegment(
            dimension=(0.5, 1.0, 1.0, 0.0, 90.0), polarization=(0, 0, 1), meshing=value
        ),
        mpj.magnet.Sphere(diameter=1.0, polarization=(0, 0, 1), meshing=value),
        mpj.magnet.Tetrahedron(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            polarization=(0, 0, 1),
            meshing=value,
        ),
        mpj.magnet.TriangularMesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            faces=[[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
            polarization=(0, 0, 1),
            meshing=value,
        ),
        mpj.current.Circle(diameter=1.0, current=1.0, meshing=value),
        mpj.current.Polyline(current=1.0, vertices=[[0, 0, 0], [1, 0, 0]], meshing=value),
        mpj.misc.Dipole(moment=(0, 0, 1), meshing=value),
        mpj.misc.Triangle(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], polarization=(0, 0, 1), meshing=value
        ),
    ]


@pytest.mark.parametrize("value", [1, 8, (2, 2, 2)])
def test_all_sources_accept_meshing(value):
    for src in _sources_with_meshing(value):
        assert src.meshing == value, type(src).__name__


def test_meshing_defaults_to_none():
    assert mpj.current.Circle(diameter=1.0, current=1.0).meshing is None
    assert mpj.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1)).meshing is None


def test_getft_uses_constructor_meshing():
    """A Circle target meshed at construction matches one meshed post-hoc."""
    from magpylib_jax import getFT

    src = mpj.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    tgt_ctor = mpj.current.Circle(diameter=1.0, current=100.0, position=(1, 0, 1), meshing=40)
    tgt_post = mpj.current.Circle(diameter=1.0, current=100.0, position=(1, 0, 1))
    tgt_post.meshing = 40
    f_ctor, t_ctor = getFT(src, tgt_ctor)
    f_post, t_post = getFT(src, tgt_post)
    np.testing.assert_allclose(np.asarray(f_ctor), np.asarray(f_post), rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(t_ctor), np.asarray(t_post), rtol=1e-12, atol=0)
