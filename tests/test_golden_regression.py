"""Fast behaviour-preservation net for the refactor.

Pins exact public getB/getH/getJ/getM values and a few gradients for every
source family against ``tests/data/golden.json`` (generated from the
parity-passing code). Runs in seconds and needs no magpylib import, so the
refactor can be checked continuously; the full magpylib parity suite remains
the physics oracle.

Regenerate intentionally with ``scripts/gen_golden.py`` only when a change is
*meant* to alter output.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

import magpylib_jax as mpj  # enables jax x64 on import

GOLDEN = json.loads((Path(__file__).parent / "data" / "golden.json").read_text())
OBS = np.array(GOLDEN["observers"])


def _make_sources():
    s = {}
    s["cuboid"] = mpj.magnet.Cuboid(
        polarization=(0.15, -0.22, 0.3), dimension=(1.1, 0.7, 1.4), position=(0.2, 0.3, -0.25)
    )
    s["cylinder"] = mpj.magnet.Cylinder(
        polarization=(0.12, -0.2, 0.28), dimension=(1.4, 1.2), position=(0.1, -0.2, 0.25)
    )
    s["cylinder_segment"] = mpj.magnet.CylinderSegment(
        polarization=(0.1, -0.2, 0.3), dimension=(0.4, 1.2, 1.1, -30.0, 110.0)
    )
    s["sphere"] = mpj.magnet.Sphere(
        polarization=(0.2, -0.15, 0.3), diameter=1.3, position=(0.2, 0.1, -0.15)
    )
    s["tetrahedron"] = mpj.magnet.Tetrahedron(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], polarization=(0.11, -0.07, 0.13)
    )
    s["triangular_mesh"] = mpj.magnet.TriangularMesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
        polarization=(0.1, 0.2, 0.3),
    )
    s["circle"] = mpj.current.Circle(current=2.5, diameter=0.9, position=(0.1, 0.2, -0.4))
    s["polyline"] = mpj.current.Polyline(
        current=1.7, vertices=[[-0.5, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0.2]]
    )
    s["triangle_sheet"] = mpj.current.TriangleSheet(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        current_densities=[[0.7, 0.1, 0.0]],
    )
    s["triangle_strip"] = mpj.current.TriangleStrip(
        vertices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], current=1.4
    )
    s["dipole"] = mpj.misc.Dipole(moment=(1.2, -0.7, 0.3), position=(0.2, -0.1, 0.5))
    s["triangle"] = mpj.misc.Triangle(
        vertices=[[-0.2, -0.1, 0.0], [0.9, 0.3, 0.2], [0.1, 0.8, -0.2]],
        polarization=(0.1, -0.2, 0.3),
    )
    s["collection"] = mpj.Collection(s["cuboid"].copy(), s["dipole"].copy())
    return s


SOURCES = _make_sources()


@pytest.mark.parametrize("name", sorted(GOLDEN["fields"]))
@pytest.mark.parametrize("field", ["B", "H", "J", "M"])
def test_golden_field(name, field):
    got = np.asarray(getattr(SOURCES[name], f"get{field}")(OBS))
    ref = np.array(GOLDEN["fields"][name][field])
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


def test_golden_gradients():
    def cuboid_bz(cx):
        return mpj.magnet.Cuboid(
            polarization=(0.15, -0.22, 0.3), dimension=(1.1, 0.7, cx)
        ).getB(OBS[0])[2]

    def circle_bz(d):
        src = mpj.current.Circle(current=2.5, diameter=d, position=(0.1, 0.2, -0.4))
        return src.getB(OBS[0])[2]

    def dipole_bx(mx):
        src = mpj.misc.Dipole(moment=(mx, -0.7, 0.3), position=(0.2, -0.1, 0.5))
        return src.getB(OBS[0])[0]

    grads = GOLDEN["grads"]
    assert float(jax.grad(cuboid_bz)(1.4)) == pytest.approx(grads["cuboid_dBz_dc"], rel=1e-7)
    assert float(jax.grad(circle_bz)(0.9)) == pytest.approx(grads["circle_dBz_dd"], rel=1e-7)
    assert float(jax.grad(dipole_bx)(1.2)) == pytest.approx(grads["dipole_dBx_dmx"], rel=1e-7)
