"""Smoke + structural tests for the matplotlib-backed ``show()``.

These run under the Agg backend so they never open a window and never call
``plt.show()``. Every source family, plus ``Sensor``, ``Collection`` and a
path object, is rendered via ``obj.show(return_fig=True)`` and checked to have
produced at least one artist without raising.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import magpylib_jax as mpj


def _artist_count(fig) -> int:
    ax = fig.axes[0]
    return (
        len(ax.collections)
        + len(ax.lines)
        + len(ax.patches)
        + len(getattr(ax, "texts", []))
    )


def _make_objects() -> dict:
    return {
        "cuboid": mpj.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1)),
        "cylinder": mpj.magnet.Cylinder(polarization=(0, 0, 1), dimension=(1, 1.5)),
        "cylinder_segment": mpj.magnet.CylinderSegment(
            polarization=(0, 0, 1), dimension=(0.5, 1.0, 1.0, -30.0, 90.0)
        ),
        "sphere": mpj.magnet.Sphere(polarization=(1, 0, 0), diameter=1.0),
        "tetrahedron": mpj.magnet.Tetrahedron(
            polarization=(0, 0, 1),
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ),
        "triangular_mesh": mpj.magnet.TriangularMesh(
            polarization=(0, 0, 1),
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            faces=[[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
        ),
        "triangle": mpj.misc.Triangle(
            polarization=(0, 0, 1), vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        ),
        "dipole": mpj.misc.Dipole(moment=(0, 0, 1)),
        "custom": mpj.misc.CustomSource(),
        "circle": mpj.current.Circle(current=1.0, diameter=1.0),
        "polyline": mpj.current.Polyline(
            current=1.0, vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0]]
        ),
        "triangle_strip": mpj.current.TriangleStrip(
            current=1.0, vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        ),
        "triangle_sheet": mpj.current.TriangleSheet(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            current_densities=[[1.0, 0.0, 0.0]],
        ),
        "sensor": mpj.Sensor(pixel=[[0, 0, 0], [0.1, 0.1, 0.1]]),
    }


@pytest.mark.parametrize("name", list(_make_objects()))
def test_show_each_object(name):
    obj = _make_objects()[name]
    fig = obj.show(return_fig=True)
    assert fig is not None
    assert len(fig.axes) == 1
    assert _artist_count(fig) >= 1
    plt.close(fig)


def test_show_rotated_object():
    """Geometry is transformed by orientation without error."""
    cub = mpj.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    cub.rotate_from_angax(45, "z")
    fig = cub.show(return_fig=True)
    assert _artist_count(fig) >= 1
    plt.close(fig)


def test_show_collection():
    coll = mpj.Collection(
        mpj.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1)),
        mpj.current.Circle(current=1.0, diameter=2.0, position=(2, 0, 0)),
        mpj.Sensor(position=(0, 2, 0)),
    )
    fig = coll.show(return_fig=True)
    assert _artist_count(fig) >= 3
    plt.close(fig)


def test_show_path_object():
    """A source with a multi-point path draws a trailing path line."""
    cub = mpj.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    cub.move([[1, 0, 0], [2, 0, 0], [3, 0, 0]], start=1)
    path = np.asarray(cub._position).reshape(-1, 3)
    assert path.shape[0] > 1
    fig = cub.show(return_fig=True)
    assert len(fig.axes[0].lines) >= 1  # the dashed path line
    plt.close(fig)


def test_module_level_show_mixed():
    a = mpj.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    b = mpj.current.Circle(current=1.0, diameter=1.0, position=(2, 0, 0))
    c = mpj.misc.Dipole(moment=(0, 0, 1), position=(0, 2, 0))
    fig = mpj.show(a, b, c, return_fig=True)
    assert _artist_count(fig) >= 3
    plt.close(fig)


def test_show_accepts_iterable():
    objs = [
        mpj.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1)),
        mpj.misc.Dipole(moment=(0, 0, 1), position=(2, 0, 0)),
    ]
    fig = mpj.show(objs, return_fig=True)
    assert _artist_count(fig) >= 2
    plt.close(fig)


def test_show_on_provided_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    src = mpj.magnet.Sphere(polarization=(0, 0, 1), diameter=1.0)
    out = mpj.show(src, ax=ax, return_fig=True)
    assert out is fig
    assert _artist_count(fig) >= 1
    plt.close(fig)


def test_show_empty_raises():
    with pytest.raises(ValueError):
        mpj.show(return_fig=True)


def test_show_bad_backend_raises():
    src = mpj.misc.Dipole(moment=(0, 0, 1))
    with pytest.raises(ValueError):
        mpj.show(src, backend="plotly", return_fig=True)


def test_show_missing_geometry_no_crash():
    """An under-specified object renders a fallback marker rather than crashing."""
    cub = mpj.magnet.Cuboid()  # no dimension / polarization
    fig = cub.show(return_fig=True)
    assert _artist_count(fig) >= 1
    plt.close(fig)


def test_show_title_via_kwargs():
    src = mpj.misc.Dipole(moment=(0, 0, 1))
    fig = mpj.show(src, return_fig=True, title="my scene")
    assert fig.axes[0].get_title() == "my scene"
    plt.close(fig)
