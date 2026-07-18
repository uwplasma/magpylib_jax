"""Tests for the ``magpylib_jax.func`` high-level functional interface.

Each ``func.*_field`` wrapper must (1) return finite arrays with the right shape,
(2) agree with the equivalent ``getB``/``getH(<type>, ...)`` call per instance,
and (3) where a magpylib reference is installed, match ``magpy.func.*_field``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib_jax as m

# --- per-function specs: (func, source_type, single-instance kwargs) ---------
# kwargs use the magpylib func parameter names; the mapping to getB kernel
# kwargs is exercised implicitly through the parity assertions below.
SINGLE_CASES = {
    "circle": (
        m.func.circle_field,
        "circle",
        dict(observers=(0.2, 0.3, 0.1), diameters=1.0, currents=1.0),
        dict(diameter=1.0, current=1.0),
    ),
    "polyline": (
        m.func.polyline_field,
        "polyline",
        dict(
            observers=(0.2, 0.3, 0.1),
            segments_start=(-0.5, -1.0, 0.0),
            segments_end=(0.5, 1.0, 0.0),
            currents=1e6,
        ),
        dict(segment_start=(-0.5, -1.0, 0.0), segment_end=(0.5, 1.0, 0.0), current=1e6),
    ),
    "cuboid": (
        m.func.cuboid_field,
        "cuboid",
        dict(observers=(0.2, 1.3, 1.1), dimensions=(1.0, 1.0, 1.0), polarizations=(0.0, 0.0, 1.0)),
        dict(dimension=(1.0, 1.0, 1.0), polarization=(0.0, 0.0, 1.0)),
    ),
    "cylinder": (
        m.func.cylinder_field,
        "cylinder",
        dict(observers=(0.2, 1.3, 1.1), dimensions=(1.0, 1.0), polarizations=(0.0, 0.0, 1.0)),
        dict(dimension=(1.0, 1.0), polarization=(0.0, 0.0, 1.0)),
    ),
    "cylinder_segment": (
        m.func.cylinder_segment_field,
        "cylindersegment",
        dict(
            observers=(0.2, 0.3, 0.1),
            dimensions=(1.0, 2.0, 1.0, 45.0, 225.0),
            polarizations=(0.0, 0.0, 1.0),
        ),
        dict(dimension=(1.0, 2.0, 1.0, 45.0, 225.0), polarization=(0.0, 0.0, 1.0)),
    ),
    "sphere": (
        m.func.sphere_field,
        "sphere",
        dict(observers=(1.2, 0.3, 0.1), diameters=1.0, polarizations=(0.0, 0.0, 1.0)),
        dict(diameter=1.0, polarization=(0.0, 0.0, 1.0)),
    ),
    "tetrahedron": (
        m.func.tetrahedron_field,
        "tetrahedron",
        dict(
            observers=(-0.2, 0.3, 0.1),
            vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            polarizations=(0.0, 0.0, 1.0),
        ),
        dict(
            vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            polarization=(0.0, 0.0, 1.0),
        ),
    ),
    "dipole": (
        m.func.dipole_field,
        "dipole",
        dict(observers=(1.2, 0.3, 0.1), moments=(1e6, 0.0, 0.0)),
        dict(moment=(1e6, 0.0, 0.0)),
    ),
    "triangle_charge": (
        m.func.triangle_charge_field,
        "triangle",
        dict(
            observers=(1.2, 0.3, 0.1),
            vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            polarizations=(0.0, 0.0, 1.0),
        ),
        dict(vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)), polarization=(0.0, 0.0, 1.0)),
    ),
    "triangle_current": (
        m.func.triangle_current_field,
        "trianglesheet",
        dict(
            observers=(1.2, 0.3, 0.1),
            vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            current_densities=(1e6, 1e6, 1e6),
        ),
        dict(
            vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            faces=((0, 1, 2),),
            current_densities=((1e6, 1e6, 1e6),),
        ),
    ),
}


@pytest.mark.parametrize("name", list(SINGLE_CASES))
@pytest.mark.parametrize("field", ["B", "H"])
def test_func_matches_getBH(name, field):
    func, src_type, func_kw, getbh_kw = SINGLE_CASES[name]
    out = np.asarray(func(field, **func_kw))
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))

    getter = m.getB if field == "B" else m.getH
    ref = np.asarray(getter(src_type, func_kw["observers"], **getbh_kw))
    np.testing.assert_allclose(out, ref, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("name", list(SINGLE_CASES))
def test_func_batched_instances(name):
    """Two instances broadcast to shape (2, 3) and match per-instance getB."""
    func, src_type, func_kw, getbh_kw = SINGLE_CASES[name]

    # Duplicate every array-valued func kwarg into two instances.
    batched = {}
    for key, val in func_kw.items():
        if key == "observers":
            batched[key] = np.stack([np.asarray(val, float), np.asarray(val, float) + 0.05])
        else:
            arr = np.asarray(val, float)
            batched[key] = np.stack([arr, arr])
    out = np.asarray(func("B", **batched))
    assert out.shape == (2, 3)
    assert np.all(np.isfinite(out))

    # Instance 0 must equal the single-instance getB at the shifted observer.
    ref0 = np.asarray(m.getB(src_type, batched["observers"][0], **getbh_kw))
    np.testing.assert_allclose(out[0], ref0, rtol=1e-10, atol=1e-12)


def test_func_orientation_and_position():
    """orientation (scipy Rotation) and position kwargs behave like getB."""
    rot = R.from_euler("z", 37.0, degrees=True)
    out = np.asarray(
        m.func.cuboid_field(
            "B",
            observers=(0.5, 0.4, 0.3),
            dimensions=(1.0, 1.0, 1.0),
            polarizations=(0.0, 0.0, 1.0),
            positions=(0.1, 0.0, -0.1),
            orientations=rot,
        )
    )
    ref = np.asarray(
        m.getB(
            "cuboid",
            (0.5, 0.4, 0.3),
            dimension=(1.0, 1.0, 1.0),
            polarization=(0.0, 0.0, 1.0),
            position=(0.1, 0.0, -0.1),
            orientation=rot,
        )
    )
    np.testing.assert_allclose(out, ref, rtol=1e-11, atol=1e-12)


def test_func_squeeze_false_keeps_axis():
    out = np.asarray(m.func.dipole_field("B", (1.2, 0.3, 0.1), (1e6, 0, 0), squeeze=False))
    assert out.shape == (1, 3)


def test_func_invalid_field_raises():
    with pytest.raises(ValueError, match="field"):
        m.func.dipole_field("J", (1, 0, 0), (1, 0, 0))


def test_func_invalid_orientation_raises():
    with pytest.raises(TypeError, match="Rotation"):
        m.func.dipole_field("B", (1, 0, 0), (1, 0, 0), orientations=(0, 0, 0, 1))


def test_func_mismatched_instances_raise():
    with pytest.raises(ValueError, match="instances"):
        m.func.circle_field("B", (0.2, 0.3, 0.1), diameters=(1.0, 1.5, 2.0), currents=(1.0, 2.0))


# --- magpylib reference parity (when installed) ------------------------------
magpylib = pytest.importorskip("magpylib")

_REF_TIGHT = {"circle", "polyline", "cuboid", "cylinder", "sphere", "tetrahedron", "dipole"}


@pytest.mark.parametrize("name", list(SINGLE_CASES))
@pytest.mark.parametrize("field", ["B", "H"])
def test_func_matches_magpylib(name, field):
    func, _src_type, func_kw, _ = SINGLE_CASES[name]
    ref_func = getattr(magpylib.func, f"{name}_field")
    out = np.asarray(func(field, **func_kw))
    ref = np.asarray(ref_func(field=field, **func_kw))
    # cylinder_segment / triangle sheets use a mesh approximation -> looser tol.
    rtol = 1e-9 if name in _REF_TIGHT else 5e-4
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=1e-9)
