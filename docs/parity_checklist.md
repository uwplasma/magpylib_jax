# Parity Checklist

This checklist tracks upstream Magpylib tests and their mirrored or equivalent coverage in `magpylib_jax`.
For the canonical mapping, see `PARITY_MATRIX.md` in the repository root.

## Upstream file coverage

| Upstream test file | Local coverage | Status |
| --- | --- | --- |
| `tests/test_getBH_interfaces.py` | `tests/upstream_mirror/test_getBH_interfaces.py` | Implemented |
| `tests/test_obj_BaseGeo.py` | `tests/upstream_mirror/test_obj_BaseGeo.py` | Implemented |
| `tests/test_obj_BaseGeo_v4motion.py` | `tests/upstream_mirror/test_obj_BaseGeo_v4motion.py` | Implemented |
| `tests/test_obj_Collection.py` | `tests/upstream_mirror/test_obj_Collection.py` | Implemented |
| `tests/test_obj_CylinderSegment.py` | `tests/upstream_mirror/test_obj_CylinderSegment.py` | Implemented |
| `tests/test_obj_TriangleStrip_Sheet.py` | `tests/upstream_mirror/test_obj_TriangleStrip_Sheet.py` | Implemented |
| `tests/test_obj_TriangularMesh.py` | `tests/upstream_mirror/test_obj_TriangularMesh.py` | Implemented |
| `tests/test_BHMJ_level.py` (mesh/sheet/strip portions) | `tests/upstream_mirror/test_BHMJ_level.py` | Implemented |
| `tests/test_obj_Sensor.py` | `tests/upstream_mirror/test_obj_Sensor.py` | Implemented |
| `tests/test_path.py` | `tests/upstream_mirror/test_path.py` | Implemented |
| `tests/test_physics_consistency.py` | `tests/upstream_mirror/test_physics_consistency.py` | Implemented |

## Parity gates

- `tests/parity_gates/test_source_profiles.py` (core profile grid coverage)
- `tests/parity_gates/test_source_boundaries.py` (inside/surface/outside neighborhood checks)

When adding new upstream mirrors, update `PARITY_MATRIX.md` and this page together.
