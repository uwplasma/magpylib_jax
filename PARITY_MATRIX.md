# Parity Matrix (magpylib -> magpylib_jax)

This matrix tracks upstream test-file parity and corresponding local checks.

## Legend
- `Implemented`: kernel/object/API ported with parity tests and CI gates.
- `Partial`: partially covered; key behavior/parity checks exist but full upstream coverage is not yet mirrored.
- `Pending`: not ported yet.

## Source-kernel parity

| Upstream capability | Upstream tests (examples) | Local equivalent tests | Status |
| --- | --- | --- | --- |
| Dipole | `tests/test_dipole_moment_testing.py` | `tests/parity/test_dipole_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| Circle current | `tests/test_field_circle.py`, `tests/test_obj_Circle.py` | `tests/parity/test_circle_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| Cuboid magnet | `tests/test_obj_Cuboid.py` | `tests/parity/test_cuboid_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| Cylinder magnet | `tests/test_field_cylinder.py`, `tests/test_obj_Cylinder.py` | `tests/parity/test_cylinder_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| Sphere magnet | `tests/test_obj_Sphere.py` | `tests/parity/test_sphere_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| Polyline current | `tests/test_obj_Polyline.py` | `tests/parity/test_polyline_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| Triangle surface | `tests/test_obj_Triangle.py` | `tests/parity/test_triangle_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| Tetrahedron magnet | `tests/test_obj_Tetrahedron.py` | `tests/parity/test_tetrahedron_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| CylinderSegment magnet | `tests/test_field_cylinder.py` (segment sections) | N/A | Pending |
| TriangleSheet / TriangleStrip currents | `tests/test_obj_TriangleStrip_Sheet.py` | N/A | Pending |
| TriangularMesh magnet | `tests/test_obj_TriangularMesh.py` | N/A | Pending |

## API and behavior parity

| Upstream area | Local tests | Status |
| --- | --- | --- |
| Object + functional dispatch | `tests/test_compatibility.py` | Implemented |
| `B/H/J/M` behavior gates by source profiles | `tests/parity_gates/test_source_profiles.py` | Implemented |
| `squeeze` behavior (core shape patterns) | `tests/parity_gates/test_source_profiles.py::test_squeeze_and_sumup_shapes` | Partial |
| Motion/path/orientation full semantics | N/A | Pending |

## Physics + regression checks

| Area | Local tests | Status |
| --- | --- | --- |
| Physics identities | `tests/physics/test_core_physics.py` | Implemented |
| Gradient/differentiability checks | `tests/differentiability/` | Implemented |
| Benchmark parity and thresholds | `scripts/benchmark_vs_magpylib.py`, `scripts/check_benchmark_thresholds.py`, CI benchmark jobs | Implemented |
