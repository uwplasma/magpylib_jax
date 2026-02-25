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
| CylinderSegment magnet | `tests/test_obj_CylinderSegment.py`, `tests/test_field_cylinder.py` | `tests/parity/test_cylinder_segment_parity.py`, `tests/upstream_mirror/test_obj_CylinderSegment.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| Sphere magnet | `tests/test_obj_Sphere.py` | `tests/parity/test_sphere_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| Polyline current | `tests/test_obj_Polyline.py` | `tests/parity/test_polyline_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| Triangle surface | `tests/test_obj_Triangle.py` | `tests/parity/test_triangle_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| TriangleSheet current | `tests/test_obj_TriangleStrip_Sheet.py` | `tests/parity/test_trianglesheet_parity.py`, `tests/upstream_mirror/test_obj_TriangleStrip_Sheet.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| TriangleStrip current | `tests/test_obj_TriangleStrip_Sheet.py` | `tests/parity/test_trianglestrip_parity.py`, `tests/upstream_mirror/test_obj_TriangleStrip_Sheet.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| TriangularMesh magnet | `tests/test_obj_TriangularMesh.py` | `tests/parity/test_triangular_mesh_parity.py`, `tests/upstream_mirror/test_obj_TriangularMesh.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| Tetrahedron magnet | `tests/test_obj_Tetrahedron.py` | `tests/parity/test_tetrahedron_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |

## Upstream file mirror mapping

| Upstream file | Local mirrored coverage | Status |
| --- | --- | --- |
| `tests/test_obj_CylinderSegment.py` | `tests/upstream_mirror/test_obj_CylinderSegment.py` | Implemented |
| `tests/test_obj_TriangleStrip_Sheet.py` | `tests/upstream_mirror/test_obj_TriangleStrip_Sheet.py` | Implemented |
| `tests/test_obj_TriangularMesh.py` | `tests/upstream_mirror/test_obj_TriangularMesh.py` | Implemented |
| `tests/test_field_cylinder.py` (segment sections) | `tests/parity/test_cylinder_segment_parity.py`, `tests/parity_gates/test_source_profiles.py` | Implemented |
| `tests/test_BHMJ_level.py` (mesh/sheet/strip portions) | `tests/parity_gates/test_source_profiles.py` | Partial |
| `tests/test_obj_Sensor.py` | `tests/upstream_mirror/test_obj_Sensor.py` | Partial |
| `tests/test_obj_Collection.py` | `tests/upstream_mirror/test_obj_Collection.py` | Partial |
| `tests/test_getBH_interfaces.py` | `tests/upstream_mirror/test_getBH_interfaces.py` | Partial |
| `tests/test_obj_BaseGeo.py` | `tests/upstream_mirror/test_obj_BaseGeo.py` | Partial |
| `tests/test_obj_BaseGeo_v4motion.py` | `tests/upstream_mirror/test_obj_BaseGeo_v4motion.py` | Partial |
| `tests/test_path.py` | `tests/upstream_mirror/test_path.py` | Partial |
| `tests/test_physics_consistency.py` | `tests/upstream_mirror/test_physics_consistency.py` | Partial |

## API and behavior parity

| Upstream area | Local tests | Status |
| --- | --- | --- |
| Object + functional dispatch | `tests/test_compatibility.py` | Implemented |
| Motion/path/orientation semantics | `tests/test_motion_shape_compat.py` | Implemented |
| Multi-source/sensor shaping and `squeeze` behavior | `tests/test_motion_shape_compat.py`, `tests/parity_gates/test_source_profiles.py::test_squeeze_and_sumup_shapes` | Implemented |
| `B/H/J/M` behavior gates by source profiles | `tests/parity_gates/test_source_profiles.py` | Implemented |

## Physics + regression checks

| Area | Local tests | Status |
| --- | --- | --- |
| Physics identities | `tests/physics/test_core_physics.py` | Implemented |
| Gradient/differentiability checks | `tests/differentiability/` | Implemented |
| Benchmark parity and thresholds | `scripts/benchmark_vs_magpylib.py`, `scripts/check_benchmark_thresholds.py`, CI benchmark jobs | Implemented |
| Kernel-level profiling + HLO/memory snapshots | `scripts/profile_kernels.py` (added), CI profiling workflow | Implemented |
