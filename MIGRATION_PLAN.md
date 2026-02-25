# Magpylib to magpylib_jax migration plan

## Mission

Deliver a fully differentiable, JAX-native replacement for Magpylib with matched physics, test parity, benchmarked performance, and production-grade docs/CI.

## Guiding constraints

- Every new kernel must be differentiable in JAX (except physical singularities where fields are undefined).
- Numerical parity is measured against upstream Magpylib before feature completion is claimed.
- CI must validate tests, lint/types, and documentation for every change.

## Implementation matrix

| Capability | Status | Parity Tests | Differentiability Tests |
| --- | --- | --- | --- |
| Dipole source (`misc.Dipole`) | Implemented | Yes | Yes |
| Circle current source (`current.Circle`) | Implemented | Yes | Yes |
| Cuboid magnet (`magnet.Cuboid`) | Implemented | Yes | Yes |
| Cylinder magnet (`magnet.Cylinder`) | Implemented | Yes | Yes |
| Sphere magnet (`magnet.Sphere`) | Implemented | Yes | Yes |
| Polyline current (`current.Polyline`) | Implemented | Yes | Partial |
| Triangle surface (`misc.Triangle`) | Implemented | Yes | Partial |
| Tetrahedron magnet (`magnet.Tetrahedron`) | Implemented | Yes | Partial |
| CylinderSegment magnet | Planned | Planned | Planned |
| TriangleSheet / TriangleStrip currents | Planned | Planned | Planned |
| TriangularMesh magnet | Planned | Planned | Planned |
| Collection/Sensor/path interfaces | Partial (Collection/Sensor + squeeze/sumup basics) | Partial | Partial |

## Phases

1. Core physics kernels
- Port source kernels one-by-one into JAX (`jit` + batch support).
- Keep each kernel isolated and benchmarked.

2. Public API compatibility
- Add object model and functional API wrappers matching Magpylib behavior.
- Add compatibility layer for common constructor and field-call patterns.

3. Validation and regression gating
- Grow parity tests to match upstream source test coverage for each implemented source.
- Add gradient stability tests with finite-difference checks.

4. Performance and memory
- Add benchmark suite (single-source, multi-source, batched observers).
- Track memory profile and kernel compile/runtime split.

5. Documentation and release
- Expand user docs, API reference, migration notes.
- Wire Read the Docs and release workflow.

## Execution status

- Phase 1 advanced: dipole, circle, cuboid, cylinder, sphere, polyline, triangle, and tetrahedron kernels implemented.
- Phase 2 advanced: expanded functional/object API + compatibility objects and basic squeeze/sumup behavior.
- Phase 3 advanced: parity/physics/differentiability coverage and strict parity gate tests in CI.
