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
| Cuboid magnet | Planned | Planned | Planned |
| Cylinder magnet | Planned | Planned | Planned |
| Sphere magnet | Planned | Planned | Planned |
| Triangle/Polyline/TriangularMesh/Tetrahedron | Planned | Planned | Planned |
| Collection/Sensor/path interfaces | Planned | Planned | Planned |

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

- Phase 1 started: dipole and circle kernels implemented.
- Phase 2 started: functional/object API scaffolding implemented.
- Phase 3 started: parity + differentiability tests for implemented sources added.

