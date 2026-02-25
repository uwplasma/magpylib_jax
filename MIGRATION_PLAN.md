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
| CylinderSegment magnet (`magnet.CylinderSegment`) | Implemented | Yes | Yes |
| Sphere magnet (`magnet.Sphere`) | Implemented | Yes | Yes |
| Polyline current (`current.Polyline`) | Implemented | Yes | Yes |
| Triangle surface (`misc.Triangle`) | Implemented | Yes | Yes |
| TriangleSheet current (`current.TriangleSheet`) | Implemented | Yes | Yes |
| TriangleStrip current (`current.TriangleStrip`) | Implemented | Yes | Yes |
| TriangularMesh magnet (`magnet.TriangularMesh`) | Implemented | Yes | Yes |
| Tetrahedron magnet (`magnet.Tetrahedron`) | Implemented | Yes | Yes |
| Collection/Sensor/path interfaces | Implemented (core workflows) | Yes | Partial |

## Phases

1. Core physics kernels
- Port source kernels one-by-one into JAX (`jit` + batch support).
- Keep each kernel isolated and benchmarked.

2. Public API compatibility
- Add object model and functional API wrappers matching Magpylib behavior.
- Add compatibility layer for constructor and field-call patterns.

3. Validation and regression gating
- Grow parity tests to mirror upstream source coverage categories.
- Add gradient stability tests with finite-difference sanity checks.

4. Performance and memory
- Benchmark suite (single-source, multi-source, batched observers).
- Kernel profiling (compile/runtime/HLO/trace/memory) with CI thresholds.

5. Documentation and release
- Expand user docs, API reference, migration notes.
- Wire Read the Docs and release workflow.

## Execution status

- Phase 1: Implemented for currently targeted source families.
- Phase 2: Implemented for core path/orientation/sensor shaping workflows.
- Phase 3: Implemented with parity gates, mirrored upstream object tests, and >90% coverage policy.
- Phase 4: Implemented baseline benchmark + profiling gates; optimization loop continues.
- Phase 5: In progress (ongoing documentation expansion and release hardening).

## Next optimization targets

1. Tighten parity tolerances around singular neighborhoods for sheet/segment kernels.
2. Continue allocation reduction and shape-specialized JIT paths for triangle/tetra/cylinder families.
3. Expand nightly profiling trend analysis and automatic regression reporting.
