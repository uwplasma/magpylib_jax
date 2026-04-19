# Overview

`magpylib_jax` is a JAX-native magnetic field library for optimization, inverse design, and differentiable physics workflows that still need a Magpylib-like object and functional API.

## Design goals

The project is built around a narrow but demanding set of requirements:

1. Closed-form source models instead of a generic field solver.
2. Full compatibility with JAX transforms where the physics is smooth.
3. High-level API parity with Magpylib where users already have object graphs, sensors, and path semantics.
4. Explicit validation against upstream Magpylib, not only local unit tests.
5. Performance observability through benchmark and profiling gates.

## What 1.0 ships

- Functional API: `getB`, `getH`, `getJ`, `getM`
- Object API: `Collection`, `Sensor`, and implemented source classes
- JIT-safe high-level field evaluation by default
- End-to-end differentiability through JAX
- Parity-oriented behavior for source shaping, path/orientation handling, pixel aggregation, and squeeze semantics

## Implemented source families

## Current-driven objects

- `current.Circle`
- `current.Polyline`
- `current.TriangleSheet`
- `current.TriangleStrip`

## Permanent-magnet objects

- `magnet.Cuboid`
- `magnet.Cylinder`
- `magnet.CylinderSegment`
- `magnet.Sphere`
- `magnet.Tetrahedron`
- `magnet.TriangularMesh`
- `misc.Triangle`
- `misc.Dipole`

## API model

The library supports two complementary ways to work:

- Object API: construct sources, sensors, and collections, then call methods such as `src.getB(obs)` or `sens.getB(collection)`.
- Functional API: call top-level functions such as `magpylib_jax.getB(...)` directly with source descriptors and arrays.

The object API is the compatibility layer. The functional and kernel layers are where performance and differentiation are concentrated.

## Validation model

Releases are gated by:

- direct numeric parity tests against upstream Magpylib,
- mirrored upstream object/API tests,
- physics-based consistency tests,
- differentiability tests,
- CI coverage enforcement,
- benchmark regression checks,
- profiling regression checks with HLO and memory snapshots.

See [Testing and Validation](testing.md) and [Parity Strategy](parity.md).

## Performance model

The implementation is optimized for repeated evaluations and differentiable outer loops:

- cached source and sensor preparation,
- cached collection flattening and orientation matrices,
- fast paths for large static circle collections,
- reusable `TriangularMesh` and `CylinderSegment` geometry preparation,
- kernel-level JIT entrypoints for hotspot source families.

See [Performance](performance.md).

## Source code entry points

- High-level API: [`src/magpylib_jax/functional.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/functional.py)
- Object base classes: [`src/magpylib_jax/core/base.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/base.py)
- Analytical kernels: [`src/magpylib_jax/core/kernels.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels.py)
- Extended kernels and hotspot JIT wrappers: [`src/magpylib_jax/core/kernels_extended.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels_extended.py)
