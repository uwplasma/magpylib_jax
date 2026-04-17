# Overview

`magpylib_jax` is a JAX-native magnetic field library aimed at optimization, inverse design, and differentiable physics workflows that still need Magpylib-like object and functional APIs.

## What 1.0 ships

- Functional API: `getB`, `getH`, `getJ`, `getM`
- Object API: `Collection`, `Sensor`, and the implemented source classes
- JIT-safe high-level field evaluation by default
- End-to-end differentiability through JAX
- Parity-oriented behavior for source shaping, path/orientation handling, and squeeze semantics

## Implemented source families

- `misc.Dipole`
- `current.Circle`
- `current.Polyline`
- `misc.Triangle`
- `current.TriangleSheet`
- `current.TriangleStrip`
- `magnet.Cuboid`
- `magnet.Cylinder`
- `magnet.CylinderSegment`
- `magnet.Sphere`
- `magnet.Tetrahedron`
- `magnet.TriangularMesh`

## Validation model

The release is gated by:

- direct numeric parity tests against upstream Magpylib,
- mirrored upstream object/API tests,
- physics and differentiability tests,
- CI coverage enforcement,
- benchmark regression checks,
- profiling regression checks with HLO and memory snapshots.

## Performance model

The current implementation is optimized for repeated object-based evaluations and differentiable outer loops:

- cached source and sensor preparation,
- cached collection flattening and orientation matrices,
- fast paths for large static circle collections,
- reusable `TriangularMesh` and `CylinderSegment` geometry preparation,
- kernel-level JIT entrypoints for hotspot source families.
