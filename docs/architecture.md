# Architecture and Source Map

This page is the shortest path from the public API to the underlying analytical kernels.

## Layered structure

The repository is organized in layers.

## Public API layer

- [`src/magpylib_jax/__init__.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/__init__.py)
- [`src/magpylib_jax/functional.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/functional.py)
- [`src/magpylib_jax/collection.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/collection.py)
- [`src/magpylib_jax/sensor.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/sensor.py)

This layer handles:

- object construction,
- compatibility-oriented `getB/getH/getJ/getM` behavior,
- source/sensor formatting,
- path and orientation semantics,
- pixel aggregation,
- squeeze and broadcasting behavior.

## Object base layer

- [`src/magpylib_jax/core/base.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/base.py)
- [`src/magpylib_jax/core/style.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/style.py)

This layer handles:

- shared object state,
- path mutation semantics,
- orientation storage and caching,
- validation of constructor and motion inputs,
- lightweight style compatibility.

## Geometry and kernel layer

- [`src/magpylib_jax/core/geometry.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/geometry.py)
- [`src/magpylib_jax/core/elliptic.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/elliptic.py)
- [`src/magpylib_jax/core/kernels.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels.py)
- [`src/magpylib_jax/core/kernels_extended.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels_extended.py)

This is where the actual field formulas live.

## Source wrappers

Current-driven sources:

- [`src/magpylib_jax/current/circle.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/current/circle.py)
- [`src/magpylib_jax/current/polyline.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/current/polyline.py)
- [`src/magpylib_jax/current/triangle_sheet.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/current/triangle_sheet.py)
- [`src/magpylib_jax/current/triangle_strip.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/current/triangle_strip.py)

Magnet sources:

- [`src/magpylib_jax/magnet/cuboid.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/cuboid.py)
- [`src/magpylib_jax/magnet/cylinder.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/cylinder.py)
- [`src/magpylib_jax/magnet/cylinder_segment.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/cylinder_segment.py)
- [`src/magpylib_jax/magnet/sphere.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/sphere.py)
- [`src/magpylib_jax/magnet/tetrahedron.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/tetrahedron.py)
- [`src/magpylib_jax/magnet/triangular_mesh.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/triangular_mesh.py)

Miscellaneous sources:

- [`src/magpylib_jax/misc/dipole.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/misc/dipole.py)
- [`src/magpylib_jax/misc/triangle.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/misc/triangle.py)
- [`src/magpylib_jax/misc/custom.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/misc/custom.py)

## How `getB` flows through the code

A typical high-level `getB` call follows this path:

1. Validate inputs and normalize source/sensor descriptors.
2. Prepare source tensors and sensor tensors, reusing caches where possible.
3. Group homogeneous source families for efficient batched evaluation.
4. Call the matching analytical kernel.
5. Rotate the resulting field back to the global frame.
6. Apply sensor aggregation and Magpylib-compatible squeeze behavior.

The orchestration lives in [`functional.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/functional.py).

## Where to profile

If the issue is:

- compile time or HLO size: start in [`scripts/profile_kernels.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/profile_kernels.py)
- high-level `getB` overhead: start in [`scripts/profile_getB_jit.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/profile_getB_jit.py)
- large loop/coil workloads: start in [`scripts/profile_wham_workload.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/profile_wham_workload.py)
