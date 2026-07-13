# Architecture and Source Map

This page is the shortest path from the public API to the underlying analytical kernels.

## Layered structure

The repository is organized in layers.

## Public API layer

- [`src/magpylib_jax/__init__.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/__init__.py) — package surface; enables JAX x64 on import.
- [`src/magpylib_jax/functional.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/functional.py) — thin re-export facade over the `fields/` package (`getB/getH/getJ/getM/getFT`).
- [`src/magpylib_jax/collection.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/collection.py)
- [`src/magpylib_jax/sensor.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/sensor.py)

This layer handles:

- object construction,
- compatibility-oriented `getB/getH/getJ/getM` behavior,
- source/sensor formatting,
- path and orientation semantics,
- pixel aggregation,
- squeeze and broadcasting behavior.

## Field engine layer (`fields/`)

- [`fields/api.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/api.py) — public `getB/getH/getJ/getM` and the `_compute_field` router.
- [`fields/prepare.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/prepare.py) — source/sensor/observer preparation and padding.
- [`fields/engine.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/engine.py) — the vectorized JIT evaluation engine (default path).
- [`fields/eager.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/eager.py) — eager reference evaluator for output modes outside JIT.
- [`fields/cache.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/cache.py) — preparation caches.
- [`fields/force.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/force.py) — `getFT`, autodiff force and torque.

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

- [`src/magpylib_jax/core/geometry.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/geometry.py) — frame transforms, cartesian↔cylindrical, pose broadcasting.
- [`src/magpylib_jax/core/kernels/`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/core/kernels) — the analytic field kernels, one module per source family (`dipole`, `circle`, `cuboid`, `cylinder`, `sphere`, `polyline`, `triangle`, `tetrahedron`, `trimesh`, `cylinder_segment`, `current_sheet`, `current_strip`), plus `elliptic` (Bulirsch `cel`), `_raycast` (mesh inside-tests), and `_safe`/`_common` helpers.

This is where the actual field formulas live. Each kernel is a pure function of
origin-local observer coordinates and is differentiable in JAX.

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

The orchestration lives in the [`fields/`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/fields)
package: `api.py` routes the call, `prepare.py` builds the batched tensors, and `engine.py`
runs the vectorized JIT kernel. Force and torque follow the same field path through
[`fields/force.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/force.py),
adding a `jax.jacfwd` of the field for the magnet gradient term.

## Where to profile

- kernel compile/runtime: [`scripts/profile_kernels.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/profile_kernels.py)
- high-level `getB` overhead: [`scripts/profile_getB_jit.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/profile_getB_jit.py)
- figures/benchmarks: [`scripts/make_figures.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/make_figures.py)
