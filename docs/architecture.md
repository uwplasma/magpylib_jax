# Architecture

This page is the shortest path from the public API you call to the analytic formula that produces a
number. Read it top to bottom and you will know where every part of a `getB` or `getFT` call lives.

## The three layers

A field computation flows through three layers. Each has a clear job, and each is a separate part
of the source tree.

```text
   Objects layer            Fields engine              Analytic kernels
   Cuboid · Collection  ->   fields/            ->      core/kernels/
   · Sensor                  (prepare · batch          (closed-form field,
   getB/getFT/show            · jit-evaluate)            differentiable)
        ^                          |                           |
        |   rotate · aggregate     |    field in local frame   |
        +------- squeeze ----------+---------------------------+
```

1. **Objects** — the friendly, Magpylib-compatible surface. Sources, `Collection`, and `Sensor`
   carry position, orientation, motion paths, and style, and expose `getB/getH/getJ/getM/getFT/show`.
2. **Fields engine** (`fields/`) — normalizes and batches inputs, runs the vectorized JIT
   evaluation, then rotates results back to the global frame and applies sensor aggregation and
   Magpylib-compatible squeeze semantics.
3. **Kernels** (`core/kernels/`) — one pure, differentiable JAX function per source family, each the
   closed-form field of a source placed at the origin.

## Objects layer

The public surface a user touches first.

- [`__init__.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/__init__.py) —
  the package namespace. It re-exports the source classes, `Collection`, `Sensor`, the field
  functions, `show`, and `mu_0`. It **never** mutates the global JAX config.
- [`functional.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/functional.py) —
  a thin facade re-exporting `getB/getH/getJ/getM/getFT` from the `fields/` package, so historical
  `from magpylib_jax.functional import getB` imports keep working.
- [`collection.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/collection.py) —
  `Collection`, which groups (and nests) sources and behaves as a single source.
- [`sensor.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/sensor.py) —
  `Sensor`, a movable observer carrying a pixel grid.

Shared object behavior — construction, path/orientation storage and caching, input validation, and
lightweight style compatibility — lives in the object base layer:

- [`core/base.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/base.py)
- [`core/style.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/style.py)

## Fields engine (`fields/`)

This is where a high-level call becomes a batched, compiled computation. The
[`fields/`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/fields) package
splits the work into focused modules:

```{list-table}
:header-rows: 1
:widths: 26 74

* - Module
  - Responsibility
* - [`api.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/api.py)
  - Public `getB/getH/getJ/getM` and the `_compute_field` router; input normalization and squeeze.
* - [`prepare.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/prepare.py)
  - Source/sensor/observer preparation, grouping of homogeneous families, and padding for batching.
* - [`engine.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/engine.py)
  - The vectorized, JIT-compiled evaluation engine — the default path.
* - [`eager.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/eager.py)
  - An eager reference evaluator used for output modes that fall outside JIT.
* - [`force.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/force.py)
  - `getFT`: force and torque by autodiff of the field.
```

## Geometry and kernels

- [`core/geometry.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/geometry.py) —
  frame transforms: cartesian ↔ cylindrical, pose broadcasting, rotate-into-local-frame and back.
- [`core/kernels/`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/core/kernels) —
  the analytic field kernels, one module per source family (`dipole`, `circle`, `cuboid`,
  `cylinder`, `cylinder_segment`, `sphere`, `polyline`, `triangle`, `tetrahedron`, `trimesh`,
  `current_sheet`, `current_strip`), plus [`elliptic.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/elliptic.py)
  (the Bulirsch `cel` elliptic integral), `_raycast` (mesh inside-tests), and the `_safe`/`_common`
  numerical helpers.

Each kernel is a pure function of origin-local observer coordinates and is differentiable in JAX.
This is where the physics lives; the closed-form derivations are collected in
[Equation models](equations.md).

### Source wrappers

Every source class is a thin wrapper that stores parameters and dispatches to its kernel.

::::{tab-set}
:::{tab-item} Magnets
- [`magnet/cuboid.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/cuboid.py)
- [`magnet/cylinder.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/cylinder.py)
- [`magnet/cylinder_segment.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/cylinder_segment.py)
- [`magnet/sphere.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/sphere.py)
- [`magnet/tetrahedron.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/tetrahedron.py)
- [`magnet/triangular_mesh.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/magnet/triangular_mesh.py)
:::
:::{tab-item} Currents
- [`current/circle.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/current/circle.py)
- [`current/polyline.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/current/polyline.py)
- [`current/triangle_sheet.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/current/triangle_sheet.py)
- [`current/triangle_strip.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/current/triangle_strip.py)
:::
:::{tab-item} Misc
- [`misc/dipole.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/misc/dipole.py)
- [`misc/triangle.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/misc/triangle.py)
- [`misc/custom.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/misc/custom.py)
:::
::::

## How a `getB` call flows through the code

A typical high-level `getB` follows this path:

1. **Validate & normalize** the source/sensor descriptors and observer array (`api.py`).
2. **Prepare** source and sensor tensors, reusing caches where possible, and pad for batching
   (`prepare.py`).
3. **Group** homogeneous source families for efficient batched evaluation.
4. **Evaluate** the matching analytic kernel in the source's local frame (`engine.py` → `core/kernels/`).
5. **Rotate** the resulting field back to the global frame (`core/geometry.py`).
6. **Aggregate & squeeze** — apply sensor pixel aggregation and Magpylib-compatible squeeze
   semantics (`api.py`).

`getFT` reuses exactly this field path through
[`fields/force.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/force.py),
adding a `jax.jacfwd` of the field for the magnet-gradient term — which is why force and torque are
exact rather than finite-difference estimates.

## Where to profile

- Kernel compile/runtime — [`scripts/profile_kernels.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/profile_kernels.py)
- High-level `getB` overhead — [`scripts/profile_getB_jit.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/profile_getB_jit.py)
- Figures & benchmarks — [`scripts/make_figures.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/make_figures.py)

For the measurement methodology and the honest CPU benchmark, see [Performance](performance.md).
