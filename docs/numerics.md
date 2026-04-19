# Numerics

This page describes the numerical strategy that sits between the analytical formulas and the public API.

## Stability strategy

The code does not assume that the closed-form expression which is shortest on paper is also the one that is most stable numerically. Several kernels therefore use explicit branch structure.

Main techniques used in the codebase:

- dimensionless scaling where available,
- explicit handling of singular locations and limit cases,
- masked special-case formulas for near-degenerate geometries,
- stable elliptic-integral helpers for circular and cylindrical sources,
- geometric precomputation for face-based source families,
- chunked accumulation for large source collections.

## Coordinate transforms and local frames

Most kernels become simpler after transforming observers into a source-local frame. The transform itself is numerical work, so the code caches orientation matrices and prepared path data where possible.

This keeps the analytical kernels focused on local-frame geometry and avoids repeatedly calling host-side rotation conversions during repeated `getB` calls.

## Singular neighborhoods

Magnetic source models are not smooth everywhere. The code treats the following sets carefully:

- observers on line-current paths,
- observers on current-sheet edges,
- observers on magnet boundaries,
- inside/outside transitions for meshes and segmented solids,
- axis limits for rotationally symmetric sources.

In these regions the implementation uses masks and branch-specific formulas to match physically expected limits and upstream Magpylib behavior.

## Differentiability

JAX differentiability is a design goal, but physics still wins over symbolic smoothness.

Practical interpretation:

- away from singular sets, gradients are expected to be reliable,
- at physical discontinuities or branch boundaries, derivatives can be undefined or numerically unstable,
- compatibility behavior on boundaries sometimes requires piecewise logic that is correct physically but not everywhere smooth.

The differentiability tests therefore focus on representative off-singularity points and on regression coverage for previously problematic kernels.

## Precision mode

The tests and profiling scripts use `jax_enable_x64=True`. That is intentional: many parity checks depend on double precision, especially for:

- near-boundary evaluation,
- mesh and tetrahedron geometry reductions,
- elliptic-integral based kernels,
- benchmark comparisons at strict tolerances.

Using x64 is strongly recommended for scientific workloads.

## High-level API numerics

The public object and functional APIs add another layer of numerical responsibility:

- path broadcasting,
- squeeze behavior,
- sensor pixel aggregation,
- source collection flattening,
- mixed-source batching.

These are not just formatting concerns. They control tensor layout, memory pressure, and how much host-side work occurs before the JAX kernels even start.

## Memory behavior

A differentiable field path can become memory-heavy when it materializes very large intermediate tensors. To control this, the code uses:

- observer-aware chunking for large circle collections,
- cached prepared source/sensor tensors,
- fixed-observer-count JIT wrappers for hotspot kernels,
- mesh geometry precomputation to reduce repeated allocations.

For details, see [Performance](performance.md).

## Validation philosophy

Numerical behavior is validated along three axes:

1. parity with upstream Magpylib,
2. physics consistency checks,
3. profiling regression gates for runtime, memory, and HLO size.

This matters because a kernel can be numerically correct but operationally unusable if compile time, memory, or shape behavior regresses.

## Relevant source files

- [`src/magpylib_jax/core/elliptic.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/elliptic.py)
- [`src/magpylib_jax/core/geometry.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/geometry.py)
- [`src/magpylib_jax/core/kernels.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels.py)
- [`src/magpylib_jax/core/kernels_extended.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels_extended.py)
- [`src/magpylib_jax/functional.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/functional.py)
