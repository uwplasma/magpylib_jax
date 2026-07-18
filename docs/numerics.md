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

magpylib_jax follows your JAX precision setting and **never mutates the global JAX config on
import**. Arrays use JAX's default float dtype: `float32` unless you enable x64. The kernels do not
hard-code `float64`, so a single toggle switches the whole library between single and double
precision:

```python
import jax
jax.config.update("jax_enable_x64", True)   # float64; needed for magpylib parity
import magpylib_jax as mpj
```

Double precision is recommended for scientific workloads and is required for bit-level magpylib
parity, especially for near-boundary evaluation, mesh/tetrahedron geometry reductions,
elliptic-integral kernels, and strict-tolerance benchmarks. The test suite enables x64 (via
`conftest.py`). `float32` is the fast default on GPU/TPU and is fine when magpylib-level accuracy
is not required.

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

## Differentiable field API

Every kernel is a pure JAX function, so `getB/getH/getJ/getM` and `getFT` are differentiable
with `jax.grad`, `jacfwd`, and `jacrev` with respect to observer positions, source pose
(position/orientation), and excitation (polarization, current, moment). Gradients are smooth and
finite everywhere the field itself is defined, which is the region that matters for optimization
and inverse design.

The exception is the **singular set** of each source — the points where the closed-form field
diverges (a dipole at its own location, a point on a current wire, the surface of a current sheet,
a magnet face or edge). There the field is physically infinite or undefined, and the gradient may
be non-finite. Optimizers should keep observers off the singular set (as they must anyway for the
field value to be meaningful). Hardening selected kernels with `custom_jvp` so that the singular
set returns a defined (zero) tangent is a planned refinement.

## Relevant source files

- [`src/magpylib_jax/core/elliptic.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/elliptic.py)
- [`src/magpylib_jax/core/geometry.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/geometry.py)
- [`src/magpylib_jax/core/kernels/`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/core/kernels)
- [`src/magpylib_jax/fields/`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/fields)
