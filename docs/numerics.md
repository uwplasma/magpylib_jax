# Numerics

This page describes the numerical strategy that sits between the analytical formulas in
[Equation models](equations.md) and the public API. The recurring theme is that the
expression which is shortest on paper is rarely the one that is most stable — or most
differentiable — in floating point, so the kernels add explicit structure for singular
neighbourhoods, gradients, coordinate frames, and memory.

## Stability strategy

Main techniques used across the kernels:

- dimensionless scaling where a natural length exists (loop radius, cylinder radius),
- explicit handling of singular locations and limit cases through boolean masks,
- masked special-case formulas for near-degenerate geometry (near-axis series in the
  cylinder, in-plane branches in the current sheet),
- stable elliptic-integral helpers (`cel` and derived $K/E/\Pi$) for circular and
  cylindrical sources,
- geometric precomputation for face-based families (triangle normals, edge vectors),
- chunked accumulation for large source collections.

## Gradient-safe primitives and the double-where trap

Masking a singular formula for its *value* is not enough under autodiff. If a kernel
writes `jnp.where(mask, safe_expr, singular_expr)`, reverse- and forward-mode AD still
differentiate **both** branches; when the unused branch evaluates to `Inf`/`NaN`, the
chain rule multiplies it by a zero cotangent and produces `0 * Inf = NaN`, which then
propagates through the whole gradient. This is the classic *double-where* trap.

magpylib_jax defuses it with two complementary measures:

- **Inner masking of the argument.** The value-level `where` is paired with a second
  `where` *inside* the dangerous op, so the unused branch is evaluated at a benign
  argument (e.g. `jnp.where(pos, x, 1.0)` before a `1/sqrt`). Both the primal and its
  tangent then stay finite.
- **`custom_jvp` safe primitives.** The delicate scalar operations carry a hand-written
  JVP whose primal is byte-for-byte identical to the naive expression but whose tangent
  is a defined, finite value at the singular argument. These live in
  [`_safe.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/_safe.py)
  and [`_common.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/_common.py):

```{list-table}
:header-rows: 1
:widths: 26 74

* - Primitive
  - Primal / tangent behaviour at the singular argument
* - `_safe_sqrt(x)`
  - $\sqrt{\max(x,0)}$; tangent $0.5/\sqrt{x}$ for $x>0$ and $0$ at $x\le0$.
* - `_norm_sqrt(s)` / `_safe_norm(v)`
  - $\sqrt{\max(s,10^{-30})}$; tangent $(\mathbf v\cdot\mathrm d\mathbf v)/\lVert\mathbf v\rVert$ for $\lVert\mathbf v\rVert>0$, else $0$.
* - `_safe_atanh(x)`
  - $\operatorname{artanh}$ clamped to $(-1,1)$; tangent $1/(1-x^2)$ inside, finite $0$ at the clamp.
* - `_safe_logabs(x)`
  - $\log\max(\lvert x\rvert,10^{-30})$; tangent $1/x$ inside, finite $0$ at $x=0$.
* - `_safe_arctan2(y,x)`
  - exact $\operatorname{atan2}$; tangent $(x\,\mathrm dy-y\,\mathrm dx)/(x^2+y^2)$, finite $0$ at the origin.
```

The dipole kernel adds a third pattern: the hard `Inf` at the source point is written by
an explicit overwrite whose gradient is severed with `jax.lax.stop_gradient`, so
`grad`/`jacfwd` on the singular set return a finite $0$ instead of `NaN`.

## Coordinate transforms and local frames

Most kernels become dramatically simpler in a source-local frame, so the field pipeline
factors the pose out of the analytics. Observers are mapped in by
`to_local_coordinates` — a rigid transform $\mathbf{o}_{\text{local}}=(\mathbf{o}-\mathbf{p})R$
for source position $\mathbf{p}$ and orientation matrix $R$ — the kernel evaluates a
canonical geometry (source at the origin, axis along $z$), and the result is rotated
back by `to_global_field` ($\mathbf{f}_{\text{global}}=\mathbf{f}_{\text{local}}R^{\top}$).
Rotationally symmetric families additionally convert to cylindrical coordinates
(`cart_to_cyl`) and back (`cyl_field_to_cart`). Orientation matrices and prepared path
tensors are cached so repeated `getB` calls do not re-run host-side rotation
conversions. See
[`core/geometry.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/geometry.py).

The current-sheet family carries this further: each triangle is brought into the
canonical *elementar sheet* frame by three composed rotations, so a single closed form
serves every triangle orientation ([`current_sheet.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/current_sheet.py)).

## Singular neighborhoods

Magnetic source models are not smooth everywhere. The kernels treat these sets with
masks and branch-specific formulas that match the physically expected limit and upstream
Magpylib behaviour:

- observers on line-current paths (polyline, circle wire),
- observers on current-sheet edges and surfaces,
- observers on magnet faces and edges,
- inside/outside transitions for meshes and segmented solids (handled by
  Möller–Trumbore ray casting in
  [`_raycast.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/_raycast.py)
  and barycentric tests for the tetrahedron),
- axis limits for rotationally symmetric sources (near-axis series in the cylinder).

## Differentiability

JAX differentiability is a design goal, but physics wins over symbolic smoothness:

- away from singular sets, gradients are reliable and exact,
- at physical discontinuities or branch boundaries derivatives can be undefined or
  numerically unstable,
- compatibility behaviour on boundaries sometimes needs piecewise logic that is correct
  physically but not everywhere smooth.

The differentiability tests therefore focus on representative off-singularity points and
carry regression coverage for kernels that were historically problematic.

## Differentiable field API

Every kernel is a pure JAX function, so `getB`/`getH`/`getJ`/`getM` and `getFT` are
differentiable with `jax.grad`, `jacfwd`, and `jacrev` with respect to observer
positions, source pose (position/orientation), and excitation (polarization, current,
moment). Thanks to the gradient-safe primitives above, gradients are finite and smooth
everywhere the field itself is defined — the region that matters for optimisation and
inverse design.

The exception is each source's **singular set**, where the closed-form field diverges (a
dipole at its own location, a point on a current wire, the surface of a current sheet, a
magnet face or edge). There the field is physically infinite or undefined; the
`custom_jvp` primitives keep the tangent finite (typically $0$) rather than `NaN`, but
the value itself is not meaningful. Optimisers should keep observers off the singular set
regardless.

## Engine and eager evaluation paths

The library has two evaluators behind the public API, which must agree to floating-point
tolerance:

- **The vectorized JIT engine** ([`fields/engine.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/engine.py))
  is the fast path. It holds a single compiled per-pose/per-chunk core
  (`_compute_field_jit_core`) that batches sources, walks the motion path with
  `jax.lax.scan`, applies masked pixel aggregation, and reuses compiled artifacts across
  calls with matching shapes.
- **The eager reference path** ([`fields/eager.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/eager.py))
  evaluates pose by pose without the fused kernel. It covers the output modes the JIT
  engine intentionally does not (callable/unsupported `pixel_agg`, non-uniform pixel
  grids, dataframe output, 4-D meshes) and serves as the correctness oracle for the
  engine.

## Chunked accumulation and memory behavior

A differentiable field path can become memory-heavy when it materialises large
intermediate tensors of shape (sources × observers × 3). To bound peak memory the code:

- pads and slices sources into fixed-size chunks and folds them with `jax.lax.scan`
  (`_slice_chunk` / `_chunk_step` in the engine), so the source axis never fully
  materialises,
- switches face accumulation in the mesh kernels from a `vmap` over faces (fast, for
  $\le 64$ faces) to an in-place `fori_loop` (constant memory) above that threshold,
- caches prepared source/sensor tensors and precomputes mesh geometry (normals, edge
  terms) once per solid,
- uses fixed-observer-count JIT wrappers for hotspot kernels to maximise cache reuse.

For end-to-end timings and memory profiles, see [Performance](performance.md).

## Precision mode

magpylib_jax follows your JAX precision setting and **never mutates the global JAX config
on import**. Arrays use JAX's default float dtype: `float32` unless you enable x64. The
kernels do not hard-code `float64`, so a single toggle switches the whole library between
single and double precision:

```python
import jax
jax.config.update("jax_enable_x64", True)   # float64; needed for magpylib parity
import magpylib_jax as mpj
```

Double precision is recommended for scientific workloads and is required for bit-level
magpylib parity, especially for near-boundary evaluation, mesh/tetrahedron geometry
reductions, elliptic-integral kernels, and strict-tolerance benchmarks. The test suite
enables x64 (via `conftest.py`). `float32` is the fast default on GPU/TPU and is fine
when magpylib-level accuracy is not required.

## High-level API numerics

The public object and functional APIs add another layer of numerical responsibility:

- path broadcasting,
- squeeze behaviour,
- sensor pixel aggregation,
- source collection flattening,
- mixed-source batching.

These are not just formatting concerns: they control tensor layout, memory pressure, and
how much host-side work occurs before the JAX kernels even start.

## Validation philosophy

Numerical behaviour is validated along three axes:

1. parity with upstream Magpylib,
2. physics consistency checks (e.g. engine vs eager agreement, field-type relations),
3. profiling regression gates for runtime, memory, and HLO size.

This matters because a kernel can be numerically correct yet operationally unusable if
compile time, memory, or shape behaviour regresses.

## Relevant source files

- [`src/magpylib_jax/core/elliptic.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/elliptic.py)
- [`src/magpylib_jax/core/geometry.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/geometry.py)
- [`src/magpylib_jax/core/kernels/_safe.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/_safe.py)
- [`src/magpylib_jax/core/kernels/_common.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/_common.py)
- [`src/magpylib_jax/core/kernels/_raycast.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/_raycast.py)
- [`src/magpylib_jax/core/kernels/`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/core/kernels)
- [`src/magpylib_jax/fields/engine.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/engine.py)
- [`src/magpylib_jax/fields/eager.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/eager.py)
