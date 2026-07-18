# Overview

`magpylib_jax` is a JAX-native library for **analytic magnetic fields you can differentiate,
compile, and optimize through**. It is a clean-room reimplementation of
[Magpylib](https://github.com/magpylib/magpylib): it keeps Magpylib's ergonomic object and
functional APIs and its closed-form source models, but replaces the numerical core with a
differentiable, JIT-compilable, `vmap`-friendly one that runs on CPU, GPU, and TPU.

The result is that the *same* field call you use for analysis drops straight into a `jax.grad`
optimization loop — no finite differences, no wrappers, no separate solver.

```python
import jax
jax.config.update("jax_enable_x64", True)   # float64, for magpylib parity
import magpylib_jax as mpj

src = mpj.magnet.Cuboid(polarization=(0, 0, 1.0), dimension=(1, 1, 1))
B = src.getB((2.0, 0.0, 0.0))                # tesla, a JAX array
```

## Why a differentiable field library

Classical field toolkits give you a number; `magpylib_jax` gives you a number *and* its exact
derivative with respect to anything the field depends on — geometry, pose, and excitation. Because
every source model is a closed-form (analytic) expression rather than a mesh or FEM solve, the
whole computation is a smooth function that JAX can transform.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🎯 Inverse design
Recover an unknown polarization, current, or geometry from measured field samples by minimizing a
mean-squared error with `jax.grad`.
:::

:::{grid-item-card} ⚙️ Force & torque
`getFT` returns exact force and torque from an autodiff of the field — no `eps` step size to tune,
unlike a finite-difference estimate.
:::

:::{grid-item-card} 🚀 Batched sweeps
`jax.vmap` evaluates one field function over thousands of source parameters or observer points with
no Python loop, then XLA fuses and compiles it.
:::

:::{grid-item-card} 🔁 Drop-in migration
`import magpylib_jax as magpy` gives you the Magpylib source classes, `Collection`, `Sensor`,
motion, and `getB/…/show/mu_0`. Often you only swap the import.
:::

::::

## Scope

All 12 source families ship, in three sub-namespaces, plus the high-level object types and the
functional/visualization entry points.

```{list-table}
:header-rows: 1
:widths: 22 78

* - Namespace
  - Sources
* - `mpj.magnet`
  - `Cuboid`, `Cylinder`, `CylinderSegment`, `Sphere`, `Tetrahedron`, `TriangularMesh`
* - `mpj.current`
  - `Circle`, `Polyline`, `TriangleSheet`, `TriangleStrip`
* - `mpj.misc`
  - `Dipole`, `Triangle`, `CustomSource`
```

On top of the sources sit the composition and observer types and the entry points:

- **Objects** — `mpj.Collection` groups sources (and nests), `mpj.Sensor` is a movable observer
  with a pixel grid.
- **Fields** — `getB` (flux density, T), `getH` (field strength, A/m), `getJ` (polarization, T),
  and `getM` (magnetization, A/m), available both as top-level functions and as object methods.
- **Force & torque** — `getFT` returns the force and torque on a target object in an external field.
- **Visualization** — `show` renders geometry and paths in 3D with Matplotlib.
- **Constants** — `mpj.mu_0`, the vacuum permeability.

### Non-goals

`magpylib_jax` deliberately does **not** reimplement everything in upstream Magpylib:

```{admonition} What is intentionally out of scope
:class: note

- The **Plotly** and **PyVista** display backends. Visualization is Matplotlib-only; `show`
  covers static 3D geometry and paths. See [Parity strategy](parity.md).
- Magpylib's `output="dataframe"` mode exists as a *compatibility* convenience, not as part of the
  jittable field graph — it returns pandas objects and cannot be traced.
- Generic numerical field solvers (FEM/BEM). Every source is an exact closed-form model; that is
  what makes the library differentiable.
```

## How it fits together

The library is organized in layers, from the friendly object API down to the pure analytic
kernels:

1. **Objects** (`Cuboid`, `Collection`, `Sensor`, …) carry position, orientation, paths, and style,
   and expose `getB/getH/getJ/getM/getFT/show`.
2. The **fields engine** (`fields/`) normalizes sources and observers, batches homogeneous source
   families, and runs the vectorized, JIT-compiled evaluation.
3. The **kernels** (`core/kernels/`) hold the closed-form field formula for each source family as a
   pure, differentiable JAX function.

To go deeper, follow the map in [Architecture](architecture.md), read the closed-form
derivations in [Equation models](equations.md), or jump to a task in the
[Examples gallery](examples/index.md). If you care about numerical precision and float32-vs-float64
behavior, start with [Precision](precision.md).

## Source code entry points

- Public functional API — [`functional.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/functional.py)
- Fields engine — [`fields/`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/fields)
- Analytic kernels — [`core/kernels/`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/core/kernels)
- Runnable examples — [`examples/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples)
