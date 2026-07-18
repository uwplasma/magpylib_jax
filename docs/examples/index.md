# Examples

This gallery is organized by **what you want to do**, not by source family. Every page pairs a short
tutorial with runnable snippets, and links the matching scripts in the top-level
[`examples/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples) folder — each script runs
standalone and prints (or plots) its result.

```{tip}
New here? Start with the [Quickstart](../quickstart.md), then come back and pick a task below.
Enable `jax.config.update("jax_enable_x64", True)` before running anything you want to compare
against Magpylib — see [Precision](../precision.md).
```

## Browse by task

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🧱 Object API
Build sources, collections, and sensors; use motion paths and pixel grids.
+++
[Object API »](object_api.md)
:::

:::{grid-item-card} 🔢 Functional API
Call `getB/getH/getJ/getM` directly with descriptors and arrays.
+++
[Functional API »](functional_api.md)
:::

:::{grid-item-card} ⚙️ Force & torque
Differentiable force and torque between objects with `getFT`.
+++
[Force & torque »](force_torque.md)
:::

:::{grid-item-card} 🎨 Visualization
Render geometry, polarization, and paths in 3D with `show`.
+++
[Visualization »](visualization.md)
:::

:::{grid-item-card} 🎯 Optimization
Inverse design and geometry fitting with `jax.grad`.
+++
[Optimization »](optimization.md)
:::

:::{grid-item-card} 🚀 Performance
Profiling, JIT, `vmap`, and large-coil workloads.
+++
[Performance »](performance.md)
:::

::::

## The `examples/` folder

The scripts are grouped by theme on GitHub:

```{list-table}
:header-rows: 1
:widths: 26 74

* - Folder
  - Contents
* - [`basics/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples/basics)
  - First field, collections & sensors, motion paths, custom sources, the source gallery.
* - [`shapes/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples/shapes)
  - Triangle meshes, convex hulls, current sheets, superposition.
* - [`force/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples/force)
  - Force intro, dipole–dipole, holding force.
* - [`visualization/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples/visualization)
  - `show` scenes, magnet rendering, field streamplots.
* - [`applications/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples/applications)
  - Air coils, Helmholtz pairs, discrete Halbach arrays.
* - [`differentiable/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples/differentiable)
  - Inverse design, geometry optimization, field Jacobians, `getFT` optimization, `jit`/`vmap`.
```

```{toctree}
:maxdepth: 1

object_api
functional_api
force_torque
visualization
optimization
performance
```
