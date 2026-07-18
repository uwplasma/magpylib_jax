---
sd_hide_title: true
---

# magpylib_jax

<div align="center">

<h1 style="border:0;margin-bottom:0.2em">magpylib_jax</h1>

**Analytic magnetic fields you can differentiate, compile, and optimize through.**

[![PyPI](https://img.shields.io/pypi/v/magpylib-jax.svg)](https://pypi.org/project/magpylib-jax/)
[![CI](https://github.com/uwplasma/magpylib_jax/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/magpylib_jax/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/uwplasma/magpylib_jax.svg)](https://github.com/uwplasma/magpylib_jax/blob/main/LICENSE)

</div>

```{image} _static/field_map.png
:alt: B-field of a cuboid magnet
:width: 60%
:align: center
```

`magpylib_jax` is a clean-room, [JAX](https://github.com/jax-ml/jax)-native reimplementation of
[Magpylib](https://github.com/magpylib/magpylib). It keeps Magpylib's ergonomic object and
functional APIs and its analytic field models, but swaps the numerical core for a
**differentiable, JIT-compilable, `vmap`-friendly** one — so the same field computation you use for
analysis drops straight into a `jax.grad` optimization loop, on CPU, GPU, or TPU.

```python
import jax
jax.config.update("jax_enable_x64", True)   # float64, for magpylib parity
import magpylib_jax as mpj

src = mpj.magnet.Cuboid(polarization=(0, 0, 1.0), dimension=(1, 1, 1))
B = src.getB((2.0, 0.0, 0.0))                 # tesla
dBz_dh = jax.grad(lambda h:
    mpj.magnet.Cuboid(polarization=(0, 0, 1.0), dimension=(1, 1, h)).getB((2, 0, 0))[2])(1.0)
```

## What you get

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} ⚡ Differentiable
Exact `grad` / `jacfwd` / `jacrev` through `getB/getH/getJ/getM` and `getFT`, w.r.t. geometry,
pose, and excitation — no finite differences.
:::

:::{grid-item-card} 🚀 Compiled & vectorized
The field core runs under `jax.jit` / `vmap` / XLA on CPU, GPU, and TPU.
:::

:::{grid-item-card} 🧲 Exact force & torque
`getFT` gives force and torque by autodiff of `m·B` — `eps`-free and exact, unlike Magpylib's
finite-difference approach.
:::

:::{grid-item-card} 🔁 Drop-in for Magpylib
Same source classes, `Collection`, `Sensor`, motion, and `getB/…/getFT/show/mu_0`. Often just swap
the import.
:::

:::{grid-item-card} 🎛️ Precision you control
Follows your JAX config: `float32` by default, `float64` for bit-level parity. The library never
mutates the global config.
:::

:::{grid-item-card} 📐 Research-grade & tested
All 12 source families, ≥95% coverage, a full Magpylib parity suite, and derivations with
citations.
:::

::::

## Where to next

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🏁 Getting started
Install, compute your first field and gradient, and pick a precision.
+++
[Quickstart »](quickstart.md)
:::

:::{grid-item-card} 📖 User guide
Tutorials: fields, motion & paths, collections, visualization, force/torque, and optimization.
+++
[Examples »](examples/index.md)
:::

:::{grid-item-card} 🧮 Theory & methods
The physical conventions, per-source closed-form models, numerics, differentiability, and the
papers behind each kernel.
+++
[Equation models »](equations.md)
:::

:::{grid-item-card} 🔧 Reference
The API reference and a clickable map from the public API down to the analytic kernels.
+++
[API reference »](reference/api.md)
:::

::::

```{toctree}
:hidden:
:caption: Getting started

overview
quickstart
precision
```

```{toctree}
:hidden:
:caption: User guide

examples/index
```

```{toctree}
:hidden:
:caption: Theory & methods

equations
numerics
references
```

```{toctree}
:hidden:
:caption: Reference

reference/api
architecture
performance
```

```{toctree}
:hidden:
:caption: Development

testing
parity
changelog
```
