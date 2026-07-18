# Testing and validation

`magpylib_jax` treats validation as a product feature, not a cleanup step. A release is only good if
it is *numerically* faithful to Magpylib, *physically* consistent, *differentiable* without
regressions, and *fast* enough to stay fast. Each of those is a distinct family of tests.

## Test taxonomy

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🎯 Parity
Direct numeric comparison of `getB/getH/getJ/getM` against upstream Magpylib for every implemented
source family, at random and profile-selected points. See [Parity strategy](parity.md).
:::

:::{grid-item-card} 🪞 Upstream mirror
Tests derived from upstream Magpylib's own suite (interfaces, `BaseGeo`, `Collection`, `Sensor`,
paths, physics consistency), so object-level behavior matches.
:::

:::{grid-item-card} 🥇 Golden regression
Frozen reference outputs guard against silent drift: a change that alters a field value has to be
deliberate.
:::

:::{grid-item-card} 🧪 Physics
Source families checked against analytical expectations and cross-source consistency relations
(e.g. `div B = 0` in current-free regions).
:::

:::{grid-item-card} ⚡ Differentiability
Representative kernels are exercised under `jax.grad`/`jacfwd`/`jacrev` so autodiff keeps working
end to end.
:::

:::{grid-item-card} 🚀 JIT
Field paths are checked under `jax.jit` (and `vmap`) to catch tracing-time and shape-polymorphism
regressions.
:::

:::{grid-item-card} 🎛️ Precision
float32 and float64 behavior is pinned so parity tolerances hold and the library respects your JAX
config. See [Precision](precision.md).
:::

:::{grid-item-card} 📦 Compat & packaging
Compatibility surface (aliases, `output="dataframe"`, `defaults`) plus a packaging test that keeps
dependencies unpinned and the Python floor at `>=3.10`.
:::

:::{grid-item-card} 📈 Coverage
The suite holds **≥95%** line coverage, enforced in CI.
:::

::::

## Upstream mirrored tests

The repository ships tests derived from upstream Magpylib categories, so that object and API
behavior tracks the original, including:

- `test_getBH_interfaces.py`
- `test_obj_BaseGeo*.py`
- `test_obj_Collection.py`
- `test_obj_Sensor.py`
- `test_path.py`
- `test_physics_consistency.py`

Coverage status per source family is tracked in [Parity strategy](parity.md).

## Running the tests

```{admonition} Editable install first
:class: tip
Install the dev extras once with `pip install -e '.[test,docs]'`, then run any of the commands below.
```

```bash
# Fast suite (skip the slow-marked tests)
pytest -m 'not slow' tests

# The whole suite, with coverage
pytest --cov=magpylib_jax tests

# A single family or a single test file
pytest tests -k cuboid
pytest tests/parity_gates

# Lint and type checks
ruff check src tests scripts
mypy src

# Build the docs the same way CI does (warnings are errors)
sphinx-build -W -b html docs docs/_build/html

# Build the distribution
python -m build
```

## CI gates

Every pull request must clear the same gates the maintainers run:

```{list-table}
:header-rows: 1
:widths: 24 76

* - Gate
  - What it enforces
* - Lint + types
  - `ruff` and `mypy` pass on `src`, `tests`, `scripts`.
* - Tests + coverage
  - The sharded test suite passes and coverage stays **≥95%**.
* - Docs
  - `sphinx-build -W` builds cleanly on the minimum supported Python.
* - Benchmark
  - Runtime stays within the thresholds in [`benchmarks/thresholds.json`](https://github.com/uwplasma/magpylib_jax/blob/main/benchmarks/thresholds.json).
```

Beyond the per-PR gates, the project also runs operational-regression checks — benchmark slowdown
thresholds, parity-error thresholds, compile-time/runtime/memory thresholds, and HLO artifacts for
hotspot inspection (see [Performance](performance.md)). Python-compatibility smoke coverage runs on
`3.10`, `3.12`, and `3.13`, with nightly full validation and nightly profiling on top.

## Packaging metadata checks

A dedicated packaging test ensures:

- dependencies in `pyproject.toml` remain unpinned,
- the Python support floor stays at `>=3.10`,
- static-analysis targets stay aligned with the supported floor.

It uses `tomllib` on Python `3.11+` and falls back to `tomli` on `3.10`.

## Relevant files

- [`tests/`](https://github.com/uwplasma/magpylib_jax/tree/main/tests)
- [`tests/parity_gates`](https://github.com/uwplasma/magpylib_jax/tree/main/tests/parity_gates)
- [`tests/upstream_mirror`](https://github.com/uwplasma/magpylib_jax/tree/main/tests/upstream_mirror)
- [`benchmarks/thresholds.json`](https://github.com/uwplasma/magpylib_jax/blob/main/benchmarks/thresholds.json)
- [`profiling/thresholds.json`](https://github.com/uwplasma/magpylib_jax/blob/main/profiling/thresholds.json)
- [`.github/workflows/ci.yml`](https://github.com/uwplasma/magpylib_jax/blob/main/.github/workflows/ci.yml)
- [`.github/workflows/full-validation.yml`](https://github.com/uwplasma/magpylib_jax/blob/main/.github/workflows/full-validation.yml)
- [`.github/workflows/profiling-nightly.yml`](https://github.com/uwplasma/magpylib_jax/blob/main/.github/workflows/profiling-nightly.yml)
