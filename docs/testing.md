# Testing and Validation

This repository treats validation as a product feature, not a cleanup step.

## Validation layers

## 1. Unit and regression tests

These cover:

- object construction and API shape behavior,
- path/orientation semantics,
- cache invalidation behavior,
- compatibility edge cases.

## 2. Physics tests

These check source families against analytical expectations or cross-source consistency relations.

## 3. Differentiability tests

These verify that representative kernels can participate in JAX autodiff without regressions.

## 4. Parity tests

These compare `magpylib_jax` outputs against upstream Magpylib for implemented source families.

## 5. Upstream mirrored tests

The repository includes mirrored tests derived from upstream Magpylib categories such as:

- `test_getBH_interfaces.py`
- `test_obj_BaseGeo*.py`
- `test_obj_Collection.py`
- `test_obj_Sensor.py`
- `test_path.py`
- `test_physics_consistency.py`

Status is tracked in [Parity Checklist](parity_checklist.md) and [`PARITY_MATRIX.md`](https://github.com/uwplasma/magpylib_jax/blob/main/PARITY_MATRIX.md).

## 6. Benchmark and profiling gates

Validation also includes operational regressions:

- benchmark slowdown thresholds,
- parity error thresholds,
- compile-time and runtime thresholds,
- memory thresholds,
- HLO artifacts for hotspot inspection.

## CI/CD matrix

The repository currently validates:

- full fast suite on GitHub Actions,
- Python compatibility smoke coverage on `3.10`, `3.12`, and `3.13`,
- docs build on the minimum supported Python version,
- nightly full validation and nightly profiling,
- release build and PyPI publish workflow.

## Packaging metadata checks

A dedicated packaging test ensures:

- dependencies in `pyproject.toml` remain unpinned,
- the Python support floor stays at `>=3.10`,
- static-analysis targets remain aligned with the supported floor.

The test uses `tomllib` on Python `3.11+` and `tomli` as a compatibility fallback on Python `3.10`.

## Useful commands

```bash
ruff check src tests scripts
mypy src
pytest -m 'not slow' tests
sphinx-build -W -b html docs docs/_build/html
python -m build
```

## Relevant files

- [`tests/`](https://github.com/uwplasma/magpylib_jax/tree/main/tests)
- [`benchmarks/thresholds.json`](https://github.com/uwplasma/magpylib_jax/blob/main/benchmarks/thresholds.json)
- [`profiling/thresholds.json`](https://github.com/uwplasma/magpylib_jax/blob/main/profiling/thresholds.json)
- [`.github/workflows/ci.yml`](https://github.com/uwplasma/magpylib_jax/blob/main/.github/workflows/ci.yml)
- [`.github/workflows/full-validation.yml`](https://github.com/uwplasma/magpylib_jax/blob/main/.github/workflows/full-validation.yml)
- [`.github/workflows/profiling-nightly.yml`](https://github.com/uwplasma/magpylib_jax/blob/main/.github/workflows/profiling-nightly.yml)
