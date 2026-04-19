# magpylib_jax

[![CI](https://github.com/uwplasma/magpylib_jax/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/magpylib_jax/actions/workflows/ci.yml)
[![Full Validation](https://github.com/uwplasma/magpylib_jax/actions/workflows/full-validation.yml/badge.svg)](https://github.com/uwplasma/magpylib_jax/actions/workflows/full-validation.yml)
[![Publish Release](https://github.com/uwplasma/magpylib_jax/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/uwplasma/magpylib_jax/actions/workflows/publish-pypi.yml)
[![PyPI version](https://img.shields.io/pypi/v/magpylib-jax.svg)](https://pypi.org/project/magpylib-jax/)
[![Python versions](https://img.shields.io/pypi/pyversions/magpylib-jax.svg)](https://pypi.org/project/magpylib-jax/)
[![Docs](https://readthedocs.org/projects/magpylib-jax/badge/?version=latest)](https://magpylib-jax.readthedocs.io/)
[![License](https://img.shields.io/github/license/uwplasma/magpylib_jax.svg)](LICENSE)

Differentiable magnetic field modeling in JAX with Magpylib-compatible APIs, parity gates, and profiling/benchmark CI. `magpylib_jax` is designed for optimization, inverse design, and simulation pipelines that need Magpylib-style ergonomics together with `jax.jit`, `jax.grad`, `jax.jacrev`, and XLA compilation.

## Why this project exists

`magpylib_jax` targets the gap between two requirements that are usually in tension:

- you want the closed-form, geometry-specific magnetic field models that make Magpylib useful,
- you also want a field pipeline that can be differentiated, compiled, and embedded in outer optimization loops.

This repository keeps the high-level user model close to upstream Magpylib while replacing the numerical core with JAX-first implementations and a JIT-safe field path.

## What you get

- End-to-end differentiable `getB/getH/getJ/getM`
- JIT-safe high-level field evaluation by default
- Magpylib-style object API: `Collection`, `Sensor`, path/orientation semantics, squeeze behavior
- Analytical kernels for dipoles, loops, line currents, polygonal current sheets, and permanent magnets
- Parity gates against upstream Magpylib, including mirrored upstream test categories
- CI/CD coverage for lint, typing, docs, parity, benchmarks, profiling, and PyPI release builds
- Python support from `3.10` onward
- Unpinned core package dependencies in `pyproject.toml`

## Implemented source families

- `misc.Dipole`
- `current.Circle`
- `current.Polyline`
- `current.TriangleSheet`
- `current.TriangleStrip`
- `misc.Triangle`
- `magnet.Cuboid`
- `magnet.Cylinder`
- `magnet.CylinderSegment`
- `magnet.Sphere`
- `magnet.Tetrahedron`
- `magnet.TriangularMesh`

## Installation

```bash
pip install magpylib-jax
```

For local development, tests, and docs:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,docs]'
pytest
```

For GPU-backed environments, install the appropriate `jax`/`jaxlib` build for your platform first, then install `magpylib-jax`.

## Quick example

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

jax.config.update("jax_enable_x64", True)

src = mpj.magnet.CylinderSegment(
    polarization=(0.1, -0.2, 0.3),
    dimension=(0.4, 1.2, 1.1, -30.0, 110.0),
)
obs = jnp.array([1.2, 0.2, 0.4])

B = src.getB(obs)


def bz(r2):
    trial = mpj.magnet.CylinderSegment(
        polarization=(0.1, -0.2, 0.3),
        dimension=(0.4, r2, 1.1, -30.0, 110.0),
    )
    return trial.getB(obs)[2]

print(B)
print(jax.grad(bz)(1.2))
```

## Documentation map

- [Overview](docs/overview.md): scope, supported objects, validation strategy, architectural intent
- [Quickstart](docs/quickstart.md): install, first field computation, first gradient, troubleshooting
- [Equation Models](docs/equations.md): field conventions, model equations, geometric reductions, derivation notes
- [Numerics](docs/numerics.md): stability, masking, singular behavior, precision, differentiation notes
- [Examples](docs/examples/index.md): object API, functional API, optimization loops, performance workflows
- [Architecture and Source Map](docs/architecture.md): clickable source-code guide to the repository internals
- [Testing and Validation](docs/testing.md): CI/CD gates, parity strategy, coverage, compatibility matrix
- [Performance](docs/performance.md): profiling workflow, hotspot kernels, JIT entrypoints, memory behavior
- [Parity Strategy](docs/parity.md)
- [Parity Checklist](docs/parity_checklist.md)
- [API Reference](docs/reference/api.md)
- [Changelog](CHANGELOG.md)

## JIT-safe `getB`

`magpylib_jax.getB/getH/getJ/getM` runs through a JIT-safe core by default. That path preserves Magpylib-style behavior while making the computational graph usable inside larger JAX programs.

Important notes:

- `output="dataframe"` is supported for compatibility, but is intentionally outside JIT.
- `pixel_agg` reducers support `mean`, `sum`, `min`, and `max` on the JIT-safe path.
- Repeated object evaluations reuse preparation caches for sources, sensors, orientation matrices, collection flattening, `TriangularMesh` geometry, and `CylinderSegment` face geometry.
- Circle-heavy workloads use a dedicated fast path to reduce host overhead and memory pressure.
- For benchmark-quality timing, use `jax.block_until_ready(...)` around the result.

## Differentiable fitting example

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

obs = jnp.array([[0.2, 0.1, 0.4], [0.5, 0.0, 0.7]])
target = jnp.array([[2.0e-4, 0.0, 3.0e-4], [1.0e-4, 0.0, 2.0e-4]])


def loss_fn(pol):
    src = mpj.magnet.Cuboid(dimension=(1.0, 0.8, 1.2), polarization=pol)
    pred = src.getB(obs)
    return jnp.mean((pred - target) ** 2)

pol = jnp.array([0.05, -0.02, 0.08])
for _ in range(50):
    pol = pol - 1e-1 * jax.grad(loss_fn)(pol)
```

## Validation and release gates

CI enforces:

- lint and type checks,
- docs build,
- `>=90%` coverage,
- sharded `pytest -m 'not slow'` test coverage,
- benchmark regression thresholds,
- profiling regression thresholds,
- Python compatibility checks on `3.10`, `3.12`, and `3.13`.

Nightly workflows additionally run the full validation suite and extended profiling artifact generation.

## Key repository files

- [`PARITY_MATRIX.md`](PARITY_MATRIX.md)
- [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md)
- [`pyproject.toml`](pyproject.toml)
- [`benchmarks/thresholds.json`](benchmarks/thresholds.json)
- [`profiling/thresholds.json`](profiling/thresholds.json)
- [`profiling/hlo_baseline.json`](profiling/hlo_baseline.json)
- [`.readthedocs.yaml`](.readthedocs.yaml)
- [`.github/workflows/publish-pypi.yml`](.github/workflows/publish-pypi.yml)

## License

BSD-2-Clause.
