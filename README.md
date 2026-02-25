# magpylib_jax

`magpylib_jax` is a JAX-native magnetic field library designed to be fully differentiable end-to-end, while tracking Magpylib functionality and numerical behavior.

## Current status

This repository is bootstrapped and includes:

- Differentiable JAX kernels for:
  - Dipole field (`misc.Dipole`, `core.dipole_hfield`)
  - Circular current loop field (`current.Circle`, `core.current_circle_hfield`)
  - Cuboid magnet field (`magnet.Cuboid`)
  - Cylinder magnet field (`magnet.Cylinder`)
- Functional API (`getH`, `getB`) and object API (source classes)
- Compatibility layer (`Collection`, `Sensor`, object/list dispatch in `getB/getH/getJ/getM`)
- Parity tests against upstream `magpylib` for implemented features
- Differentiability tests (`grad`, `jacrev`)
- Benchmark script scaffold for performance and parity checks
- CI for tests, lint, types, and docs
- Read the Docs configuration + Sphinx docs scaffold

## Design goals

- JAX-first, vectorized, `jit`/`vmap` friendly kernels
- Differentiable with respect to observers and source parameters (away from physical singularities)
- Numerical parity against Magpylib for all supported source types
- Memory-aware kernels for large batched field evaluations

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,docs]'
pytest
```

## Project roadmap

See [MIGRATION_PLAN.md](MIGRATION_PLAN.md) for the phased migration and validation matrix.

## License

BSD-2-Clause.
