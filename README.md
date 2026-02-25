# magpylib_jax

Differentiable magnetic field modeling in JAX, with Magpylib-style APIs and parity-focused validation.

`magpylib_jax` is built for optimization and inverse design workflows where you need:
- analytical magnetic field models,
- automatic differentiation through source and observer parameters,
- explicit numerical parity checks against upstream [Magpylib](https://github.com/magpylib/magpylib),
- reproducible CI gates for physics, parity, performance, and documentation.

## Current capabilities

Implemented kernels and objects:
- `misc.Dipole`
- `current.Circle`
- `magnet.Cuboid`
- `magnet.Cylinder`
- `magnet.Sphere`
- `current.Polyline`
- `misc.Triangle`
- `magnet.Tetrahedron`

Implemented compatibility utilities:
- functional API: `getB/getH/getJ/getM`
- object API and additive containers: `Collection`, `Sensor`
- parity behavior gates (`B/H/J/M`, inside/outside/singular profiles)
- benchmark regression thresholds in CI

Pending ports (tracked in [PARITY_MATRIX.md](PARITY_MATRIX.md)):
- `CylinderSegment`
- `TriangleSheet/TriangleStrip`
- `TriangularMesh`

## Why JAX here?

- End-to-end differentiability for geometry and material parameters.
- Efficient vectorization and JIT compilation for large observer batches.
- Stable testing + profiling infrastructure to preserve correctness while optimizing kernels.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,docs]'
pytest
```

## Example

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

src = mpj.magnet.Cuboid(
    polarization=(0.1, -0.2, 0.3),
    dimension=(1.0, 0.8, 1.2),
)
obs = jnp.array([0.2, 0.1, 0.5])

B = src.getB(obs)

# Differentiate Bz with respect to cuboid x-size
def bz(dim_x):
    s = mpj.magnet.Cuboid(
        polarization=(0.1, -0.2, 0.3),
        dimension=(dim_x, 0.8, 1.2),
    )
    return s.getB(obs)[2]

grad_bz = jax.grad(bz)(1.0)
```

## Testing and parity standards

CI currently enforces:
- test suite + strict parity profile gates,
- lint and type checks,
- documentation build,
- coverage threshold (`>=90%`),
- benchmark thresholds (error + runtime slowdown bounds).

See:
- [PARITY_MATRIX.md](PARITY_MATRIX.md)
- [MIGRATION_PLAN.md](MIGRATION_PLAN.md)

## Documentation

Detailed docs include equations, numerical methods, examples, parity strategy, testing, and performance notes:
- local: `docs/`
- Read the Docs config: [`.readthedocs.yaml`](.readthedocs.yaml)

## License

BSD-2-Clause.
