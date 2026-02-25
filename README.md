# magpylib_jax

Differentiable magnetic field modeling in JAX with Magpylib-compatible APIs, parity gates, and profiling/benchmark CI.

`magpylib_jax` targets optimization and inverse-design workflows where you need:
- analytical or high-fidelity kernel models,
- reliable `jax.grad/jacrev` differentiation through source and observer parameters,
- behavior parity checks against upstream [Magpylib](https://github.com/magpylib/magpylib),
- reproducible performance and memory regression tracking.

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

## Compatibility coverage

- Functional API: `getB/getH/getJ/getM`
- Object API: `Collection`, `Sensor`, source classes above
- Path/motion/orientation coverage for core source/sensor workflows
- Shape/squeeze behavior parity checks vs Magpylib
- Upstream-file mirrored object tests (`test_obj_*` categories)

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

src = mpj.magnet.CylinderSegment(
    polarization=(0.1, -0.2, 0.3),
    dimension=(0.4, 1.2, 1.1, -30.0, 110.0),
)
obs = jnp.array([1.2, 0.2, 0.4])

B = src.getB(obs)

def bz(r2):
    s = mpj.magnet.CylinderSegment(
        polarization=(0.1, -0.2, 0.3),
        dimension=(0.4, r2, 1.1, -30.0, 110.0),
    )
    return s.getB(obs)[2]

print(jax.grad(bz)(1.2))
```

## Validation + profiling gates

CI enforces:
- full test suite + strict parity gates,
- lint + type checks,
- docs build,
- coverage threshold (`>=90%`),
- benchmark thresholds (runtime slowdown + parity error),
- kernel profiling thresholds (compile/runtime/parity/memory) with HLO, trace, and memory snapshots.

Key files:
- [`PARITY_MATRIX.md`](PARITY_MATRIX.md)
- [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md)
- [`benchmarks/thresholds.json`](benchmarks/thresholds.json)
- [`profiling/thresholds.json`](profiling/thresholds.json)

## Documentation

Detailed docs are in `docs/` (equations, numerics, examples, parity strategy, testing, performance, API).
Read the Docs config is in [`.readthedocs.yaml`](.readthedocs.yaml).

## License

BSD-2-Clause.
