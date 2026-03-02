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
- Fixed-observer-count JIT entrypoints for hotspot kernels (triangle*, tetrahedron, mesh)
- JIT-safe `getB/getH/getJ/getM` path with parity checks against legacy behavior

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

## JIT-safe `getB`

`magpylib_jax.getB/getH/getJ/getM` now run through a JIT-safe core by default. This keeps the API identical to Magpylib while enabling end-to-end differentiation and compilation.

Notes:
- When `pixel_agg` is a string, supported reducers are `mean`, `sum`, `min`, `max`. Other reducers fall back to the legacy path.
- `output="dataframe"` is still supported, but runs outside JIT (matches Magpylib semantics).

## Differentiable fitting (mini loop)

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

## Validation + profiling gates

CI enforces:
- fast sharded test suite (`-m 'not slow'`) + strict parity gates,
- lint + type checks,
- docs build,
- coverage threshold (`>=90%`),
- benchmark thresholds (runtime slowdown + parity error),
- kernel profiling thresholds (compile/runtime/parity/memory) with HLO, trace, and memory snapshots.

Nightly `full-validation.yml` runs the complete suite (including `slow` upstream-mirror parity tests).

Key files:
- [`PARITY_MATRIX.md`](PARITY_MATRIX.md)
- [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md)
- [`benchmarks/thresholds.json`](benchmarks/thresholds.json)
- [`profiling/thresholds.json`](profiling/thresholds.json)
- [`profiling/hlo_baseline.json`](profiling/hlo_baseline.json)

### Profiling sweep (local)

```bash
python scripts/profile_kernels.py --n-observers 512 --repeats 1 \
  --output profiling/profile.local.json --output-dir profiling/artifacts/local
python scripts/check_profiling_thresholds.py profiling/profile.local.json
python scripts/check_hlo_diffs.py profiling/profile.local.json profiling/hlo_baseline.json
```

Update `profiling/thresholds.json` when intentional kernel changes shift HLO size or runtime.

## Documentation

Detailed docs are in `docs/` (equations, numerics, examples, parity strategy, testing, performance, API).
Read the Docs config is in [`.readthedocs.yaml`](.readthedocs.yaml).

## License

BSD-2-Clause.
