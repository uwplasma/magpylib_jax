# Performance

## Benchmark pipeline

Benchmarks compare `magpylib_jax` vs upstream `magpylib` for implemented source types.

Scripts:
- `scripts/benchmark_vs_magpylib.py`
- `scripts/check_benchmark_thresholds.py`

Thresholds:
- `benchmarks/thresholds.json`

## What is measured

- Median runtime over repeated runs.
- JAX compile time for each source type.
- Max absolute parity error in Tesla.

## CI regression gates

- Error thresholds per source type.
- Runtime slowdown limits relative to upstream Magpylib.
