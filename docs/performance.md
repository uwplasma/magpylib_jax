# Performance

## Benchmark pipeline

Benchmarks compare `magpylib_jax` vs upstream `magpylib` for all currently implemented source families.

Scripts:
- `scripts/benchmark_vs_magpylib.py`
- `scripts/check_benchmark_thresholds.py`

Thresholds:
- `benchmarks/thresholds.json`

## Kernel profiling pipeline

Kernel-level profiling tracks compile/runtime/parity/memory and captures XLA artifacts.

Scripts:
- `scripts/profile_kernels.py`
- `scripts/check_profiling_thresholds.py`

Artifacts produced per source family:
- JAX trace (`jax.profiler.trace`)
- HLO dump (`compiler_ir(..., dialect="hlo")`)
- device memory profile snapshot (`jax.profiler.save_device_memory_profile`)

Thresholds:
- `profiling/thresholds.json`

## What is measured

- JAX compile time per source type.
- Steady-state runtime (median over repeated runs).
- Max absolute parity error (Tesla).
- Peak process memory (bytes).

## Updating thresholds

When kernel changes intentionally alter HLO size, compile time, or runtime:

1. Run `scripts/profile_kernels.py` with the standard observer count.
2. Review the JSON output and HLO artifacts.
3. Update `profiling/thresholds.json` with conservative budgets.
4. Re-run `scripts/check_profiling_thresholds.py` to confirm.

## CI regression gates

- `ci.yml`: benchmark regression + profiling regression jobs.
- `profiling-nightly.yml`: nightly profiling with artifact upload.
