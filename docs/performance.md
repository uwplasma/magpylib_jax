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
- `scripts/check_hlo_diffs.py`

Kernel-level JIT entrypoints (fixed observer count):
- `magpylib_jax.core.kernels_extended.magnet_trimesh_bfield_jit`
- `magpylib_jax.core.kernels_extended.magnet_cylinder_segment_bfield_jit`
- `magpylib_jax.core.kernels_extended.triangle_bfield_jit`
- `magpylib_jax.core.kernels_extended.current_trisheet_bfield_jit`
- `magpylib_jax.core.kernels_extended.current_tristrip_bfield_jit`
- `magpylib_jax.core.kernels_extended.tetrahedron_bfield_jit`

These wrappers cache compilation per `(kernel, observer_count)` and provide a stable handle for
HLO/compile/runtime/memory budgets. They are used by the profiling scripts and can also be used
directly in performance-critical pipelines with fixed observer grids.

Artifacts produced per source family:
- JAX trace (`jax.profiler.trace`)
- HLO dump (`compiler_ir(..., dialect="hlo")`)
- device memory profile snapshot (`jax.profiler.save_device_memory_profile`)

Thresholds:
- `profiling/thresholds.json`
- `profiling/hlo_baseline.json`

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
5. Update `profiling/hlo_baseline.json` if the HLO hashes change intentionally, then run
   `scripts/check_hlo_diffs.py` to verify the new baseline.

For new JIT entrypoints, add thresholds under matching keys (e.g., `triangle_jit`, `tetrahedron_jit`)
and keep the budgets conservative until a profiling sweep establishes tighter bounds.

## CI regression gates

- `ci.yml`: benchmark regression + profiling regression jobs.
- `profiling-nightly.yml`: nightly profiling with artifact upload.
