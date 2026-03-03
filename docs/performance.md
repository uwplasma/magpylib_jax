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
- `scripts/profile_getB_jit.py` (end-to-end JIT-safe `getB`)

Kernel-level JIT entrypoints (fixed observer count):
- `magpylib_jax.core.kernels_extended.current_circle_bfield_jit`
- `magpylib_jax.core.kernels_extended.current_polyline_bfield_jit`
- `magpylib_jax.core.kernels_extended.magnet_trimesh_bfield_jit`
- `magpylib_jax.core.kernels_extended.magnet_trimesh_bfield_jit_faces`
- `magpylib_jax.core.kernels_extended.magnet_trimesh_bfield_jit_faces_precomp`
- `magpylib_jax.core.kernels_extended.magnet_cylinder_segment_bfield_jit`
- `magpylib_jax.core.kernels_extended.magnet_cylinder_segment_bfield_jit_faces`
- `magpylib_jax.core.kernels_extended.triangle_bfield_jit`
- `magpylib_jax.core.kernels_extended.current_trisheet_bfield_jit`
- `magpylib_jax.core.kernels_extended.current_tristrip_bfield_jit`
- `magpylib_jax.core.kernels_extended.tetrahedron_bfield_jit`

These wrappers cache compilation per `(kernel, observer_count)` and provide a stable handle for
HLO/compile/runtime/memory budgets. They are used by the profiling scripts and can also be used
directly in performance-critical pipelines with fixed observer grids.

## JIT-safe getB path

The high-level `getB/getH/getJ/getM` API runs through a JIT-safe core by default. This allows
end-to-end compilation and differentiation across source parameters and observer grids while
preserving Magpylib-compatible shapes and squeeze behavior.

### Circle-heavy collection fast path

For circle-only object collections (for example, large coil stacks), the JIT core now uses:
- observer-aware source chunking (memory-bounded intermediate tensors),
- a single-sensor/identity-rotation fast path for reduced transform overhead,
- source-preparation caching for repeated calls on unchanged circle collections.

This significantly reduces steady-state runtime and peak memory pressure for workloads that call
`getB` repeatedly with the same source object graph and observer grid.

### Object preparation caches

The default JIT-safe `getB/getH/getJ/getM` path also caches the expensive host-side preparation
steps that sit in front of the kernel math:
- generalized source preparation caches keyed by stable object cache tokens,
- sensor preparation caches keyed by sensor identity, path, pixel layout, and handedness,
- cached orientation matrices on `BaseGeo` objects to avoid repeated `Rotation.as_matrix()` work,
- cached `Collection` flatten/source/sensor lists with dirty propagation on add/remove,
- cached `TriangularMesh` oriented faces and face geometry reuse,
- precomputed `CylinderSegment` face geometry inside the high-level JIT path.

These caches are invalidated by object mutations and are covered by dedicated cache-regression
tests so repeated calls stay fast without sacrificing parity.

### Tiny-batch circle fast path

Small observer grids used with very large circle collections were still paying a host-side cost for
rebuilding singleton path stacks on every call. The prepared state now caches single-path source and
sensor tensors directly as JAX arrays, so repeated small-grid evaluations bypass that formatting
work instead of rebuilding thousands of `(1, 3)` and `(1, 3, 3)` objects per call.

This is especially relevant for coil-stack workloads where:
- source count is very large,
- observer count is small,
- source and sensor paths are static.

The change keeps the public API unchanged while making the prepared state more JAX-friendly for
outer-loop `jax.jit` usage.

### WHAM-style workload

For the WHAM-style double-coil workload used during development, the reproducible profiling entry
point is:
- `scripts/profile_wham_workload.py`

That script compares upstream `magpylib` and `magpylib_jax` on a small observer grid and a larger
`(101, 31, 3)` grid, and also records the cost of converting the large JAX result back to NumPy.

Profiling remains focused on kernel entrypoints for stable, comparable HLO baselines. The JIT-safe
`getB` path is validated via parity tests against the legacy implementation and is suitable for
application-level JIT usage.

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
