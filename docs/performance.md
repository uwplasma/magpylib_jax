# Performance

This page covers both how the library is optimized and how performance is measured.

## Performance model

The repository has two distinct performance layers:

- kernel performance: analytical source-family implementations,
- high-level orchestration performance: source/sensor preparation, path handling, batching, and field assembly.

Both matter. A fast kernel can still produce a slow user-facing `getB` if host-side preparation dominates.

## High-level `getB` optimizations

The JIT-safe `getB/getH/getJ/getM` path includes:

- generalized source preparation caches keyed by object cache tokens,
- sensor preparation caches keyed by identity, path, pixel layout, and handedness,
- cached orientation matrices on `BaseGeo`,
- cached `Collection` flatten/source/sensor lists with dirty propagation,
- cached `TriangularMesh` oriented faces and face geometry reuse,
- precomputed `CylinderSegment` face geometry inside the high-level JIT path,
- circle-heavy collection fast paths,
- singleton-path caching for tiny observer batches.

## Kernel-level profiling pipeline

Scripts:

- [`scripts/profile_kernels.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/profile_kernels.py)
- [`scripts/check_profiling_thresholds.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/check_profiling_thresholds.py)
- [`scripts/check_hlo_diffs.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/check_hlo_diffs.py)
- [`scripts/profile_getB_jit.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/profile_getB_jit.py)
- [`scripts/profile_wham_workload.py`](https://github.com/uwplasma/magpylib_jax/blob/main/scripts/profile_wham_workload.py)

Artifacts produced per source family:

- JAX trace (`jax.profiler.trace`)
- HLO dump (`compiler_ir(..., dialect="hlo")`)
- device memory profile snapshot (`jax.profiler.save_device_memory_profile`)

## Fixed-observer-count JIT entrypoints

Hotspot wrappers live in [`core/kernels_extended.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels_extended.py) and cache compilation by observer count.

Representative examples:

- `current_circle_bfield_jit`
- `current_polyline_bfield_jit`
- `triangle_bfield_jit`
- `current_trisheet_bfield_jit`
- `current_tristrip_bfield_jit`
- `tetrahedron_bfield_jit`
- `magnet_trimesh_bfield_jit_faces_precomp`
- `magnet_cylinder_segment_bfield_jit`

These are mainly for profiling and specialized high-throughput workloads. They are not the default user entry point.

## What is measured

- JAX compile time per source type
- steady-state runtime (median over repeated runs)
- max absolute parity error
- peak process memory
- HLO size and hash for inspection

## HLO hash checks

Exact HLO hashes are useful for inspection and trend tracking, but they are intentionally treated as report-only in CI/nightly because unpinned JAX/XLA versions can change compiler output structure without changing correctness.

The hard gating remains on:

- parity error,
- compile/runtime thresholds,
- memory thresholds,
- benchmark thresholds,
- tests and docs.

## Threshold files

- [`benchmarks/thresholds.json`](https://github.com/uwplasma/magpylib_jax/blob/main/benchmarks/thresholds.json)
- [`profiling/thresholds.json`](https://github.com/uwplasma/magpylib_jax/blob/main/profiling/thresholds.json)
- [`profiling/thresholds_getB_jit.json`](https://github.com/uwplasma/magpylib_jax/blob/main/profiling/thresholds_getB_jit.json)

## Typical workflow after a kernel change

1. Run `profile_kernels.py` and inspect compile/runtime/memory deltas.
2. Run `profile_getB_jit.py` if the change can affect the high-level path.
3. Compare parity outputs.
4. Update thresholds only when the change is intentional and justified.
5. Keep HLO baselines as observability aids, not as the only regression signal.
