# Performance Examples

## Local kernel profiling

```bash
python scripts/profile_kernels.py \
  --n-observers 512 \
  --repeats 1 \
  --output profiling/profile.local.json \
  --output-dir profiling/artifacts/local
```

## End-to-end JIT-safe `getB` profiling

```bash
python scripts/profile_getB_jit.py \
  --repeats 3 \
  --output profiling/getB_jit.local.json \
  --output-dir profiling/artifacts/getB_jit
```

## WHAM-style coil workload

```bash
python scripts/profile_wham_workload.py
```

That script compares upstream `magpylib` and `magpylib_jax` on a representative double-coil workload and also records the cost of converting the JAX result back to NumPy.

## What to inspect in the artifacts

- runtime and compile time in the JSON summaries,
- HLO dumps under `profiling/.../hlo/`,
- trace directories under `profiling/.../trace/`,
- memory snapshots under `profiling/.../memory/`.

## When to use kernel JIT entrypoints directly

If your application has a fixed observer count and repeatedly evaluates a single source family, the specialized wrappers in `core.kernels_extended` can be useful for isolating compile/runtime behavior.

For most users, the high-level `getB` path is the right default.
