# Performance

This page is the hands-on companion to the [Performance guide](../performance.md): the batching
patterns you write in your own code, and the profiling scripts the project uses to keep the library
fast. For the *why* behind JIT, `vmap`, and async timing, read the guide first.

## Batch with `jax.vmap` + `jax.jit`

The field core is `vmap`-friendly and `jit`-compilable, so you can evaluate one field function over
a whole batch of source parameters with no Python loop. Compile once, then time the compiled call
with `block_until_ready` — never the first, tracing call:

```python
import time
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

OBS = jnp.array([[0.2, 0.1, 0.4], [0.5, 0.0, 0.7], [-0.3, 0.2, 0.5]])

def field(pol):
    return mpj.magnet.Cuboid(dimension=(1.0, 0.8, 1.2), polarization=pol).getB(OBS)

pols = jax.random.normal(jax.random.PRNGKey(0), (256, 3)) * 0.5
batched = jax.jit(jax.vmap(field))

out = jax.block_until_ready(batched(pols))          # compile + warm up (not timed)

t0 = time.perf_counter()
for _ in range(20):
    out = jax.block_until_ready(batched(pols))
per_call_ms = (time.perf_counter() - t0) / 20 * 1e3
print(out.shape, f"{per_call_ms:.3f} ms")           # (256, 3, 3)
```

Full script:
[`examples/differentiable/jit_vmap_batching.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/differentiable/jit_vmap_batching.py).

```{admonition} The two timing rules
:class: important
1. Never time the first call — it includes tracing and compilation.
2. Always `block_until_ready` before reading the clock, because JAX dispatch is asynchronous.
```

## Profiling scripts

The repository ships profiling drivers you can run locally to inspect compile time, runtime, memory,
and HLO for each source family and for the high-level path.

::::{tab-set}
:::{tab-item} Kernels
```bash
python scripts/profile_kernels.py \
  --n-observers 512 \
  --repeats 1 \
  --output profiling/profile.local.json \
  --output-dir profiling/artifacts/local
```
:::
:::{tab-item} High-level getB
```bash
python scripts/profile_getB_jit.py \
  --repeats 3 \
  --output profiling/getB_jit.local.json \
  --output-dir profiling/artifacts/getB_jit
```
:::
:::{tab-item} WHAM workload
```bash
python scripts/profile_wham_workload.py
```
:::
::::

The WHAM script compares upstream `magpylib` and `magpylib_jax` on a representative double-coil
workload and records the cost of converting the JAX result back to NumPy.

## What the artifacts contain

- runtime and compile time in the JSON summaries,
- HLO dumps under `profiling/.../hlo/`,
- trace directories under `profiling/.../trace/`,
- device-memory snapshots under `profiling/.../memory/`.

Interpret them against the thresholds in
[`profiling/thresholds.json`](https://github.com/uwplasma/magpylib_jax/blob/main/profiling/thresholds.json)
and
[`profiling/thresholds_getB_jit.json`](https://github.com/uwplasma/magpylib_jax/blob/main/profiling/thresholds_getB_jit.json).

## When to reach for kernel JIT entrypoints

If your application has a **fixed observer count** and repeatedly evaluates a **single source
family**, the specialized wrappers in `magpylib_jax.core.kernels` cache compilation by observer
count and are handy for isolating compile/runtime behavior. For everything else, the high-level
`getB` path is the right default — it already runs through a JIT-safe core.

## Larger workloads

Coil and array applications stress the batching and superposition paths:

- [`examples/applications/coil.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/applications/coil.py) — air coils and a Helmholtz pair.
- [`examples/applications/halbach.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/applications/halbach.py) — a discrete Halbach cylinder from rotated cuboids.

## Where to go next

- [Performance guide](../performance.md) — the methodology, the CPU benchmark, and the caching model.
- [Optimization](optimization.md) — the same JIT patterns inside a gradient-descent loop.
