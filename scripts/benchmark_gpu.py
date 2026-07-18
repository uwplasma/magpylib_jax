"""Device-aware micro-benchmark for magpylib_jax (CPU / GPU / TPU).

Self-contained: needs only ``magpylib-jax`` (which pulls in JAX + NumPy). Run it on
whatever backend JAX picks up — the report prints the active device, so a run on a
GPU/TPU host self-documents.

Google Colab (GPU runtime → Runtime ▸ Change runtime type ▸ GPU)::

    !pip install -q magpylib-jax
    !python -c "import urllib.request; urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/uwplasma/magpylib_jax/main/scripts/benchmark_gpu.py',
        'benchmark_gpu.py')"
    !python benchmark_gpu.py            # add --x64 for float64

Locally::

    python scripts/benchmark_gpu.py [--x64] [--sizes 1000 10000 100000 1000000]

It reports, per observer-count:
  * forward  — time for one jitted ``getB`` over the batch,
  * field+∇  — time for ``value_and_grad`` of a scalar field summary w.r.t. the
    3-component polarization (a full field evaluation *and* its exact gradient).
Both are jitted once and timed warm; wrap in ``block_until_ready`` so we measure
compute, not async dispatch.
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x64", action="store_true", help="enable float64 (magpylib parity)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    import jax

    if args.x64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    import magpylib_jax as mpj

    backend = jax.default_backend().upper()
    dtype = "float64" if jax.config.jax_enable_x64 else "float32"
    print(f"magpylib_jax {mpj.__version__} | JAX {jax.__version__} | backend {backend} | {dtype}")
    print(f"devices: {jax.devices()}\n")

    def timeit(fn) -> float:
        jax.block_until_ready(fn())  # warm / compile
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(fn())
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) * 1e3  # ms

    dim = (1.0, 1.0, 1.0)
    pol0 = jnp.array([0.1, 0.2, 0.3])

    print("| observers | forward getB (ms) | field + ∇ (ms) |")
    print("|---|---|---|")
    for n in args.sizes:
        obs = jnp.asarray(np.random.default_rng(0).normal(size=(n, 3)) + np.array([1.5, 0, 0]))

        fwd = jax.jit(lambda o=obs: mpj.magnet.Cuboid(polarization=pol0, dimension=dim).getB(o))
        t_fwd = timeit(lambda f=fwd: f())

        def scalar(pol, o=obs):
            return jnp.sum(mpj.magnet.Cuboid(polarization=pol, dimension=dim).getB(o) ** 2)

        vg = jax.jit(jax.value_and_grad(scalar))
        t_vg = timeit(lambda v=vg: v(pol0))

        print(f"| {n:>9,} | {t_fwd:>17.2f} | {t_vg:>14.2f} |")


if __name__ == "__main__":
    main()
