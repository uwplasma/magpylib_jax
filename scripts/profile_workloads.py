"""Profile representative magpylib_jax workloads (device-aware).

Measures steady-state (warmed, ``block_until_ready``) runtime for the common
usage patterns — eager object ``getB`` per source family, a collection, and
``getFT`` — plus first-call compile time and peak device memory. Prints a report
that names the active backend, so a GPU/TPU run self-documents.

Run: ``python scripts/profile_workloads.py [--x64] [--n 100000]``
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x64", action="store_true", help="float64 (magpylib parity)")
    ap.add_argument("--n", type=int, default=100000, help="number of observers")
    ap.add_argument("--repeats", type=int, default=6)
    args = ap.parse_args()

    import jax

    if args.x64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np

    import magpylib_jax as mpj
    from magpylib_jax import getFT

    dev = jax.default_backend().upper()
    dtype = "float64" if jax.config.jax_enable_x64 else "float32"
    print(
        f"magpylib_jax {mpj.__version__} | JAX {jax.__version__} | "
        f"{dev} | {dtype} | n={args.n:,}\n"
    )

    obs = jnp.asarray(np.random.default_rng(0).normal(size=(args.n, 3)) + np.array([1.5, 0, 0]))

    def warm_run(fn):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        compile_ms = (time.perf_counter() - t0) * 1e3
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(fn())
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) * 1e3, compile_ms

    sources = {
        "Cuboid": mpj.magnet.Cuboid(polarization=(0.1, 0.2, 0.3), dimension=(1, 1, 1)),
        "Cylinder": mpj.magnet.Cylinder(polarization=(0, 0, 1), dimension=(1, 1)),
        "CylinderSegment": mpj.magnet.CylinderSegment(
            polarization=(0, 0, 1), dimension=(0.5, 1, 1, 0, 90)
        ),
        "Sphere": mpj.magnet.Sphere(polarization=(0, 0, 1), diameter=1),
        "Tetrahedron": mpj.magnet.Tetrahedron(
            polarization=(0, 0, 1), vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ),
        "Dipole": mpj.misc.Dipole(moment=(0, 0, 1)),
        "Circle": mpj.current.Circle(current=1, diameter=1),
        "Polyline": mpj.current.Polyline(current=1, vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
    }

    print("Eager object getB (steady-state / first-call compile):")
    print("| source | run (ms) | compile (ms) |")
    print("|---|---|---|")
    for name, s in sources.items():
        run, comp = warm_run(lambda s=s: s.getB(obs))
        print(f"| {name} | {run:.2f} | {comp:.0f} |")

    coll = mpj.Collection(
        sources["Cuboid"].copy(),
        mpj.misc.Dipole(moment=(0, 0, 1), position=(2, 0, 0)),
        mpj.current.Circle(current=1, diameter=1, position=(0, 2, 0)),
    )
    run, comp = warm_run(lambda: coll.getB(obs))
    print(f"| Collection(3) | {run:.2f} | {comp:.0f} |")

    print("\ngetFT (eager):")
    src = mpj.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    tgt = mpj.misc.Dipole(moment=(0, 0, 1), position=(2, 0, 0))
    run, comp = warm_run(lambda: getFT(src, tgt))
    print(f"  dipole target: {run:.1f} ms (compile {comp:.0f} ms)")

    print("\nField + gradient w.r.t. polarization (value_and_grad, jitted):")
    def scalar(pol):
        return jnp.sum(mpj.magnet.Cuboid(polarization=pol, dimension=(1, 1, 1)).getB(obs) ** 2)
    vg = jax.jit(jax.value_and_grad(scalar))
    run, comp = warm_run(lambda: vg(jnp.array([0.1, 0.2, 0.3])))
    print(f"  {run:.2f} ms (compile {comp:.0f} ms)")

    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            print(f"\npeak device bytes: {stats.get('peak_bytes_in_use', 0) / 1e6:.1f} MB")
    except Exception:
        pass


if __name__ == "__main__":
    main()
