"""Profile a WHAM-style large circle-collection workload against Magpylib."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


def _build_wham(mod):
    dz_hf = 14.3e-3 * 8
    r_in_hf = 0.5 * 86e-3
    r_out_hf = 0.5 * 730e-3
    nz = 8
    nr = 310
    current = 2000 * 17.0 / 17.51

    coll = mod.Collection()
    circle_ctor = mod.current.Circle if hasattr(mod, "current") else mod.Circle
    for z in np.linspace(-dz_hf / 2, dz_hf / 2, nz):
        for r in np.linspace(r_in_hf, r_out_hf, nr):
            coll.add(circle_ctor(current=current, diameter=2 * r, position=(0, 0, z)))
    coll.position = (0, 0, -0.98)
    coll2 = coll.copy(position=(0, 0, 0.98))
    return mod.Collection(coll, coll2)


def _timed(fn, *, repeats: int) -> tuple[list[float], object]:
    values: list[float] = []
    out_last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        values.append(time.perf_counter() - t0)
        out_last = out
    return values, out_last


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile WHAM-like circle-stack workload.")
    parser.add_argument("--output", type=Path, default=Path("profiling/wham_profile.json"))
    parser.add_argument("--small-repeats", type=int, default=3)
    parser.add_argument("--large-repeats", type=int, default=3)
    args = parser.parse_args()

    import magpylib as magpy

    import magpylib_jax as mpj

    small_grid = np.array(
        [[(x, 0, z) for x in np.linspace(0, 6, 2)] for z in np.linspace(-3, 3, 5)]
    )
    large_grid = np.array(
        [[(x, 0, z) for x in np.linspace(0, 6, 31)] for z in np.linspace(-3, 3, 101)]
    )

    ref = _build_wham(magpy)
    new = _build_wham(mpj)

    warm_small = mpj.getB(new, small_grid)
    if hasattr(warm_small, "block_until_ready"):
        warm_small.block_until_ready()
    warm_large = mpj.getB(new, large_grid)
    if hasattr(warm_large, "block_until_ready"):
        warm_large.block_until_ready()

    ref_small_vals, ref_small = _timed(
        lambda: magpy.getB(ref, small_grid),
        repeats=args.small_repeats,
    )
    new_small_vals, new_small = _timed(
        lambda: mpj.getB(new, small_grid),
        repeats=args.small_repeats,
    )
    ref_large_vals, ref_large = _timed(lambda: magpy.getB(ref, large_grid), repeats=1)
    new_large_vals, new_large = _timed(
        lambda: mpj.getB(new, large_grid),
        repeats=args.large_repeats,
    )

    t0 = time.perf_counter()
    new_large_np = np.asarray(new_large)
    to_numpy_s = time.perf_counter() - t0

    result = {
        "small_grid": {
            "magpylib_s": ref_small_vals,
            "magpylib_jax_s": new_small_vals,
            "max_abs_diff": float(
                np.max(np.abs(np.asarray(ref_small) - np.asarray(new_small)))
            ),
        },
        "large_grid": {
            "magpylib_s": ref_large_vals,
            "magpylib_jax_s": new_large_vals,
            "max_abs_diff": float(
                np.max(np.abs(np.asarray(ref_large) - np.asarray(new_large_np)))
            ),
            "to_numpy_s": to_numpy_s,
            "shape": list(new_large_np.shape),
        },
    }

    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
