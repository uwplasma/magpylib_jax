"""Benchmark magpylib_jax against magpylib for currently implemented source types."""

from __future__ import annotations

import json
import time

import jax
import magpylib as magpy
import numpy as np

import magpylib_jax as mpj

jax.config.update("jax_enable_x64", True)


def _timeit(fn, repeats: int = 5) -> float:
    vals: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals))


def main() -> None:
    rng = np.random.default_rng(42)
    observers = rng.normal(size=(20_000, 3))

    dip_ref = magpy.misc.Dipole(moment=(1.0, -0.3, 0.2))
    dip_new = mpj.Dipole(moment=(1.0, -0.3, 0.2))

    circ_ref = magpy.current.Circle(current=2.0, diameter=1.2)
    circ_new = mpj.Circle(current=2.0, diameter=1.2)

    # JIT warmup
    _ = circ_new.getB(observers[:32])

    dip_time_ref = _timeit(lambda: dip_ref.getB(observers))
    dip_time_new = _timeit(lambda: dip_new.getB(observers))
    circ_time_ref = _timeit(lambda: circ_ref.getB(observers))
    circ_time_new = _timeit(lambda: circ_new.getB(observers))

    dip_err = float(np.max(np.abs(np.asarray(dip_new.getB(observers)) - dip_ref.getB(observers))))
    circ_err = float(
        np.max(np.abs(np.asarray(circ_new.getB(observers)) - circ_ref.getB(observers)))
    )

    report = {
        "n_observers": int(observers.shape[0]),
        "dipole": {
            "magpylib_s": dip_time_ref,
            "magpylib_jax_s": dip_time_new,
            "speedup_vs_magpylib": dip_time_ref / dip_time_new,
            "max_abs_error_T": dip_err,
        },
        "circle": {
            "magpylib_s": circ_time_ref,
            "magpylib_jax_s": circ_time_new,
            "speedup_vs_magpylib": circ_time_ref / circ_time_new,
            "max_abs_error_T": circ_err,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
