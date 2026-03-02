import time

import numpy as np

import magpylib_jax as mpj


def test_perf_smoke_large_batch_executes_quickly() -> None:
    rng = np.random.default_rng(123)
    observers = rng.normal(size=(50_000, 3))
    src = mpj.Circle(current=2.0, diameter=1.0)

    # Warm up to exclude one-time compilation overhead from the runtime gate.
    _ = src.getB(observers)

    t0 = time.perf_counter()
    _ = src.getB(observers)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0
