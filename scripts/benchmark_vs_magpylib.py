"""Benchmark magpylib_jax against magpylib for implemented source types."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import jax
import magpylib as magpy
import numpy as np

import magpylib_jax as mpj

jax.config.update("jax_enable_x64", True)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


def _timeit(fn, repeats: int) -> float:
    values: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        values.append(time.perf_counter() - t0)
    return float(np.median(values))


def _compile_time(fn) -> float:
    t0 = time.perf_counter()
    out = fn()
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    return float(time.perf_counter() - t0)


def _entry(name: str, ref_fn, new_fn, repeats: int, observers: np.ndarray) -> dict[str, float]:
    compile_s = _compile_time(new_fn)
    time_ref = _timeit(ref_fn, repeats)
    time_new = _timeit(new_fn, repeats)
    err = float(np.max(np.abs(np.asarray(new_fn()) - ref_fn())))

    return {
        "name": name,
        "magpylib_s": time_ref,
        "magpylib_jax_s": time_new,
        "magpylib_jax_compile_s": compile_s,
        "speedup_vs_magpylib": time_ref / time_new,
        "max_abs_error_T": err,
        "n_observers": int(observers.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-observers", type=int, default=8_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    observers = rng.normal(size=(args.n_observers, 3))

    entries: dict[str, dict[str, float]] = {}

    dip_ref = magpy.misc.Dipole(moment=(1.0, -0.3, 0.2))
    dip_new = mpj.misc.Dipole(moment=(1.0, -0.3, 0.2))
    entries["dipole"] = _entry(
        "dipole",
        lambda: dip_ref.getB(observers),
        lambda: dip_new.getB(observers),
        args.repeats,
        observers,
    )

    circ_ref = magpy.current.Circle(current=2.0, diameter=1.2)
    circ_new = mpj.current.Circle(current=2.0, diameter=1.2)
    entries["circle"] = _entry(
        "circle",
        lambda: circ_ref.getB(observers),
        lambda: circ_new.getB(observers),
        args.repeats,
        observers,
    )

    cub_ref = magpy.magnet.Cuboid(polarization=(0.15, -0.22, 0.3), dimension=(1.1, 0.7, 1.4))
    cub_new = mpj.magnet.Cuboid(polarization=(0.15, -0.22, 0.3), dimension=(1.1, 0.7, 1.4))
    entries["cuboid"] = _entry(
        "cuboid",
        lambda: cub_ref.getB(observers),
        lambda: cub_new.getB(observers),
        args.repeats,
        observers,
    )

    cyl_ref = magpy.magnet.Cylinder(polarization=(0.11, -0.07, 0.13), dimension=(1.4, 1.2))
    cyl_new = mpj.magnet.Cylinder(polarization=(0.11, -0.07, 0.13), dimension=(1.4, 1.2))
    entries["cylinder"] = _entry(
        "cylinder",
        lambda: cyl_ref.getB(observers),
        lambda: cyl_new.getB(observers),
        args.repeats,
        observers,
    )

    sph_ref = magpy.magnet.Sphere(polarization=(0.11, -0.07, 0.13), diameter=1.3)
    sph_new = mpj.magnet.Sphere(polarization=(0.11, -0.07, 0.13), diameter=1.3)
    entries["sphere"] = _entry(
        "sphere",
        lambda: sph_ref.getB(observers),
        lambda: sph_new.getB(observers),
        args.repeats,
        observers,
    )

    line_vertices = [(-0.5, -0.1, 0.3), (0.2, 0.4, 0.7), (0.9, 0.6, -0.2), (1.2, -0.3, 0.1)]
    line_ref = magpy.current.Polyline(current=1.7, vertices=line_vertices)
    line_new = mpj.current.Polyline(current=1.7, vertices=line_vertices)
    entries["polyline"] = _entry(
        "polyline",
        lambda: line_ref.getB(observers),
        lambda: line_new.getB(observers),
        args.repeats,
        observers,
    )

    tri_vertices = [(-0.2, -0.1, 0.0), (0.9, 0.3, 0.2), (0.1, 0.8, -0.2)]
    tri_ref = magpy.misc.Triangle(vertices=tri_vertices, polarization=(0.2, -0.1, 0.3))
    tri_new = mpj.misc.Triangle(vertices=tri_vertices, polarization=(0.2, -0.1, 0.3))
    entries["triangle"] = _entry(
        "triangle",
        lambda: tri_ref.getB(observers),
        lambda: tri_new.getB(observers),
        args.repeats,
        observers,
    )

    tet_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    tet_ref = magpy.magnet.Tetrahedron(vertices=tet_vertices, polarization=(0.1, -0.2, 0.3))
    tet_new = mpj.magnet.Tetrahedron(vertices=tet_vertices, polarization=(0.1, -0.2, 0.3))
    entries["tetrahedron"] = _entry(
        "tetrahedron",
        lambda: tet_ref.getB(observers),
        lambda: tet_new.getB(observers),
        args.repeats,
        observers,
    )

    report = {
        "n_observers": int(observers.shape[0]),
        "repeats": args.repeats,
        **entries,
    }

    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
