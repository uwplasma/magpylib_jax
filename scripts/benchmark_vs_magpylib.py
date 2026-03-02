"""Benchmark magpylib_jax against magpylib for implemented source types."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import magpylib as magpy
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_enable_x64", True)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
import magpylib_jax as mpj  # noqa: E402


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
    observers_jax = jnp.asarray(observers, dtype=jnp.float64)
    new_jit = jax.jit(new_fn)

    compile_s = _compile_time(lambda: new_jit(observers_jax))
    time_ref = _timeit(lambda: ref_fn(observers), repeats)
    time_new = _timeit(lambda: new_jit(observers_jax), repeats)
    err = float(np.max(np.abs(np.asarray(new_jit(observers_jax)) - ref_fn(observers))))

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
        lambda obs: dip_ref.getB(obs),
        lambda obs: dip_new.getB(obs),
        args.repeats,
        observers,
    )

    circ_ref = magpy.current.Circle(current=2.0, diameter=1.2)
    circ_new = mpj.current.Circle(current=2.0, diameter=1.2)
    entries["circle"] = _entry(
        "circle",
        lambda obs: circ_ref.getB(obs),
        lambda obs: circ_new.getB(obs),
        args.repeats,
        observers,
    )

    cub_ref = magpy.magnet.Cuboid(polarization=(0.15, -0.22, 0.3), dimension=(1.1, 0.7, 1.4))
    cub_new = mpj.magnet.Cuboid(polarization=(0.15, -0.22, 0.3), dimension=(1.1, 0.7, 1.4))
    entries["cuboid"] = _entry(
        "cuboid",
        lambda obs: cub_ref.getB(obs),
        lambda obs: cub_new.getB(obs),
        args.repeats,
        observers,
    )

    cyl_ref = magpy.magnet.Cylinder(polarization=(0.11, -0.07, 0.13), dimension=(1.4, 1.2))
    cyl_new = mpj.magnet.Cylinder(polarization=(0.11, -0.07, 0.13), dimension=(1.4, 1.2))
    entries["cylinder"] = _entry(
        "cylinder",
        lambda obs: cyl_ref.getB(obs),
        lambda obs: cyl_new.getB(obs),
        args.repeats,
        observers,
    )

    sph_ref = magpy.magnet.Sphere(polarization=(0.11, -0.07, 0.13), diameter=1.3)
    sph_new = mpj.magnet.Sphere(polarization=(0.11, -0.07, 0.13), diameter=1.3)
    entries["sphere"] = _entry(
        "sphere",
        lambda obs: sph_ref.getB(obs),
        lambda obs: sph_new.getB(obs),
        args.repeats,
        observers,
    )

    line_vertices = [(-0.5, -0.1, 0.3), (0.2, 0.4, 0.7), (0.9, 0.6, -0.2), (1.2, -0.3, 0.1)]
    line_ref = magpy.current.Polyline(current=1.7, vertices=line_vertices)
    line_new = mpj.current.Polyline(current=1.7, vertices=line_vertices)
    entries["polyline"] = _entry(
        "polyline",
        lambda obs: line_ref.getB(obs),
        lambda obs: line_new.getB(obs),
        args.repeats,
        observers,
    )

    tri_vertices = [(-0.2, -0.1, 0.0), (0.9, 0.3, 0.2), (0.1, 0.8, -0.2)]
    tri_ref = magpy.misc.Triangle(vertices=tri_vertices, polarization=(0.2, -0.1, 0.3))
    tri_new = mpj.misc.Triangle(vertices=tri_vertices, polarization=(0.2, -0.1, 0.3))
    entries["triangle"] = _entry(
        "triangle",
        lambda obs: tri_ref.getB(obs),
        lambda obs: tri_new.getB(obs),
        args.repeats,
        observers,
    )

    tet_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    tet_ref = magpy.magnet.Tetrahedron(vertices=tet_vertices, polarization=(0.1, -0.2, 0.3))
    tet_new = mpj.magnet.Tetrahedron(vertices=tet_vertices, polarization=(0.1, -0.2, 0.3))
    entries["tetrahedron"] = _entry(
        "tetrahedron",
        lambda obs: tet_ref.getB(obs),
        lambda obs: tet_new.getB(obs),
        args.repeats,
        observers,
    )

    cylseg_ref = magpy.magnet.CylinderSegment(
        polarization=(0.1, -0.2, 0.3),
        dimension=(0.4, 1.2, 1.1, -30.0, 110.0),
    )
    cylseg_new = mpj.magnet.CylinderSegment(
        polarization=(0.1, -0.2, 0.3),
        dimension=(0.4, 1.2, 1.1, -30.0, 110.0),
    )
    entries["cylindersegment"] = _entry(
        "cylindersegment",
        lambda obs: cylseg_ref.getB(obs),
        lambda obs: cylseg_new.getB(obs),
        args.repeats,
        observers,
    )

    sheet_vertices = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    sheet_faces = [[0, 1, 2], [1, 2, 3]]
    sheet_cds = [[0.7, 0.1, 0.0], [0.7, 0.1, 0.0]]
    sheet_ref = magpy.current.TriangleSheet(
        vertices=sheet_vertices,
        faces=sheet_faces,
        current_densities=sheet_cds,
    )
    sheet_new = mpj.current.TriangleSheet(
        vertices=sheet_vertices,
        faces=sheet_faces,
        current_densities=sheet_cds,
    )
    entries["trianglesheet"] = _entry(
        "trianglesheet",
        lambda obs: sheet_ref.getB(obs),
        lambda obs: sheet_new.getB(obs),
        args.repeats,
        observers,
    )

    strip_vertices = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]]
    strip_ref = magpy.current.TriangleStrip(vertices=strip_vertices, current=1.4)
    strip_new = mpj.current.TriangleStrip(vertices=strip_vertices, current=1.4)
    entries["trianglestrip"] = _entry(
        "trianglestrip",
        lambda obs: strip_ref.getB(obs),
        lambda obs: strip_new.getB(obs),
        args.repeats,
        observers,
    )

    mesh_vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mesh_faces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]
    mesh_ref = magpy.magnet.TriangularMesh(
        vertices=mesh_vertices,
        faces=mesh_faces,
        polarization=(0.1, -0.2, 0.3),
        reorient_faces=False,
        check_open=False,
    )
    mesh_new = mpj.magnet.TriangularMesh(
        vertices=mesh_vertices,
        faces=mesh_faces,
        polarization=(0.1, -0.2, 0.3),
        reorient_faces=False,
    )
    entries["triangularmesh"] = _entry(
        "triangularmesh",
        lambda obs: mesh_ref.getB(obs),
        lambda obs: mesh_new.getB(obs),
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
