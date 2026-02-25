"""Kernel-level profiling with parity, compile/runtime, HLO, trace, and memory snapshots."""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import time
from pathlib import Path

import jax
import magpylib as magpy
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_enable_x64", True)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
import magpylib_jax as mpj  # noqa: E402


def _median_runtime(fn, repeats: int) -> float:
    values: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        values.append(time.perf_counter() - t0)
    return float(np.median(values))


def _compile_time(jit_fn, observers: np.ndarray) -> float:
    t0 = time.perf_counter()
    out = jit_fn(observers)
    out.block_until_ready()
    return float(time.perf_counter() - t0)


def _hlo_text(jit_fn, observers: np.ndarray) -> str:
    lowered = jit_fn.lower(observers)
    hlo = lowered.compiler_ir(dialect="hlo")
    if hasattr(hlo, "as_hlo_text"):
        return hlo.as_hlo_text()
    return str(hlo)


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes.
    if usage < 10_000_000:
        return int(usage * 1024)
    return int(usage)


def _profile_entry(
    name: str,
    ref_fn,
    new_fn,
    observers: np.ndarray,
    repeats: int,
    out_dir: Path,
) -> dict[str, float | str]:
    jit_new = jax.jit(new_fn)

    compile_s = _compile_time(jit_new, observers)
    runtime_s = _median_runtime(lambda: jit_new(observers), repeats)
    ref_out = ref_fn(observers)
    new_out = np.asarray(jit_new(observers))
    max_abs_error = float(np.max(np.abs(new_out - ref_out)))

    hlo_dir = out_dir / "hlo"
    trace_dir = out_dir / "trace"
    mem_dir = out_dir / "memory"
    hlo_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    mem_dir.mkdir(parents=True, exist_ok=True)

    hlo_path = hlo_dir / f"{name}.hlo.txt"
    hlo_path.write_text(_hlo_text(jit_new, observers), encoding="utf-8")

    trace_path = trace_dir / name
    trace_path.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_path), create_perfetto_link=False):
        _ = jit_new(observers).block_until_ready()

    mem_path = mem_dir / f"{name}.memory.prof"
    try:
        jax.profiler.save_device_memory_profile(str(mem_path))
    except Exception:  # pragma: no cover - backend-dependent
        mem_path.write_text("unavailable\n", encoding="utf-8")

    return {
        "name": name,
        "compile_time_s": compile_s,
        "steady_state_runtime_s": runtime_s,
        "max_abs_parity_error_T": max_abs_error,
        "peak_memory_bytes": _peak_rss_bytes(),
        "hlo_path": str(hlo_path),
        "trace_path": str(trace_path),
        "memory_profile_path": str(mem_path),
    }


def _profiles(observers: np.ndarray) -> dict[str, tuple]:
    tri_vertices = [(-0.2, -0.1, 0.0), (0.9, 0.3, 0.2), (0.1, 0.8, -0.2)]
    tetra_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    strip_vertices = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]]
    sheet_vertices = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    sheet_faces = [[0, 1, 2], [1, 2, 3]]
    sheet_cds = [[0.7, 0.1, 0.0], [0.7, 0.1, 0.0]]
    mesh_vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mesh_faces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]

    refs_news = {
        "dipole": (
            magpy.misc.Dipole(moment=(1.0, -0.3, 0.2)),
            mpj.misc.Dipole(moment=(1.0, -0.3, 0.2)),
        ),
        "circle": (
            magpy.current.Circle(current=2.0, diameter=1.2),
            mpj.current.Circle(current=2.0, diameter=1.2),
        ),
        "cuboid": (
            magpy.magnet.Cuboid(polarization=(0.15, -0.22, 0.3), dimension=(1.1, 0.7, 1.4)),
            mpj.magnet.Cuboid(polarization=(0.15, -0.22, 0.3), dimension=(1.1, 0.7, 1.4)),
        ),
        "cylinder": (
            magpy.magnet.Cylinder(polarization=(0.11, -0.07, 0.13), dimension=(1.4, 1.2)),
            mpj.magnet.Cylinder(polarization=(0.11, -0.07, 0.13), dimension=(1.4, 1.2)),
        ),
        "cylindersegment": (
            magpy.magnet.CylinderSegment(
                polarization=(0.1, -0.2, 0.3),
                dimension=(0.4, 1.2, 1.1, -30.0, 110.0),
            ),
            mpj.magnet.CylinderSegment(
                polarization=(0.1, -0.2, 0.3),
                dimension=(0.4, 1.2, 1.1, -30.0, 110.0),
            ),
        ),
        "sphere": (
            magpy.magnet.Sphere(polarization=(0.11, -0.07, 0.13), diameter=1.3),
            mpj.magnet.Sphere(polarization=(0.11, -0.07, 0.13), diameter=1.3),
        ),
        "polyline": (
            magpy.current.Polyline(
                current=1.7,
                vertices=[(-0.5, -0.1, 0.3), (0.2, 0.4, 0.7), (0.9, 0.6, -0.2), (1.2, -0.3, 0.1)],
            ),
            mpj.current.Polyline(
                current=1.7,
                vertices=[(-0.5, -0.1, 0.3), (0.2, 0.4, 0.7), (0.9, 0.6, -0.2), (1.2, -0.3, 0.1)],
            ),
        ),
        "triangle": (
            magpy.misc.Triangle(vertices=tri_vertices, polarization=(0.2, -0.1, 0.3)),
            mpj.misc.Triangle(vertices=tri_vertices, polarization=(0.2, -0.1, 0.3)),
        ),
        "trianglesheet": (
            magpy.current.TriangleSheet(
                vertices=sheet_vertices,
                faces=sheet_faces,
                current_densities=sheet_cds,
            ),
            mpj.current.TriangleSheet(
                vertices=sheet_vertices,
                faces=sheet_faces,
                current_densities=sheet_cds,
            ),
        ),
        "trianglestrip": (
            magpy.current.TriangleStrip(vertices=strip_vertices, current=1.4),
            mpj.current.TriangleStrip(vertices=strip_vertices, current=1.4),
        ),
        "triangularmesh": (
            magpy.magnet.TriangularMesh(
                vertices=mesh_vertices,
                faces=mesh_faces,
                polarization=(0.1, -0.2, 0.3),
                reorient_faces=False,
                check_open=False,
            ),
            mpj.magnet.TriangularMesh(
                vertices=mesh_vertices,
                faces=mesh_faces,
                polarization=(0.1, -0.2, 0.3),
                reorient_faces=False,
            ),
        ),
        "tetrahedron": (
            magpy.magnet.Tetrahedron(vertices=tetra_vertices, polarization=(0.1, -0.2, 0.3)),
            mpj.magnet.Tetrahedron(vertices=tetra_vertices, polarization=(0.1, -0.2, 0.3)),
        ),
    }

    return {
        name: (
            lambda obs, src=src_ref: src.getB(obs),
            lambda obs, src=src_new: src.getB(obs),
        )
        for name, (src_ref, src_new) in refs_news.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-observers", type=int, default=4000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("profiling"))
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    observers = rng.normal(size=(args.n_observers, 3))

    entries: dict[str, dict[str, float | str]] = {}
    for name, (ref_fn, new_fn) in _profiles(observers).items():
        entries[name] = _profile_entry(
            name=name,
            ref_fn=ref_fn,
            new_fn=new_fn,
            observers=observers,
            repeats=args.repeats,
            out_dir=args.output_dir,
        )

    report = {
        "n_observers": int(observers.shape[0]),
        "repeats": int(args.repeats),
        **entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
