"""Kernel-level profiling with parity, compile/runtime, HLO, trace, and memory snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import time
from collections.abc import Callable
from pathlib import Path

import jax
import magpylib as magpy
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_enable_x64", True)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


_JIT_CACHE: dict[tuple[str, int], Callable] = {}


def _jit_for_profile(name: str, fn: Callable, n_obs: int) -> Callable:
    """Cache JIT-compiled entrypoints by name and observer count."""
    key = (name, n_obs)
    if key not in _JIT_CACHE:
        _JIT_CACHE[key] = jax.jit(fn)
    return _JIT_CACHE[key]


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
    hlo_dir = out_dir / "hlo"
    trace_dir = out_dir / "trace"
    mem_dir = out_dir / "memory"
    hlo_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    mem_dir.mkdir(parents=True, exist_ok=True)

    jit_new = None
    compile_s = 0.0
    runtime_s = 0.0
    hlo_path = hlo_dir / f"{name}.hlo.txt"
    trace_path = trace_dir / name
    mem_path = mem_dir / f"{name}.memory.prof"
    hlo_text = "unavailable\n"
    hlo_size = 0
    hlo_hash = "unavailable"

    try:
        jit_new = _jit_for_profile(name, new_fn, int(observers.shape[0]))
        compile_s = _compile_time(jit_new, observers)
        runtime_s = _median_runtime(lambda: jit_new(observers), repeats)
        new_out = np.asarray(jit_new(observers))
        hlo_text = _hlo_text(jit_new, observers)
        hlo_path.write_text(hlo_text, encoding="utf-8")
        hlo_bytes = hlo_text.encode("utf-8")
        hlo_size = len(hlo_bytes)
        hlo_hash = hashlib.sha256(hlo_bytes).hexdigest()

        trace_path.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(trace_path), create_perfetto_link=False):
            _ = jit_new(observers).block_until_ready()

        try:
            jax.profiler.save_device_memory_profile(str(mem_path))
        except Exception:  # pragma: no cover - backend-dependent
            mem_path.write_text("unavailable\n", encoding="utf-8")
    except Exception as err:  # pragma: no cover - profiling fallback
        logging.warning("JIT profiling failed for %s: %s", name, err)
        runtime_s = _median_runtime(lambda: new_fn(observers), repeats)
        new_out = np.asarray(new_fn(observers))
        hlo_path.write_text(hlo_text, encoding="utf-8")
        trace_path.mkdir(parents=True, exist_ok=True)
        mem_path.write_text("unavailable\n", encoding="utf-8")

    ref_out = ref_fn(observers)
    max_abs_error = float(np.max(np.abs(new_out - ref_out)))

    return {
        "name": name,
        "compile_time_s": compile_s,
        "steady_state_runtime_s": runtime_s,
        "max_abs_parity_error_T": max_abs_error,
        "peak_memory_bytes": _peak_rss_bytes(),
        "hlo_size_bytes": hlo_size,
        "hlo_hash": hlo_hash,
        "hlo_path": str(hlo_path),
        "trace_path": str(trace_path),
        "memory_profile_path": str(mem_path),
    }


def _profiles(observers: np.ndarray) -> dict[str, tuple]:
    import magpylib_jax as mpj
    from magpylib_jax.core.kernels_extended import (
        current_circle_bfield_jit,
        current_trisheet_bfield_jit,
        current_tristrip_bfield_jit,
        current_polyline_bfield_jit,
        magnet_trimesh_bfield_jit_faces_precomp,
        precompute_cylinder_segment_geometry,
        precompute_trimesh_geometry,
        tetrahedron_bfield_jit,
        triangle_bfield_jit,
    )

    tri_vertices = [(-0.2, -0.1, 0.0), (0.9, 0.3, 0.2), (0.1, 0.8, -0.2)]
    tetra_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    strip_vertices = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]]
    sheet_vertices = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    sheet_faces = [[0, 1, 2], [1, 2, 3]]
    sheet_cds = [[0.7, 0.1, 0.0], [0.7, 0.1, 0.0]]
    mesh_vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mesh_faces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]
    polyline_vertices = [
        (-0.5, -0.1, 0.3),
        (0.2, 0.4, 0.7),
        (0.9, 0.6, -0.2),
        (1.2, -0.3, 0.1),
    ]

    mesh_tris = np.array(mesh_vertices, dtype=float)[np.array(mesh_faces, dtype=int)]
    mesh_arr, mesh_nvec, mesh_L, mesh_l1, mesh_l2 = precompute_trimesh_geometry(mesh_tris)
    cyl_dim = np.array((0.4, 1.2, 1.1, -30.0, 110.0), dtype=float)
    cyl_mesh, cyl_nvec, cyl_L, cyl_l1, cyl_l2 = precompute_cylinder_segment_geometry(cyl_dim)
    polyline_start = np.array(polyline_vertices[:-1], dtype=float)
    polyline_end = np.array(polyline_vertices[1:], dtype=float)

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
                vertices=polyline_vertices,
            ),
            mpj.current.Polyline(
                current=1.7,
                vertices=polyline_vertices,
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
                check_open=False,
            ),
        ),
        "tetrahedron": (
            magpy.magnet.Tetrahedron(vertices=tetra_vertices, polarization=(0.1, -0.2, 0.3)),
            mpj.magnet.Tetrahedron(vertices=tetra_vertices, polarization=(0.1, -0.2, 0.3)),
        ),
    }

    profiles = {
        name: (
            lambda obs, src=src_ref: src.getB(obs),
            lambda obs, src=src_new: src.getB(obs),
        )
        for name, (src_ref, src_new) in refs_news.items()
    }

    profiles.update(
        {
            "triangle_jit": (
                lambda obs: magpy.misc.Triangle(
                    vertices=tri_vertices, polarization=(0.2, -0.1, 0.3)
                ).getB(obs),
                lambda obs: triangle_bfield_jit(
                    obs, tri_vertices, np.array((0.2, -0.1, 0.3))
                ),
            ),
            "circle_jit": (
                lambda obs: magpy.current.Circle(current=2.0, diameter=1.2).getB(obs),
                lambda obs: current_circle_bfield_jit(
                    obs, np.array(1.2, dtype=float), np.array(2.0, dtype=float)
                ),
            ),
            "polyline_jit": (
                lambda obs: magpy.current.Polyline(
                    current=1.7,
                    vertices=polyline_vertices,
                ).getB(obs),
                lambda obs: current_polyline_bfield_jit(
                    obs, polyline_start, polyline_end, np.array(1.7, dtype=float)
                ),
            ),
            "trianglesheet_jit": (
                lambda obs: magpy.current.TriangleSheet(
                    vertices=sheet_vertices, faces=sheet_faces, current_densities=sheet_cds
                ).getB(obs),
                lambda obs: current_trisheet_bfield_jit(
                    obs,
                    np.array(sheet_vertices, dtype=float),
                    np.array(sheet_faces, dtype=int),
                    np.array(sheet_cds, dtype=float),
                ),
            ),
            "trianglestrip_jit": (
                lambda obs: magpy.current.TriangleStrip(
                    vertices=strip_vertices, current=1.4
                ).getB(obs),
                lambda obs: current_tristrip_bfield_jit(
                    obs, np.array(strip_vertices, dtype=float), np.array(1.4, dtype=float)
                ),
            ),
            "tetrahedron_jit": (
                lambda obs: magpy.magnet.Tetrahedron(
                    vertices=tetra_vertices, polarization=(0.1, -0.2, 0.3)
                ).getB(obs),
                lambda obs: tetrahedron_bfield_jit(
                    obs, np.array(tetra_vertices, dtype=float), np.array((0.1, -0.2, 0.3))
                ),
            ),
            "triangularmesh_jit": (
                lambda obs: magpy.magnet.TriangularMesh(
                    vertices=mesh_vertices,
                    faces=mesh_faces,
                    polarization=(0.1, -0.2, 0.3),
                    reorient_faces=False,
                    check_open=False,
                ).getB(obs),
                lambda obs: magnet_trimesh_bfield_jit_faces_precomp(
                    obs,
                    mesh_arr,
                    np.array((0.1, -0.2, 0.3), dtype=float),
                    mesh_nvec,
                    mesh_L,
                    mesh_l1,
                    mesh_l2,
                    in_out="auto",
                ),
            ),
            "cylindersegment_jit": (
                lambda obs: magpy.magnet.CylinderSegment(
                    polarization=(0.1, -0.2, 0.3),
                    dimension=(0.4, 1.2, 1.1, -30.0, 110.0),
                ).getB(obs),
                lambda obs: magnet_trimesh_bfield_jit_faces_precomp(
                    obs,
                    cyl_mesh,
                    np.array((0.1, -0.2, 0.3), dtype=float),
                    cyl_nvec,
                    cyl_L,
                    cyl_l1,
                    cyl_l2,
                    in_out="auto",
                ),
            ),
        }
    )

    return profiles


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
