"""End-to-end profiling for magpylib_jax.getB under jax.jit."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import time
from pathlib import Path

import jax
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_enable_x64", True)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage < 10_000_000:
        return int(usage * 1024)
    return int(usage)


def _median_runtime(fn, repeats: int) -> float:
    values: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        values.append(time.perf_counter() - t0)
    return float(np.median(values))


def _hlo_text(jit_fn) -> str:
    lowered = jit_fn.lower()
    hlo = lowered.compiler_ir(dialect="hlo")
    if hasattr(hlo, "as_hlo_text"):
        return hlo.as_hlo_text()
    return str(hlo)


def _build_sources_sensors():
    import magpylib_jax as mpj

    src1 = mpj.magnet.Cuboid(dimension=(1.0, 0.8, 1.2), polarization=(0.1, -0.05, 0.2))
    src1.position = [(0.0, 0.0, 0.0), (0.15, -0.05, 0.0)]

    src2 = mpj.magnet.CylinderSegment(
        polarization=(0.08, -0.12, 0.18),
        dimension=(0.3, 1.0, 0.9, -40.0, 95.0),
    )
    src2.position = (0.1, 0.2, -0.1)

    strip_vertices = [
        (0.0, 0.0, 0.0),
        (0.2, 0.5, 0.0),
        (0.6, 0.1, 0.1),
        (1.0, 0.6, -0.1),
    ]
    src3 = mpj.current.TriangleStrip(vertices=strip_vertices, current=1.4)

    grid = np.stack(np.meshgrid(np.linspace(-0.05, 0.05, 3), np.linspace(-0.05, 0.05, 3), [0.0], indexing="ij"), axis=-1)
    sens1 = mpj.Sensor(pixel=grid, position=[(0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.2, 0.0)])
    sens2 = mpj.Sensor(pixel=(0.12, 0.03, -0.04), position=(0.0, 0.0, 0.0))

    sources = [src1, src2, src3]
    sensors = [sens1, sens2]
    return sources, sensors


def _profile_getB(repeats: int, out_dir: Path) -> dict[str, float | str]:
    import magpylib_jax as mpj

    sources, sensors = _build_sources_sensors()

    def getB_fn():
        return mpj.getB(sources, sensors, pixel_agg="mean", squeeze=False)

    jit_fn = jax.jit(getB_fn)

    compile_s = 0.0
    runtime_s = 0.0
    hlo_text = "unavailable\n"
    hlo_size = 0
    hlo_hash = "unavailable"

    hlo_dir = out_dir / "hlo"
    trace_dir = out_dir / "trace"
    mem_dir = out_dir / "memory"
    hlo_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    mem_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    out = jit_fn()
    out.block_until_ready()
    compile_s = float(time.perf_counter() - t0)

    runtime_s = _median_runtime(jit_fn, repeats)
    _ = np.asarray(jit_fn())

    hlo_text = _hlo_text(jit_fn)
    hlo_path = hlo_dir / "getB_jit.hlo.txt"
    hlo_path.write_text(hlo_text, encoding="utf-8")
    hlo_bytes = hlo_text.encode("utf-8")
    hlo_size = len(hlo_bytes)
    hlo_hash = hashlib.sha256(hlo_bytes).hexdigest()

    trace_path = trace_dir / "getB_jit"
    trace_path.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_path), create_perfetto_link=False):
        _ = jit_fn().block_until_ready()

    mem_path = mem_dir / "getB_jit.memory.prof"
    try:
        jax.profiler.save_device_memory_profile(str(mem_path))
    except Exception:  # pragma: no cover
        mem_path.write_text("unavailable\n", encoding="utf-8")

    return {
        "name": "getB_jit",
        "compile_time_s": compile_s,
        "steady_state_runtime_s": runtime_s,
        "peak_memory_bytes": _peak_rss_bytes(),
        "hlo_size_bytes": hlo_size,
        "hlo_hash": hlo_hash,
        "hlo_path": str(hlo_path),
        "trace_path": str(trace_path),
        "memory_profile_path": str(mem_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile JIT-safe getB end-to-end.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("profiling/getB_jit.local.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("profiling/artifacts/getB_jit"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"getB_jit": _profile_getB(args.repeats, args.output_dir)}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
