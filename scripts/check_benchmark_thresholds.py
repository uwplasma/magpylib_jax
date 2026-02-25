"""Validate benchmark results against committed thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("thresholds", type=Path)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))

    errors = []

    max_err = thresholds.get("max_abs_error_T", {})
    max_slowdown = thresholds.get("max_runtime_slowdown_vs_magpylib", {})

    for source, err_limit in max_err.items():
        observed = benchmark[source]["max_abs_error_T"]
        if observed > err_limit:
            errors.append(
                f"{source}: max_abs_error_T {observed:.3e} exceeds limit {err_limit:.3e}"
            )

    for source, slowdown_limit in max_slowdown.items():
        ref_s = benchmark[source]["magpylib_s"]
        new_s = benchmark[source]["magpylib_jax_s"]
        slowdown = new_s / ref_s if ref_s > 0 else float("inf")
        if slowdown > slowdown_limit:
            errors.append(
                f"{source}: runtime slowdown {slowdown:.3f} exceeds limit {slowdown_limit:.3f}"
            )

    if errors:
        details = "\n".join(errors)
        raise SystemExit(f"Benchmark thresholds failed:\n{details}")

    print("Benchmark thresholds passed.")


if __name__ == "__main__":
    main()
