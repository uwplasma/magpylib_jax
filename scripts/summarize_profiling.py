"""Summarize profiling metrics against thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiling", type=Path)
    parser.add_argument("thresholds", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    profile = json.loads(args.profiling.read_text(encoding="utf-8"))
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))

    summary: dict[str, dict[str, dict[str, float]]] = {}

    for key, limits in thresholds.items():
        for source, limit in limits.items():
            observed = profile[source][key]
            entry = summary.setdefault(source, {})
            entry[key] = {
                "observed": float(observed),
                "limit": float(limit),
                "delta": float(observed - limit),
                "ratio": float(observed / limit) if limit != 0 else float("inf"),
            }

    out = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is None:
        print(out)
    else:
        args.output.write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
