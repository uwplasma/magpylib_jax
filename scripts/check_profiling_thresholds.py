"""Validate profiling report against regression thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiling", type=Path)
    parser.add_argument("thresholds", type=Path)
    args = parser.parse_args()

    profile = json.loads(args.profiling.read_text(encoding="utf-8"))
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))

    errors: list[str] = []

    for key, limits in thresholds.items():
        for source, limit in limits.items():
            observed = profile[source][key]
            if observed > limit:
                errors.append(
                    f"{source}: {key} {observed:.6g} exceeds limit {limit:.6g}"
                )

    if errors:
        raise SystemExit("Profiling thresholds failed:\n" + "\n".join(errors))

    print("Profiling thresholds passed.")


if __name__ == "__main__":
    main()
