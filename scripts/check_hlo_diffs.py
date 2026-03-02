"""Validate hotspot HLO hashes against a baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiling", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()

    profile = json.loads(args.profiling.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    errors: list[str] = []
    for name, expected_value in baseline.items():
        if name not in profile:
            errors.append(f"{name}: missing from profiling report")
            continue
        observed_hash = profile[name].get("hlo_hash", "unavailable")
        if observed_hash == "unavailable":
            errors.append(f"{name}: HLO hash unavailable")
            continue
        if isinstance(expected_value, list):
            expected_hashes = [str(value) for value in expected_value]
        else:
            expected_hashes = [str(expected_value)]
        if observed_hash not in expected_hashes:
            errors.append(
                f"{name}: hlo_hash {observed_hash} does not match baseline "
                f"{', '.join(expected_hashes)}"
            )

    if errors:
        raise SystemExit("HLO hash checks failed:\n" + "\n".join(errors))

    print("HLO hash checks passed.")


if __name__ == "__main__":
    main()
