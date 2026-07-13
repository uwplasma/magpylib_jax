"""Smoke test: every example script imports and its ``main()`` runs without raising.

Discovers all ``examples/**/*.py`` scripts, imports each in isolation (with Matplotlib forced to the
non-interactive Agg backend), and -- if the module defines ``main()`` -- calls it and asserts it
returns something. This keeps the runnable examples green in CI without duplicating their logic.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# Headless plotting for any script that builds a figure.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = sorted(p for p in EXAMPLES_DIR.rglob("*.py") if "_output" not in p.parts)


def _module_id(path: Path) -> str:
    return path.relative_to(EXAMPLES_DIR).with_suffix("").as_posix().replace("/", ".")


def test_examples_discovered() -> None:
    assert SCRIPTS, "no example scripts found"


@pytest.mark.parametrize("script", SCRIPTS, ids=_module_id)
def test_example_runs(script: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"examples.{_module_id(script)}", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # runs module-level code, not the __main__ guard

    if hasattr(module, "main"):
        result = module.main()
        assert result is not None, f"{script.name}: main() returned None"

    import matplotlib.pyplot as plt

    plt.close("all")
