from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _is_unpinned(requirement: str) -> bool:
    requirement_head = requirement.split(";", 1)[0]
    return all(op not in requirement_head for op in ("<", ">", "=", "~", "!"))


def test_project_dependencies_are_unpinned() -> None:
    data = _load_pyproject()
    deps = data["project"]["dependencies"]
    assert deps
    assert all(_is_unpinned(dep) for dep in deps)


def test_optional_dependencies_are_unpinned() -> None:
    data = _load_pyproject()
    optional = data["project"]["optional-dependencies"]
    assert optional
    for group in optional.values():
        assert all(_is_unpinned(dep) for dep in group)


def test_python_floor_is_310() -> None:
    data = _load_pyproject()
    assert data["project"]["requires-python"] == ">=3.10"


def test_static_analysis_targets_python310() -> None:
    data = _load_pyproject()
    assert data["tool"]["ruff"]["target-version"] == "py310"
    assert data["tool"]["mypy"]["python_version"] == "3.10"
