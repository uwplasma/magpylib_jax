# Changelog

## 1.0.1

- Restored the nightly `Full Validation` workflow to Python 3.11 so the upstream mirror tests
  run in an upstream-compatible environment.
- Kept Python 3.10 package compatibility while avoiding false workflow failures from upstream
  dependency drift.
- Added a coverage badge to the repository README.
- Simplified release publishing so PyPI uploads are driven by GitHub releases.

## 1.0.0

`magpylib_jax` 1.0.0 is the first stable release.

- Stable functional and object APIs for the implemented source families.
- JIT-safe `getB/getH/getJ/getM` by default, with end-to-end differentiability through JAX.
- Broad parity coverage against upstream Magpylib, including mirrored upstream object tests.
- CI and nightly validation for lint, typing, docs, coverage, parity, benchmarks, and profiling.
- Release automation for GitHub releases and PyPI publishing.

The repository root changelog used for GitHub releases is also available on GitHub:
[CHANGELOG.md](https://github.com/uwplasma/magpylib_jax/blob/main/CHANGELOG.md).
