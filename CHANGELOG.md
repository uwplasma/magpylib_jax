# Changelog

## 1.0.1 - 2026-04-20

### Release maintenance

- Restored the `Full Validation` workflow to Python 3.11 so the mirrored upstream Magpylib
  suite runs against an upstream-compatible dependency set.
- Kept Python 3.10 support in package metadata and CI compatibility coverage without forcing
  the nightly upstream-mirror workflow onto an unsupported upstream combination.
- Added a coverage badge to the README and kept the PyPI badges aligned with the published
  package name `magpylib-jax`.
- Simplified the PyPI release workflow so publishing is driven from GitHub releases and
  manual dispatch, using trusted publishing and avoiding duplicate release jobs from both
  tag-push and release events.

## 1.0.0 - 2026-04-17

`magpylib_jax` 1.0.0 is the first stable release of the project.

### Highlights

- Stable functional and object APIs for the currently implemented Magpylib-compatible source families:
  `Dipole`, `Circle`, `Polyline`, `Triangle`, `TriangleSheet`, `TriangleStrip`,
  `Cuboid`, `Cylinder`, `CylinderSegment`, `Sphere`, `Tetrahedron`, and `TriangularMesh`.
- JIT-safe `getB/getH/getJ/getM` by default, with end-to-end differentiability through JAX.
- Compatibility coverage for collection, sensor, motion, orientation, path, shaping, and squeeze semantics.
- Large parity suite with direct numeric comparisons against upstream Magpylib and mirrored upstream object tests.
- Profiling, benchmark, coverage, lint, type-check, and documentation gates in CI and nightly workflows.
- High-level performance work for repeated object evaluations, including preparation caches,
  circle-stack fast paths, `TriangularMesh` geometry reuse, and `CylinderSegment` precomputation reuse.

### Packaging and release

- Published package metadata for PyPI distribution.
- GitHub release automation and PyPI publish workflow with distribution build and `twine check`.

### Notes

- `output="dataframe"` remains supported, but intentionally runs outside the JIT path to preserve Magpylib-compatible semantics.
- For GPU-backed environments, install the desired JAX runtime first and then install `magpylib-jax`.
