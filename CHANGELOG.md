# Changelog

## 2.1.0 - 2026-07-13

### Added

- **`getFT` now supports all magnet and triangle-current targets** — Cylinder, CylinderSegment,
  Tetrahedron, TriangularMesh, TriangleStrip, and TriangleSheet, in addition to Dipole/Sphere/
  Cuboid/Circle/Polyline. Meshing replicates magpylib's `target_meshing`; parity vs `magpy.getFT`
  is ~1e-10 (magnets) and ~1e-16 (currents). Only `Triangle` (surface charge) and `CustomSource`
  remain unsupported targets.
- **getFT tutorial and show() gallery** docs pages, and per-kernel citations plus a force/torque
  derivation in the equations reference.
- **Device-aware benchmark** — `scripts/make_figures.py` labels the benchmark panels with the
  active JAX backend (CPU/GPU/TPU), so re-running on a GPU host self-documents the device.

### Fixed

- **All sources accept a `meshing` constructor kwarg** (magpylib parity). In 2.0.0 only magnet
  sources stored `meshing` and current/misc sources raised `TypeError` on `meshing=...`; `getFT`
  reads `target.meshing`, so this now works uniformly (`meshing` is handled once in `BaseGeo`).
  Found by smoke-testing the published wheel.

### Notes

- `getFT` is jittable and differentiable when geometry/`meshing` are static (only excitation/pose
  traced): `jax.jit(jax.grad(loss))` over a `getFT`-based loss compiles once.

## 2.0.0 - 2026-07-13

A major refactor for legibility, differentiability, and a new force/torque feature. The public
object and functional APIs are unchanged; behaviour is verified against the same magpylib parity
suite.

### Added

- **`getFT` — autodiff force and torque.** Magnetic force on a magnet is computed as `∇(m·B)`
  via `jax.jacfwd` (exact, no finite-difference step `eps`), and current force as `(I dL)×B`.
  `getFT` is itself differentiable. Supports `Dipole`, `Sphere`, `Cuboid`, `Circle` and
  `Polyline` targets; validated against magpylib and an analytic dipole–dipole reference.
- **`show()` — matplotlib 3-D visualization.** `mpj.show(*objects)` and every object's `.show()`
  render magnets (shaded bodies + polarization arrow), currents (loops/lines + direction arrow),
  dipoles, triangular meshes, sensors (with pixels and local axes), collections, and motion paths.
- **Hardened gradients at singularities** via `custom_jvp` safe ops: fields near a source's
  singular set now yield finite gradients instead of NaN, with the primal field unchanged.
- README showcase with generated figures (`scripts/make_figures.py`): field map, inverse-design
  loss, `getFT` force curve, and an honest CPU benchmark (jitted forward and field+gradient).

### Changed

- **x64 enabled on import.** `import magpylib_jax` now sets `jax_enable_x64`, so fields and
  gradients are float64 by default (previously non-pytest users silently ran float32). Import is
  warning-free. Set `jax.config.update("jax_enable_x64", False)` afterwards to opt out.
- **Numerical core restructured for legibility.** `functional.py` (2418 LOC) split into a `fields/`
  package (`api`, `prepare`, `engine`, `eager`, `cache`) with a thin re-export facade;
  `kernels.py` + `kernels_extended.py` (2567 LOC) split into a `core/kernels/` package with one
  module per source family. Four overlapping field evaluators consolidated to a documented JIT
  engine plus an eager reference path.
- The 52 duplicated `getB/getH/getJ/getM` methods across source classes collapsed into one
  `BaseSource` mixin driven by a per-class `_field_kwargs()`.
- CI simplified: the brittle profiling/HLO-baseline gates moved to the nightly workflow; PR CI runs
  lint, types, the full test suite with coverage (≥95%), docs, and a benchmark check.

### Removed

- Dead code (`vgetB`, `vgetH`, and unused observer/mesh helpers) and duplicated kernels.

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
