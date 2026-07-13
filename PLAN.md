# magpylib_jax — Refactor & Roadmap Plan

**Status of this document:** authoritative plan (supersedes the old `MIGRATION_PLAN.md`).
**Last revised:** 2026-07-13.
**Baseline at revision:** 284 tests passing, **92.01 %** coverage, full suite ~6.5 min, magpylib parity target **5.2.3**, package version 1.0.1.

---

## 1. Mission

A JAX-native, **end-to-end differentiable** reimplementation of [magpylib](https://github.com/magpylib/magpylib) that is:

- **Correct & validated** — numerical parity with upstream magpylib 5.x on every source family, verified in CI.
- **Differentiable** — clean, finite gradients through `jax.grad`/`jacrev`/`jacfwd` w.r.t. geometry, pose, and excitation, *including at/near singularities* (no NaN-gradient traps).
- **Fast & lean on memory** — competitive with (and where possible beating) magpylib on batched evaluation; usable inside `jit`/`vmap` optimization loops.
- **Slim & legible** — fewer files, fewer lines, intuitive names, one obvious way to do each thing. A newcomer can read the source and understand it.
- **Fully documented** — equations with derivations, tutorials, worked examples, design decisions, model spec, and citations.
- **Research-grade** — reproducible, tested, packaged, released.

## 2. Honest current state (from the 2026-07-13 audit)

**What works:** all 12 magpylib source families are implemented with parity + differentiability tests; object API, `Collection`, `Sensor`, path/orientation motion, and the functional `getB/H/J/M` all mirror magpylib. CI runs lint, types, sharded tests, docs, benchmark and profiling regression, and PyPI release.

**What is wrong / debt (the reason for this refactor):**

| # | Problem | Evidence | Impact |
|---|---|---|---|
| D1 | **x64 is never enabled in-package** — every non-pytest user silently runs float32, and import emits a `float64` `UserWarning`. | `kernels_extended.py:799,1479` build module-level `float64` arrays; only `tests/conftest.py` sets x64. | **Correctness** bug + bad first impression. |
| D2 | **Four overlapping field-evaluation engines** in `functional.py` (2418 LOC). | `_compute_field_jit`, `_compute_field_legacy`, `_evaluate_core_field`, `_get_field_from_type`. | Bloat, drift risk, hard to reason about. |
| D3 | **52 duplicated `getB/H/J/M` methods** (13 classes × 4), each identical but for a type string. | e.g. `magnet/cuboid.py:66-156`. | ~700 LOC of copy-paste. |
| D4 | **32-argument vmapped `per_source`** switch; adding a source touches 4 places. | `functional.py:1198,1465-1531`. | Change-averse, unreadable. |
| D5 | **No `custom_jvp`/`stop_gradient` anywhere**; singular kernels use output-only `jnp.where`, `nan_to_num`, `clip`-based "safe" ops. | circle ring, cuboid edges, dipole `r=0`, current sheet. | Silent **NaN / zero / exploding gradients** at singularities. |
| D6 | **Dead & duplicated code.** | `vgetB/vgetH`, `_extract_observers`, `_reshape_observer_field`; `cel` implemented twice (`elliptic.py` + `kernels.py`), circle bfield twice, two `_broadcast_vec*`. | Bloat, low coverage. |
| D7 | **Mutable cache-tracking layer** on `BaseGeo` (`__setattr__` version bumping) to serve two module-global LRU prep caches + four unbounded jit caches. | `base.py:528-604`, `functional.py:114-119`, `kernels_extended.py:24-27`. | Global mutable state, subtle correctness surface, per-attr overhead. |
| D8 | **mypy disabled** on the 5 messiest files. | `pyproject.toml:85-93`. | Types unchecked where most needed. |
| D9 | **Brittle CI** — many recent commits only "fix/stabilize/relax" profiling & HLO-baseline gates. | git log. | Maintenance churn, red-herring failures. |
| D10 | **README has no plots / showcase**; docs lack derivations & citations depth. | `README.md`. | Undersells the project vs the vmec_jax bar. |
| D11 | **Parity gap: no `getFT`** (force & torque). magpylib 5.2 computes it by finite differences (`eps`); JAX autodiff does it exactly. | upstream `field_FT.py`. | Missing feature *and* a flagship differentiability demo. |

## 2b. Progress log

- ✅ Revised plan (this file) replaces `MIGRATION_PLAN.md`.
- ✅ Golden-value regression net (`tests/test_golden_regression.py`) — fast behaviour gate.
- ✅ D1 x64 enabled in-package; import warning-free; float64 by default.
- ✅ D3 field mixin — 52 `getB/H/J/M` methods → one `BaseSource` mixin (−~1000 LOC).
- ✅ D6 (kernels) `kernels.py` + `kernels_extended.py` → `core/kernels/` package (16 family
  modules + `_common`/`_safe`/`_raycast`); circle/broadcast de-duped; `cel` kept specialized.
- ✅ D2/D4 `functional.py` → `fields/` package (`api/prepare/cache/engine/eager`) + facade.
- ✅ D11 `getFT` — autodiff force/torque (`fields/force.py`), exact ∇B via `jacfwd` (no eps),
  dipole/sphere/cuboid/circle/polyline targets, analytic dipole parity < 1e-8.
- ✅ D8 mypy — green (kernels + source classes checked; dynamic `fields/*` engine ignored,
  as the old `functional.py` was).
- ✅ 95% coverage — **95.38%**, 428 tests + full magpylib parity; `--cov-fail-under=95`.
- ✅ D9 CI — profiling/HLO gates → nightly; four directory-partitioned test shards + coverage
  combine; lint/types/docs/benchmark.
- ✅ docs/README — showcase README with figures, refreshed API/architecture/numerics, 2.0.0 changelog.
- ✅ Released as **2.0.0** (version bumped; GitHub release/PyPI publish left to rogeriojorge).
- ✅ **D5** safe-op `custom_jvp` — singular kernels (dipole/cuboid; circle/sheet via shared
  helpers) now give finite gradients on their singular set; primal bit-stable.
- ✅ **D7** removed the mutable `__setattr__` cache-tracking layer and the prep caches.
- ✅ **`show()`** — matplotlib 3-D visualization of all sources/sensors/collections/paths;
  display is no longer out of scope.

**Follow-ups noted:** `getFT` compiles a fresh `jacfwd` graph per call (like the object-API
`getB`, it is differentiable but not itself jit-cached); fine for correctness/grad, a perf tuning
target. `engine.py`/`prepare.py` remain large (single big functions); the 32-arg `per_source`
(D4) could later become a pytree.

**Architecture note (pragmatic deviation):** the public namespaces `magnet/`, `current/`,
`misc/` are kept (they already map cleanly to magpylib and are intuitive) rather than folded
into `objects/`. Kernels live in `core/kernels/`. The field engine is split into `fields/`
with `functional.py` retained as a thin re-export facade for import stability. The four
overlapping evaluators are consolidated to **two documented paths**: one vectorized JIT engine
(`fields/engine.py`, the default) and one eager reference (`fields/eager.py`) for output modes
outside JIT (callable/unsupported `pixel_agg`, non-uniform pixel grids, dataframe, 4-D meshes,
pairwise observers). Dead evaluators removed.

## 3. Target architecture (slimmer, one obvious path)

```
magpylib_jax/
  __init__.py            # public API; enables x64; version
  config.py              # MU0, dtype policy, x64 setup           (was constants.py + _types.py)
  fields/
    __init__.py          # getB/getH/getJ/getM, getFT            (public functional API)
    engine.py            # ONE jit evaluation engine (SoA + per-type pytree, no legacy fork)
    prepare.py           # source/sensor/observer preparation, path padding
    convert.py           # B<->H, J, M relations + mu0 (single source of truth)
  kernels/               # pure, origin-local analytic field kernels, one file per family
    __init__.py
    elliptic.py          # Bulirsch cel + K/E/Pi (the ONLY copy)
    dipole.py  circle.py  polyline.py
    cuboid.py  cylinder.py  cylinder_segment.py  sphere.py
    triangle.py  triangle_sheet.py  triangle_strip.py
    tetrahedron.py  triangular_mesh.py
    _safe.py             # shared custom_jvp'd safe ops (sqrt, norm, atanh, log)
  geometry.py            # frame transforms, cart<->cyl, pose broadcasting
  objects/               # thin object model
    base.py              # BaseGeo/BaseTransform + ONE getB/H/J/M/FT mixin (kills D3)
    magnet.py current.py misc.py   # source classes (small; geometry + excitation only)
    sensor.py  collection.py
    style.py
```

Guiding rules:
- **One engine.** Delete the legacy/`_evaluate_core_field`/`_get_field_from_type` forks (D2). Non-jit-able cases (`output="dataframe"`, 4-D pixel meshes) become thin post/pre-processing around the same engine, not parallel implementations.
- **Per-type pytree, not 32 args** (D4): the engine vmaps one `dict`/dataclass of per-source params; adding a source touches one registry entry.
- **One B/H/J/M conversion** in `fields/convert.py` (D6): kernels return their native field; conversion + `mu0` live in exactly one place.
- **One `cel`** in `kernels/elliptic.py` (D6).
- **Safe ops with real gradients** in `kernels/_safe.py` (D5): `custom_jvp`-backed `safe_sqrt`, `safe_norm`, `safe_atanh`, `safe_log`, so singular points return finite, physically-correct (or zero) tangents.
- **No mutable object cache by default** (D7): rely on `jax.jit`'s own compilation cache keyed on shapes/dtypes. If a prep cache is retained for the eager path, it must be optional and not require `__setattr__` interception.

**Non-goals (documented, not built):** interactive 3-D `show()` (matplotlib/plotly/pyvista) and the visual style system beyond what `describe()`/labels need. `matplotlib` is used only for docs/README figures, not runtime.

## 4. Workstreams & phases

Each phase ends green: `ruff`, `mypy`, full `pytest` parity, and (from Phase 2 on) ≥95 % coverage. Commits are incremental and authored by **rogeriojorge** (no Claude co-author trailer).

**Phase 0 — Safety net (before touching source).**
- Snapshot current public behavior in a golden-value regression test (fields for every source at fixed inputs, plus a gradient value), so the refactor is provably behavior-preserving. Keep the full magpylib parity suite as the outer check.

**Phase 1 — Correctness & quick wins (low risk, high value).**
- D1: enable x64 in `__init__.py`; make module constants dtype-safe; kill the import warning. Verify float32 fallback still works if a user forces it off.
- D6: delete dead code (`vgetB/vgetH`, `_extract_observers`, `_reshape_observer_field`); de-duplicate `cel`, circle bfield, broadcast helpers.
- D3: collapse the 52 `getB/H/J/M` methods into one `BaseSource` mixin.

**Phase 2 — Engine unification & file split.**
- D2/D4: one jit engine with per-type pytree params; delete legacy forks; split `functional.py` → `fields/*` and `kernels_extended.py` → `kernels/*` along family seams.
- D7: remove the `__setattr__` cache-tracking layer; simplify or drop prep caches.
- Re-run parity + golden tests after each extraction.

**Phase 3 — Differentiability hardening.**
- D5: introduce `kernels/_safe.py` custom_jvp ops; retrofit circle, cuboid, dipole, triangle, current-sheet kernels. Add gradient-at-singularity tests (finite, matches finite-difference away from the singular set; zero/defined on it).

**Phase 4 — New capability: `getFT`.**
- D11: implement force & torque via autodiff of the interaction energy / field gradient (exact, no `eps`), matching magpylib's `getFT` interface and values (validated against magpylib's FD result within tolerance). Showcase as the flagship differentiability demo.

**Phase 5 — Types, coverage, CI.**
- D8: re-enable mypy on the cleaned files; fix annotations.
- D4-coverage: add physics/numerics/future-proofing tests to reach **≥95 %**; bump `--cov-fail-under` to 95.
- D9: simplify CI — keep lint/types/tests/coverage/docs/benchmark; demote or remove the brittle HLO-baseline & fragile profiling thresholds (or make them non-blocking, nightly-only).

**Phase 6 — Docs & README showcase (vmec_jax bar).**
- D10: rewrite README to *show* the library — quickstart, differences vs magpylib, advantages, and **plots** (field maps, gradient/optimization demo, benchmark bars, force/torque). Refresh docs: equations with derivations & citations, tutorials, examples, design decisions, model spec, API reference.

**Phase 7 — Release.**
- Version bump (behavior change → **2.0.0**), changelog, tag & PyPI release — all as rogeriojorge.

## 5. Definition of done

- [ ] `import magpylib_jax` is warning-free and runs float64 by default.
- [ ] Source LOC materially reduced; no file > ~500 LOC; no dead/duplicate engines.
- [ ] One field engine; one B/H/J/M conversion; one `cel`; one `getB/H/J/M/FT` mixin.
- [ ] Clean finite gradients at/near every documented singularity (tested).
- [ ] `getFT` implemented and validated against magpylib.
- [ ] mypy clean on all of `src/`.
- [ ] ≥95 % coverage; full magpylib parity suite green; benchmark not regressed.
- [ ] README with plots + docs (equations/derivations/citations/tutorials/design).
- [ ] Released as 2.0.0 by rogeriojorge.

## 6. Decisions (confirmed 2026-07-13)

1. **x64 policy** — ✅ **enable globally in-package** (matches magpylib, fixes D1). Kernels stay float64; a user may still force x64 off.
2. **Refactor appetite** — ✅ **full engine-collapse + directory restructure** (this plan). Ships as **2.0.0**.
3. **`getFT`** — ✅ **build now** (Phase 4), validated against magpylib, featured in README.
4. **CI** — ✅ **simplify**: keep lint/types/tests/coverage/docs + a simple benchmark; make HLO-baseline & tight profiling thresholds non-blocking / nightly-only.
