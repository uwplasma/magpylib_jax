# Parity strategy

`magpylib_jax` is a clean-room reimplementation of Magpylib, so "does it agree with Magpylib?" is a
first-class, continuously tested question — not a hope. This page explains what parity means here,
how it is enforced, the drop-in compatibility surface, and the few things that are *intentionally*
different.

## What parity means

For every implemented source family, `magpylib_jax` reproduces upstream Magpylib's field to within
a strict numerical tolerance, and reproduces its object/API behavior (shapes, squeeze semantics,
path handling, pixel aggregation) test-for-test.

Parity is validated in x64 (`float64`), because that is Magpylib's precision. In the default
`float32` mode you get the usual single-precision agreement instead — enable double precision for
bit-level comparison:

```python
import jax
jax.config.update("jax_enable_x64", True)   # required for float64 parity
import magpylib_jax as mpj
```

See [Precision](precision.md) for how the dtype follows your JAX config.

## The per-source workflow

Each source family goes through the same pipeline before it is considered at parity:

1. Port the kernel/object implementation to JAX.
2. Add **parity tests** against upstream Magpylib at random points and profile points.
3. Add **mirrored upstream-file tests** for object behavior.
4. Add **physics** and **differentiability** tests.
5. Gate the behavior in CI with strict parity markers and threshold checks.

Until every step lands, a family stays marked as partial and is not promoted to a hard gate.

## What is covered

All 12 source families are implemented and exercised against Magpylib:

```{list-table}
:header-rows: 1
:widths: 22 78

* - Namespace
  - Families
* - `magnet`
  - `Cuboid`, `Cylinder`, `CylinderSegment`, `Sphere`, `Tetrahedron`, `TriangularMesh`
* - `current`
  - `Circle`, `Polyline`, `TriangleSheet`, `TriangleStrip`
* - `misc`
  - `Dipole`, `Triangle`, `CustomSource`
```

The full field surface — `getB`, `getH`, `getJ`, `getM`, and `getFT` — plus `Collection`, `Sensor`,
motion paths, and pixel aggregation is covered. See [Testing and validation](testing.md) for the
complete taxonomy and how to run each layer.

## Drop-in compatibility surface

Migration from Magpylib is usually a one-line import change:

```python
import magpylib_jax as magpy      # instead of: import magpylib as magpy
```

The following keep the same names and semantics:

- **Sources** — `magpy.magnet.*`, `magpy.current.*`, `magpy.misc.*`.
- **Objects** — `Collection` (grouping and nesting) and `Sensor` (pixel grids, handedness).
- **Fields** — `getB/getH/getJ/getM` as both functions and methods, plus `getFT`.
- **Motion** — `position`, `orientation`, `move`, `rotate`, `rotate_from_*`, and path/edge-padding
  semantics.
- **Functional & low-level** — the high-level `func.*_field` interface and the `core.*` field
  functions (magpylib's capitalized names such as `magnet_cuboid_Bfield` are provided as aliases).
- **Utilities** — `show`, `mu_0`, `defaults`, `SUPPORTED_PLOTTING_BACKENDS`, and the
  `MagpylibBadUserInput` / `MagpylibMissingInput` exceptions.

```{admonition} One difference to keep in mind
:class: note
Fields come back as **JAX arrays**, not NumPy arrays. Wrap a call in `np.asarray(...)` when you need
a NumPy result (for example to hand off to pandas or Matplotlib). Everything else follows Magpylib.
```

## What is intentionally different

A few upstream features are out of scope by design, so their absence is expected, not a parity gap:

```{list-table}
:header-rows: 1
:widths: 34 66

* - Area
  - Difference
* - Display backends
  - Only the **Matplotlib** `show` backend is implemented. Plotly and PyVista are not shipped.
* - `output="dataframe"`
  - Provided as a compatibility convenience that returns pandas objects; it is **not** part of the
    jittable field graph and cannot be traced.
* - `CustomSource.field_func`
  - Takes a single `observers` argument and returns `B` directly, and is evaluated on the raw global
    observers (it does not apply the object pose to the field). Bake pose into the field function if
    you need it.
* - Result dtype
  - Fields are JAX arrays following your JAX precision config, not NumPy `float64` by default.
```

## Where the tests live

- [`tests/parity_gates`](https://github.com/uwplasma/magpylib_jax/tree/main/tests/parity_gates) —
  direct numeric comparison against Magpylib.
- [`tests/upstream_mirror`](https://github.com/uwplasma/magpylib_jax/tree/main/tests/upstream_mirror) —
  mirrored upstream object/API tests.
- The rest of the suite lives under [`tests/`](https://github.com/uwplasma/magpylib_jax/tree/main/tests).
