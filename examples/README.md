# magpylib_jax examples

Runnable scripts that (a) port the portable examples from
[Magpylib's user guide](https://magpylib.readthedocs.io/) — the API is drop-in, just
`import magpylib_jax as magpy` — and (b) showcase what magpylib_jax adds on top of Magpylib:
exact autodiff, `jit`/`vmap`, and differentiable `getFT` force/torque.

Every script is self-contained, uses small inputs (each runs in well under ~2 s), needs no network
or external data files, and prints a concise result. Scripts that plot do so only when run directly
(`python examples/<...>.py`); they force the non-interactive Matplotlib "Agg" backend and save a PNG
into `examples/_output/`. Each script exposes a `main()` that returns its numeric results, which is
what the smoke test in `tests/test_examples.py` calls.

Run one with:

```bash
python examples/differentiable/inverse_design.py
```

## Catalogue

### `basics/` — object API fundamentals (ported)
| script | shows |
|---|---|
| `first_field.py` | The "three lines to a field" basic call, B/H/J/M on a grid, and multi-source/multi-sensor combination indexing. |
| `sources_gallery.py` | One of every supported source family (magnets, currents, misc) and its field at a probe point. |
| `collections_sensors.py` | Grouping windings into a `Collection` (a Helmholtz pair) and reading the field with a `Sensor`. |
| `motion_paths.py` | Absolute paths, relative `move`/`rotate`, merging paths into a spiral, and edge-padding. |
| `magnet_scan.py` | Forward model of a cylinder magnet swept over a fixed sensor (the portable half of "Modeling a Real Magnet"). |
| `custom_source.py` | A magnetic-monopole `CustomSource` and a four-charge quadrupole `Collection`. |

### `shapes/` — building magnets and current sheets (ported)
| script | shows |
|---|---|
| `triangle.py` | A cuboctahedron magnet from `misc.Triangle` faces and the same body as a `TriangularMesh`. |
| `convex_hull.py` | `TriangularMesh.from_ConvexHull` — a pyramid magnet from a point cloud. |
| `superposition.py` | Cut-out by opposing polarizations: two `Cylinder`s reproduce a `CylinderSegment` ring. |
| `current_sheet.py` | `TriangleStrip` (a Möbius ribbon) and `TriangleSheet` (a meshed surface current). |

### `force/` — force & torque with `getFT` (ported)
| script | shows |
|---|---|
| `force_intro.py` | The minimal `getFT(cube, loop)` call and the effect of the `pivot` point on torque. |
| `holding_force.py` | Holding force of a magnet on a soft-magnetic plate via the method of images. |
| `dipole_dipole.py` | `getFT` convergence of a meshed current loop against the exact dipole-dipole force (reciprocity). |

### `visualization/` — `show()` and Matplotlib (ported)
| script | shows |
|---|---|
| `show_scene.py` | `Collection.show()` of a magnet, current loop, dipole, and sensor (saved headless). |
| `streamplot_field.py` | Matplotlib `streamplot` of a cuboid magnet's B-field slice. |
| `magnet_show.py` | Rendering several magnet types together with `show()`. |

### `applications/` — assemblies (ported)
| script | shows |
|---|---|
| `coil.py` | Air coils from `Circle` loops and a `Polyline` spiral, combined into a Helmholtz pair with a homogeneity map. |
| `halbach.py` | A discrete Halbach cylinder from rotated `Cuboid`s and its field slice. |

### `differentiable/` — the magpylib_jax value-add (autodiff / jit / vmap)
| script | shows |
|---|---|
| `inverse_design.py` | Recover a magnet's polarization from field samples with `jax.grad` gradient descent. |
| `optimize_geometry.py` | Tune a magnet dimension to hit a target field, using `jax.jit(jax.grad(...))`. |
| `jit_vmap_batching.py` | `jax.vmap` a field over many source parameters / observers and `jax.jit` it; the `block_until_ready` timing pattern. |
| `getft_optimization.py` | Solve for a force equilibrium (levitation height where `F_z = weight`) via `jax.grad` of a `getFT` objective. |
| `field_jacobian.py` | `jax.jacrev`/`jacfwd` for the field-gradient tensor `dB/dr` and the source sensitivity `dB/d(polarization)`. |

## Magpylib examples ported vs. skipped

**Ported** (drop-in, adapted to small/headless inputs): field computation, modeling magnets, working
with collections, custom source, working with paths, triangular meshes, convex hull, superposition,
current sheets, force & torque basics, holding force, floating/dipole force (as a `getFT`
convergence study), Matplotlib streamplot, magnet colors/`show`, coils, and Halbach arrays.

**Not ported** (out of scope for this JAX port), with reasons:

| Magpylib example | reason |
|---|---|
| `examples_shapes_pyvista`, `examples_vis_pv_streamlines` | PyVista streamlines / meshing backend not shipped. |
| `examples_vis_animations`, `examples_tutorial_paths` (Plotly animation parts) | Plotly animation backend not shipped; only the static Matplotlib `show()` is. |
| `examples_vis_subplots`, `examples_vis_vectorfield` | Rely on the Plotly `show_context`/subplot and pixel-field styling not shipped. |
| `examples_shapes_cad` | CAD/STL import (`pyvista`/`trimesh` file loading) out of scope. |
| `examples_misc_field_interpolation` | Field interpolation helper not part of this port. |
| `examples_misc_image_method`, `examples_misc_inhom` | Image-method and inhomogeneous-material (material-response) workflows out of scope. |
| `examples_misc_equivalent`, `examples_misc_compound` | Depend on advanced graphics/compound tooling beyond the core field API. |
| `examples_app_pcb`, `examples_app_end_of_shaft`, `examples_app_scales` | Load external datasets / heavy Plotly dashboards; not portable as fast offline scripts. |
