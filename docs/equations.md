# Equation Models

This project uses analytical field expressions and geometric reductions rather than a generic PDE solve. The kernels are organized by source family, but they share a common field convention and local-coordinate reduction strategy.

## Field conventions

The code distinguishes four quantities:

- magnetic flux density: $\mathbf{B}$
- magnetic field strength: $\mathbf{H}$
- polarization: $\mathbf{J}$
- magnetization: $\mathbf{M}$

For linear vacuum conversions used throughout the library:

$$
\mathbf{B} = \mu_0 \mathbf{H}, \qquad \mathbf{J} = \mu_0 \mathbf{M}.
$$

For permanent magnets, the implementation usually computes a field representation in terms of polarization and then converts to the requested output quantity.

## Common reduction pattern

Most kernels follow the same three-step structure:

1. Transform observers into a local frame attached to the source.
2. Evaluate a closed-form field model in that frame.
3. Rotate the result back to global coordinates.

If the source has a motion path, the reduction is repeated path-wise with Magpylib-compatible broadcasting and squeeze semantics.

## Dipole model

For dipole moment $\mathbf{m}$ and observation vector $\mathbf{r}$:

$$
\mathbf{H}(\mathbf{r}) = \frac{1}{4\pi}
\left(
\frac{3(\mathbf{m}\cdot\mathbf{r})\mathbf{r}}{\lVert \mathbf{r} \rVert^5}
-
\frac{\mathbf{m}}{\lVert \mathbf{r} \rVert^3}
\right),
$$

and then

$$
\mathbf{B}(\mathbf{r}) = \mu_0 \mathbf{H}(\mathbf{r}).
$$

This is the far-field reference used explicitly for `misc.Dipole` and implicitly as a limiting comparison for other source families.

## Sphere with uniform polarization

A uniformly polarized sphere has a particularly simple decomposition:

- inside the sphere, the magnetic field is uniform,
- outside the sphere, the field is equivalent to that of a dipole with moment proportional to the sphere volume.

With polarization $\mathbf{J}$:

$$
\mathbf{B}_{\text{inside}} = \frac{2}{3}\mathbf{J},
$$

while outside the sphere the implementation uses the dipole-equivalent form.

## Current line and polyline models

For line currents the code uses Biot-Savart closed forms on each segment and sums them.

For a current element,

$$
\mathrm{d}\mathbf{B} = \frac{\mu_0 I}{4\pi}
\frac{\mathrm{d}\mathbf{l} \times \mathbf{r}}{\lVert \mathbf{r} \rVert^3}.
$$

The polyline implementation reduces the source to a set of finite segments and accumulates the analytical segment fields. This gives a path that is exact for the piecewise-linear geometry and differentiable away from the segment singular set.

## Circular loop model

The circle kernel uses the standard cylindrical-coordinate analytical reduction with complete elliptic integrals. The implementation relies on a robust elliptic helper in [`core/elliptic.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/elliptic.py) and handles the axis separately to avoid unstable generic formulas.

The important implementation point is not the textbook formula itself, but the combination of:

- cylindrical reduction,
- axis masks,
- x64-friendly evaluation,
- JAX-compatible branching.

## Cuboid, triangle, tetrahedron, and mesh models

Permanent-magnet kernels are built from surface-charge or solid-angle formulations.

## Triangle

The triangle kernel is the basic surface element. For a uniformly polarized planar triangle, the field can be written in terms of edge contributions and a solid-angle term. That representation is reused directly or indirectly by several higher-level sources.

## Tetrahedron

A tetrahedron is reduced to a sum over oriented triangle faces. The implementation precomputes shared geometry terms per face and accumulates them in JAX, which improves both memory behavior and compilation reuse.

## Triangular mesh

A `TriangularMesh` is reduced to its oriented faces. The inside/outside behavior, open-mesh handling, and face orientation semantics are treated as compatibility-sensitive behavior and are validated against upstream Magpylib.

## TriangleSheet and TriangleStrip

Current-sheet kernels map each triangle into a canonical local frame with vertices of the form

$$
(0,0,0), \quad (u_1,0,0), \quad (u_2, v_2, 0),
$$

then apply the analytical sheet-field expressions triangle by triangle. Edge, in-plane, and off-plane limits are handled separately because the physically correct limit is piecewise.

## Cylinder and CylinderSegment models

The cylinder family combines analytical radial/axial reductions with explicit boundary handling.

## Cylinder

The full cylinder kernel is treated as a rotationally symmetric magnet and uses cylindrical reduction with elliptic-integral building blocks.

## CylinderSegment

`CylinderSegment` is more difficult because symmetry is reduced: the source has radial, axial, and azimuthal boundaries. The implementation therefore uses a face-based decomposition with precomputed geometry and specialized JIT entrypoints for hotspot profiling.

## Derivation strategy in code

The codebase does not try to reproduce textbook derivations line by line inside docstrings. Instead, it encodes derivation structure as reusable geometry reductions:

- local-frame transforms,
- oriented face decompositions,
- edge and solid-angle terms,
- stable branch masks for singular neighborhoods.

That structure is easier to validate and profile than directly transcribing symbolic expressions into monolithic formulas.

## Where to read the implementation

- Core analytical kernels: [`src/magpylib_jax/core/kernels.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels.py)
- Extended and mesh/segment kernels: [`src/magpylib_jax/core/kernels_extended.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels_extended.py)
- Geometric coordinate helpers: [`src/magpylib_jax/core/geometry.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/geometry.py)
- High-level field assembly: [`src/magpylib_jax/functional.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/functional.py)
