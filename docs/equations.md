# Equation Models

This project uses analytical field expressions and geometric reductions rather than a generic PDE solve. The kernels are organized by source family, but they share a common field convention and local-coordinate reduction strategy.

The closed-form models follow the same analytical literature that Magpylib is built on. Each source section below cites the specific paper it derives from, and the full bibliography is collected in the References section at the end of the page. The overall library design mirrors Magpylib (Ortner & Coliado Bandeira, 2020).

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

This is the far-field reference used explicitly for `misc.Dipole` and implicitly as a limiting comparison for other source families. It is the standard point-dipole expression and is the same limiting form used by Magpylib (Ortner & Coliado Bandeira, 2020).

## Sphere with uniform polarization

A uniformly polarized sphere has a particularly simple decomposition:

- inside the sphere, the magnetic field is uniform,
- outside the sphere, the field is equivalent to that of a dipole with moment proportional to the sphere volume.

With polarization $\mathbf{J}$:

$$
\mathbf{B}_{\text{inside}} = \frac{2}{3}\mathbf{J},
$$

while outside the sphere the implementation uses the dipole-equivalent form. Both limits are classical results and are the closed forms used for the sphere in Magpylib (Ortner & Coliado Bandeira, 2020).

## Current line and polyline models

For line currents the code uses Biot-Savart closed forms on each segment and sums them.

For a current element,

$$
\mathrm{d}\mathbf{B} = \frac{\mu_0 I}{4\pi}
\frac{\mathrm{d}\mathbf{l} \times \mathbf{r}}{\lVert \mathbf{r} \rVert^3}.
$$

The polyline implementation reduces the source to a set of finite segments and accumulates the analytical segment fields. This gives a path that is exact for the piecewise-linear geometry and differentiable away from the segment singular set. The finite-segment closed form is the standard Biot-Savart line result used by Magpylib (Ortner & Coliado Bandeira, 2020).

## Circular loop model

The circle kernel uses the standard cylindrical-coordinate analytical reduction with complete elliptic integrals. The off-axis field of a circular current loop is given exactly by Simpson, Lane, Immer & Youngquist (2001) in terms of complete elliptic integrals. The implementation relies on a robust elliptic helper in [`core/elliptic.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/elliptic.py) that evaluates the integrals with the Bulirsch (1965) `cel` algorithm, and handles the axis separately to avoid unstable generic formulas.

The important implementation point is not the textbook formula itself, but the combination of:

- cylindrical reduction,
- axis masks,
- x64-friendly evaluation,
- JAX-compatible branching.

## Cuboid, triangle, tetrahedron, and mesh models

Permanent-magnet kernels are built from surface-charge or solid-angle formulations. The homogeneously magnetized cuboid uses the closed-form prism field of Camacho & Sosa (2013); the polyhedral surface-charge treatment underlying the triangle, tetrahedron, and mesh kernels follows Guptasarma & Singh (1999) and Fabbri (2008).

## Triangle

The triangle kernel is the basic surface element. For a uniformly polarized planar triangle, the field can be written in terms of edge contributions and a solid-angle term, following the polyhedron surface-charge formulation of Guptasarma & Singh (1999) and Fabbri (2008). That representation is reused directly or indirectly by several higher-level sources.

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

The full cylinder kernel is treated as a rotationally symmetric magnet and uses cylindrical reduction with elliptic-integral building blocks, following the axially and diametrally magnetized bar-magnet solution of Derby & Olbert (2010).

## CylinderSegment

`CylinderSegment` is more difficult because symmetry is reduced: the source has radial, axial, and azimuthal boundaries. The general cylinder-tile field used here is the full analytical solution of Slanovc, Ortner, Moridi, Abert & Suess (2022), from which full cylinders, rings, and ring segments follow as special cases. The implementation therefore uses a face-based decomposition with precomputed geometry and specialized JIT entrypoints for hotspot profiling.

## Derivation strategy in code

The codebase does not try to reproduce textbook derivations line by line inside docstrings. Instead, it encodes derivation structure as reusable geometry reductions:

- local-frame transforms,
- oriented face decompositions,
- edge and solid-angle terms,
- stable branch masks for singular neighborhoods.

That structure is easier to validate and profile than directly transcribing symbolic expressions into monolithic formulas.

## Where to read the implementation

- Core analytical kernels: [`src/magpylib_jax/core/kernels.py`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/core/kernels)
- Extended and mesh/segment kernels: [`src/magpylib_jax/core/kernels/`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/core/kernels)
- Geometric coordinate helpers: [`src/magpylib_jax/core/geometry.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/geometry.py)
- High-level field assembly: [`src/magpylib_jax/functional.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/functional.py)

## Force and torque

The force and torque on a magnetic target are derived from the same field models. For a magnetic moment $\mathbf{m}$ placed in an external flux density $\mathbf{B}$, the potential energy is

$$
U = -\,\mathbf{m}\cdot\mathbf{B},
$$

so the force is the negative gradient of the energy. Because $\mathbf{m}$ is fixed, this is equivalent to a gradient acting on the field,

$$
\mathbf{F} = -\nabla U = \nabla(\mathbf{m}\cdot\mathbf{B}) = (\mathbf{m}\cdot\nabla)\,\mathbf{B},
$$

where the last equality uses $\nabla\times\mathbf{B}=\mathbf{0}$ in the source-free region around the target. The torque about the target reference point is

$$
\mathbf{T} = \mathbf{m}\times\mathbf{B} + \mathbf{r}\times\mathbf{F},
$$

with $\mathbf{m}\times\mathbf{B}$ the intrinsic aligning torque and $\mathbf{r}\times\mathbf{F}$ the lever-arm term about the chosen `pivot` (here $\mathbf{r}$ is the offset of the acting cell from the pivot). For a distributed magnet the cell contributions are summed; for a current target the Laplace force $\mathbf{F}=\oint I\,\mathrm{d}\boldsymbol{\ell}\times\mathbf{B}$ replaces the moment gradient.

The key implementation difference from Magpylib (Ortner & Coliado Bandeira, 2020) is that `getFT` evaluates $\nabla(\mathbf{m}\cdot\mathbf{B})$ with `jax.jacfwd` rather than a finite-difference stencil, so the result is exact and free of a step-size parameter. The worked example is in [Force and torque (getFT)](examples/force_torque.md).

## References

Full citations for the analytical models above. Author-year mentions in the text refer to this list.

- **Ortner, M., & Coliado Bandeira, L. G. (2020).** Magpylib: A free Python package for magnetic field computation. *SoftwareX*, 11, 100466. [doi:10.1016/j.softx.2020.100466](https://doi.org/10.1016/j.softx.2020.100466)
- **Camacho, J. M., & Sosa, V. (2013).** Alternative method to calculate the magnetic field of permanent magnets with azimuthal symmetry. *Revista Mexicana de Física E*, 59(1), 8–17. [Open access (SciELO)](https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1870-35422013000100002)
- **Derby, N., & Olbert, S. (2010).** Cylindrical magnets and ideal solenoids. *American Journal of Physics*, 78(3), 229–235. [doi:10.1119/1.3256157](https://doi.org/10.1119/1.3256157)
- **Slanovc, F., Ortner, M., Moridi, M., Abert, C., & Suess, D. (2022).** Full analytical solution for the magnetic field of uniformly magnetized cylinder tiles. *Journal of Magnetism and Magnetic Materials*, 559, 169482. [doi:10.1016/j.jmmm.2022.169482](https://doi.org/10.1016/j.jmmm.2022.169482)
- **Simpson, J. C., Lane, J. E., Immer, C. D., & Youngquist, R. C. (2001).** Simple analytic expressions for the magnetic field of a circular current loop. NASA Technical Reports Server, document 20010038494. [NTRS record](https://ntrs.nasa.gov/citations/20010038494)
- **Guptasarma, D., & Singh, B. (1999).** New scheme for computing the magnetic field resulting from a uniformly magnetized arbitrary polyhedron. *Geophysics*, 64(1), 70–74. [doi:10.1190/1.1444531](https://doi.org/10.1190/1.1444531)
- **Fabbri, M. (2008).** Magnetic flux density and vector potential of uniform polyhedral sources. *IEEE Transactions on Magnetics*, 44(1), 32–36. [doi:10.1109/TMAG.2007.908698](https://doi.org/10.1109/TMAG.2007.908698)
- **Bulirsch, R. (1965).** Numerical calculation of elliptic integrals and elliptic functions. *Numerische Mathematik*, 7(1), 78–90. [doi:10.1007/BF01397975](https://doi.org/10.1007/BF01397975)
