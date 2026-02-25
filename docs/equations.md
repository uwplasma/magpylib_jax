# Equation Models

This project uses closed-form analytical field expressions (no mesh-based PDE solve in core kernels).

## Examples of implemented models

- Dipole field:
  - \(\mathbf{H}(\mathbf{r}) = \frac{1}{4\pi}\left(\frac{3(\mathbf{m}\cdot\mathbf{r})\mathbf{r}}{\|\mathbf{r}\|^5}-\frac{\mathbf{m}}{\|\mathbf{r}\|^3}\right)\)
  - \(\mathbf{B}=\mu_0\mathbf{H}\)
- Sphere (uniform polarization):
  - Inside: \(\mathbf{B}=\frac{2}{3}\mathbf{J}\)
  - Outside: equivalent dipole expression with moment scaling by sphere radius.
- Line current segment/polyline:
  - Biot-Savart closed forms per segment, summed over segments.
- Triangle sheet / strip:
  - Analytical current-sheet field using an elementar triangle in a local frame.
  - Each triangle is translated + rotated into \((0,0,0)\), \((u_1,0,0)\), \((u_2,v_2,0)\).
  - Piecewise formulas handle off-sheet, in-plane, and edge cases with \(\arctan\)/\(\operatorname{atanh}\)/log terms.
- Triangle and tetrahedron:
  - Surface-charge / solid-angle based formulations.

For exact implementation details, see:
- `src/magpylib_jax/core/kernels.py`
- `src/magpylib_jax/core/kernels_extended.py`
