# References

The analytic field models in magpylib_jax come from the magnetostatics literature. This page
collects the primary sources; each is cited from the relevant kernel in
[Equation models](equations.md) and, where useful, from the source code.

## Field models

- **Magpylib (the reference library and conventions).**
  M. Ortner and L. G. Coliado Bandeira, *Magpylib: A free Python package for magnetic field
  computation*, **SoftwareX** 11, 100466 (2020).
  [doi:10.1016/j.softx.2020.100466](https://doi.org/10.1016/j.softx.2020.100466)

- **Cuboid / rectangular prism magnet.**
  R. Ravaud and G. Lemarquand — analytic prism fields; see also
  J. M. Camacho and V. Sosa, *Alternative method to calculate the magnetic field of permanent
  magnets with azimuthal symmetry*, **Rev. Mex. Fís. E** 59(1), 8–17 (2013).

- **Cylinder (axial + diametral).**
  N. Derby and S. Olbert, *Cylindrical magnets and ideal solenoids*,
  **Am. J. Phys.** 78(3), 229–235 (2010).
  [doi:10.1119/1.3256157](https://doi.org/10.1119/1.3256157)

- **Cylinder segment / general cylinder tile.**
  P. Slanovc, M. Ortner, M. Moridi, C. Abert, and D. Suess, *Full analytical solution for the
  magnetic field of uniformly magnetized cylinder tiles*, **J. Magn. Magn. Mater.** 559, 169482
  (2022). [doi:10.1016/j.jmmm.2022.169482](https://doi.org/10.1016/j.jmmm.2022.169482)

- **Current loop (circle).**
  J. Simpson, J. Lane, C. Immer, and R. Youngquist, *Simple analytic expressions for the magnetic
  field of a circular current loop*, NASA Technical Reports Server, 20010038494 (2001).

- **Triangle surface charge / polyhedral magnets.**
  D. Guptasarma and B. Singh, *New scheme for computing the magnetic field resulting from a
  uniformly magnetized arbitrary polyhedron*, **Geophysics** 64(1), 70–74 (1999).
  [doi:10.1190/1.1444531](https://doi.org/10.1190/1.1444531);
  M. Fabbri, *Magnetic flux density and vector potential of uniform polyhedral sources*,
  **IEEE Trans. Magn.** 44(1), 32–36 (2008).
  [doi:10.1109/TMAG.2007.908698](https://doi.org/10.1109/TMAG.2007.908698)

## Numerical methods

- **Complete elliptic integrals (Bulirsch `cel`).**
  R. Bulirsch, *Numerical calculation of elliptic integrals and elliptic functions*,
  **Numer. Math.** 7(1), 78–90 (1965).
  [doi:10.1007/BF01397975](https://doi.org/10.1007/BF01397975)

- **Automatic differentiation / the JAX system.**
  J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula,
  A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang, *JAX: composable transformations of
  Python+NumPy programs* (2018). <https://github.com/jax-ml/jax>

## Citing magpylib_jax

If magpylib_jax supports your research, please cite this repository together with the Magpylib
paper above. A `CITATION` entry will accompany a tagged release; until then, cite the GitHub
repository and the release version you used.
