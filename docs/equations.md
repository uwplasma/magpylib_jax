# Equation models

This is the physics reference for magpylib_jax. Every source uses an *analytical*
field expression or an exact geometric reduction of one — there is no PDE solve and
no meshed finite-element step (except where a solid is deliberately discretised into
analytic surface elements, noted below). The closed forms follow the same
magnetostatics literature as Magpylib ([Ortner & Coliado Bandeira, 2020](https://doi.org/10.1016/j.softx.2020.100466)),
collected in [References](references.md).

Each kernel is a pure JAX function living in
[`src/magpylib_jax/core/kernels/`](https://github.com/uwplasma/magpylib_jax/tree/main/src/magpylib_jax/core/kernels).
The subsections below link the exact file, state the geometry parameters and the
governing expression, name the numerical method, and mark the **singular set** where
the field is infinite or undefined.

## Conventions and units

magpylib_jax is SI throughout and distinguishes four magnetic quantities:

- magnetic flux density $\mathbf{B}$ in tesla (T),
- magnetic field strength $\mathbf{H}$ in ampere per metre (A/m),
- magnetic polarization $\mathbf{J}$ in tesla (T),
- magnetization $\mathbf{M}$ in ampere per metre (A/m).

In the linear vacuum regime used everywhere in the library,

$$
\mathbf{B} = \mu_0 \mathbf{H} + \mathbf{J}, \qquad \mathbf{J} = \mu_0 \mathbf{M},
$$

with the vacuum permeability $\mu_0 = 4\pi\times 10^{-7}\ \mathrm{T\,m/A}$
(`magpylib_jax.constants.MU0`). Outside magnetic material $\mathbf{J}=\mathbf{0}$ and
the two field quantities collapse to $\mathbf{B}=\mu_0\mathbf{H}$.

:::{note}
Permanent magnets are parameterised by their **polarization** $\mathbf{J}$ (the
Magpylib convention), not by $\mathbf{M}$. A source of polarization $\mathbf{J}$ and a
current source share the same $\mathbf{H}$ field; they differ only by the material term
$\mathbf{J}$ that is added inside the magnet body.
:::

## Field types and the B/H/J/M dispatch

Each family exposes `getB` / `getH` / `getJ` / `getM`. Internally every kernel has one
*native* quantity and derives the rest algebraically, so the four public outputs are
always mutually consistent.

```{list-table}
:header-rows: 1
:widths: 22 16 62

* - Family
  - Native
  - Derivation of the other quantities
* - Magnets (cuboid, cylinder, sphere, triangle, tetrahedron, mesh, cylinder segment)
  - $\mathbf{B}$
  - $\mathbf{J}=\mathbf{J}_0$ inside the body and $\mathbf{0}$ outside; $\mathbf{H}=(\mathbf{B}-\mathbf{J})/\mu_0$; $\mathbf{M}=\mathbf{J}/\mu_0$.
* - Currents (circle, polyline, triangle sheet, triangle strip) and the dipole
  - $\mathbf{H}$
  - $\mathbf{B}=\mu_0\mathbf{H}$; $\mathbf{J}=\mathbf{M}=\mathbf{0}$ (no material).
```

The material term makes the inside/outside distinction load-bearing: for a magnet,
$\mathbf{B}$ is continuous across a face while $\mathbf{H}$ jumps by $-\mathbf{J}/\mu_0$,
and $\mathbf{J}$ is a hard indicator of the body interior. Kernels therefore carry an
`in_out` flag (`"auto"`, `"inside"`, `"outside"`); `"auto"` performs the geometric
point-in-solid test, while the explicit modes skip it when the caller already knows the
region (see [`_common.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/_common.py)).

## Per-source closed-form models

### Dipole

[`dipole.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/dipole.py)

**Geometry** — a point moment $\mathbf{m}$ (A·m²) at the origin, observed at $\mathbf{r}$.

**Model** — the textbook point dipole,

$$
\mathbf{H}(\mathbf{r}) = \frac{1}{4\pi}
\left(
\frac{3(\mathbf{m}\cdot\mathbf{r})\,\mathbf{r}}{\lVert\mathbf{r}\rVert^{5}}
-\frac{\mathbf{m}}{\lVert\mathbf{r}\rVert^{3}}
\right),
\qquad \mathbf{B}=\mu_0\mathbf{H}.
$$

**Method** — direct evaluation; negative powers are fed a $\mathbf{r}$ kept strictly
positive by masking, and the physical divergence at the origin is restored by an
explicit overwrite whose gradient is frozen with `stop_gradient`.

**Singular set** — the source point $\mathbf{r}=\mathbf{0}$.

**Reference** — the far-field limit of every magnet family; standard form used by
[Ortner & Coliado Bandeira (2020)](https://doi.org/10.1016/j.softx.2020.100466).

### Sphere

[`sphere.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/sphere.py)

**Geometry** — a uniformly polarized sphere of diameter $d$ (radius $R=d/2$),
polarization $\mathbf{J}$, centred at the origin.

**Model** — exact and piecewise. Inside the sphere the flux density is uniform,

$$
\mathbf{B}_{\text{in}} = \tfrac{2}{3}\,\mathbf{J},
$$

and outside it is exactly a dipole with moment $\mathbf{m}=\tfrac{4}{3}\pi R^{3}\mathbf{M}$,

$$
\mathbf{B}_{\text{out}}(\mathbf{r}) = \frac{R^{3}}{3}
\left(\frac{3(\mathbf{J}\cdot\mathbf{r})\,\mathbf{r}}{\lVert\mathbf{r}\rVert^{5}}
-\frac{\mathbf{J}}{\lVert\mathbf{r}\rVert^{3}}\right).
$$

**Method** — a `where` on $r>R$ selects the two closed forms; the radial norm goes
through the singularity-safe `_safe_norm`.

**Singular set** — none in practice (the interior is smooth and the exterior is a
dipole evaluated away from its centre). The surface $r=R$ is the $\mathbf{B}$-continuous
boundary where $\mathbf{H}$ and $\mathbf{J}$ jump.

**Reference** — classical uniformly-magnetized-sphere result;
[Ortner & Coliado Bandeira (2020)](https://doi.org/10.1016/j.softx.2020.100466).

### Cuboid

[`cuboid.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/cuboid.py)

**Geometry** — a homogeneously polarized rectangular prism of side lengths
$(2a,2b,2c)$ centred at the origin, polarization $\mathbf{J}=(J_x,J_y,J_z)$.

**Model** — the surface-charge prism field. Each polarization component contributes
diagonal ($\arctan$) terms and off-diagonal ($\log$) terms built from the eight corner
distances $\sqrt{(x\pm a)^2+(y\pm b)^2+(z\pm c)^2}$; symmetry sign tables
(`qsigns`) fold the observer into the first octant. The three field components are
combinations of the two field families

$$
F_1 \sim \sum \pm\arctan\!\frac{(\cdot)(\cdot)}{(\cdot)\,\rho},
\qquad
F_2 \sim \sum \pm\log\bigl[(\cdot)+\rho\bigr],
$$

evaluated with the gradient-safe `_safe_arctan2` and `_safe_logabs`.

**Method** — closed form; octant folding plus safe transcendental ops. $\mathbf{H}$
subtracts the interior $\mathbf{J}$; $\mathbf{J}/\mathbf{M}$ are interior indicators.

**Singular set** — the edges of the prism (faces are $\mathbf{B}$-continuous but edge
limits are direction-dependent). The mask logic in `_cuboid_masks` flags surface and
edge points.

**Reference** — [Camacho & Sosa (2013)](https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1870-35422013000100002);
also Ravaud–Lemarquand prism fields (see [References](references.md)).

### Circular loop

[`circle.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/circle.py)

**Geometry** — a filamentary current loop of diameter $d$ carrying current $I$, in the
$xy$ plane, centred at the origin.

**Model** — the off-axis loop field in cylindrical coordinates $(r,\phi,z)$, expressed
through complete elliptic integrals. Writing $x_0 = z^2+(r+1)^2$ (in units of the loop
radius) and $k^2 = 4r/x_0$, the radial and axial components reduce to Bulirsch
`cel`-form combinations that the kernel evaluates with a fused, vectorised CEL
iteration (`_cel_iter`) rather than assembling $K$, $E$, $\Pi$ separately.

**Method** — cylindrical reduction + Bulirsch `cel` (see [Elliptic integrals](#elliptic-integrals));
dimensionless scaling by the radius; separate masks for the on-axis limit ($r=0$) and
the degenerate zero-radius loop.

**Singular set** — the wire itself, $\{r=R,\ z=0\}$, flagged by `mask_singular`.

**Reference** — [Simpson, Lane, Immer & Youngquist (2001)](https://ntrs.nasa.gov/citations/20010038494).

### Cylinder

[`cylinder.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/cylinder.py)

**Geometry** — a homogeneously polarized finite cylinder of diameter $d$ and height
$h$ (half-height $z_0$), centred at the origin.

**Model** — the polarization is split into an **axial** part $J_z$ and a **diametral**
(transverse) part $J_{xy}$, each with its own closed form, then recombined. The axial
field uses `cel` directly; the diametral field uses the complete integrals $K$, $E$,
$\Pi$ built on `cel`. A small-radius series ($r<0.05$, terms up to $r^5$) replaces the
general formula near the axis where the elliptic arguments become ill-conditioned:

$$
H_r^{\text{small}} = -\tfrac14\cos\phi\,(T_1 + 9T_2 + 25T_3),\quad
H_\phi^{\text{small}} = \tfrac14\sin\phi\,(T_1 + 3T_2 + 5T_3),
$$

with $T_i$ the successive series terms in $r$.

**Method** — axial/diametral decomposition; `cel`, `ellipk`, `ellipe`, `ellippi`;
near-axis series; interior $\mathbf{J}$ handling for the transverse (in $\mathbf{B}$)
and axial (in $\mathbf{H}$) components.

**Singular set** — the top/bottom rim edges $\{r=R,\ |z|=z_0\}$, flagged by
`_cylinder_masks`.

**Reference** — [Derby & Olbert (2010)](https://doi.org/10.1119/1.3256157).

### Cylinder segment

[`cylinder_segment.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/cylinder_segment.py)

**Geometry** — a ring-sector (tile) with inner/outer radii $r_1,r_2$, height $h$ and
azimuth span $[\phi_1,\phi_2]$; the dimension is the 5-tuple
$(r_1,r_2,h,\phi_1,\phi_2)$.

**Model** — because the tile has radial, axial *and* azimuthal boundaries, symmetry is
too low for a compact bar-magnet form. The kernel therefore **discretises the closed
boundary into an oriented triangular surface mesh** (outer/inner hulls, top/bottom
caps, two azimuthal cut planes; default $n_\phi=96$) and evaluates the exact
polyhedral surface-charge field over those faces via the [triangular mesh](#triangular-mesh)
kernel. The point-in-solid test for the material term is done analytically in $(r,\phi,z)$.

**Method** — structured meshing (`_build_cylinder_segment_mesh`) + surface-charge
triangle field; geometry precomputation shared with the mesh path.

**Singular set** — the tile edges and faces; interior determined analytically.

**Reference** — the general cylinder-tile solution of
[Slanovc, Ortner, Moridi, Abert & Suess (2022)](https://doi.org/10.1016/j.jmmm.2022.169482)
(here realised through the equivalent surface-charge discretisation).

### Triangle

[`triangle.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/triangle.py)

**Geometry** — a planar triangle $(\mathbf{v}_0,\mathbf{v}_1,\mathbf{v}_2)$ carrying a
uniform surface charge $\sigma=\mathbf{J}\cdot\hat{\mathbf{n}}$ (the building block of
all polyhedral magnets).

**Model** — the surface-charge field combines a **solid-angle** term along the face
normal with three **edge line-integral** terms,

$$
\mathbf{B} = \frac{\sigma}{4\pi}\Bigl(\hat{\mathbf{n}}\,\Omega(\mathbf{r})
- \hat{\mathbf{n}}\times\sum_{\text{edges}} I_{e}\,\mathbf{L}_{e}\Bigr),
$$

where $\Omega$ is the solid angle subtended by the triangle at the observer
(`_solid_angle`, via $2\arctan(N/D)$) and $I_e$ are logarithmic edge integrals. A
second logarithmic form ($\texttt{integ2}$) is used when the first denominator
collapses on an edge line.

**Method** — solid angle + edge integrals; `nan_to_num` cleans exact-boundary hits;
`_triangle_geom_terms` precomputes normals and edge vectors so tetrahedra and meshes
reuse them.

**Singular set** — the triangle edges and the plane of the triangle (where the
solid angle is discontinuous).

**Reference** — [Guptasarma & Singh (1999)](https://doi.org/10.1190/1.1444531);
[Fabbri (2008)](https://doi.org/10.1109/TMAG.2007.908698).

### Tetrahedron

[`tetrahedron.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/tetrahedron.py)

**Geometry** — a uniformly polarized tetrahedron given by four vertices.

**Model** — a tetrahedron is the sum of its four oriented triangular faces. A chirality
check (`_check_tetra_chirality`, via the vertex-matrix determinant) fixes outward
normals; the four `triangle` surface-charge fields are `vmap`-summed, and the interior
material term is added using a barycentric point-in-tetra test
(`_points_inside_tetra`, solving $\mathbf{r}=\mathbf{v}_0+M\boldsymbol\lambda$ for
$\boldsymbol\lambda\ge0,\ \sum\lambda\le1$).

**Method** — oriented-face decomposition + precomputed triangle geometry; barycentric
inside test.

**Singular set** — the tetrahedron edges and faces.

**Reference** — [Guptasarma & Singh (1999)](https://doi.org/10.1190/1.1444531);
[Fabbri (2008)](https://doi.org/10.1109/TMAG.2007.908698).

### Triangular mesh

[`trimesh.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/trimesh.py)

**Geometry** — a closed, consistently oriented triangular surface mesh (shape
$(n_{\text{faces}},3,3)$) of uniform polarization $\mathbf{J}$.

**Model** — the mesh field is the sum of its oriented-face surface-charge triangle
fields. The interior material term needs a robust point-in-mesh test, done by
**Möller–Trumbore ray casting**: a ray from a fixed exterior point to the observer is
intersected against every face, and an odd crossing count marks an interior point
(with a bounding-box prefilter and edge/coincidence tolerances). See
[`_raycast.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/_raycast.py).

**Method** — face accumulation (`vmap` for $\le 64$ faces, an in-place `fori_loop`
above that to cap memory), ray-cast inside test, and a masked variant for
padded/variable face counts.

**Singular set** — mesh faces and edges; open or non-orientable meshes have undefined
interior and are a compatibility-sensitive case validated against Magpylib.

**Reference** — [Guptasarma & Singh (1999)](https://doi.org/10.1190/1.1444531);
[Fabbri (2008)](https://doi.org/10.1109/TMAG.2007.908698).

### Polyline

[`polyline.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/polyline.py)

**Geometry** — a piecewise-linear current path given by segment endpoints
$(\mathbf{p}_1,\mathbf{p}_2)$ and current $I$.

**Model** — the finite straight-segment Biot–Savart field. Each element contributes

$$
\mathrm{d}\mathbf{B} = \frac{\mu_0 I}{4\pi}\frac{\mathrm{d}\boldsymbol\ell\times\mathbf{r}}{\lVert\mathbf{r}\rVert^{3}},
$$

and the closed-form integral over a segment is assembled from the perpendicular foot
point $\mathbf{p}_4$, the perpendicular distance, and the two end-angle sines
$\sin\theta_{1,2}$; the transverse direction is $\hat{\mathbf{e}}_B\propto(\mathbf{p}_2-\mathbf{p}_1)\times\mathbf{o}_4$.
Segment fields are `vmap`-summed.

**Method** — per-segment analytic Biot–Savart with normalisation by segment length;
validity masks for zero-length segments, on-line observers, and non-finite endpoints.

**Singular set** — the wire, i.e. observers on any segment line.

**Reference** — standard finite-segment Biot–Savart;
[Ortner & Coliado Bandeira (2020)](https://doi.org/10.1016/j.softx.2020.100466).

### Triangle sheet and triangle strip

[`current_sheet.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/current_sheet.py) ·
[`current_strip.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/kernels/current_strip.py)

**Geometry** — a triangle carrying a uniform surface current density $\mathbf{K}$
(A/m). A `TriangleStrip` is a fan of such triangles sharing edges; per-triangle current
densities are derived so the net current $I$ flows along the strip.

**Model** — each triangle is mapped by three rotations into a canonical *elementar
sheet* frame with vertices

$$
(0,0,0),\quad (u_1,0,0),\quad (u_2,v_2,0),
$$

(`_triangle_coordinate_transform`), where the sheet field has a closed form built from
$\arctan$ and $\operatorname{artanh}$ terms of the five corner distances. The physically
correct limit is genuinely piecewise, so the kernel separates the **off-plane**
($z\neq0$), **in-plane interior**, and several **edge/in-plane** cases with explicit
masks; a 7-point symmetric triangle **quadrature** (`_TRI_Q_W`, `_TRI_Q_L`) provides a
stable fallback for the delicate in-plane region.

**Method** — canonical-frame reduction + closed-form sheet field with gradient-safe
`_safe_atanh`/`_safe_logabs`/`_safe_sqrt`; degenerate-triangle guards; Gaussian triangle
quadrature fallback.

**Singular set** — the sheet surface and its edges.

**Reference** — polyhedral / surface-current formulation in the spirit of
[Guptasarma & Singh (1999)](https://doi.org/10.1190/1.1444531) and
[Fabbri (2008)](https://doi.org/10.1109/TMAG.2007.908698) (see [References](references.md)).

## Elliptic integrals

[`core/elliptic.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/core/elliptic.py)

The circle and cylinder families reduce to complete elliptic integrals, evaluated
through Bulirsch's single primitive

$$
\operatorname{cel}(k_c,p,c,s) =
\int_{0}^{\pi/2}
\frac{c\cos^{2}\theta + s\sin^{2}\theta}
{(\cos^{2}\theta + p\sin^{2}\theta)\sqrt{\cos^{2}\theta + k_c^{2}\sin^{2}\theta}}
\,\mathrm{d}\theta,
$$

computed by the descending-Landen `cel` iteration (a `lax.fori_loop` with a
convergence mask, so it is JIT- and `vmap`-friendly and stable in both float32 and
float64). The standard complete integrals are then special cases:

$$
K(m) = \operatorname{cel}(\sqrt{1-m},\,1,\,1,\,1),\quad
E(m) = \operatorname{cel}(\sqrt{1-m},\,1,\,1,\,1-m),\quad
\Pi(n,m) = \operatorname{cel}(\sqrt{1-m},\,1-n,\,1,\,1).
$$

The circle kernel calls a fused CEL iteration directly; the cylinder kernel uses the
derived `ellipk`/`ellipe`/`ellippi` wrappers.

**Reference** — [Bulirsch (1965)](https://doi.org/10.1007/BF01397975).

## Force and torque

[`fields/force.py`](https://github.com/uwplasma/magpylib_jax/blob/main/src/magpylib_jax/fields/force.py)

`getFT` computes the force $\mathbf{F}$ (N) and torque $\mathbf{T}$ (N·m) on each
target. A target is discretised into cells (mirroring Magpylib's `target_meshing`), each
cell carrying either a magnetic moment $\mathbf{m}$ (magnet) or a current vector
$I\,\mathrm{d}\boldsymbol\ell$ (current).

For a **magnet** cell the energy in the source field is $U=-\mathbf{m}\cdot\mathbf{B}$,
so the force is the gradient

$$
\mathbf{F} = -\nabla U = \nabla(\mathbf{m}\cdot\mathbf{B}) = (\mathbf{m}\cdot\nabla)\,\mathbf{B},
$$

the last step using $\nabla\times\mathbf{B}=\mathbf{0}$ in the source-free region
around the target. In code this is exactly `jnp.einsum("nk,nki->ni", m, J)` with
$J_{ki}=\partial B_k/\partial x_i$ obtained by `jax.jacfwd` of `getB` — no
finite-difference step and no `eps` parameter. The torque about the target `pivot`
(default the centroid) is

$$
\mathbf{T} = \mathbf{m}\times\mathbf{B} + (\mathbf{r}-\mathbf{r}_{\text{pivot}})\times\mathbf{F},
$$

with $\mathbf{m}\times\mathbf{B}$ the intrinsic aligning torque and the second term the
lever arm of each cell. For a **current** cell there is no dipole moment, so the Laplace
force replaces the gradient,

$$
\mathbf{F} = (I\,\mathrm{d}\boldsymbol\ell)\times\mathbf{B},\qquad \mathbf{T}=(\mathbf{r}-\mathbf{r}_{\text{pivot}})\times\mathbf{F},
$$

with zero intrinsic torque. Cell contributions are summed per target.

:::{tip}
Because every kernel is differentiable and the gradient uses forward-mode autodiff, the
magnet force is exact to machine precision and independent of a stencil size — the key
difference from Magpylib's finite-difference `getFT`
([Ortner & Coliado Bandeira, 2020](https://doi.org/10.1016/j.softx.2020.100466)).
`getFT` is itself differentiable and composes inside optimisation loops.
:::

The worked example is [Force and torque (getFT)](examples/force_torque.md).

## Further reading

Full bibliographic entries with DOIs for every model cited above — and the JAX and
Bulirsch numerics references — are collected in [References](references.md).
