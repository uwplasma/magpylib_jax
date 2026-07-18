# Quickstart

This page takes you from an empty environment to your first field, your first gradient, a
multi-source scene, a force computation, and a 3D render — all with runnable snippets.

## Install

::::{tab-set}
:::{tab-item} PyPI
```bash
pip install magpylib-jax
```
:::
:::{tab-item} Development
```bash
git clone https://github.com/uwplasma/magpylib_jax
cd magpylib_jax
pip install -e '.[test,docs]'
```
:::
::::

Requirements:

- Python `3.10+`
- JAX (a CPU build is pulled in automatically; `numpy` and `scipy` come with it)
- For GPU/TPU, install the matching `jax`/`jaxlib` build for your platform **first**, then install
  `magpylib-jax`.

:::{admonition} Precision follows your JAX config
:class: tip

Arrays use JAX's default float dtype — `float32` unless you opt in to double precision. For
bit-level parity with Magpylib (which is `float64`), enable x64 **before** you use the library:

```python
import jax
jax.config.update("jax_enable_x64", True)
import magpylib_jax as mpj
```

The package never mutates the global JAX config on import. See [Precision](precision.md) for the
full story.
:::

## First field

The magnetic field of a source is three lines away. `getB` returns flux density in tesla as a JAX
array.

```python
import jax
jax.config.update("jax_enable_x64", True)
import magpylib_jax as mpj

loop = mpj.current.Circle(current=1.0, diameter=1.0)
B = loop.getB((0.0, 0.0, 0.0))
print(B)          # [0. 0. 1.2566...e-06] T  (mu_0 * I / d on axis at the center)
```

Every source also exposes `getH` (A/m), `getJ` (T), and `getM` (A/m). Observers can be a single
point, a list of points, or any array whose last axis has length 3:

```python
import numpy as np
import magpylib_jax as mpj

cyl = mpj.magnet.Cylinder(polarization=(0.5, 0.5, 0.0), dimension=(0.04, 0.02))
grid = np.stack(np.meshgrid(np.linspace(-0.05, 0.05, 30),
                            np.linspace(-0.05, 0.05, 30), [0.0], indexing="ij"), axis=-1)
B = cyl.getB(grid)     # shape (30, 30, 1, 3)
```

## First gradient

Because `getB` is a smooth JAX function, its derivative with respect to *any* input is one
`jax.grad` away. Here we differentiate the on-axis field of a loop with respect to the probe height:

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

loop = mpj.current.Circle(current=1.0, diameter=1.0)

def bz_at_z(z):
    return loop.getB(jnp.array([0.0, 0.0, z]))[2]

print(jax.grad(bz_at_z)(0.25))     # dBz/dz, exact — no finite differences
```

The same works for geometry and excitation. A tiny inverse-design loop — recover an unknown cuboid
polarization from field samples — is just gradient descent:

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

obs = jnp.array([[0.2, 0.1, 0.4], [0.5, 0.0, 0.7], [-0.3, 0.2, 0.5]])

def field(pol):
    return mpj.magnet.Cuboid(dimension=(1.0, 0.8, 1.2), polarization=pol).getB(obs)

target = field(jnp.array([0.35, -0.20, 0.80]))          # synthetic measurements
loss = jax.jit(lambda p: jnp.mean((field(p) - target) ** 2))
grad = jax.jit(jax.grad(loss))

pol = jnp.array([0.05, 0.05, 0.05])                     # wrong initial guess
for _ in range(150):
    pol = pol - 2.0 * grad(pol)
print(pol)          # -> approaches [0.35, -0.20, 0.80]
```

See the full script:
[`examples/differentiable/inverse_design.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/differentiable/inverse_design.py).

## Collections and sensors

Group sources into a `Collection` (which behaves as a single source) and read the field through a
movable `Sensor`:

```python
import magpylib_jax as mpj

coil = mpj.current.Circle(current=1.2, diameter=0.6)
magnet = mpj.magnet.Cuboid(dimension=(0.4, 0.3, 0.2),
                           polarization=(0.0, 0.0, 0.7),
                           position=(0.0, 0.0, 0.5))
scene = mpj.Collection(coil, magnet)

sensor = mpj.Sensor(pixel=[(0.0, 0.0, 0.2), (0.0, 0.1, 0.2)])
B = sensor.getB(scene)          # field at both pixels, from both sources combined
```

Sources and sensors carry vectorized **paths** (position + orientation arrays); a single field call
runs over the whole path. See
[`examples/basics/collections_sensors.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/basics/collections_sensors.py)
and
[`examples/basics/motion_paths.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/basics/motion_paths.py).

## Force and torque

`getFT(source, target)` returns the `(force, torque)` on `target` in the field of `source`. It is
differentiable, so you can optimize *through* it:

```python
import magpylib_jax as mpj

magnet = mpj.misc.Dipole(moment=(0.0, 0.0, 300.0))
loop = mpj.current.Circle(diameter=0.04, current=-500.0,
                          position=(0.0, 0.0, 0.03), meshing=8)

force, torque = mpj.getFT(magnet, loop)
print(force)        # newtons; force[2] is the vertical (levitation) component
```

More in the [force and torque](examples/force_torque.md) tutorial and
[`examples/force/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples/force).

## Visualize with `show`

`show` renders geometry, polarization, and paths in 3D with Matplotlib:

```python
import magpylib_jax as mpj

sphere = mpj.magnet.Sphere(polarization=(1, 1, 1), diameter=1.0, position=(-1.5, 1, 0))
cube = mpj.magnet.Cuboid(polarization=(1, 0, 0), dimension=(1, 0.3, 0.3), position=(1.5, 1, 0))
seg = mpj.magnet.CylinderSegment(polarization=(1, 0, 0), dimension=(1.7, 2.0, 0.3, -145, -35))

fig = mpj.show(sphere, cube, seg, return_fig=True)
```

```{admonition} Matplotlib only
:class: note
The Plotly and PyVista backends from upstream Magpylib are out of scope. See
[Parity strategy](parity.md#what-is-intentionally-different).
```

## Where to next

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 📖 Examples
Task-organized tutorials with runnable scripts.
+++
[Examples gallery »](examples/index.md)
:::

:::{grid-item-card} 🎛️ Precision
float32 vs float64, and how to match Magpylib.
+++
[Precision »](precision.md)
:::

:::{grid-item-card} 🚀 Performance
JIT, `vmap`, GPU/TPU, and honest benchmarks.
+++
[Performance »](performance.md)
:::

:::{grid-item-card} 🔧 Architecture
From the public API to the analytic kernels.
+++
[Architecture »](architecture.md)
:::

::::
