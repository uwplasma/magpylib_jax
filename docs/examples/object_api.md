# Object API

The object API is the Magpylib-compatible way to work: construct sources, give them position and
orientation, group them into collections, place sensors, and call `getB/getH/getJ/getM` as methods.
It is the most readable entry point and the easiest migration path from Magpylib.

```{admonition} Runnable scripts
:class: seealso
Everything here is distilled from
[`examples/basics/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples/basics). Run any
script directly (`python examples/basics/first_field.py`) to see printed output and, where relevant,
a saved figure.
```

## A source in three lines

Every source exposes `getB` (T), `getH` (A/m), `getJ` (T), and `getM` (A/m). Observers can be a
single point, a list of points, or any array whose last axis is length 3.

```python
import magpylib_jax as mpj

src = mpj.magnet.Sphere(polarization=(0.1, 0.0, 0.2), diameter=1.5)
B = src.getB([[0.1, 0.2, 0.3], [0.5, 0.0, 1.0]])
print(B.shape)      # (2, 3) — one field vector per observer
```

The four fields of one source come from the matching methods — see
[`examples/basics/first_field.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/basics/first_field.py),
which evaluates `B/H/J/M` of a diametrically polarized cylinder on a grid:

```python
import numpy as np
import magpylib_jax as mpj

cyl = mpj.magnet.Cylinder(polarization=(0.5, 0.5, 0.0), dimension=(0.04, 0.02))
grid = np.stack(np.meshgrid(np.linspace(-0.05, 0.05, 30),
                            np.linspace(-0.05, 0.05, 30), [0.0], indexing="ij"), axis=-1)
fields = {name: getattr(cyl, f"get{name}")(grid) for name in "BHJM"}
```

## Collections of mixed sources

A `Collection` groups sources (and nests other collections) and behaves as a single source — the
fields superpose:

```python
import magpylib_jax as mpj

loop = mpj.current.Circle(current=1.2, diameter=1.0)
dip = mpj.misc.Dipole(moment=(0.0, 0.0, 1.0))
mag = mpj.magnet.Cuboid(dimension=(0.4, 0.2, 0.2), polarization=(0.0, 0.0, 0.5),
                        position=(0.0, 0.0, 0.4))

col = mpj.Collection(loop, dip, mag)
B = col.getB([(0.1, 0.0, 0.2), (0.2, 0.0, 0.3)])     # sum of all three sources
```

You can grow a collection incrementally with `add`, `copy` it (optionally with a new pose), and even
combine collections with `+`. See the Helmholtz pair in
[`examples/basics/collections_sensors.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/basics/collections_sensors.py):

```python
import numpy as np
import magpylib_jax as mpj

coil1 = mpj.Collection()
for z in np.linspace(-0.0005, 0.0005, 5):
    coil1.add(mpj.current.Circle(current=1.0, diameter=0.02, position=(0.0, 0.0, z)))
coil1.position = (0.0, 0.0, -0.005)
coil2 = coil1.copy(position=(0.0, 0.0, 0.005))
helmholtz = coil1 + coil2
```

## Sensors and pixel grids

A `Sensor` is a movable observer with its own pixel layout. Read a source (or a whole collection)
*through* the sensor, and optionally reduce the pixels with `pixel_agg`:

```python
import numpy as np
import magpylib_jax as mpj

pixel = np.stack(np.meshgrid(np.linspace(-0.1, 0.1, 3),
                             np.linspace(-0.1, 0.1, 3), [0.0], indexing="ij"), axis=-1)
sensor = mpj.Sensor(pixel=pixel, position=(0.0, 0.0, 0.3))
source = mpj.current.Circle(current=5.0, diameter=0.5)

B = sensor.getB(source, pixel_agg="mean")     # mean over the 3x3 pixel grid
```

## Motion paths

Position and orientation are vectorized **paths**: a single field call runs once over the whole
path. Set an absolute path, or build one with relative `move`/`rotate` operations:

```python
import magpylib_jax as mpj

src = mpj.current.Circle(current=1.0, diameter=0.5)
src.position = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.2), (0.0, 0.0, 0.4)]
obs = [(0.1, 0.0, 0.0), (0.1, 0.0, 0.2), (0.1, 0.0, 0.4)]
B = src.getB(obs, squeeze=False)              # field at each path step
```

Relative motion, merged rotations (spirals), and the edge-padding rule that keeps a shorter path
static while a longer one keeps moving are all covered in
[`examples/basics/motion_paths.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/basics/motion_paths.py).

## Custom sources

`misc.CustomSource` wraps a user field function, so you can drop any closed-form field into the
object API. In `magpylib_jax`, `field_func` takes a single `observers` argument and returns `B`
directly:

```python
import jax.numpy as jnp
import magpylib_jax as mpj

def monopole(observers):                       # unphysical, but instructive
    r = jnp.atleast_2d(jnp.asarray(observers, dtype=jnp.float64))
    return r / jnp.linalg.norm(r, axis=-1, keepdims=True) ** 3

mono = mpj.misc.CustomSource(field_func=monopole)
B = mono.getB((1.0, 0.0, 0.0))                 # ~ [1, 0, 0]
```

See the four-pole quadrupole in
[`examples/basics/custom_source.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/basics/custom_source.py).

## Where to go next

- [Functional API](functional_api.md) — the same fields without building objects.
- [Optimization](optimization.md) — differentiate through any of the calls above.
- The full source-family tour in
  [`examples/basics/sources_gallery.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/basics/sources_gallery.py)
  and the shape builders in
  [`examples/shapes/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples/shapes).
