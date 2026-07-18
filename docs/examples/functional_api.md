# Functional API

The functional API skips object construction: call `getB/getH/getJ/getM` directly with a source
descriptor and an observer array. It is the leanest path for one-off evaluations, batched sweeps,
and code where you would rather pass plain arrays than build and mutate objects.

Two call styles are accepted:

- **By type string** — `mpj.getB("cuboid", observers=..., **params)`.
- **By object** — `mpj.getB(src, observers)` with a source (or list of sources) you already built.

```{admonition} Same engine, same results
:class: note
The functional and object APIs share the fields engine, so results are identical. Objects add pose,
paths, style, and caching on top. See [Architecture](../architecture.md).
```

## By type string

Pass the family name and its parameters as keywords:

```python
import magpylib_jax as mpj

B = mpj.getB(
    "cuboid",
    observers=[[0.2, 0.1, 0.4]],
    polarization=(0.1, -0.2, 0.3),
    dimension=(1.0, 0.8, 1.2),
)

B_loop = mpj.getB(
    "circle",
    observers=[[0.0, 0.0, 0.2], [0.1, 0.0, 0.2]],
    current=2.0,
    diameter=0.6,
)
```

## By object

If you already have a source object, hand it (or a list) straight to the field functions. All four
fields share the same signature:

```python
import magpylib_jax as mpj

src = mpj.magnet.Cuboid(dimension=(0.5, 0.5, 0.5), polarization=(0.0, 0.0, 1.0))
obs = [(0.2, 0.0, 0.0)]

B = mpj.getB(src, obs)     # flux density   (T)
H = mpj.getH(src, obs)     # field strength  (A/m)
J = mpj.getJ(src, obs)     # polarization    (T)
M = mpj.getM(src, obs)     # magnetization   (A/m)
```

Multiple sources and multiple sensors produce one output index per input axis — see
[`examples/basics/first_field.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/basics/first_field.py):

```python
import numpy as np
import magpylib_jax as mpj

cube = mpj.magnet.Cuboid(polarization=(0.0, 0.0, 1.0), dimension=(0.1, 0.1, 0.1))
sens1 = mpj.Sensor(pixel=[(0.0, 0.0, 0.0), (0.005, 0.0, 0.0)])
sens2 = sens1.copy()
combo = mpj.getB([cube, cube.copy()], [sens1, sens2])
print(combo.shape)         # (2 sources, 2 sensors, 2 pixels, 3 components)
```

## Squeeze semantics

By default the output is squeezed to drop length-1 axes, matching Magpylib. Pass `squeeze=False` to
keep the full `(sources, paths, sensors, pixels, 3)` layout — useful when you need a stable shape in
a compiled or batched pipeline:

```python
import magpylib_jax as mpj

src = mpj.current.Circle(current=1.0, diameter=0.5)
B = mpj.getB(src, [(0.1, 0.0, 0.0)], squeeze=False)
```

## DataFrame compatibility output

`output="dataframe"` returns a pandas frame for drop-in compatibility with Magpylib workflows:

```python
import magpylib_jax as mpj

src = mpj.current.Circle(current=1.0, diameter=0.4)
df = mpj.getB(src, [(0.1, 0.0, 0.1)], output="dataframe")
```

```{admonition} Not part of the jittable graph
:class: warning
`output="dataframe"` is intentionally a compatibility path: it returns Python/pandas objects and
cannot be traced by `jax.jit`. Use the default array output inside compiled or differentiated code.
See [Parity strategy](../parity.md#what-is-intentionally-different).
```

## Where to go next

- [Object API](object_api.md) — add pose, paths, sensors, and caching.
- [Optimization](optimization.md) — wrap any functional call in `jax.grad`.
- [Performance](../performance.md) — JIT and `vmap` your functional calls.
