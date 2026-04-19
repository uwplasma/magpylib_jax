# Object API Examples

## Single source

```python
import magpylib_jax as mpj

src = mpj.magnet.Sphere(polarization=(0.1, 0.0, 0.2), diameter=1.5)
B = src.getB([[0.1, 0.2, 0.3], [0.5, 0.0, 1.0]])
```

## Collection with mixed sources

```python
import magpylib_jax as mpj

loop = mpj.current.Circle(current=1.2, diameter=1.0)
dip = mpj.misc.Dipole(moment=(0.0, 0.0, 1.0))
mag = mpj.magnet.Cuboid(
    dimension=(0.4, 0.2, 0.2),
    polarization=(0.0, 0.0, 0.5),
    position=(0.0, 0.0, 0.4),
)

col = mpj.Collection(loop, dip, mag)
B = col.getB([(0.1, 0.0, 0.2), (0.2, 0.0, 0.3)])
```

## Sensor with pixel grid

```python
import numpy as np
import magpylib_jax as mpj

pixel = np.stack(
    np.meshgrid(np.linspace(-0.1, 0.1, 3), np.linspace(-0.1, 0.1, 3), [0.0], indexing="ij"),
    axis=-1,
)

sensor = mpj.Sensor(pixel=pixel, position=(0.0, 0.0, 0.3))
source = mpj.current.Circle(current=5.0, diameter=0.5)
B = sensor.getB(source, pixel_agg="mean")
```

## Path semantics

```python
import magpylib_jax as mpj

src = mpj.current.Circle(current=1.0, diameter=0.5)
src.position = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.2), (0.0, 0.0, 0.4)]
obs = [(0.1, 0.0, 0.0), (0.1, 0.0, 0.2), (0.1, 0.0, 0.4)]
B = src.getB(obs, squeeze=False)
```
