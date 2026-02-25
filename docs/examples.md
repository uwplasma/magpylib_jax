# Examples

## Object API

```python
import magpylib_jax as mpj

src = mpj.magnet.Sphere(polarization=(0.1, 0.0, 0.2), diameter=1.5)
B = src.getB([[0.1, 0.2, 0.3], [0.5, 0.0, 1.0]])
```

## Functional API

```python
import magpylib_jax as mpj

B = mpj.getB(
    "cuboid",
    observers=[[0.2, 0.1, 0.4]],
    polarization=(0.1, -0.2, 0.3),
    dimension=(1.0, 0.8, 1.2),
)
```

## Collection + Sensor

```python
import magpylib_jax as mpj

src1 = mpj.current.Circle(current=1.2, diameter=1.0)
src2 = mpj.misc.Dipole(moment=(0.0, 0.0, 1.0))
col = mpj.Collection([src1, src2])
sens = mpj.Sensor(pixel=[(0.1, 0.2, 0.3), (0.2, 0.0, 0.4)])

B = sens.getB(col)
```
