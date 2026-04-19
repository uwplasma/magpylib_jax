# Functional API Examples

## Direct cuboid call

```python
import magpylib_jax as mpj

B = mpj.getB(
    "cuboid",
    observers=[[0.2, 0.1, 0.4]],
    polarization=(0.1, -0.2, 0.3),
    dimension=(1.0, 0.8, 1.2),
)
```

## Direct circle call

```python
import magpylib_jax as mpj

B = mpj.getB(
    "circle",
    observers=[[0.0, 0.0, 0.2], [0.1, 0.0, 0.2]],
    current=2.0,
    diameter=0.6,
)
```

## Querying `H`, `J`, and `M`

```python
import magpylib_jax as mpj

src = mpj.magnet.Cuboid(dimension=(0.5, 0.5, 0.5), polarization=(0.0, 0.0, 1.0))
obs = [(0.2, 0.0, 0.0)]

B = mpj.getB(src, obs)
H = mpj.getH(src, obs)
J = mpj.getJ(src, obs)
M = mpj.getM(src, obs)
```

## DataFrame compatibility output

```python
import magpylib_jax as mpj

src = mpj.current.Circle(current=1.0, diameter=0.4)
df = mpj.getB(src, [(0.1, 0.0, 0.1)], output="dataframe")
```

`output="dataframe"` is intentionally a compatibility path, not part of the jittable field graph.
