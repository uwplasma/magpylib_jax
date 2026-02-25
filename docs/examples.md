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

## Differentiable optimization loop

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

obs = jnp.array([[0.2, 0.1, 0.4], [0.5, 0.0, 0.7]])
target = jnp.array([[2.0e-4, 0.0, 3.0e-4], [1.0e-4, 0.0, 2.0e-4]])

def loss_fn(pol):
    src = mpj.magnet.Cuboid(
        dimension=(1.0, 0.8, 1.2),
        polarization=pol,
    )
    pred = src.getB(obs)
    return jnp.mean((pred - target) ** 2)

pol = jnp.array([0.05, -0.02, 0.08])
lr = 1e-1
for _ in range(50):
    grad = jax.grad(loss_fn)(pol)
    pol = pol - lr * grad
```
