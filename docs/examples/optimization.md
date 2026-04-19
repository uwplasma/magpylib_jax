# Optimization Examples

## Fitting cuboid polarization

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

obs = jnp.array([[0.2, 0.1, 0.4], [0.5, 0.0, 0.7]])
target = jnp.array([[2.0e-4, 0.0, 3.0e-4], [1.0e-4, 0.0, 2.0e-4]])


def loss_fn(pol):
    src = mpj.magnet.Cuboid(dimension=(1.0, 0.8, 1.2), polarization=pol)
    pred = src.getB(obs)
    return jnp.mean((pred - target) ** 2)

pol = jnp.array([0.05, -0.02, 0.08])
for _ in range(50):
    pol = pol - 1e-1 * jax.grad(loss_fn)(pol)
```

## Fitting multiple source parameters

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

obs = jnp.array(
    [
        [0.2, 0.1, 0.4],
        [0.5, 0.0, 0.7],
        [-0.1, 0.3, 0.2],
        [0.3, -0.2, 0.6],
    ]
)
target = jnp.array(
    [
        [2.0e-4, 0.0, 3.0e-4],
        [1.0e-4, 0.0, 2.0e-4],
        [1.5e-4, 0.5e-4, 2.2e-4],
        [0.8e-4, -0.2e-4, 1.7e-4],
    ]
)


def loss_fn(params):
    pol1 = params[0:3]
    pos1 = params[3:6]
    pol2 = params[6:9]
    pos2 = params[9:12]

    src1 = mpj.magnet.Cuboid(dimension=(1.0, 0.8, 1.2), polarization=pol1, position=pos1)
    src2 = mpj.magnet.Cuboid(dimension=(0.6, 0.6, 0.6), polarization=pol2, position=pos2)
    pred = mpj.Collection(src1, src2).getB(obs)
    return jnp.mean((pred - target) ** 2)

params = jnp.array([0.05, -0.02, 0.08, 0.0, 0.0, 0.0, 0.03, 0.01, 0.04, 0.2, 0.1, -0.1])
for _ in range(80):
    params = params - 5e-2 * jax.grad(loss_fn)(params)
```

## Practical note

For large optimization loops:

- prefer x64,
- keep observer layouts static when possible,
- reuse object graphs if the optimization variables can be isolated cleanly,
- profile both compile time and steady-state runtime.
