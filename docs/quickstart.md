# Quickstart

## Install

```bash
pip install -e '.[test,docs]'
```

## Example

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

src = mpj.Circle(current=1.0, diameter=1.0)
obs = jnp.array([0.1, 0.2, 0.3])

B = src.getB(obs)

def bz_at_z(z: float) -> jax.Array:
    return src.getB(jnp.array([0.0, 0.0, z]))[2]

grad_bz = jax.grad(bz_at_z)(0.25)
```
