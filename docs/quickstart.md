# Quickstart

## Install

```bash
pip install magpylib-jax
```

For local development:

```bash
pip install -e '.[test,docs]'
```

If you use a GPU-backed JAX environment, install the desired `jax`/`jaxlib` build first and then install `magpylib-jax`.

## Example

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import magpylib_jax as mpj

src = mpj.Circle(current=1.0, diameter=1.0)
obs = jnp.array([0.1, 0.2, 0.3])

B = src.getB(obs)

def bz_at_z(z: float) -> jax.Array:
    return src.getB(jnp.array([0.0, 0.0, z]))[2]

grad_bz = jax.grad(bz_at_z)(0.25)
```
