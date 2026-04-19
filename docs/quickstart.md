# Quickstart

## Requirements

- Python `3.10+`
- JAX CPU install, or a platform-specific GPU/accelerator JAX install
- `numpy` and `scipy` are installed automatically with the package

## Install

```bash
pip install magpylib-jax
```

For development, tests, and docs:

```bash
pip install -e '.[test,docs]'
```

If you use a GPU-backed JAX environment, install the desired `jax` and `jaxlib` build for your platform first, then install `magpylib-jax`.

## First field computation

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

jax.config.update("jax_enable_x64", True)

src = mpj.Circle(current=1.0, diameter=1.0)
obs = jnp.array([0.1, 0.2, 0.3])

B = src.getB(obs)
print(B)
```

## First gradient

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

src = mpj.Circle(current=1.0, diameter=1.0)


def bz_at_z(z: float):
    return src.getB(jnp.array([0.0, 0.0, z]))[2]

print(jax.grad(bz_at_z)(0.25))
```

## First collection and sensor example

```python
import magpylib_jax as mpj

coil = mpj.current.Circle(current=1.2, diameter=0.6)
magnet = mpj.magnet.Cuboid(
    dimension=(0.4, 0.3, 0.2),
    polarization=(0.0, 0.0, 0.7),
    position=(0.0, 0.0, 0.5),
)
collection = mpj.Collection(coil, magnet)
sensor = mpj.Sensor(pixel=[(0.0, 0.0, 0.2), (0.0, 0.1, 0.2)])

B = sensor.getB(collection)
```

## JIT-safe high-level API

The public `getB/getH/getJ/getM` functions run through a JIT-safe core by default.

```python
import jax
import magpylib_jax as mpj

sources = [
    mpj.current.Circle(current=1.0, diameter=0.5),
    mpj.magnet.Sphere(polarization=(0.1, 0.0, 0.0), diameter=0.2),
]
sensor = mpj.Sensor(pixel=(0.0, 0.0, 0.1))

compiled = jax.jit(lambda: mpj.getB(sources, sensor, squeeze=False))
out = compiled()
```

## Troubleshooting

## `output="dataframe"` is not jittable

That is expected. It is a compatibility path that intentionally returns Python/pandas objects.

## I need exact upstream behavior around surfaces or boundaries

Run the parity suite for the relevant source family and inspect the matching tests in:

- [Parity Checklist](parity_checklist.md)
- [`tests/parity_gates`](https://github.com/uwplasma/magpylib_jax/tree/main/tests/parity_gates)
- [`tests/upstream_mirror`](https://github.com/uwplasma/magpylib_jax/tree/main/tests/upstream_mirror)

## I need to understand which code path is hot

Start with [Performance](performance.md) and the scripts in [`scripts/`](https://github.com/uwplasma/magpylib_jax/tree/main/scripts).
