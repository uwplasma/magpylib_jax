# Choosing precision (float32 vs float64)

magpylib_jax follows **your** JAX precision setting and never changes the global JAX config on
import — you pick the precision exactly like any other JAX code. The one rule: **set it before you
build any array** (that is, before the first `getB`/`getFT`/object construction).

`````{tab-set}
````{tab-item} float64 (parity)
Full Magpylib parity and the tightest gradients — recommended for scientific work:

```python
import jax
jax.config.update("jax_enable_x64", True)   # do this first
import magpylib_jax as mpj

mpj.magnet.Cuboid(polarization=(0, 0, 1.0), dimension=(1, 1, 1)).getB((2, 0, 0)).dtype
# -> float64
```
````

````{tab-item} float32 (default)
JAX's default — faster and lower-memory, ideal on GPU/TPU and in ML pipelines. Just don't enable
x64:

```python
import magpylib_jax as mpj   # no x64 -> float32

mpj.magnet.Cuboid(polarization=(0, 0, 1.0), dimension=(1, 1, 1)).getB((2, 0, 0)).dtype
# -> float32
```
````
`````

| | float32 (default) | float64 (`jax_enable_x64=True`) |
|---|---|---|
| speed / memory | faster, lighter (especially GPU/TPU) | slower, 2× memory |
| Magpylib parity | ~1e-6 relative | bit-level (~1e-15) |
| how to select | do nothing | `jax.config.update("jax_enable_x64", True)` before use |

Internally, the kernels never hard-code `float64`: array dtypes use JAX's canonical default
(`jax.numpy.float_`), so a single flag switches the entire library between single and double
precision with no truncation warning. The test suite runs with x64 enabled via `conftest.py`.

```{note}
Why not enable x64 on import? Because `jax.config.update` is *process-global*: it would force every
JAX computation in your program into float64 (slower on GPU/TPU) and override other libraries'
choices. A library should respect your configuration, not mutate it.
```
