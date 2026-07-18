# Optimization

This is the reason `magpylib_jax` exists. Because `getB/getH/getJ/getM` and `getFT` are smooth JAX
functions, fitting a source to data — or sizing a geometry to hit a target field — is plain gradient
descent. No finite differences, no adjoint solver, no wrappers: just `jax.grad`.

```{raw} html
<video controls loop muted playsinline width="100%" src="../_static/movie_optimization.mp4"></video>
```

*Above: a magnet parameter converging to its target under gradient descent, driven entirely by the
exact `jax.grad` of the field.*

```{admonition} Runnable scripts
:class: seealso
The examples on this page are distilled from
[`examples/differentiable/`](https://github.com/uwplasma/magpylib_jax/tree/main/examples/differentiable).
Enable x64 for parity-grade results: `jax.config.update("jax_enable_x64", True)`.
```

## Inverse design: recover a polarization

The flagship pattern. Generate synthetic field samples from a magnet with an unknown polarization,
then recover it from a wrong initial guess by minimizing the mean-squared field error. Compiling the
loss and its gradient with `jax.jit` means you trace once and every step reuses the compiled kernel.

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

OBS = jnp.array([[0.2, 0.1, 0.4], [0.5, 0.0, 0.7], [-0.3, 0.2, 0.5],
                 [0.1, -0.4, 0.6], [0.4, 0.3, 0.3], [-0.2, -0.1, 0.8]])

def field(pol):
    return mpj.magnet.Cuboid(dimension=(1.0, 0.8, 1.2), polarization=pol).getB(OBS)

target = field(jnp.array([0.35, -0.20, 0.80]))       # synthetic measurements
loss = jax.jit(lambda p: jnp.mean((field(p) - target) ** 2))
grad = jax.jit(jax.grad(loss))

pol = jnp.array([0.05, 0.05, 0.05])                  # deliberately wrong start
for _ in range(150):
    pol = pol - 2.0 * grad(pol)
print(pol)                                           # -> [0.35, -0.20, 0.80]
```

Expected: the recovered polarization matches the true vector to within `~1e-3` T, and the loss falls
by several orders of magnitude. Full script:
[`examples/differentiable/inverse_design.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/differentiable/inverse_design.py).

## Geometry optimization: size a magnet

Gradients flow through *geometry* too. Here we solve for the height of a cuboid so the axial field
at a fixed probe hits a target, using a compiled `jax.jit(jax.grad(...))`:

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

PROBE = jnp.array([0.0, 0.0, 0.03])      # 30 mm above the magnet center
TARGET_BZ = 2.0e-2                        # tesla

def bz(height):
    dim = jnp.stack([jnp.asarray(0.02), jnp.asarray(0.02), height])
    return mpj.magnet.Cuboid(dimension=dim, polarization=(0.0, 0.0, 1.2)).getB(PROBE)[2]

grad = jax.jit(jax.grad(lambda h: (bz(h) - TARGET_BZ) ** 2))

height = jnp.asarray(0.005)               # 5 mm initial guess
for _ in range(200):
    height = jnp.clip(height - 5.0 * grad(height), 1e-3, 0.1)
print(float(height) * 1e3, "mm")          # converges to the height that hits TARGET_BZ
```

Full script:
[`examples/differentiable/optimize_geometry.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/differentiable/optimize_geometry.py).

## Fitting several parameters at once

Nothing changes when the parameter vector grows — pack everything into one array and let autodiff
handle the bookkeeping. This fits the polarizations and positions of two cuboids in a collection:

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

obs = jnp.array([[0.2, 0.1, 0.4], [0.5, 0.0, 0.7], [-0.1, 0.3, 0.2], [0.3, -0.2, 0.6]])
target = jnp.array([[2.0e-4, 0.0, 3.0e-4], [1.0e-4, 0.0, 2.0e-4],
                    [1.5e-4, 0.5e-4, 2.2e-4], [0.8e-4, -0.2e-4, 1.7e-4]])

def loss(params):
    src1 = mpj.magnet.Cuboid(dimension=(1.0, 0.8, 1.2), polarization=params[0:3], position=params[3:6])
    src2 = mpj.magnet.Cuboid(dimension=(0.6, 0.6, 0.6), polarization=params[6:9], position=params[9:12])
    return jnp.mean((mpj.Collection(src1, src2).getB(obs) - target) ** 2)

grad = jax.jit(jax.grad(loss))
params = jnp.array([0.05, -0.02, 0.08, 0.0, 0.0, 0.0, 0.03, 0.01, 0.04, 0.2, 0.1, -0.1])
for _ in range(80):
    params = params - 5e-2 * grad(params)
```

## Optimizing through force and torque

`getFT` is differentiable, so you can optimize a *force* balance. The levitation example solves for
the equilibrium height where an upward magnetic force balances a load, using the exact `dF_z/dh`
from `jax.value_and_grad` in a Newton iteration — something a finite-difference `getFT` cannot supply
directly:

```python
import jax
import jax.numpy as jnp
import magpylib_jax as mpj

BASE = mpj.misc.Dipole(moment=(0.0, 0.0, 300.0))
WEIGHT = 1.5

def force_z(height):
    loop = mpj.current.Circle(diameter=0.04, current=-500.0, meshing=8,
                              position=jnp.stack([jnp.asarray(0.0), jnp.asarray(0.0), height]))
    return mpj.getFT(BASE, loop)[0][2]

force_and_grad = jax.jit(jax.value_and_grad(force_z))
height = jnp.asarray(0.03)
for _ in range(6):
    fz, dfz = force_and_grad(height)
    height = jnp.clip(height - (fz - WEIGHT) / dfz, 0.02, 0.15)   # Newton step
```

Full script:
[`examples/differentiable/getft_optimization.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/differentiable/getft_optimization.py).

## Practical notes

```{admonition} Getting reliable, fast optimization loops
:class: tip

- **Enable x64** for well-conditioned gradients and parity-grade fits.
- **Compile the loss and its gradient** with `jax.jit` so tracing happens once, not per step.
- **Keep shapes static** — a changing observer count or source count forces recompilation.
- **Isolate the variables you optimize**; reuse the fixed parts of the object graph.
- **Profile both** compile time and steady-state runtime — see [Performance](../performance.md).
```

## Where to go next

- Field Jacobians (`dB/dr`, `dB/dJ`) via `jax.jacfwd`/`jacrev` in
  [`examples/differentiable/field_jacobian.py`](https://github.com/uwplasma/magpylib_jax/blob/main/examples/differentiable/field_jacobian.py).
- Batched sweeps in [Performance](../performance.md) and
  [Performance examples](performance.md).
- The [force and torque](force_torque.md) tutorial for more `getFT` patterns.
