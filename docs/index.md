# magpylib_jax

JAX-native, end-to-end differentiable magnetic field computation with Magpylib-compatible APIs.

`magpylib_jax` is organized around three goals:

- keep the high-level Magpylib user model familiar,
- provide analytical field kernels that remain differentiable through JAX,
- make correctness and performance observable through parity tests, benchmarks, and profiling artifacts.

## Start here

- New user: [Quickstart](quickstart.md)
- Looking for the mathematical model: [Equation Models](equations.md)
- Looking for implementation details: [Architecture and Source Map](architecture.md)
- Looking for examples: [Examples](examples/index.md)
- Looking for validation guarantees: [Testing and Validation](testing.md)
- Looking for performance guidance: [Performance](performance.md)

```{toctree}
:maxdepth: 2
:caption: User Guide

overview
quickstart
equations
numerics
architecture
examples/index
testing
performance
parity
parity_checklist
changelog
reference/api
roadmap
```
