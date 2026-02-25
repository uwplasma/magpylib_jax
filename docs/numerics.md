# Numerics

## Stability strategies

- Dimensionless scaling where available (e.g. current loops, segment formulas).
- Explicit handling of singular locations (on-source, edges, corners).
- Masked special-case formulas for near-degenerate geometries.
- Bulirsch CEL-based elliptic integral routines for robust cylinder/circle expressions.
- Current sheet kernels use explicit in-plane/edge masks to zero the field on the sheet
  and to switch to the correct edge-limit expressions.

## Differentiability considerations

- Kernels are JAX-first and vectorized.
- Non-smooth physics boundaries (on edges/surfaces/singular points) are handled by masked branches.
- Gradients are expected to be reliable away from physical singularities.

## Precision

- `jax_enable_x64=True` in tests.
- Constants and tolerances are chosen to match upstream Magpylib behavior where practical.
