# Parity Strategy

`magpylib_jax` follows a source-by-source parity process:

1. Port analytical kernel into JAX.
2. Add parity tests vs Magpylib for random and profile points.
3. Add physics and differentiability checks.
4. Add benchmark and CI threshold gates.

The live tracking table is in:
- `PARITY_MATRIX.md`
- `MIGRATION_PLAN.md`

Pending capabilities are explicitly listed until parity tests and CI gates are in place.
