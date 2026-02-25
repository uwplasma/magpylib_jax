# Parity Strategy

`magpylib_jax` follows a source-by-source parity workflow:

1. Port kernel/object implementation in JAX.
2. Add parity tests against upstream Magpylib (random + profile points).
3. Add mirrored upstream-file tests for object behavior.
4. Add physics and differentiability tests.
5. Gate behavior in CI with strict parity markers and threshold checks.

Live tracking tables:
- `PARITY_MATRIX.md`
- `MIGRATION_PLAN.md`

For pending work, entries remain marked `Partial`/`Pending` until parity and CI gates are in place.
