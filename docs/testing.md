# Testing and Validation

## Test layers

- Parity tests (`tests/parity/`): direct numeric comparison against upstream Magpylib.
- Upstream-mirror tests (`tests/upstream_mirror/`): mirrored coverage for selected upstream object test files.
- Parity gates (`tests/parity_gates/`): profile checks across inside/surface/outside/singular neighborhoods for `B/H/J/M`.
- Physics tests (`tests/physics/`): identity checks from electromagnetic theory.
- Differentiability tests (`tests/differentiability/`): finite-gradient sanity checks.
- Compatibility tests (`tests/test_compatibility.py`, `tests/test_motion_shape_compat.py`): API and shape/path/orientation behavior.

## Coverage policy

The test suite enforces at least **90% code coverage** in CI.

Fast CI uses `-m "not slow"` for PR/push feedback. Extended upstream-mirror stress tests are marked
`slow` and executed in scheduled full validation.

## Running locally

```bash
pytest
pytest -m parity_gate
pytest -m slow
```
