# Testing and Validation

## Test layers

- Parity tests (`tests/parity/`): direct numeric comparison against upstream Magpylib.
- Parity gates (`tests/parity_gates/`): behavior profiles across inside/surface/outside/singular points for `B/H/J/M`.
- Physics tests (`tests/physics/`): identity checks from electromagnetic theory.
- Differentiability tests (`tests/differentiability/`): finite gradient sanity checks.
- Compatibility tests (`tests/test_compatibility.py`): object/functional API behavior and shaping.

## Coverage policy

The test suite enforces at least **90% code coverage** in CI.

## Running locally

```bash
pytest
pytest -m parity_gate
```
