"""Field-evaluation package: public API plus the JIT and eager engines."""

from __future__ import annotations

from magpylib_jax.fields.api import _compute_field, getB, getH, getJ, getM

__all__ = ["getB", "getH", "getJ", "getM", "_compute_field"]
