"""Differentiable JAX-native magnetic field toolkit."""

from magpylib_jax.current import Circle
from magpylib_jax.functional import getB, getH
from magpylib_jax.misc import Dipole

__all__ = ["Circle", "Dipole", "getB", "getH"]
