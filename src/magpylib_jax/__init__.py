"""Differentiable JAX-native magnetic field toolkit."""

from magpylib_jax import current, magnet, misc
from magpylib_jax.collection import Collection
from magpylib_jax.current import Circle, Polyline
from magpylib_jax.functional import getB, getH, getJ, getM
from magpylib_jax.magnet import Cuboid, Cylinder, Sphere, Tetrahedron
from magpylib_jax.misc import Dipole, Triangle
from magpylib_jax.sensor import Sensor

__all__ = [
    "Circle",
    "Collection",
    "Cuboid",
    "Cylinder",
    "Dipole",
    "Polyline",
    "Sensor",
    "Sphere",
    "Tetrahedron",
    "Triangle",
    "current",
    "getB",
    "getH",
    "getJ",
    "getM",
    "magnet",
    "misc",
]
