"""Differentiable JAX-native magnetic field toolkit."""

from magpylib_jax import current, magnet, misc
from magpylib_jax.collection import Collection
from magpylib_jax.core.base import MagpylibBadUserInput, MagpylibMissingInput
from magpylib_jax.current import Circle, Polyline, TriangleSheet, TriangleStrip
from magpylib_jax.functional import getB, getH, getJ, getM
from magpylib_jax.magnet import (
    Cuboid,
    Cylinder,
    CylinderSegment,
    Sphere,
    Tetrahedron,
    TriangularMesh,
)
from magpylib_jax.misc import CustomSource, Dipole, Triangle
from magpylib_jax.sensor import Sensor

__version__ = "1.0.0"

__all__ = [
    "Circle",
    "Collection",
    "Cuboid",
    "Cylinder",
    "CylinderSegment",
    "CustomSource",
    "Dipole",
    "Polyline",
    "Sensor",
    "Sphere",
    "Tetrahedron",
    "Triangle",
    "TriangleSheet",
    "TriangleStrip",
    "TriangularMesh",
    "__version__",
    "current",
    "getB",
    "getH",
    "getJ",
    "getM",
    "MagpylibBadUserInput",
    "MagpylibMissingInput",
    "magnet",
    "misc",
]
