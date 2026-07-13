"""Differentiable JAX-native magnetic field toolkit."""

import jax as _jax

# magpylib works in SI double precision; enable JAX x64 on import so field
# values and gradients match upstream out of the box. Must run before any
# submodule builds arrays. Users who prefer single precision can set
# ``jax.config.update("jax_enable_x64", False)`` after importing this package.
_jax.config.update("jax_enable_x64", True)

from magpylib_jax import current, magnet, misc
from magpylib_jax.collection import Collection
from magpylib_jax.core.base import MagpylibBadUserInput, MagpylibMissingInput
from magpylib_jax.current import Circle, Polyline, TriangleSheet, TriangleStrip
from magpylib_jax.functional import getB, getFT, getH, getJ, getM
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

__version__ = "1.0.1"

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
    "getFT",
    "getH",
    "getJ",
    "getM",
    "MagpylibBadUserInput",
    "MagpylibMissingInput",
    "magnet",
    "misc",
]
