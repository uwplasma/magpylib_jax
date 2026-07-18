"""Differentiable JAX-native magnetic field toolkit.

Precision follows your JAX configuration: arrays use JAX's default float dtype
(``float32`` unless you enable x64). For bit-level parity with magpylib (which is
float64), enable double precision **before** using the library::

    import jax
    jax.config.update("jax_enable_x64", True)
    import magpylib_jax as mpj

This package never mutates the global JAX config on import.
"""

from magpylib_jax import current, magnet, misc
from magpylib_jax.collection import Collection
from magpylib_jax.compat import (
    SUPPORTED_PLOTTING_BACKENDS,
    defaults,
    mu_0,
    show_context,
)
from magpylib_jax.core.base import MagpylibBadUserInput, MagpylibMissingInput
from magpylib_jax.current import Circle, Polyline, TriangleSheet, TriangleStrip
from magpylib_jax.display import show
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

__version__ = "3.0.0"

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
    "SUPPORTED_PLOTTING_BACKENDS",
    "defaults",
    "magnet",
    "misc",
    "mu_0",
    "show",
    "show_context",
]
