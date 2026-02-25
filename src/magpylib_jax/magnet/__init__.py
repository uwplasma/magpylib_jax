"""Magnet source objects."""

from magpylib_jax.magnet.cuboid import Cuboid
from magpylib_jax.magnet.cylinder import Cylinder
from magpylib_jax.magnet.cylinder_segment import CylinderSegment
from magpylib_jax.magnet.sphere import Sphere
from magpylib_jax.magnet.tetrahedron import Tetrahedron
from magpylib_jax.magnet.triangular_mesh import TriangularMesh

__all__ = ["Cuboid", "Cylinder", "CylinderSegment", "Sphere", "Tetrahedron", "TriangularMesh"]
