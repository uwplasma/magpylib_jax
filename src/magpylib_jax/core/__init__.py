"""Core differentiable magnetic-field kernels."""

from magpylib_jax.core.kernels import (
    current_circle_hfield,
    dipole_hfield,
    magnet_cuboid_bfield,
    magnet_cuboid_hfield,
    magnet_cylinder_bfield,
    magnet_cylinder_hfield,
)
from magpylib_jax.core.kernels_extended import (
    current_polyline_hfield,
    magnet_sphere_bfield,
    magnet_sphere_hfield,
    tetrahedron_bfield,
    tetrahedron_hfield,
    triangle_bfield,
    triangle_hfield,
)

__all__ = [
    "current_circle_hfield",
    "dipole_hfield",
    "magnet_cuboid_bfield",
    "magnet_cuboid_hfield",
    "magnet_cylinder_bfield",
    "magnet_cylinder_hfield",
    "magnet_sphere_bfield",
    "magnet_sphere_hfield",
    "current_polyline_hfield",
    "triangle_bfield",
    "triangle_hfield",
    "tetrahedron_bfield",
    "tetrahedron_hfield",
]
