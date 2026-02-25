"""Core differentiable magnetic-field kernels."""

from magpylib_jax.core.kernels import (
    current_circle_hfield,
    dipole_hfield,
    magnet_cuboid_bfield,
    magnet_cuboid_hfield,
    magnet_cylinder_bfield,
    magnet_cylinder_hfield,
)

__all__ = [
    "current_circle_hfield",
    "dipole_hfield",
    "magnet_cuboid_bfield",
    "magnet_cuboid_hfield",
    "magnet_cylinder_bfield",
    "magnet_cylinder_hfield",
]
