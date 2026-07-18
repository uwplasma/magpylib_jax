"""Core differentiable magnetic-field kernels.

Besides the JAX-native lowercase kernels, this namespace exposes
magpylib-compatible capitalized aliases (``magnet_cuboid_Bfield`` etc.) that map
onto the same underlying kernels, so ``magpylib_jax.core`` mirrors
``magpylib.core``. The capitalized names are plain references to the lowercase
kernels; they share the exact same call signature.
"""

from magpylib_jax.core.kernels import (
    current_circle_hfield,
    current_polyline_hfield,
    current_triangle_sheet_hfield,
    current_trisheet_hfield,
    current_tristrip_hfield,
    dipole_hfield,
    magnet_cuboid_bfield,
    magnet_cuboid_hfield,
    magnet_cylinder_axial_bfield,
    magnet_cylinder_bfield,
    magnet_cylinder_diametral_hfield,
    magnet_cylinder_hfield,
    magnet_cylinder_segment_bfield,
    magnet_cylinder_segment_hfield,
    magnet_sphere_bfield,
    magnet_sphere_hfield,
    magnet_trimesh_bfield,
    magnet_trimesh_hfield,
    tetrahedron_bfield,
    tetrahedron_hfield,
    triangle_bfield,
    triangle_hfield,
)

# --- magpylib-compatible capitalized aliases (same call signature) -----------
magnet_cuboid_Bfield = magnet_cuboid_bfield
magnet_sphere_Bfield = magnet_sphere_bfield
dipole_Hfield = dipole_hfield
current_circle_Hfield = current_circle_hfield
current_polyline_Hfield = current_polyline_hfield
triangle_Bfield = triangle_bfield
magnet_cylinder_axial_Bfield = magnet_cylinder_axial_bfield
magnet_cylinder_diametral_Hfield = magnet_cylinder_diametral_hfield
magnet_cylinder_segment_Hfield = magnet_cylinder_segment_hfield
current_sheet_Hfield = current_triangle_sheet_hfield

__all__ = [
    # lowercase JAX-native kernels
    "current_circle_hfield",
    "dipole_hfield",
    "magnet_cuboid_bfield",
    "magnet_cuboid_hfield",
    "magnet_cylinder_bfield",
    "magnet_cylinder_hfield",
    "magnet_cylinder_axial_bfield",
    "magnet_cylinder_diametral_hfield",
    "magnet_sphere_bfield",
    "magnet_sphere_hfield",
    "current_polyline_hfield",
    "current_triangle_sheet_hfield",
    "current_trisheet_hfield",
    "current_tristrip_hfield",
    "magnet_cylinder_segment_bfield",
    "magnet_cylinder_segment_hfield",
    "magnet_trimesh_bfield",
    "magnet_trimesh_hfield",
    "triangle_bfield",
    "triangle_hfield",
    "tetrahedron_bfield",
    "tetrahedron_hfield",
    # magpylib-compatible capitalized aliases
    "magnet_cuboid_Bfield",
    "magnet_sphere_Bfield",
    "dipole_Hfield",
    "current_circle_Hfield",
    "current_polyline_Hfield",
    "triangle_Bfield",
    "magnet_cylinder_axial_Bfield",
    "magnet_cylinder_diametral_Hfield",
    "magnet_cylinder_segment_Hfield",
    "current_sheet_Hfield",
]
