"""A gallery of every supported source family and its field at a probe point.

magpylib_jax ships the same twelve source classes as Magpylib. This script builds one of each,
evaluates ``getB`` at a common observer, and prints the result so you can see the whole catalogue
at a glance.
"""

from __future__ import annotations

import numpy as np

import magpylib_jax as magpy

PROBE = (0.03, 0.02, 0.04)


def _sources() -> dict:
    tri_verts = ((-0.01, -0.01, 0.0), (0.01, -0.01, 0.0), (0.0, 0.02, 0.0))
    mesh_pts = np.array([(-1, -1, -1), (1, -1, -1), (0, 1, -1), (0, 0, 1)], dtype=float) / 100
    return {
        "magnet.Cuboid": magpy.magnet.Cuboid(
            polarization=(0, 0, 1.0), dimension=(0.01, 0.01, 0.01)
        ),
        "magnet.Cylinder": magpy.magnet.Cylinder(polarization=(0, 0, 1.0), dimension=(0.01, 0.01)),
        "magnet.CylinderSegment": magpy.magnet.CylinderSegment(
            polarization=(0, 0, 1.0), dimension=(0.004, 0.006, 0.01, 0, 270)
        ),
        "magnet.Sphere": magpy.magnet.Sphere(polarization=(0, 0, 1.0), diameter=0.01),
        "magnet.Tetrahedron": magpy.magnet.Tetrahedron(
            polarization=(0, 0, 1.0), vertices=mesh_pts
        ),
        "magnet.TriangularMesh": magpy.magnet.TriangularMesh.from_ConvexHull(
            polarization=(0, 0, 1.0), points=mesh_pts
        ),
        "current.Circle": magpy.current.Circle(current=100.0, diameter=0.02),
        "current.Polyline": magpy.current.Polyline(
            current=100.0, vertices=[(-0.01, 0, 0), (0.01, 0, 0), (0.01, 0.01, 0)]
        ),
        "misc.Dipole": magpy.misc.Dipole(moment=(0.0, 0.0, 1.0)),
        "misc.Triangle": magpy.misc.Triangle(polarization=(0, 0, 1.0), vertices=tri_verts),
    }


def main() -> dict:
    results = {}
    for name, src in _sources().items():
        b = np.asarray(src.getB(PROBE))
        results[name] = b
        print(f"{name:24s} B = [{b[0]: .3e} {b[1]: .3e} {b[2]: .3e}] T")
    return {"probe": PROBE, "fields": results}


if __name__ == "__main__":
    main()
