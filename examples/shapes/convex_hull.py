"""Convex-hull magnets: a pyramid from a point cloud.

Ports "Convex Hull". ``TriangularMesh.from_ConvexHull`` wraps ``scipy.spatial.ConvexHull`` to build
a closed, correctly oriented magnet from the smallest convex shape enclosing a set of points -- here
the five corners of a pyramid.
"""

from __future__ import annotations

import numpy as np

import magpylib_jax as magpy


def main() -> dict:
    points = np.array([(-2, -2, 0), (-2, 2, 0), (2, -2, 0), (2, 2, 0), (0, 0, 3)]) / 100.0
    pyramid = magpy.magnet.TriangularMesh.from_ConvexHull(
        polarization=(0.0, 0.0, 1.0), points=points
    )

    b_above = np.asarray(pyramid.getB((0.0, 0.0, 0.05)))
    b_inside = np.asarray(pyramid.getB((0.0, 0.0, 0.01)))

    print(f"hull vertices:   {points.shape[0]}")
    print(f"B above apex:    {b_above} T")
    print(f"B inside body:   {b_inside} T  (carries the polarization)")

    return {"b_above": b_above, "b_inside": b_inside}


if __name__ == "__main__":
    main()
