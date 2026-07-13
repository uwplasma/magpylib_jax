"""Triangular meshes: a cuboctahedron magnet two ways.

Ports "Triangular Meshes". The field of a homogeneously polarized body equals that of its charged
hull. We build a cuboctahedron first as a ``Collection`` of ``misc.Triangle`` faces, then as a
single ``magnet.TriangularMesh`` from the same vertices/faces, and check that the two agree.
"""

from __future__ import annotations

import numpy as np

import magpylib_jax as magpy

POLARIZATION = (0.1, 0.2, 0.3)
PROBE = (0.03, 0.01, 0.02)


def main() -> dict:
    triangles_cm = [
        ([0, 1, -1], [-1, 1, 0], [1, 1, 0]),
        ([0, 1, 1], [1, 1, 0], [-1, 1, 0]),
        ([0, 1, 1], [-1, 0, 1], [0, -1, 1]),
        ([0, 1, 1], [0, -1, 1], [1, 0, 1]),
        ([0, 1, -1], [1, 0, -1], [0, -1, -1]),
        ([0, 1, -1], [0, -1, -1], [-1, 0, -1]),
        ([0, -1, 1], [-1, -1, 0], [1, -1, 0]),
        ([0, -1, -1], [1, -1, 0], [-1, -1, 0]),
        ([-1, 1, 0], [-1, 0, -1], [-1, 0, 1]),
        ([-1, -1, 0], [-1, 0, 1], [-1, 0, -1]),
        ([1, 1, 0], [1, 0, 1], [1, 0, -1]),
        ([1, -1, 0], [1, 0, -1], [1, 0, 1]),
        ([0, 1, 1], [-1, 1, 0], [-1, 0, 1]),
        ([0, 1, 1], [1, 0, 1], [1, 1, 0]),
        ([0, 1, -1], [-1, 0, -1], [-1, 1, 0]),
        ([0, 1, -1], [1, 1, 0], [1, 0, -1]),
        ([0, -1, -1], [-1, -1, 0], [-1, 0, -1]),
        ([0, -1, -1], [1, 0, -1], [1, -1, 0]),
        ([0, -1, 1], [-1, 0, 1], [-1, -1, 0]),
        ([0, -1, 1], [1, -1, 0], [1, 0, 1]),
    ]
    triangles = np.array(triangles_cm) / 100.0  # cm -> m

    # Version 1: a collection of charged triangle faces.
    faces_coll = magpy.Collection(
        *[magpy.misc.Triangle(polarization=POLARIZATION, vertices=t) for t in triangles]
    )
    b_faces = np.asarray(faces_coll.getB(PROBE))

    # Version 2: the same body as a single validated TriangularMesh (built from the faces).
    mesh = magpy.magnet.TriangularMesh.from_ConvexHull(
        polarization=POLARIZATION, points=triangles.reshape(-1, 3)
    )
    b_mesh = np.asarray(mesh.getB(PROBE))

    print(f"triangle faces:       {len(faces_coll.sources)}")
    print(f"B (triangle faces):   {b_faces} T")
    print(f"B (TriangularMesh):   {b_mesh} T")
    print(f"agreement:            {np.linalg.norm(b_faces - b_mesh):.2e} T")

    return {"b_faces": b_faces, "b_mesh": b_mesh}


if __name__ == "__main__":
    main()
