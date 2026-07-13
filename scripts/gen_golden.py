"""Generate golden regression values from the CURRENT (parity-passing) code.

These pin exact public behaviour so the refactor can be proven
behaviour-preserving in seconds without the full magpylib parity suite.
"""

import json

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import magpylib_jax as mpj  # noqa: E402

OBS = np.array(
    [
        [1.2, 0.2, 0.4],
        [-0.7, 0.9, -0.6],
        [0.3, -1.1, 0.8],
        [2.0, 2.0, 2.0],
    ]
)


def make_sources():
    """(name, object) pairs covering every source family + a Collection."""
    srcs = {}
    srcs["cuboid"] = mpj.magnet.Cuboid(
        polarization=(0.15, -0.22, 0.3), dimension=(1.1, 0.7, 1.4), position=(0.2, 0.3, -0.25)
    )
    srcs["cylinder"] = mpj.magnet.Cylinder(
        polarization=(0.12, -0.2, 0.28), dimension=(1.4, 1.2), position=(0.1, -0.2, 0.25)
    )
    srcs["cylinder_segment"] = mpj.magnet.CylinderSegment(
        polarization=(0.1, -0.2, 0.3), dimension=(0.4, 1.2, 1.1, -30.0, 110.0)
    )
    srcs["sphere"] = mpj.magnet.Sphere(
        polarization=(0.2, -0.15, 0.3), diameter=1.3, position=(0.2, 0.1, -0.15)
    )
    srcs["tetrahedron"] = mpj.magnet.Tetrahedron(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        polarization=(0.11, -0.07, 0.13),
    )
    srcs["triangular_mesh"] = mpj.magnet.TriangularMesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
        polarization=(0.1, 0.2, 0.3),
    )
    srcs["circle"] = mpj.current.Circle(current=2.5, diameter=0.9, position=(0.1, 0.2, -0.4))
    srcs["polyline"] = mpj.current.Polyline(
        current=1.7, vertices=[[-0.5, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0.2]]
    )
    srcs["triangle_sheet"] = mpj.current.TriangleSheet(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        current_densities=[[0.7, 0.1, 0.0]],
    )
    srcs["triangle_strip"] = mpj.current.TriangleStrip(
        vertices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], current=1.4
    )
    srcs["dipole"] = mpj.misc.Dipole(moment=(1.2, -0.7, 0.3), position=(0.2, -0.1, 0.5))
    srcs["triangle"] = mpj.misc.Triangle(
        vertices=[[-0.2, -0.1, 0.0], [0.9, 0.3, 0.2], [0.1, 0.8, -0.2]],
        polarization=(0.1, -0.2, 0.3),
    )
    srcs["collection"] = mpj.Collection(srcs["cuboid"].copy(), srcs["dipole"].copy())
    return srcs


def main():
    out = {"observers": OBS.tolist(), "fields": {}, "grads": {}}
    for name, src in make_sources().items():
        rec = {}
        for field in ("B", "H", "J", "M"):
            val = np.asarray(getattr(src, f"get{field}")(OBS))
            rec[field] = val.tolist()
        out["fields"][name] = rec

    # A few gradient anchors (d B_z / d scalar-param) through the object API.
    def cuboid_bz(cx):
        s = mpj.magnet.Cuboid(polarization=(0.15, -0.22, 0.3), dimension=(1.1, 0.7, cx))
        return s.getB(OBS[0])[2]

    def circle_bz(d):
        s = mpj.current.Circle(current=2.5, diameter=d, position=(0.1, 0.2, -0.4))
        return s.getB(OBS[0])[2]

    def dipole_bx(mx):
        s = mpj.misc.Dipole(moment=(mx, -0.7, 0.3), position=(0.2, -0.1, 0.5))
        return s.getB(OBS[0])[0]

    out["grads"]["cuboid_dBz_dc"] = float(jax.grad(cuboid_bz)(1.4))
    out["grads"]["circle_dBz_dd"] = float(jax.grad(circle_bz)(0.9))
    out["grads"]["dipole_dBx_dmx"] = float(jax.grad(dipole_bx)(1.2))
    return out


if __name__ == "__main__":
    data = main()
    with open("tests/data/golden.json", "w") as f:
        json.dump(data, f, indent=1)
    print("golden written:", len(data["fields"]), "sources,", len(data["grads"]), "grads")
