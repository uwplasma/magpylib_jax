"""Force and torque basics with getFT.

Ports "Force and Torque Basics". ``getFT(sources, targets)`` returns the force (N) and torque (N.m)
on each target. Current targets are meshed via ``target.meshing``. We reproduce the minimal
cube-and-loop example and then show how the torque depends on the ``pivot`` point while the force is
invariant.

Unlike Magpylib's finite-difference ``getFT``, magpylib_jax computes the magnet gradient by
autodiff, so there is no ``eps`` step size and the result is exact.
"""

from __future__ import annotations

import numpy as np

import magpylib_jax as magpy


def main() -> dict:
    # Minimal example: force/torque on a current loop near a cubical magnet.
    cube = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.01), polarization=(0.7, 0.7, 0.7))
    loop = magpy.current.Circle(
        diameter=0.02, current=1e3, position=(0.0, 0.0, 0.001), meshing=40
    )
    f, t = magpy.getFT(cube, loop)
    f, t = np.asarray(f), np.asarray(t)
    print(f"Force:  {f} N")
    print(f"Torque: {t} N*m")

    # Pivot dependence: force is invariant, torque picks up a lever-arm term r x F.
    # The two stator coils are passed as a source list and their contributions summed.
    rotor = magpy.magnet.Cuboid(dimension=(0.2, 0.2, 0.2), polarization=(0, 0, 1.0), meshing=64)
    stator = [
        magpy.current.Circle(diameter=0.2, current=1e3, position=(0.1, 0, 0.15)),
        magpy.current.Circle(diameter=0.2, current=-1e3, position=(-0.1, 0, 0.15)),
    ]
    pivots = {"intrinsic": None, "centroid": "centroid", "axle_z=-0.5": (0, 0, -0.5)}
    torques = {}
    force_ref = None
    for label, pivot in pivots.items():
        fp, tp = magpy.getFT(stator, rotor, pivot=pivot)
        fp, tp = np.asarray(fp).sum(axis=0), np.asarray(tp).sum(axis=0)  # sum over sources
        torques[label] = tp
        force_ref = fp
        print(f"pivot {label:12s} -> |F| {np.linalg.norm(fp):7.2f} N, Ty {tp[1]:7.2f} N*m")

    return {"force_loop": f, "torque_loop": t, "torques": torques, "force_rotor": force_ref}


if __name__ == "__main__":
    main()
