"""getFT convergence: meshed current loop vs. exact dipole force.

Adapts the convergence study from "Force and Torque Basics". By reciprocity the force between a
dipole and a current loop can be computed exactly (loop as a single source acting on the dipole) or
numerically (dipole as a source acting on the meshed loop). The numerical result converges to the
exact one as ``meshing`` increases, with opposite sign.
"""

from __future__ import annotations

import numpy as np

import magpylib_jax as magpy


def main() -> dict:
    dipole = magpy.misc.Dipole(moment=(0.0, 0.0, 1e6), position=(1.0, 1.0, 1.0))
    loop = magpy.current.Circle(diameter=1.0, current=1.0)

    # Exact: loop is the source, dipole (single cell) is the target.
    f_exact, _ = magpy.getFT(loop, dipole, pivot=(0, 0, 0))
    f_exact = np.asarray(f_exact)

    print(f"exact |F|: {np.linalg.norm(f_exact):.4e} N")
    errors = {}
    for meshing in (8, 40, 200):
        loop.meshing = meshing
        f_num, _ = magpy.getFT(dipole, loop, pivot=(0, 0, 0))  # opposite sign
        err = np.linalg.norm(np.asarray(f_num) + f_exact) / np.linalg.norm(f_exact)
        errors[meshing] = float(err)
        print(f"meshing {meshing:>4}: relative force error {err:.2e}")

    return {"f_exact": f_exact, "errors": errors}


if __name__ == "__main__":
    main()
