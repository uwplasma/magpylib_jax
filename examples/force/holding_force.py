"""Magnetic holding force via the method of images.

Ports "Magnetic Holding Force". The pull force needed to detach a magnet from a soft-magnetic plate
equals the attraction between the magnet and its mirror image across the plate surface. We reproduce
the Supermagnete N45 cuboid whose datasheet holding force is ~350 g.
"""

from __future__ import annotations

import numpy as np

import magpylib_jax as magpy


def main() -> dict:
    cube = magpy.magnet.Cuboid(
        dimension=(5e-3, 2.5e-3, 1e-3),
        polarization=(0.0, 0.0, 1.33),  # N45 remanence ~1.32-1.36 T
        meshing=100,
    )
    mirror = cube.copy(position=(0.0, 0.0, 1e-3))  # image across the plate surface

    f, _ = magpy.getFT(mirror, cube)
    fz = float(np.asarray(f)[2])
    holding_g = fz * 100.0  # N -> grams-force (approx, g ~ 9.81)

    print(f"Holding force: {fz:.3e} N")
    print(f"Holding force: {holding_g:.1f} g  (datasheet ~350 g)")

    return {"force_z": fz, "holding_g": holding_g}


if __name__ == "__main__":
    main()
