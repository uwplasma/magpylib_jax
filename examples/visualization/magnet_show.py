"""Render several magnet types together with show().

Adapts "Magnet Colors": magpylib_jax renders magnet polarization with a shaded body and arrow. We
place a sphere, a thin cuboid, and a cylinder segment in one scene and save the Matplotlib render.
Interactive plotly/pyvista color-scheme styling from upstream is not shipped, so this focuses on the
static ``show()``.

Run directly (or via the smoke test) to save the figure to ``examples/_output/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import magpylib_jax as magpy


def main() -> dict:
    sphere = magpy.magnet.Sphere(polarization=(1, 1, 1), diameter=1.0, position=(-1.5, 1, 0))
    cube = magpy.magnet.Cuboid(
        polarization=(1, 0, 0), dimension=(1, 0.3, 0.3), position=(1.5, 1, 0)
    )
    cyl = magpy.magnet.CylinderSegment(polarization=(1, 0, 0), dimension=(1.7, 2.0, 0.3, -145, -35))

    fig = magpy.show(sphere, cube, cyl, return_fig=True)
    out = Path(__file__).resolve().parent.parent / "_output" / "magnet_show.png"
    fig.savefig(out, dpi=90)

    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"rendered {3} magnets -> {out}")
    return {"path": out}


if __name__ == "__main__":
    main()
