"""Render a mixed scene with show().

Ports the README/visualization ``show()`` demo. magpylib_jax ships a Matplotlib ``show()`` backend:
magnets render as shaded bodies with a polarization arrow, currents as loops, dipoles as moment
arrows, and sensors as markers with their pixel grid. We build a small scene and save it headlessly
with ``return_fig=True``.

Run directly (or via the smoke test) to save the figure to ``examples/_output/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import magpylib_jax as magpy


def main() -> dict:
    scene = magpy.Collection(
        magpy.magnet.Cuboid(polarization=(0, 0, 1.0), dimension=(1, 1, 1)),
        magpy.current.Circle(diameter=2.0, current=100.0, position=(1.5, 0, -1)),
        magpy.misc.Dipole(moment=(0, 0, 1.0), position=(1.5, 0, 1)),
        magpy.Sensor(position=(0, 0, -1.2), pixel=[[0, 0, 0]]),
    )
    fig = scene.show(return_fig=True)

    out = Path(__file__).resolve().parent.parent / "_output" / "show_scene.png"
    fig.savefig(out, dpi=90)

    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"scene children: {len(scene.children)}")
    print(f"saved {out}")
    return {"n_children": len(scene.children), "path": out}


if __name__ == "__main__":
    main()
