"""First field computation with the object API.

Ports the opening of Magpylib's "Field Computation (B, H, J, M)" tutorial:

1. the "three lines to a field" minimal call,
2. the four fields B / H / J / M of a diametrically polarized cylinder on a grid, and
3. a multi-source / multi-sensor call whose output carries one index per input axis.

Run directly to also save a 2x2 streamplot of the four fields to ``examples/_output/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import magpylib_jax as magpy


def main() -> dict:
    # 1. The magnetic field is only three lines of code away.
    loop = magpy.current.Circle(current=1.0, diameter=1.0)
    b_center = np.asarray(loop.getB((0.0, 0.0, 0.0)))

    # 2. B/H/J/M of a diametrically polarized cylinder on a symmetry-plane grid.
    cyl = magpy.magnet.Cylinder(polarization=(0.5, 0.5, 0.0), dimension=(0.04, 0.02))
    ts = np.linspace(-0.05, 0.05, 30)
    grid = np.stack(np.meshgrid(ts, ts, [0.0], indexing="ij"), axis=-1).reshape(30, 30, 3)
    fields = {name: np.asarray(getattr(cyl, f"get{name}")(grid)) for name in "BHJM"}

    # 3. Multiple sources and sensors -> field for every combination.
    cube = magpy.magnet.Cuboid(polarization=(0.0, 0.0, 1.0), dimension=(0.1, 0.1, 0.1))
    sens1 = magpy.Sensor(pixel=[(0.0, 0.0, 0.0), (0.005, 0.0, 0.0)])
    sens2 = sens1.copy()
    combo = np.asarray(magpy.getB([cube, cube.copy()], [sens1, sens2]))

    print(f"Loop B at center:           {b_center} T")
    for name, arr in fields.items():
        print(f"|{name}| grid max:              {np.linalg.norm(arr, axis=2).max():.4e}")
    print(f"[2 sources x 2 sensors] B shape: {combo.shape}")

    return {"b_center": b_center, "fields": fields, "combo_shape": combo.shape}


def _plot(fields: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.linspace(-0.05, 0.05, 30)
    x, y = np.meshgrid(ts, ts, indexing="ij")
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    cmaps = {"B": "spring_r", "H": "winter_r", "J": "summer_r", "M": "autumn_r"}
    for ax, (name, arr) in zip(axes.ravel(), fields.items(), strict=True):
        fx, fy = arr[:, :, 0], arr[:, :, 1]
        ax.streamplot(x.T, y.T, fx.T, fy.T, color=np.log(np.linalg.norm(arr, axis=2).T + 1e-30),
                      cmap=cmaps[name])
        ax.set(title=f"{name}-field", aspect=1)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "_output" / "first_field.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


if __name__ == "__main__":
    result = main()
    print("saved", _plot(result["fields"]))
