"""A discrete Halbach cylinder from rotated cuboids.

Ports "Halbach Magnets". A discrete Halbach array approximates the ideal Halbach magnetization
(polarization rotating twice per turn) with a ring of cuboids: each cube is placed on a circle and
its polarization is rotated by twice its angular position, concentrating the field on one side.

Run directly (or via the smoke test) to save the field slice to ``examples/_output/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import magpylib_jax as magpy


def _make_halbach(n: int = 10) -> magpy.Collection:
    halbach = magpy.Collection()
    for a in np.linspace(0, 360, n, endpoint=False):
        cube = magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(1, 0, 0), position=(2.3, 0, 0)
        )
        cube.rotate_from_angax(a, "z", anchor=0)  # move around the ring
        cube.rotate_from_angax(a, "z")            # extra self-rotation -> Halbach pattern
        halbach.add(cube)
    return halbach


def main() -> dict:
    halbach = _make_halbach()
    ts = np.linspace(-3.5, 3.5, 60)
    x, y = np.meshgrid(ts, ts, indexing="ij")
    grid = np.stack([x, y, np.zeros_like(x)], axis=-1)
    b = np.asarray(halbach.getB(grid))
    amp = np.linalg.norm(b, axis=2)

    # Halbach signature: strong field on one side of the bore, weak on the other.
    b_in = np.linalg.norm(np.asarray(halbach.getB((0.0, 0.0, 0.0))))
    b_out = np.linalg.norm(np.asarray(halbach.getB((3.4, 0.0, 0.0))))

    print(f"cubes in ring:   {len(halbach.sources)}")
    print(f"|B| at center:   {b_in:.3e}")
    print(f"|B| outside:     {b_out:.3e}")

    return {"amp": amp, "x": x, "y": y, "b_center": b_in, "b_outside": b_out}


def _plot(x: np.ndarray, y: np.ndarray, b: np.ndarray, amp: np.ndarray) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    cf = ax.contourf(x.T, y.T, amp.T, levels=40, cmap="coolwarm")
    ax.streamplot(x.T, y.T, b[:, :, 0].T, b[:, :, 1].T, color="k", density=1.4, linewidth=0.8)
    fig.colorbar(cf, ax=ax, label="|B|")
    ax.set(title="Discrete Halbach field", xlabel="x", ylabel="y", aspect=1)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "_output" / "halbach.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


if __name__ == "__main__":
    result = main()
    halb = _make_halbach()
    ts = np.linspace(-3.5, 3.5, 60)
    gx, gy = np.meshgrid(ts, ts, indexing="ij")
    grid = np.stack([gx, gy, np.zeros_like(gx)], axis=-1)
    bfield = np.asarray(halb.getB(grid))
    print("saved", _plot(gx, gy, bfield, result["amp"]))
