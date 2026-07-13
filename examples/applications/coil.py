"""Air coils and a Helmholtz pair.

Ports "Coils". A coil is modeled as a stack of circular ``Circle`` current loops (or, more
faithfully, a ``Polyline`` spiral). Two coils on a common axis form a Helmholtz pair with a nearly
uniform central field; we map its homogeneity error over a small central region.

Run directly (or via the smoke test) to save the homogeneity map to ``examples/_output/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import magpylib_jax as magpy


def _make_coil() -> magpy.Collection:
    coil = magpy.Collection()
    for z in np.linspace(-1, 1, 5):
        for r in np.linspace(4, 5, 3):
            coil.add(magpy.current.Circle(current=10.0, diameter=2 * r, position=(0, 0, z)))
    return coil


def main() -> dict:
    # Model 1: loop-stack coil. Model 2: a Polyline spiral (single winding path).
    coil = _make_coil()
    ts = np.linspace(-8, 8, 200)
    spiral = magpy.current.Polyline(
        current=100.0,
        vertices=np.c_[5 * np.cos(ts * 2 * np.pi), 5 * np.sin(ts * 2 * np.pi), ts],
    )
    b_spiral = np.asarray(spiral.getB((0.0, 0.0, 0.0)))

    # Helmholtz pair: two coils at +/- 5 on the axis.
    coil.position = (0, 0, 5)
    helmholtz = magpy.Collection(coil, coil.copy(position=(0, 0, -5)))

    ts2 = np.linspace(-3, 3, 20)
    y, z = np.meshgrid(ts2, ts2, indexing="ij")
    grid = np.stack([np.zeros_like(y), y, z], axis=-1)
    b = np.asarray(helmholtz.getB(grid))
    b0 = np.asarray(helmholtz.getB((0.0, 0.0, 0.0)))
    err = np.linalg.norm((b - b0) / np.linalg.norm(b0), axis=2)

    print(f"loops in one coil:    {len(coil.sources)}")
    print(f"spiral B at center:   {b_spiral} T")
    print(f"Helmholtz B0:         {b0} T")
    print(f"max homogeneity err:  {err.max():.2%} over +/-3 region")

    return {"b0": b0, "err_grid": err, "y": y, "z": z}


def _plot(y: np.ndarray, z: np.ndarray, err: np.ndarray) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    cf = ax.contourf(y.T, z.T, err.T * 100, levels=20)
    fig.colorbar(cf, ax=ax, label="% of B0")
    ax.set(title="Helmholtz homogeneity error", xlabel="y", ylabel="z", aspect=1)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "_output" / "coil.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


if __name__ == "__main__":
    result = main()
    print("saved", _plot(result["y"], result["z"], result["err_grid"]))
