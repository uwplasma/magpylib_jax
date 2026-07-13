"""Matplotlib streamplot of a cuboid magnet's field slice.

Ports "Matplotlib Streamplot". Streamlines show two components of the field in a plane, with the
amplitude encoded as color and line width. Here we slice a cuboid magnet in the x-z plane.

Run directly (or via the smoke test) to save the figure to ``examples/_output/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import magpylib_jax as magpy


def main() -> dict:
    cube = magpy.magnet.Cuboid(polarization=(0.5, 0.0, 0.5), dimension=(0.02, 0.02, 0.02))
    ts = np.linspace(-0.05, 0.05, 40)
    x, z = np.meshgrid(ts, ts, indexing="ij")
    grid = np.stack([x, np.zeros_like(x), z], axis=-1)
    b = np.asarray(cube.getB(grid))
    norm = np.linalg.norm(b, axis=2)

    print(f"grid points:  {b.shape[0] * b.shape[1]}")
    print(f"|B| range:    {norm.min():.3e} .. {norm.max():.3e} T")

    _plot(x, z, b, norm)
    return {"b": b, "norm_max": float(norm.max())}


def _plot(x: np.ndarray, z: np.ndarray, b: np.ndarray, norm: np.ndarray) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_norm = np.log10(norm.T + 1e-30)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    strm = ax.streamplot(x.T, z.T, b[:, :, 0].T, b[:, :, 2].T, density=1.5,
                         color=log_norm, linewidth=1.0, cmap="autumn")
    fig.colorbar(strm.lines, ax=ax, label="log10 |B|")
    ax.plot([0.01, 0.01, -0.01, -0.01, 0.01], [0.01, -0.01, -0.01, 0.01, 0.01], "k--")
    ax.set(xlabel="x (m)", ylabel="z (m)", aspect=1)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "_output" / "streamplot_field.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
