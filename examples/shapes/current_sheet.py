"""Current sheets: a Mobius strip and a meshed surface current.

Ports "Current Sheets". ``TriangleStrip`` builds a ribbon from an alternating top/bottom vertex
list (here a Mobius band); ``TriangleSheet`` builds an arbitrary surface current from a triangle
mesh with a per-face current-density vector (here a swirling density on a square patch).

Run directly to save the strip's B-field slice to ``examples/_output/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import magpylib_jax as magpy


def _mobius_vertices(width: float, n: int = 60) -> np.ndarray:
    u = np.linspace(0, 4 * np.pi, n)
    x = (1 + width * np.cos(u / 2)) * np.cos(u)
    y = (1 + width * np.cos(u / 2)) * np.sin(u)
    z = width * np.sin(u / 2)
    return np.stack([x, y, z], axis=1)


def _build_strip(n: int = 60) -> magpy.current.TriangleStrip:
    verts = np.zeros((2 * n, 3))
    verts[::2] = _mobius_vertices(0.0, n)
    verts[1::2] = _mobius_vertices(0.4, n)
    return magpy.current.TriangleStrip(vertices=verts, current=1.0)


def _build_sheet(n: int = 8) -> magpy.current.TriangleSheet:
    ts = np.linspace(-1, 1, n)
    vertices = np.array([(x, y, 0.0) for x in ts for y in ts])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = j + i * n
            faces.append((idx, idx + n, idx + n + 1))
            faces.append((idx, idx + n + 1, idx + 1))
    faces = np.array(faces, dtype=int)
    centers = vertices[faces].mean(axis=1)
    # Swirling (curl-like) surface current density, growing toward the edges.
    cds = np.stack([centers[:, 1], -centers[:, 0], np.zeros(len(centers))], axis=1)
    return magpy.current.TriangleSheet(current_densities=cds, vertices=vertices, faces=faces)


def main() -> dict:
    strip = _build_strip()
    sheet = _build_sheet()

    ts = np.linspace(-2, 2, 24)
    x, z = np.meshgrid(ts, ts, indexing="ij")
    grid = np.stack([x, np.zeros_like(x), z], axis=-1)
    b_strip = np.asarray(strip.getB(grid))
    strip_max = np.linalg.norm(b_strip, axis=2).max()

    b_sheet_axis = np.asarray(sheet.getB((0.0, 0.0, 0.3)))

    print(f"Mobius strip triangles: {strip.vertices.shape[0] - 2}")
    print(f"strip |B| max on slice: {strip_max:.3e} T")
    print(f"sheet B on axis:        {b_sheet_axis} T")

    return {"strip_max": strip_max, "b_sheet_axis": b_sheet_axis, "strip": strip, "slice": b_strip}


def _plot(x: np.ndarray, z: np.ndarray, b_strip: np.ndarray) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    color = np.log10(np.linalg.norm(b_strip, axis=2).T + 1e-30)
    ax.streamplot(x.T, z.T, b_strip[:, :, 0].T, b_strip[:, :, 2].T, density=1.4, color=color,
                  cmap="viridis_r")
    ax.set(title="Mobius current strip B-field", xlabel="x", ylabel="z", aspect=1)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "_output" / "current_sheet.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


if __name__ == "__main__":
    result = main()
    ts = np.linspace(-2, 2, 24)
    gx, gz = np.meshgrid(ts, ts, indexing="ij")
    print("saved", _plot(gx, gz, result["slice"]))
