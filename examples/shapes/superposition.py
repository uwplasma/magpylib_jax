"""Superposition and cut-out: a ring from two cylinders.

Ports "Superposition". Because magnetostatics without material response is linear, overlapping
magnets add their polarizations. Two concentric cylinders with opposite polarization cancel in the
overlap, leaving a hollow ring -- which we check against the exact ``CylinderSegment`` solution.

Run directly to save the |M| cross-section to ``examples/_output/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import magpylib_jax as magpy


def main() -> dict:
    # Cut-out: outer +z cylinder minus an inner -z cylinder = magnetized ring wall.
    inner = magpy.magnet.Cylinder(polarization=(0, 0, -0.1), dimension=(0.04, 0.05))
    outer = magpy.magnet.Cylinder(polarization=(0, 0, 0.1), dimension=(0.06, 0.05))
    ring = inner + outer

    # Exact reference: a full-annulus CylinderSegment (0..360 deg).
    seg = magpy.magnet.CylinderSegment(
        polarization=(0, 0, 0.1), dimension=(0.02, 0.03, 0.05, 0, 360)
    )

    probe = (0.025, 0.0, 0.0)  # inside the ring wall
    b_ring = np.asarray(ring.getB(probe))
    b_seg = np.asarray(seg.getB(probe))

    m_center = np.linalg.norm(np.asarray(ring.getM((0.0, 0.0, 0.0))))  # hollow -> 0
    m_wall = np.linalg.norm(np.asarray(ring.getM((0.025, 0.0, 0.0))))  # magnetized

    print(f"B superposition ring: {b_ring} T")
    print(f"B CylinderSegment:    {b_seg} T")
    print(f"agreement:            {np.linalg.norm(b_ring - b_seg):.2e} T")
    print(f"|M| hollow center:    {m_center:.3e} A/m")
    print(f"|M| ring wall:        {m_wall:.3e} A/m")

    return {"b_ring": b_ring, "b_seg": b_seg, "ring": ring}


def _plot(ring) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.linspace(-0.04, 0.04, 60)
    x, y = np.meshgrid(ts, ts, indexing="ij")
    grid = np.stack([x, y, np.zeros_like(x)], axis=-1)
    m = np.linalg.norm(np.asarray(ring.getM(grid)), axis=2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(x.T, y.T, m.T, cmap="hot_r", levels=20)
    ax.set(title="|M| in xy-plane (cut-out ring)", xlabel="x", ylabel="y", aspect=1)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "_output" / "superposition.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


if __name__ == "__main__":
    result = main()
    print("saved", _plot(result["ring"]))
