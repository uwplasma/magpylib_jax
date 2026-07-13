"""Forward model of a magnet swept over a sensor.

Ports the portable, offline half of Magpylib's "Modeling a Real Magnet" tutorial: a cylindrical
magnet is moved along a line above a fixed 3D sensor, and the three field components are recorded as
a function of magnet position (the measured-vs-simulated comparison in the original needs an
external data file, which we omit to keep this runnable offline).

Run directly to save the Bx/By/Bz scan to ``examples/_output/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import magpylib_jax as magpy


def main() -> dict:
    # Testbench: magnet centered 7 mm above the sensor, swept along x (all lengths in mm scale).
    xs = np.linspace(-15e-3, 15e-3, 61)
    sensor = magpy.Sensor(position=(0.0, 0.0, 0.0))
    magnet = magpy.magnet.Cylinder(
        polarization=(0.0, 0.0, 1.2),  # ~1200 mT axial remanence
        dimension=(8e-3, 2e-3),
        position=[(x, 0.0, 7e-3) for x in xs],
    )
    bsim = np.asarray(sensor.getB(magnet)) * 1e3  # T -> mT

    print(f"scan points:      {bsim.shape[0]}")
    print(f"peak |Bz|:        {np.abs(bsim[:, 2]).max():.2f} mT at x=0")
    print(f"Bx antisymmetry:  {np.abs(bsim[0, 0] + bsim[-1, 0]):.2e} mT (near 0)")

    return {"xs": xs, "bsim": bsim}


def _plot(xs: np.ndarray, bsim: np.ndarray) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for i, (ax, lab) in enumerate(zip(axes, ["Bx (mT)", "By (mT)", "Bz (mT)"], strict=True)):
        ax.plot(xs * 1e3, bsim[:, i], color="b")
        ax.set(title=lab, xlabel="magnet position (mm)")
        ax.grid(color="0.85")
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "_output" / "magnet_scan.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


if __name__ == "__main__":
    result = main()
    print("saved", _plot(result["xs"], result["bsim"]))
