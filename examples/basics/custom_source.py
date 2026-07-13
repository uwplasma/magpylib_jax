"""A user-defined CustomSource: the magnetic monopole.

Ports the "Custom Source" tutorial. A ``CustomSource`` wraps a user field function, letting you drop
any closed-form field into the object API. Here we implement the (unphysical but instructive)
magnetic monopole ``B = Qm * (r - r0) / |r - r0|^3`` and superpose four of them into a quadrupole.

Notes on the magpylib_jax API:
* ``field_func`` takes a single ``observers`` argument and returns B directly (as JAX arrays); this
  is slightly simpler than upstream Magpylib's ``field_func(field, observers)`` form.
* This lightweight ``CustomSource`` evaluates ``field_func`` on the raw (global) observers, i.e. it
  does not apply the object ``position``/``orientation`` to the field. So we bake each pole's
  location straight into its field function and superpose by summing (magnetostatics is linear).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import magpylib_jax as magpy


def _monopole_field(charge: float, center=(0.0, 0.0, 0.0)):
    center = jnp.asarray(center, dtype=jnp.float64)

    def field(observers):
        r = jnp.atleast_2d(jnp.asarray(observers, dtype=jnp.float64)) - center
        return charge * r / jnp.linalg.norm(r, axis=-1, keepdims=True) ** 3

    return field


def main() -> dict:
    # A single positive monopole at the origin.
    mono = magpy.misc.CustomSource(field_func=_monopole_field(1.0))
    b_unit = np.asarray(mono.getB((1.0, 0.0, 0.0)))[0]

    # A quadrupole: two positive and two negative charges, superposed by summation.
    charges = [(1.0, (1, 0, 0)), (1.0, (-1, 0, 0)), (-1.0, (0, 0, 1)), (-1.0, (0, 0, -1))]
    poles = [magpy.misc.CustomSource(field_func=_monopole_field(q, p)) for q, p in charges]
    b_probe = sum(np.asarray(pole.getB((0.5, 0.3, 0.2))) for pole in poles)[0]

    print(f"monopole B at (1,0,0): {b_unit} T  (expect [1,0,0])")
    print(f"quadrupole B at probe: {b_probe} T")

    return {"b_unit": b_unit, "b_probe": b_probe, "poles": poles}


def _plot(poles) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.linspace(-2, 2, 60)
    x, z = np.meshgrid(ts, ts, indexing="ij")
    grid = np.stack([x, np.zeros_like(x), z], axis=-1)
    b = sum(np.asarray(pole.getB(grid)) for pole in poles)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.streamplot(x.T, z.T, b[:, :, 0].T, b[:, :, 2].T, density=1.4, color="k")
    ax.set(title="Quadrupole field", xlabel="x", ylabel="z", aspect=1)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "_output" / "custom_source.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


if __name__ == "__main__":
    result = main()
    print("saved", _plot(result["poles"]))
