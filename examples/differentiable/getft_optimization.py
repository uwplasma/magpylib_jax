"""Force equilibrium with getFT gradients: a levitation height.

``getFT`` is differentiable, so we can optimize *through* a force computation. A current loop floats
above a fixed dipole magnet; the loop's current is chosen so the vertical force is repulsive and
decays with height. We solve for the equilibrium height where that upward force balances a load,
``F_z(h) = weight``, using the exact derivative ``dF_z/dh`` from ``jax.value_and_grad`` (a Newton
iteration). Finite-difference ``getFT`` cannot supply this derivative directly.

The whole force-and-derivative function is wrapped in a single ``jax.jit`` so it compiles once and
every Newton step reuses it -- the pattern the README recommends for ``getFT``-based losses.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import magpylib_jax as magpy

WEIGHT = 1.5  # load to support (N)
BASE = magpy.misc.Dipole(moment=(0.0, 0.0, 300.0))  # fixed source magnet


def _force_z(height: jnp.ndarray) -> jnp.ndarray:
    """Upward magnetic force on the floating current loop at a given height (m)."""
    loop = magpy.current.Circle(
        diameter=0.04,
        current=-500.0,  # sign chosen so the force is repulsive (upward)
        position=jnp.stack([jnp.asarray(0.0), jnp.asarray(0.0), height]),
        meshing=8,
    )
    return magpy.getFT(BASE, loop)[0][2]


def main() -> dict:
    # One compiled function returns both F_z and its exact derivative dF_z/dh.
    force_and_grad = jax.jit(jax.value_and_grad(_force_z))

    height = jnp.asarray(0.03)
    history = []
    for _ in range(6):
        fz, dfz = force_and_grad(height)
        # Newton step on the equilibrium condition F_z(h) - weight = 0.
        height = jnp.clip(height - (fz - WEIGHT) / dfz, 0.02, 0.15)
        history.append((float(height), float(fz)))

    fz, dfz = force_and_grad(height)
    fz, height = float(fz), float(height)

    print(f"target load (weight):   {WEIGHT:.3f} N")
    print(f"equilibrium height:     {height * 1e3:.2f} mm")
    print(f"force at equilibrium:   {fz:.4f} N")
    print(f"dF_z/dh (stiffness):    {float(dfz):.1f} N/m")
    print(f"residual:               {abs(fz - WEIGHT):.2e} N")

    return {"height": height, "force_z": fz, "weight": WEIGHT, "history": history}


if __name__ == "__main__":
    main()
