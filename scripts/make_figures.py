"""Generate the figures used in the README and docs.

Outputs PNGs into ``docs/_static/``:
- ``field_map.png``       B-field streamlines around a cuboid magnet.
- ``optimization.png``    Differentiable inverse-design: fit a dipole to a target field.
- ``benchmark.png``       Batched getB runtime vs magpylib, and gradient timing.
- ``force_distance.png``  Autodiff getFT force between two magnets vs separation.

Run: ``python scripts/make_figures.py``  (needs matplotlib + magpylib).
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import magpylib_jax as mpj

OUT = Path(__file__).resolve().parent.parent / "docs" / "_static"
OUT.mkdir(parents=True, exist_ok=True)

# Consistent, theme-neutral palette.
C_JAX = "#3b7dd8"      # magpylib_jax
C_REF = "#e08a1e"      # magpylib
C_ACCENT = "#d1495b"
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_field_map() -> None:
    """Streamlines of B in the x-z plane around a cuboid magnet."""
    src = mpj.magnet.Cuboid(polarization=(0.0, 0.0, 1.0), dimension=(1.0, 1.0, 1.0))
    n = 60
    xs = np.linspace(-2.5, 2.5, n)
    zs = np.linspace(-2.5, 2.5, n)
    X, Z = np.meshgrid(xs, zs)
    pts = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=1)
    B = np.asarray(src.getB(pts)).reshape(n, n, 3)
    Bx, Bz = B[..., 0], B[..., 2]
    mag = np.log10(np.linalg.norm(B[..., ::2], axis=-1) + 1e-6)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    strm = ax.streamplot(X, Z, Bx, Bz, color=mag, cmap="viridis", density=1.4, linewidth=0.8)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1.0, 1.0, color="#333", alpha=0.85, zorder=5))
    ax.set(xlabel="x (m)", ylabel="z (m)", title="B-field of a cuboid magnet (x–z plane)")
    ax.set_aspect("equal")
    fig.colorbar(strm.lines, ax=ax, label=r"$\log_{10}\,|B|$ (T)")
    fig.tight_layout()
    fig.savefig(OUT / "field_map.png")
    plt.close(fig)
    print("wrote field_map.png")


def fig_optimization() -> None:
    """Differentiable inverse design: recover a dipole moment from field samples."""
    obs = jnp.asarray(np.random.default_rng(0).normal(scale=0.8, size=(40, 3)) + np.array([2.0, 0, 0]))
    true_moment = jnp.array([0.5, -0.3, 0.8])

    def field(moment):
        return mpj.misc.Dipole(moment=moment).getB(obs)

    target = field(true_moment)

    def loss(moment):
        return jnp.mean((field(moment) - target) ** 2)

    grad = jax.jit(jax.grad(loss))
    m = jnp.array([0.0, 0.0, 0.0])
    hist = [float(loss(m))]
    lr = 5.0
    for _ in range(60):
        m = m - lr * grad(m)
        hist.append(float(loss(m)))

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.semilogy(hist, color=C_JAX, lw=2)
    ax.set(xlabel="gradient step", ylabel="MSE loss (T$^2$)",
           title="Inverse design via jax.grad\n(recovering a dipole moment)")
    rec = ", ".join(f"{v:.2f}" for v in np.asarray(m))
    tru = ", ".join(f"{v:.2f}" for v in np.asarray(true_moment))
    ax.text(0.96, 0.94, f"true:  ({tru})\nfit:   ({rec})", transform=ax.transAxes,
            ha="right", va="top", family="monospace", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    fig.tight_layout()
    fig.savefig(OUT / "optimization.png")
    plt.close(fig)
    print("wrote optimization.png")


def _time_call(fn, repeats=5):
    fn()  # warmup / compile
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def fig_benchmark() -> None:
    """Honest CPU comparison: forward throughput, and field+gradient cost.

    Left: pure forward getB on CPU — magpylib's NumPy core is faster here; JAX's
    XLA-CPU dispatch carries overhead (JAX's real throughput win is on GPU/TPU).
    Right: the cost of a field *and* its gradient w.r.t. the 3 polarization
    components. magpylib has no autodiff, so a gradient needs central finite
    differences = 6 extra forward evals (and is only approximate); magpylib_jax
    gets the exact gradient from one reverse pass via value_and_grad.
    """
    import magpylib as magpy

    sizes = [100, 1000, 10000, 100000]
    t_jax, t_ref = [], []
    for n in sizes:
        obs = np.random.default_rng(1).normal(size=(n, 3)) + np.array([1.5, 0, 0])
        obs_j = jnp.asarray(obs)
        src_j = mpj.magnet.Cuboid(polarization=(0.1, 0.2, 0.3), dimension=(1.0, 1.0, 1.0))
        src_r = magpy.magnet.Cuboid(polarization=(0.1, 0.2, 0.3), dimension=(1.0, 1.0, 1.0))
        getb_j = jax.jit(lambda o, s=src_j: s.getB(o))
        t_jax.append(_time_call(lambda o=obs_j, f=getb_j: jax.block_until_ready(f(o))))
        t_ref.append(_time_call(lambda o=obs, s=src_r: s.getB(o)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    x = np.arange(len(sizes))
    w = 0.38
    ax1.bar(x - w / 2, np.array(t_ref) * 1e3, w, label="magpylib (NumPy)", color=C_REF)
    ax1.bar(x + w / 2, np.array(t_jax) * 1e3, w, label="magpylib_jax (XLA-CPU)", color=C_JAX)
    ax1.set(xticks=x, xticklabels=[f"{s:,}" for s in sizes], xlabel="observers",
            ylabel="getB time (ms)", title="Forward field, CPU")
    ax1.set_yscale("log")
    ax1.legend(frameon=False, fontsize=9)

    # Right: cost of field + gradient w.r.t. polarization (3 params).
    gsizes = [100, 1000, 10000, 100000]
    t_grad_jax, t_grad_fd = [], []
    for n in gsizes:
        obs = np.random.default_rng(2).normal(size=(n, 3)) + np.array([1.5, 0, 0])
        obs_j = jnp.asarray(obs)

        def scalar_jax(pol, o=obs_j):
            return jnp.sum(mpj.magnet.Cuboid(polarization=pol, dimension=(1.0, 1.0, 1.0)).getB(o) ** 2)

        vg = jax.jit(jax.value_and_grad(scalar_jax))
        t_grad_jax.append(_time_call(lambda: jax.block_until_ready(vg(jnp.array([0.1, 0.2, 0.3])))))

        def scalar_ref(pol, o=obs):
            return float(np.sum(magpy.magnet.Cuboid(polarization=tuple(pol), dimension=(1., 1., 1.)).getB(o) ** 2))

        def fd_grad():
            p0 = np.array([0.1, 0.2, 0.3])
            g = np.zeros(3)
            h = 1e-6
            for i in range(3):
                pp = p0.copy(); pp[i] += h
                pm = p0.copy(); pm[i] -= h
                g[i] = (scalar_ref(pp) - scalar_ref(pm)) / (2 * h)
            return g
        t_grad_fd.append(_time_call(fd_grad, repeats=3))

    xg = np.arange(len(gsizes))
    ax2.bar(xg - w / 2, np.array(t_grad_fd) * 1e3, w, label="magpylib (6× FD, approx.)", color=C_REF)
    ax2.bar(xg + w / 2, np.array(t_grad_jax) * 1e3, w, label="magpylib_jax (autodiff, exact)", color=C_JAX)
    ax2.set(xticks=xg, xticklabels=[f"{s:,}" for s in gsizes], xlabel="observers",
            ylabel="field + ∇ time (ms)", title="Field + gradient w.r.t. polarization")
    ax2.set_yscale("log")
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "benchmark.png")
    plt.close(fig)
    print("benchmark fwd jax(ms)=", [round(t * 1e3, 2) for t in t_jax],
          "ref(ms)=", [round(t * 1e3, 2) for t in t_ref])
    print("benchmark grad jax(ms)=", [round(t * 1e3, 2) for t in t_grad_jax],
          "fd(ms)=", [round(t * 1e3, 2) for t in t_grad_fd])


def fig_force_distance() -> None:
    """Autodiff getFT: force between two axially-aligned dipoles vs separation."""
    from magpylib_jax import getFT

    src = mpj.misc.Dipole(moment=(0.0, 0.0, 1.0))
    ds = np.linspace(0.5, 3.0, 25)
    fz = []
    for d in ds:
        tgt = mpj.misc.Dipole(moment=(0.0, 0.0, 1.0), position=(0.0, 0.0, float(d)))
        F, _T = getFT(src, tgt)
        fz.append(float(np.asarray(F)[2]))
    # analytic coaxial dipole force: F_z = -3 mu0 m1 m2 / (2 pi d^4)
    mu0 = 4e-7 * np.pi
    fz_ana = -3 * mu0 * 1.0 * 1.0 / (2 * np.pi * ds ** 4)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(ds, fz_ana, "-", color="#888", lw=4, alpha=0.6, label="analytic")
    ax.plot(ds, fz, "o", color=C_JAX, ms=5, label="getFT (autodiff)")
    ax.set(xlabel="separation d (m)", ylabel="$F_z$ (N)",
           title="Force between coaxial dipoles\n(exact autodiff getFT)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "force_distance.png")
    plt.close(fig)
    print("wrote force_distance.png")


if __name__ == "__main__":
    fig_field_map()
    fig_optimization()
    fig_force_distance()
    fig_benchmark()
    print("all figures written to", OUT)
