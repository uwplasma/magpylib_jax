"""magpylib vs magpylib_jax performance comparison plots (honest, CPU).

Generates ``docs/_static/perf_hero.png`` — a multi-panel comparison used at the top
of the README — and prints the numbers (including an optimization-loop wall-clock
comparison) so the README text can quote real figures.

Run: ``python scripts/make_benchmark_plots.py``  (needs matplotlib + magpylib).
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

import magpylib as magpy  # noqa: E402

import magpylib_jax as mpj  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "_static"
OUT.mkdir(parents=True, exist_ok=True)
C_JAX = "#2f6fd0"
C_REF = "#e8833a"
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10.5,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
RNG = np.random.default_rng(0)


def _median_ms(fn, repeats=7):
    jax.block_until_ready(fn())  # warm / compile
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())  # blocks on every array in the returned pytree
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def bench_forward(sizes):
    """Batched forward getB, magpylib vs magpylib_jax (eager object API, warm)."""
    jx, rf = [], []
    kw = dict(polarization=(0.1, 0.2, 0.3), dimension=(1.0, 1.0, 1.0))
    for n in sizes:
        obs = RNG.normal(size=(n, 3)) + np.array([1.5, 0, 0])
        oj = jnp.asarray(obs)
        sj = mpj.magnet.Cuboid(**kw)
        sr = magpy.magnet.Cuboid(**kw)
        jx.append(_median_ms(lambda oj=oj, sj=sj: sj.getB(oj)))
        rf.append(_median_ms(lambda obs=obs, sr=sr: sr.getB(obs)))
    return np.array(jx), np.array(rf)


def bench_grad(sizes):
    """Cost of field + gradient w.r.t. polarization: magpylib FD vs autodiff."""
    jx, rf = [], []
    for n in sizes:
        obs = RNG.normal(size=(n, 3)) + np.array([1.5, 0, 0])
        oj = jnp.asarray(obs)

        def scalar_j(pol, oj=oj):
            src = mpj.magnet.Cuboid(polarization=pol, dimension=(1.0, 1.0, 1.0))
            return jnp.sum(src.getB(oj) ** 2)

        vg = jax.jit(jax.value_and_grad(scalar_j))
        p0 = jnp.array([0.1, 0.2, 0.3])
        jx.append(_median_ms(lambda vg=vg, p0=p0: vg(p0)))

        def scalar_r(pol, obs=obs):
            src = magpy.magnet.Cuboid(polarization=tuple(pol), dimension=(1.0, 1.0, 1.0))
            return float(np.sum(src.getB(obs) ** 2))

        def fd(scalar_r=scalar_r):
            p = np.array([0.1, 0.2, 0.3])
            g = np.zeros(3)
            h = 1e-6
            for i in range(3):
                pp = p.copy()
                pp[i] += h
                pm = p.copy()
                pm[i] -= h
                g[i] = (scalar_r(pp) - scalar_r(pm)) / (2 * h)
            return g
        rf.append(_median_ms(fd, repeats=3))
    return np.array(jx), np.array(rf)


def bench_assembly(counts):
    """Field of a K-magnet assembly over 20k observers, vs magpylib."""
    obs = RNG.normal(size=(20000, 3)) + np.array([2.5, 0, 0])
    oj = jnp.asarray(obs)
    jx, rf = [], []
    for k in counts:
        ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
        pos = np.stack([np.cos(ang), np.sin(ang), np.zeros(k)], axis=1)
        pol = np.stack([np.cos(2 * ang), np.sin(2 * ang), np.zeros(k)], axis=1)
        cj = mpj.Collection(*[
            mpj.magnet.Cuboid(polarization=tuple(pol[i]), dimension=(0.4, 0.4, 0.4),
                              position=tuple(pos[i])) for i in range(k)])
        cr = magpy.Collection(*[
            magpy.magnet.Cuboid(polarization=tuple(pol[i]), dimension=(0.4, 0.4, 0.4),
                                position=tuple(pos[i])) for i in range(k)])
        jx.append(_median_ms(lambda oj=oj, cj=cj: cj.getB(oj)))
        rf.append(_median_ms(lambda obs=obs, cr=cr: cr.getB(obs)))
    return np.array(jx), np.array(rf)


def bench_sweep(counts):
    """Parameter sweep: field for K geometries at 2k observers. vmap vs Python loop."""
    obs = RNG.normal(size=(2000, 3)) + np.array([1.5, 0, 0])
    oj = jnp.asarray(obs)
    jx, rf = [], []
    for k in counts:
        dims = jnp.asarray(RNG.uniform(0.5, 1.5, size=(k, 3)))

        def sweep_j(dims=dims, oj=oj):
            def f(d):
                return mpj.magnet.Cuboid(polarization=(0, 0, 1.0), dimension=d).getB(oj)
            return jax.vmap(f)(dims)

        sweep_j = jax.jit(sweep_j)
        jx.append(_median_ms(sweep_j))
        dims_np = np.asarray(dims)

        def sweep_r(dims_np=dims_np, obs=obs):
            return np.stack([
                magpy.magnet.Cuboid(polarization=(0, 0, 1.0), dimension=tuple(d)).getB(obs)
                for d in dims_np])
        rf.append(_median_ms(sweep_r, repeats=3))
    return np.array(jx), np.array(rf)


def main():
    fsz = [1000, 10000, 100000, 1000000]
    fj, fr = bench_forward(fsz)
    gsz = [1000, 10000, 100000]
    gj, gr = bench_grad(gsz)
    acnt = [2, 8, 32]
    aj, ar = bench_assembly(acnt)
    scnt = [10, 50, 200]
    sj, sr = bench_sweep(scnt)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2))
    dev = jax.default_backend().upper()
    fig.suptitle(f"magpylib_jax vs magpylib  ·  {dev}  ·  float64", fontsize=13, y=0.98)

    def bars(ax, xs, jx, rf, xlabel, title, log=True):
        x = np.arange(len(xs))
        w = 0.38
        ax.bar(x - w / 2, rf, w, label="magpylib", color=C_REF)
        ax.bar(x + w / 2, jx, w, label="magpylib_jax", color=C_JAX)
        ax.set(xticks=x, xticklabels=[f"{v:,}" for v in xs], xlabel=xlabel, ylabel="time (ms)",
               title=title)
        if log:
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        speedup = float(np.max(np.asarray(rf) / np.asarray(jx)))
        ax.text(0.97, 0.93, f"up to {speedup:.0f}× faster", transform=ax.transAxes,
                ha="right", color=C_JAX, fontweight="bold", fontsize=11)

    bars(axes[0, 0], fsz, fj, fr, "observers", "Field  (getB)")
    bars(axes[0, 1], gsz, gj, gr, "observers", "Field + gradient")
    bars(axes[1, 0], acnt, aj, ar, "magnets in assembly", "Assembly field  (Collection.getB)")
    bars(axes[1, 1], scnt, sj, sr, "geometries swept", "Parameter sweep  (vmap vs loop)")

    fig.text(0.5, 0.012,
             "Warmed, compiled steady-state runtime.  Field + gradient: magpylib uses 6-point "
             "finite differences (approximate); magpylib_jax uses exact autodiff.",
             ha="center", fontsize=8.2, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT / "perf_hero.png")
    plt.close(fig)

    print("wrote perf_hero.png")
    print("forward (ms) jax:", [round(v, 2) for v in fj], "ref:", [round(v, 2) for v in fr])
    print("field+grad (ms) jax:", [round(v, 2) for v in gj], "ref:", [round(v, 2) for v in gr])
    print("assembly (ms) jax:", [round(v, 2) for v in aj], "ref:", [round(v, 2) for v in ar])
    print("sweep (ms) jax:", [round(v, 2) for v in sj], "ref:", [round(v, 2) for v in sr])


if __name__ == "__main__":
    main()
