"""Generate the animated figures (movies) used in the docs.

Outputs compressed MP4s into ``docs/_static/``:
- ``movie_field.mp4``         B-field streamlines of a cuboid magnet as it rotates.
- ``movie_optimization.mp4``  Differentiable inverse design converging to a target field.

Run: ``python scripts/make_movies.py``  (needs matplotlib + ffmpeg).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

import magpylib_jax as mpj  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "_static"
OUT.mkdir(parents=True, exist_ok=True)
C_JAX = "#2f6fd0"
plt.rcParams.update({"figure.dpi": 90, "font.size": 10})

# Small, web-friendly encode: H.264, downscaled, moderate CRF.
_WRITER = animation.FFMpegWriter(
    fps=20,
    codec="libx264",
    bitrate=-1,
    # yuv420p needs even width/height; pad odd figure sizes up to the next even pixel.
    extra_args=[
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p", "-crf", "30", "-preset", "veryfast",
    ],
)


def movie_field() -> None:
    """Streamlines of B in the x–z plane while a cuboid magnet rotates about y."""
    n = 45
    xs = np.linspace(-2.5, 2.5, n)
    zs = np.linspace(-2.5, 2.5, n)
    X, Z = np.meshgrid(xs, zs)
    pts = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=1)
    angles = np.linspace(0, 180, 36)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))

    def frame(k: int):
        ax.clear()
        ang = float(angles[k])
        src = mpj.magnet.Cuboid(polarization=(0, 0, 1.0), dimension=(1.0, 1.0, 1.6))
        src.rotate_from_angax(ang, "y")
        B = np.asarray(src.getB(pts)).reshape(n, n, 3)
        mag = np.log10(np.linalg.norm(B[..., ::2], axis=-1) + 1e-6)
        ax.streamplot(X, Z, B[..., 0], B[..., 2], color=mag, cmap="viridis",
                      density=1.2, linewidth=0.7, arrowsize=0.7)
        # draw the rotated magnet cross-section (a, c) rotated about y
        a, c = 0.5, 0.8
        corners = np.array([[-a, -c], [a, -c], [a, c], [-a, c]])
        th = np.deg2rad(ang)
        rot = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        poly = corners @ rot.T
        ax.fill(poly[:, 0], poly[:, 1], color="#333", alpha=0.85, zorder=5)
        ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xlabel="x (m)", ylabel="z (m)",
               title=f"Rotating cuboid magnet — {ang:.0f}°")
        ax.set_aspect("equal")
        return ax.collections

    anim = animation.FuncAnimation(fig, frame, frames=len(angles), blit=False)
    anim.save(OUT / "movie_field.mp4", writer=_WRITER)
    plt.close(fig)
    print("wrote movie_field.mp4")


def movie_optimization() -> None:
    """Inverse design: a dipole moment fit converging to a target field."""
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.normal(scale=0.6, size=(60, 3)) + np.array([1.5, 0.3, -0.2]))
    true_m = jnp.array([0.4, -0.3, 0.7])
    pos = (0.1, -0.1, 0.05)

    def field(m):
        return mpj.misc.Dipole(moment=m, position=pos).getB(obs)

    target = field(true_m)
    scale = jnp.sqrt(jnp.mean(target ** 2))

    def loss(m):
        return jnp.mean(((field(m) - target) / scale) ** 2)

    grad = jax.jit(jax.grad(loss))
    m = jnp.array([0.0, 0.0, 0.0])
    hist, traj = [float(loss(m))], [np.asarray(m)]
    for _ in range(40):
        m = m - 0.2 * grad(m)
        hist.append(float(loss(m)))
        traj.append(np.asarray(m))
    traj = np.array(traj)
    tgt = np.asarray(target)
    obs_np = np.asarray(obs)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.2, 3.8))

    def frame(k: int):
        axL.clear(); axR.clear()
        axL.semilogy(hist[: k + 1], color=C_JAX, lw=2)
        axL.set(xlim=(0, 40), ylim=(1e-23, 2), xlabel="gradient step",
                ylabel="normalized loss", title="Differentiable inverse design")
        axL.grid(alpha=0.25)
        pred = np.asarray(field(jnp.asarray(traj[k])))
        # 2D quiver (x–z) of predicted (blue) vs target (grey) at each observer
        axR.quiver(obs_np[:, 0], obs_np[:, 2], tgt[:, 0], tgt[:, 2],
                   color="#bbb", angles="xy", scale=8e-3, width=0.006, label="target")
        axR.quiver(obs_np[:, 0], obs_np[:, 2], pred[:, 0], pred[:, 2],
                   color=C_JAX, angles="xy", scale=8e-3, width=0.004, label="fit")
        axR.set(xlabel="x (m)", ylabel="z (m)", title=f"step {k}   loss={hist[k]:.1e}")
        axR.legend(loc="upper right", fontsize=8, frameon=False)
        fig.tight_layout()
        return []

    anim = animation.FuncAnimation(fig, frame, frames=len(traj), blit=False)
    anim.save(OUT / "movie_optimization.mp4", writer=_WRITER)
    plt.close(fig)
    print("wrote movie_optimization.mp4")


if __name__ == "__main__":
    movie_field()
    movie_optimization()
    for mp4 in ("movie_field.mp4", "movie_optimization.mp4"):
        kb = (OUT / mp4).stat().st_size / 1024
        print(f"{mp4}: {kb:.0f} KB")
