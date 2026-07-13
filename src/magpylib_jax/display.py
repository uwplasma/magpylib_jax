"""Matplotlib-backed 3D visualization for magpylib_jax objects.

The public entry point is :func:`show`, which renders any mix of source
objects, :class:`~magpylib_jax.sensor.Sensor` instances and
:class:`~magpylib_jax.collection.Collection` containers into a single
``mpl_toolkits.mplot3d`` axes. Each object is drawn at its current pose
(``position`` + ``orientation``); objects that carry a path (position length
> 1) additionally show the path as a faint trailing line.

Matplotlib is imported lazily inside :func:`show` so that the rest of the
package does not depend on it at import time; a helpful error is raised if it
is missing. Surface resolutions are deliberately low to keep rendering (and
the test suite) fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

# Category colors -- magnets, currents and misc sources are visually distinct.
_COLORS = {
    "magnet": "#d1495b",
    "current": "#3b7dd8",
    "dipole": "#8b5cf6",
    "sensor": "#2a9d8f",
    "custom": "#8a8a8a",
}
_POL_ARROW = "#111111"      # polarization / moment arrow
_CURRENT_ARROW = "#123e78"  # current-direction arrow

# Deliberately coarse surface resolutions (fast tests, readable figures).
_N_THETA = 18
_N_PHI = 14
_N_LAT = 9


def _pose(obj: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rotation_matrix, last_position, full_path)`` for ``obj``.

    The body is drawn at the final path pose; ``full_path`` (n, 3) lets the
    caller draw a trailing path line when it has more than one point.
    """
    path = np.asarray(obj._position, dtype=float).reshape(-1, 3)
    rot_mats = np.asarray(obj._orientation_matrix, dtype=float).reshape(-1, 3, 3)
    return rot_mats[-1], path[-1], path


def _apply(local: np.ndarray, rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Rotate/translate local coordinates ``(..., 3)`` into the global frame."""
    arr = np.asarray(local, dtype=float)
    flat = arr.reshape(-1, 3) @ rot.T + pos
    return flat.reshape(arr.shape)


def _surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, rot: np.ndarray, pos: np.ndarray):
    """Transform a ``plot_surface`` mesh into the global frame."""
    pts = np.stack([x, y, z], axis=-1)
    tp = _apply(pts, rot, pos)
    return tp[..., 0], tp[..., 1], tp[..., 2]


class _Bounds:
    """Accumulate rendered points to derive equal-aspect axis limits."""

    def __init__(self) -> None:
        self._pts: list[np.ndarray] = []

    def add(self, pts: np.ndarray) -> None:
        arr = np.asarray(pts, dtype=float).reshape(-1, 3)
        if arr.size:
            self._pts.append(arr)

    def apply(self, ax: Any) -> None:
        if not self._pts:
            lo = np.array([-1.0, -1.0, -1.0])
            hi = np.array([1.0, 1.0, 1.0])
        else:
            allpts = np.concatenate(self._pts, axis=0)
            lo = allpts.min(axis=0)
            hi = allpts.max(axis=0)
        center = (lo + hi) / 2.0
        span = float((hi - lo).max())
        if span <= 0:
            span = 1.0
        r = span / 2.0 * 1.15
        ax.set_xlim(center[0] - r, center[0] + r)
        ax.set_ylim(center[1] - r, center[1] + r)
        ax.set_zlim(center[2] - r, center[2] + r)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:  # pragma: no cover - very old matplotlib
            pass


def _arrow(ax: Any, start: np.ndarray, vec: np.ndarray, color: str, lw: float = 2.2) -> None:
    ax.quiver(
        float(start[0]), float(start[1]), float(start[2]),
        float(vec[0]), float(vec[1]), float(vec[2]),
        color=color, linewidth=lw, arrow_length_ratio=0.25,
    )


def _char_size(bounds_pts: np.ndarray, fallback: float = 1.0) -> float:
    arr = np.asarray(bounds_pts, dtype=float).reshape(-1, 3)
    if arr.shape[0] < 2:
        return fallback
    span = float((arr.max(axis=0) - arr.min(axis=0)).max())
    return span if span > 0 else fallback


# --------------------------------------------------------------------------- #
# Per-object renderers
# --------------------------------------------------------------------------- #

def _polarization(obj: Any) -> np.ndarray | None:
    try:
        pol = obj._polarization
    except Exception:
        pol = getattr(obj, "polarization", None)
    if pol is None:
        return None
    return np.asarray(pol, dtype=float)


def _add_poly(ax: Any, tris: np.ndarray, color: str, bounds: _Bounds, alpha: float = 0.55) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = [list(map(tuple, tri)) for tri in np.asarray(tris, dtype=float)]
    coll = Poly3DCollection(faces, facecolor=color, edgecolor="#2a2a2a",
                            linewidths=0.4, alpha=alpha)
    ax.add_collection3d(coll)
    bounds.add(np.asarray(tris, dtype=float).reshape(-1, 3))


def _draw_cuboid(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    dim = getattr(obj, "dimension", None)
    if dim is None:
        _draw_marker(ax, obj, bounds, _COLORS["magnet"])
        return
    a, b, c = (np.asarray(dim, dtype=float) / 2.0)
    corners = np.array([
        [-a, -b, -c], [a, -b, -c], [a, b, -c], [-a, b, -c],
        [-a, -b, c], [a, -b, c], [a, b, c], [-a, b, c],
    ])
    gc = _apply(corners, rot, pos)
    face_idx = [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4),
    ]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = [[tuple(gc[i]) for i in f] for f in face_idx]
    coll = Poly3DCollection(faces, facecolor=_COLORS["magnet"], edgecolor="#2a2a2a",
                            linewidths=0.5, alpha=0.5)
    ax.add_collection3d(coll)
    bounds.add(gc)
    _magnet_arrow(ax, obj, pos, gc)


def _magnet_arrow(ax: Any, obj: Any, pos: np.ndarray, body_pts: np.ndarray) -> None:
    pol = _polarization(obj)
    if pol is None or not np.any(pol):
        return
    rot, _, _ = _pose(obj)
    direction = rot @ (pol / np.linalg.norm(pol))
    size = _char_size(body_pts)
    vec = direction * size
    _arrow(ax, pos - vec / 2.0, vec, _POL_ARROW, lw=2.0)


def _draw_cylinder(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    dim = getattr(obj, "dimension", None)
    if dim is None:
        _draw_marker(ax, obj, bounds, _COLORS["magnet"])
        return
    d, h = np.asarray(dim, dtype=float)
    r = d / 2.0
    th = np.linspace(0, 2 * np.pi, _N_THETA)
    z = np.array([-h / 2.0, h / 2.0])
    Th, Z = np.meshgrid(th, z)
    X, Y = r * np.cos(Th), r * np.sin(Th)
    gx, gy, gz = _surface(X, Y, Z, rot, pos)
    ax.plot_surface(gx, gy, gz, color=_COLORS["magnet"], alpha=0.45, linewidth=0)
    bounds.add(np.stack([gx, gy, gz], axis=-1))
    # end caps
    rr = np.array([0.0, r])
    Rr, Thc = np.meshgrid(rr, th)
    Xc, Yc = Rr * np.cos(Thc), Rr * np.sin(Thc)
    for zc in (-h / 2.0, h / 2.0):
        cx, cy, cz = _surface(Xc, Yc, np.full_like(Xc, zc), rot, pos)
        ax.plot_surface(cx, cy, cz, color=_COLORS["magnet"], alpha=0.45, linewidth=0)
        bounds.add(np.stack([cx, cy, cz], axis=-1))
    _magnet_arrow(ax, obj, pos, np.stack([gx, gy, gz], axis=-1).reshape(-1, 3))


def _draw_cylinder_segment(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    dim = getattr(obj, "dimension", None)
    if dim is None:
        _draw_marker(ax, obj, bounds, _COLORS["magnet"])
        return
    r1, r2, h, phi1, phi2 = np.asarray(dim, dtype=float)
    phi = np.linspace(np.deg2rad(phi1), np.deg2rad(phi2), _N_PHI)
    z = np.array([-h / 2.0, h / 2.0])
    all_pts = []

    def _emit(X, Y, Z):
        gx, gy, gz = _surface(X, Y, Z, rot, pos)
        ax.plot_surface(gx, gy, gz, color=_COLORS["magnet"], alpha=0.45, linewidth=0)
        all_pts.append(np.stack([gx, gy, gz], axis=-1))

    Phi, Z = np.meshgrid(phi, z)
    for rad in (r1, r2):  # inner + outer walls
        _emit(rad * np.cos(Phi), rad * np.sin(Phi), Z)
    rr = np.array([r1, r2])
    Rr, Php = np.meshgrid(rr, phi)
    for zc in (-h / 2.0, h / 2.0):  # top + bottom faces
        _emit(Rr * np.cos(Php), Rr * np.sin(Php), np.full_like(Rr, zc))
    Rz, Zc = np.meshgrid(rr, z)
    for pc in (np.deg2rad(phi1), np.deg2rad(phi2)):  # end caps
        _emit(Rz * np.cos(pc), Rz * np.sin(pc), Zc)

    body = np.concatenate([p.reshape(-1, 3) for p in all_pts], axis=0)
    bounds.add(body)
    _magnet_arrow(ax, obj, pos, body)


def _draw_sphere(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    diameter = getattr(obj, "diameter", None)
    if diameter is None:
        _draw_marker(ax, obj, bounds, _COLORS["magnet"])
        return
    r = float(np.asarray(diameter, dtype=float)) / 2.0
    u = np.linspace(0, 2 * np.pi, _N_THETA)
    v = np.linspace(0, np.pi, _N_LAT)
    X = r * np.outer(np.cos(u), np.sin(v))
    Y = r * np.outer(np.sin(u), np.sin(v))
    Z = r * np.outer(np.ones_like(u), np.cos(v))
    gx, gy, gz = _surface(X, Y, Z, rot, pos)
    ax.plot_surface(gx, gy, gz, color=_COLORS["magnet"], alpha=0.45, linewidth=0)
    bounds.add(np.stack([gx, gy, gz], axis=-1))
    _magnet_arrow(ax, obj, pos, np.stack([gx, gy, gz], axis=-1).reshape(-1, 3))


def _draw_tetrahedron(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    verts = getattr(obj, "vertices", None)
    if verts is None:
        _draw_marker(ax, obj, bounds, _COLORS["magnet"])
        return
    gv = _apply(np.asarray(verts, dtype=float), rot, pos)
    tris = np.array([gv[[0, 1, 2]], gv[[0, 1, 3]], gv[[0, 2, 3]], gv[[1, 2, 3]]])
    _add_poly(ax, tris, _COLORS["magnet"], bounds)
    _magnet_arrow(ax, obj, gv.mean(axis=0), gv)


def _draw_triangular_mesh(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    mesh = getattr(obj, "mesh", None)
    if mesh is None or np.asarray(mesh).size == 0:
        _draw_marker(ax, obj, bounds, _COLORS["magnet"])
        return
    tris = _apply(np.asarray(mesh, dtype=float), rot, pos)
    _add_poly(ax, tris, _COLORS["magnet"], bounds)
    _magnet_arrow(ax, obj, pos + tris.reshape(-1, 3).mean(axis=0) - pos, tris.reshape(-1, 3))


def _draw_triangle(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    verts = getattr(obj, "vertices", None)
    if verts is None:
        _draw_marker(ax, obj, bounds, _COLORS["magnet"])
        return
    gv = _apply(np.asarray(verts, dtype=float), rot, pos)
    _add_poly(ax, gv[None, :, :], _COLORS["magnet"], bounds, alpha=0.6)
    _magnet_arrow(ax, obj, gv.mean(axis=0), gv)


def _draw_dipole(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    moment = getattr(obj, "moment", None)
    if moment is None or not np.any(np.asarray(moment, dtype=float)):
        _draw_marker(ax, obj, bounds, _COLORS["dipole"])
        return
    m = np.asarray(moment, dtype=float)
    direction = rot @ (m / np.linalg.norm(m))
    scale = _char_size(bounds._pts[-1] if bounds._pts else pos[None, :]) if bounds._pts else 1.0
    scale = max(scale, 1.0)
    vec = direction * scale
    _arrow(ax, pos - vec / 2.0, vec, _COLORS["dipole"], lw=2.6)
    ax.scatter([pos[0]], [pos[1]], [pos[2]], color=_COLORS["dipole"], s=25)
    bounds.add(np.stack([pos - vec / 2.0, pos + vec / 2.0]))


def _draw_circle(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    diameter = getattr(obj, "diameter", None)
    if diameter is None:
        _draw_marker(ax, obj, bounds, _COLORS["current"])
        return
    r = float(np.asarray(diameter, dtype=float)) / 2.0
    th = np.linspace(0, 2 * np.pi, 48)
    ring = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], axis=1)
    gr = _apply(ring, rot, pos)
    ax.plot(gr[:, 0], gr[:, 1], gr[:, 2], color=_COLORS["current"], linewidth=2.4)
    bounds.add(gr)
    # current-direction arrow (tangent at theta=0), sign follows current sign.
    cur = float(np.asarray(getattr(obj, "current", 1.0) or 1.0, dtype=float))
    sign = 1.0 if cur >= 0 else -1.0
    tang = rot @ np.array([0.0, sign, 0.0]) * (2 * np.pi * r / 12.0)
    _arrow(ax, gr[0], tang, _CURRENT_ARROW, lw=2.0)


def _draw_polyline(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    verts = getattr(obj, "vertices", None)
    if verts is None:
        _draw_marker(ax, obj, bounds, _COLORS["current"])
        return
    gv = _apply(np.asarray(verts, dtype=float), rot, pos)
    ax.plot(gv[:, 0], gv[:, 1], gv[:, 2], color=_COLORS["current"], linewidth=2.4)
    bounds.add(gv)
    cur = float(np.asarray(getattr(obj, "current", 1.0) or 1.0, dtype=float))
    sign = 1.0 if cur >= 0 else -1.0
    for i in range(len(gv) - 1):
        mid = (gv[i] + gv[i + 1]) / 2.0
        seg = (gv[i + 1] - gv[i]) * sign
        n = np.linalg.norm(seg)
        if n > 0:
            _arrow(ax, mid - seg * 0.15, seg * 0.3, _CURRENT_ARROW, lw=1.6)


def _draw_triangle_strip(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    verts = getattr(obj, "vertices", None)
    if verts is None:
        _draw_marker(ax, obj, bounds, _COLORS["current"])
        return
    v = _apply(np.asarray(verts, dtype=float), rot, pos)
    tris = np.stack([v[:-2], v[1:-1], v[2:]], axis=1)
    _add_poly(ax, tris, _COLORS["current"], bounds, alpha=0.5)


def _draw_triangle_sheet(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    verts = getattr(obj, "vertices", None)
    faces = getattr(obj, "faces", None)
    if verts is None or faces is None:
        _draw_marker(ax, obj, bounds, _COLORS["current"])
        return
    v = _apply(np.asarray(verts, dtype=float), rot, pos)
    tris = v[np.asarray(faces, dtype=int)]
    _add_poly(ax, tris, _COLORS["current"], bounds, alpha=0.5)


def _draw_sensor(ax: Any, obj: Any, bounds: _Bounds) -> None:
    rot, pos, _ = _pose(obj)
    ax.scatter([pos[0]], [pos[1]], [pos[2]], color=_COLORS["sensor"], s=45, marker="s")
    bounds.add(pos[None, :])
    pix = getattr(obj, "pixel", None)
    if pix is not None:
        px = np.asarray(pix, dtype=float).reshape(-1, 3)
        gp = _apply(px, rot, pos)
        ax.scatter(gp[:, 0], gp[:, 1], gp[:, 2], color=_COLORS["sensor"], s=12)
        bounds.add(gp)
        size = _char_size(gp, fallback=1.0) * 0.4 + 1e-9
    else:
        size = 0.5
    # local coordinate axes (handedness flips the third axis)
    hand = -1.0 if getattr(obj, "handedness", "right") == "left" else 1.0
    axes = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, hand]]) * size
    for a in axes:
        _arrow(ax, pos, rot @ a, _COLORS["sensor"], lw=1.4)


def _draw_custom(ax: Any, obj: Any, bounds: _Bounds) -> None:
    _draw_marker(ax, obj, bounds, _COLORS["custom"])


def _draw_marker(ax: Any, obj: Any, bounds: _Bounds, color: str) -> None:
    _rot, pos, _ = _pose(obj)
    ax.scatter([pos[0]], [pos[1]], [pos[2]], color=color, s=40, marker="o")
    label = getattr(obj, "style_label", None) or type(obj).__name__
    ax.text(float(pos[0]), float(pos[1]), float(pos[2]), f"  {label}", fontsize=7)
    bounds.add(pos[None, :])


_RENDERERS = {
    "cuboid": (_draw_cuboid, "magnet"),
    "cylinder": (_draw_cylinder, "magnet"),
    "cylindersegment": (_draw_cylinder_segment, "magnet"),
    "sphere": (_draw_sphere, "magnet"),
    "tetrahedron": (_draw_tetrahedron, "magnet"),
    "triangularmesh": (_draw_triangular_mesh, "magnet"),
    "triangle": (_draw_triangle, "magnet"),
    "dipole": (_draw_dipole, "dipole"),
    "circle": (_draw_circle, "current"),
    "polyline": (_draw_polyline, "current"),
    "trianglestrip": (_draw_triangle_strip, "current"),
    "trianglesheet": (_draw_triangle_sheet, "current"),
    "custom": (_draw_custom, "custom"),
}


def _draw_path(ax: Any, obj: Any, bounds: _Bounds) -> None:
    path = np.asarray(obj._position, dtype=float).reshape(-1, 3)
    if path.shape[0] > 1:
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#999999",
                linestyle="--", linewidth=1.0, alpha=0.7)
        bounds.add(path)


def _render_object(ax: Any, obj: Any, bounds: _Bounds, categories: set[str]) -> None:
    """Dispatch a single object to its renderer, recursing into collections."""
    if getattr(obj, "_is_collection", False):
        for child in getattr(obj, "children", []):
            _render_object(ax, child, bounds, categories)
        return

    _draw_path(ax, obj, bounds)

    if getattr(obj, "_is_sensor", False):
        _draw_sensor(ax, obj, bounds)
        categories.add("sensor")
        return

    source_type = getattr(obj, "_source_type", "")
    entry = _RENDERERS.get(source_type)
    if entry is None:
        _draw_marker(ax, obj, bounds, _COLORS["custom"])
        categories.add("custom")
        return
    renderer, category = entry
    renderer(ax, obj, bounds)
    categories.add(category)


def show(
    *objects: Any,
    backend: str = "matplotlib",
    ax: Any = None,
    return_fig: bool = False,
    style: Any = None,
    **kwargs: Any,
) -> Figure | None:
    """Render magpylib_jax objects in a single 3D matplotlib axes.

    Parameters
    ----------
    *objects:
        Sources, :class:`~magpylib_jax.sensor.Sensor` instances and/or
        :class:`~magpylib_jax.collection.Collection` containers to draw. At
        least one object is required.
    backend:
        Only ``"matplotlib"`` is supported.
    ax:
        Existing 3D axes to draw on. If ``None`` a new figure/axes is created.
    return_fig:
        When ``True`` return the :class:`matplotlib.figure.Figure` (and never
        call ``plt.show()``); useful for tests and headless rendering.
    style:
        Optional mapping of display options. ``title`` sets the axes title.
    **kwargs:
        Accepted for magpylib compatibility; ``title`` is honored.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure when ``return_fig`` is ``True``, otherwise ``None``.
    """
    if backend != "matplotlib":
        raise ValueError(
            f"Unsupported backend {backend!r}; only 'matplotlib' is available."
        )

    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d proj)
    except ImportError as exc:  # pragma: no cover - exercised only without mpl
        raise ImportError(
            "magpylib_jax.show() requires matplotlib. Install it with "
            "`pip install matplotlib`."
        ) from exc

    # Flatten a single iterable argument, e.g. show([a, b]).
    if len(objects) == 1 and not hasattr(objects[0], "_position"):
        candidate = objects[0]
        if isinstance(candidate, (list, tuple)):
            objects = tuple(candidate)

    if not objects:
        raise ValueError("show() requires at least one object to display.")

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6.5, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.get_figure()

    bounds = _Bounds()
    categories: set[str] = set()
    for obj in objects:
        _render_object(ax, obj, bounds, categories)

    bounds.apply(ax)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    opts = dict(style) if isinstance(style, dict) else {}
    opts.update({k: v for k, v in kwargs.items() if k == "title"})
    ax.set_title(opts.get("title", "magpylib_jax.show"))

    if categories:
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLORS[c],
                   markersize=9, label=c)
            for c in sorted(categories)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.6)

    if return_fig:
        return fig

    backend_name = matplotlib.get_backend().lower()
    interactive = "agg" not in backend_name and "template" not in backend_name
    if created_fig and interactive:  # pragma: no cover - GUI only
        plt.show()
    return None
