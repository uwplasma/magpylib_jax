"""JAX-native, autodiff magnetic force and torque (``getFT``).

This module implements :func:`getFT`, a differentiable analogue of magpylib's
``magpy.getFT`` (magpylib 5.2.x). The force on a magnet cell is the gradient of
the magnetic dipole energy ``m . B`` computed with :func:`jax.jacfwd` (exact
autodiff, no finite differences), and the force on a current cell is the Laplace
force ``(I dL) x B``. Because the whole pipeline is built from the differentiable
:func:`magpylib_jax.getB` plus ``jacfwd``, ``getFT`` is itself differentiable and
can be used directly inside optimisation loops.

Advantage over magpylib: magpylib evaluates the field gradient of magnet targets
with symmetric finite differences of step ``eps``; here the gradient is obtained
by forward-mode autodiff, so the magnet result does *not* depend on ``eps`` and is
exact to machine precision.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R

from magpylib_jax.constants import MU0
from magpylib_jax.fields.api import getB

__all__ = ["getFT"]


# ---------------------------------------------------------------------------
# Meshing: local cell points + per-cell moment (magnet) or current-vector.
# ---------------------------------------------------------------------------
def _cuboid_cell_counts(meshing: Any, dimension: np.ndarray) -> tuple[int, int, int]:
    """Number of grid divisions per axis (mirrors magpylib target_meshing)."""
    if not isinstance(meshing, (int, np.integer)):
        n1, n2, n3 = (int(v) for v in meshing)
        return n1, n2, n3
    target_elems = int(meshing)
    if target_elems == 1:
        return 1, 1, 1
    a, b, c = (float(v) for v in dimension)
    cell_size = (a * b * c / target_elems) ** (1 / 3)
    n1 = max(1, int(np.round(a / cell_size)))
    n2 = max(1, int(np.round(b / cell_size)))
    n3 = max(1, int(np.round(c / cell_size)))
    return n1, n2, n3


# ---------------------------------------------------------------------------
# NumPy replicas of magpylib's target_meshing (static geometry). These mirror
# ``magpylib._src.obj_classes.target_meshing`` cell-for-cell so getFT matches
# ``magpy.getFT``. Cell *centers* and per-cell *volumes/areas* are pure geometry
# (computed in NumPy, static w.r.t. the integer ``target.meshing``); the caller
# multiplies them by the JAX excitation (magnetization / current) to build the
# differentiable per-cell moments or current-vectors.
# ---------------------------------------------------------------------------
def _apportion_triple(triple: np.ndarray, min_val: int = 1, max_iter: int = 30) -> np.ndarray:
    """Apportion a triple so ``min_val`` is respected and the product is kept."""
    triple = np.abs(np.array(triple, dtype=float))
    count = 0
    while any(n < min_val for n in triple) and count < max_iter:
        count += 1
        amin, amax = triple.argmin(), triple.argmax()
        factor = min_val / triple[amin]
        if triple[amax] >= factor * min_val:
            triple /= factor**0.5
            triple[amin] *= factor**1.5
    return triple


def _cells_from_dimension(dim: Any, target_elems: int, min_val: int = 1) -> np.ndarray:
    """Divide a dimension triple into a near-cubic (n1, n2, n3) cell count."""
    from itertools import product

    elems = np.prod(target_elems)
    funcs = [np.ceil, np.floor]
    elems = max(min_val**3, elems)
    x, y, z = np.abs(dim)
    a = x ** (2 / 3) * (elems / y / z) ** (1 / 3)
    b = y ** (2 / 3) * (elems / x / z) ** (1 / 3)
    c = z ** (2 / 3) * (elems / x / y) ** (1 / 3)
    a, b, c = _apportion_triple((a, b, c), min_val=min_val)
    epsilon = elems
    result = [funcs[0](k) for k in (a, b, c)]
    for fn in product(*[funcs] * 3):
        res = [f(k) for f, k in zip(fn, (a, b, c), strict=False)]
        epsilon_new = elems - np.prod(res)
        if np.abs(epsilon_new) <= epsilon and all(r >= min_val for r in res):
            epsilon = np.abs(epsilon_new)
            result = res
    return np.array(result).astype(int)


def _cylinder_cells(
    r1: float, r2: float, h: float, phi1: float, phi2: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cylinder(-segment) cell centers + volumes (mirrors _target_mesh_cylinder)."""
    al = (r2 + r1) * 3.14 * (phi2 - phi1) / 360  # arclen = D*pi*arcratio
    dim = al, r2 - r1, h
    nphi, nr, nh = _cells_from_dimension(dim, n)
    r = np.linspace(r1, r2, nr + 1)
    dh = h / nh
    cells: list = []
    volumes: list = []
    for r_ind in range(nr):
        nphi_r = max(1, int(r[r_ind + 1] / ((r1 + r2) / 2) * nphi))
        phi = np.linspace(phi1, phi2, nphi_r + 1)
        for h_ind in range(nh):
            pos_h = dh * h_ind - h / 2 + dh / 2
            if nr >= 3 and r[r_ind] == 0 and phi2 - phi1 == 360:
                cells.append((0.0, 0.0, pos_h))
                volumes.append(np.pi * r[r_ind + 1] ** 2 * dh)
            else:
                for phi_ind in range(nphi_r):
                    radial_coord = (r[r_ind] + r[r_ind + 1]) / 2
                    angle_coord = (phi[phi_ind] + phi[phi_ind + 1]) / 2
                    cells.append(
                        (
                            radial_coord * np.cos(np.deg2rad(angle_coord)),
                            radial_coord * np.sin(np.deg2rad(angle_coord)),
                            pos_h,
                        )
                    )
                    volumes.append(
                        np.pi
                        * (r[r_ind + 1] ** 2 - r[r_ind] ** 2)
                        * dh
                        / nphi_r
                        * (phi2 - phi1)
                        / 360
                    )
    return np.array(cells, dtype=float), np.array(volumes, dtype=float)


def _tetrahedron_cells(
    n_points: int, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedron cell centers + volumes (mirrors _target_mesh_tetrahedron)."""
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    v3 = vertices[3] - vertices[0]
    tet_volume = abs(np.linalg.det(np.column_stack([v1, v2, v3]))) / 6.0

    if n_points == 1:
        centroid = np.mean(vertices, axis=0)
        return np.array([centroid]), np.array([tet_volume])

    def points_for_n_div(n: int) -> int:
        return (n + 1) * (n + 2) * (n + 3) // 6

    n_div_estimate = max(1, int(np.round((6 * n_points) ** (1 / 3) - 1.5)))
    best_n_div = n_div_estimate
    best_diff = abs(points_for_n_div(n_div_estimate) - n_points)
    for test_n_div in range(max(1, n_div_estimate - 2), n_div_estimate + 4):
        diff = abs(points_for_n_div(test_n_div) - n_points)
        if diff < best_diff:
            best_diff = diff
            best_n_div = test_n_div
    n_div = best_n_div

    pts_list = []
    for i in range(n_div + 1):
        for j in range(n_div + 1 - i):
            for k in range(n_div + 1 - i - j):
                latt = n_div - i - j - k
                if latt >= 0:
                    point = (
                        (i / n_div) * vertices[0]
                        + (j / n_div) * vertices[1]
                        + (k / n_div) * vertices[2]
                        + (latt / n_div) * vertices[3]
                    )
                    pts_list.append(point)
    pts = np.array(pts_list)
    volumes = np.full(len(pts), tet_volume / len(pts))
    return pts, volumes


def _subdiv(triangles: np.ndarray, splits: np.ndarray) -> np.ndarray:
    """Subdivide triangles by longest-edge bisection (mirrors _subdiv)."""
    n_sub = 2**splits
    n_tot = int(np.sum(n_sub))
    ends = np.cumsum(n_sub)
    starts = np.r_[0, ends[:-1]]
    TRIA = np.empty((n_tot, 3, 3))
    TRIA[starts] = triangles
    MASK = np.zeros((n_tot), dtype=bool)
    MASK[starts] = True
    for i in range(int(max(splits))):
        mask_split = i == splits
        for start in starts[mask_split]:
            MASK[start : start + 2**i] = False
        triangles = TRIA[MASK]
        mask_split = i < splits
        for start in starts[mask_split]:
            MASK[start : start + 2 ** (i + 1)] = True
        A = triangles[:, 0]
        B = triangles[:, 1]
        C = triangles[:, 2]
        d2_AB = np.sum((B - A) ** 2, axis=1)
        d2_BC = np.sum((C - B) ** 2, axis=1)
        d2_CA = np.sum((A - C) ** 2, axis=1)
        case1 = (d2_AB >= d2_BC) * (d2_AB >= d2_CA)
        case2 = d2_BC >= d2_CA
        case3 = ~(case1 | case2)
        new_triangles = np.empty((len(triangles), 2, 3, 3), dtype=float)
        if np.any(case1):
            new_triangles[case1, 0, 0] = A[case1]
            new_triangles[case1, 0, 1] = (A[case1] + B[case1]) / 2.0
            new_triangles[case1, 0, 2] = C[case1]
            new_triangles[case1, 1, 0] = (A[case1] + B[case1]) / 2.0
            new_triangles[case1, 1, 1] = B[case1]
            new_triangles[case1, 1, 2] = C[case1]
        if np.any(case2):
            new_triangles[case2, 0, 0] = B[case2]
            new_triangles[case2, 0, 1] = (B[case2] + C[case2]) / 2.0
            new_triangles[case2, 0, 2] = A[case2]
            new_triangles[case2, 1, 0] = (B[case2] + C[case2]) / 2.0
            new_triangles[case2, 1, 1] = C[case2]
            new_triangles[case2, 1, 2] = A[case2]
        if np.any(case3):
            new_triangles[case3, 0, 0] = C[case3]
            new_triangles[case3, 0, 1] = (C[case3] + A[case3]) / 2.0
            new_triangles[case3, 0, 2] = B[case3]
            new_triangles[case3, 1, 0] = (C[case3] + A[case3]) / 2.0
            new_triangles[case3, 1, 1] = A[case3]
            new_triangles[case3, 1, 2] = B[case3]
        TRIA[MASK] = new_triangles.reshape(-1, 3, 3)
    return TRIA


def _triangle_current_cells(
    triangles: np.ndarray, n_target: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine triangles; return (centroids, per-orig-tri splits, per-cell areas).

    Mirrors ``_target_mesh_triangle_current`` geometry: current-vectors are then
    ``repeat(cds, 2**splits) * areas`` (built with JAX by the caller).
    """
    n_tria = len(triangles)
    surfaces = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    splits = np.zeros(n_tria, dtype=int)
    while n_tria < n_target:
        idx = int(np.argmax(surfaces))
        surfaces[idx] /= 2.0
        splits[idx] += 1
        n_tria = int(np.sum(2**splits))
    surfaces = np.repeat(surfaces, 2**splits)
    triangles = _subdiv(triangles, splits)
    centroids = np.mean(triangles, axis=1)
    return centroids, splits, surfaces


def _create_regular_grid(
    min_c: np.ndarray, max_c: np.ndarray, spacing: float
) -> np.ndarray:
    """Regular cubic grid covering a box (mirrors target_meshing._create_grid)."""
    x0, y0, z0 = (float(v) for v in min_c)
    x1, y1, z1 = (float(v) for v in max_c)
    nx = int(np.ceil((x1 - x0) / spacing))
    ny = int(np.ceil((y1 - y0) / spacing))
    nz = int(np.ceil((z1 - z0) / spacing))
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    x = cx + spacing * (np.arange(nx + 1) - nx // 2)
    y = cy + spacing * (np.arange(ny + 1) - ny // 2)
    z = cz + spacing * (np.arange(nz + 1) - nz // 2)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _generate_ft_mesh(target: Any) -> tuple[jnp.ndarray, jnp.ndarray, str]:
    """Return ``(pts_local (n,3), weights (n,3), kind)`` in the target frame.

    ``kind`` is ``"magnet"`` (weights are cell magnetic moments in A.m^2) or
    ``"current"`` (weights are current vectors ``I . dL`` in A.m). Cell points
    and weights are expressed in the target LOCAL frame; the caller rotates by
    the target orientation and translates by the target position.
    """
    # Local imports avoid an import cycle at module load time.
    from magpylib_jax.current import (
        Circle,
        Polyline,
        TriangleSheet,
        TriangleStrip,
    )
    from magpylib_jax.magnet import (
        Cuboid,
        Cylinder,
        CylinderSegment,
        Sphere,
        Tetrahedron,
        TriangularMesh,
    )
    from magpylib_jax.misc import Dipole

    if isinstance(target, Dipole):
        pts = jnp.zeros((1, 3), dtype=float)
        moment = jnp.asarray(target.moment, dtype=float).reshape(1, 3)
        return pts, moment, "magnet"

    if isinstance(target, Sphere):
        pts = jnp.zeros((1, 3), dtype=float)
        d = jnp.asarray(target.diameter, dtype=float)
        volume = (4.0 / 3.0) * jnp.pi * (d / 2.0) ** 3
        magnetization = jnp.asarray(target._polarization, dtype=float) / MU0
        moment = (volume * magnetization).reshape(1, 3)
        return pts, moment, "magnet"

    if isinstance(target, Cuboid):
        dim = np.asarray(target.dimension, dtype=float)
        meshing = getattr(target, "meshing", None)
        if meshing is None:
            meshing = 1
        n1, n2, n3 = _cuboid_cell_counts(meshing, dim)
        a, b, c = (float(v) for v in dim)
        xs = np.linspace(-a / 2, a / 2, n1 + 1)
        ys = np.linspace(-b / 2, b / 2, n2 + 1)
        zs = np.linspace(-c / 2, c / 2, n3 + 1)
        xc = (xs[:-1] + xs[1:]) / 2 if n1 > 1 else np.array([0.0])
        yc = (ys[:-1] + ys[1:]) / 2 if n2 > 1 else np.array([0.0])
        zc = (zs[:-1] + zs[1:]) / 2 if n3 > 1 else np.array([0.0])
        grid = np.array([(x, y, z) for x in xc for y in yc for z in zc])
        pts = jnp.asarray(grid, dtype=float)
        n_cells = pts.shape[0]
        volume = jnp.asarray(a * b * c, dtype=float)
        magnetization = jnp.asarray(target._polarization, dtype=float) / MU0
        cell_moment = (volume / n_cells) * magnetization
        moments = jnp.broadcast_to(cell_moment, (n_cells, 3))
        return pts, moments, "magnet"

    if isinstance(target, Circle):
        meshing = getattr(target, "meshing", None)
        n = int(meshing) if meshing is not None else 100
        if n < 3:
            raise ValueError(
                f"Circle target meshing must be an integer >= 3; received {n!r}."
            )
        r = jnp.asarray(target.diameter, dtype=float) / 2.0
        i0 = jnp.asarray(target.current, dtype=float)
        k = jnp.arange(n + 1, dtype=float)
        # equal-area polygon radius
        r1 = r * jnp.sqrt((2 * jnp.pi) / (n * jnp.sin(2 * jnp.pi / n)))
        vx = r1 * jnp.cos(2 * jnp.pi * k / n)
        vy = r1 * jnp.sin(2 * jnp.pi * k / n)
        midx = (vx[:-1] + vx[1:]) / 2
        midy = (vy[:-1] + vy[1:]) / 2
        midz = jnp.zeros((n,), dtype=float)
        tx = vx[1:] - vx[:-1]
        ty = vy[1:] - vy[:-1]
        pts = jnp.stack((midx, midy, midz), axis=1)
        cvecs = jnp.stack((tx, ty, midz), axis=1) * i0
        return pts, cvecs, "current"

    if isinstance(target, Polyline):
        verts = jnp.asarray(target.vertices, dtype=float)
        i0 = jnp.asarray(target.current, dtype=float)
        meshing = getattr(target, "meshing", None)
        n_points = int(meshing) if meshing is not None else (verts.shape[0] - 1)
        seg_vecs = verts[1:] - verts[:-1]
        n_segments = int(seg_vecs.shape[0])
        seg_lengths = np.asarray(jnp.linalg.norm(seg_vecs, axis=1))
        total_length = float(seg_lengths.sum())
        points_per_segment = np.ones(n_segments, dtype=int)
        remaining = n_points - n_segments
        if remaining > 0 and total_length > 0:
            extra = np.round((seg_lengths / total_length) * remaining).astype(int)
            points_per_segment += extra
        pts_list = []
        cvec_list = []
        for seg_idx in range(n_segments):
            n_pts = int(points_per_segment[seg_idx])
            start = verts[seg_idx]
            vec = seg_vecs[seg_idx]
            for j in range(n_pts):
                frac = (2 * j + 1) / (2 * n_pts)
                pts_list.append(start + vec * frac)
                cvec_list.append(vec / n_pts * i0)
        pts = jnp.stack(pts_list, axis=0)
        cvecs = jnp.stack(cvec_list, axis=0)
        return pts, cvecs, "current"

    if isinstance(target, CylinderSegment):
        meshing = int(getattr(target, "meshing", None) or 1)
        r1, r2, h, phi1, phi2 = (float(v) for v in np.asarray(target.dimension, dtype=float))
        pts_np, vols_np = _cylinder_cells(r1, r2, h, phi1, phi2, meshing)
        pts = jnp.asarray(pts_np, dtype=float)
        magnetization = jnp.asarray(target._polarization, dtype=float) / MU0
        moments = jnp.asarray(vols_np, dtype=float)[:, None] * magnetization
        return pts, moments, "magnet"

    if isinstance(target, Cylinder):
        meshing = int(getattr(target, "meshing", None) or 1)
        d, h = (float(v) for v in np.asarray(target.dimension, dtype=float))
        pts_np, vols_np = _cylinder_cells(0.0, d / 2.0, h, 0.0, 360.0, meshing)
        pts = jnp.asarray(pts_np, dtype=float)
        magnetization = jnp.asarray(target._polarization, dtype=float) / MU0
        moments = jnp.asarray(vols_np, dtype=float)[:, None] * magnetization
        return pts, moments, "magnet"

    if isinstance(target, Tetrahedron):
        meshing = int(getattr(target, "meshing", None) or 1)
        verts_np = np.asarray(target.vertices, dtype=float)
        pts_np, vols_np = _tetrahedron_cells(meshing, verts_np)
        pts = jnp.asarray(pts_np, dtype=float)
        magnetization = jnp.asarray(target._polarization, dtype=float) / MU0
        moments = jnp.asarray(vols_np, dtype=float)[:, None] * magnetization
        return pts, moments, "magnet"

    if isinstance(target, TriangularMesh):
        meshing = int(getattr(target, "meshing", None) or 1)
        mesh_np = np.asarray(target.mesh, dtype=float)  # (m,3,3) oriented triangles
        volume = float(target.volume)
        magnetization = jnp.asarray(target._polarization, dtype=float) / MU0
        if meshing == 1:
            # area-weighted surface centroid (barycenter), single cell
            ctri = mesh_np.mean(axis=1)
            area = 0.5 * np.linalg.norm(
                np.cross(mesh_np[:, 1] - mesh_np[:, 0], mesh_np[:, 2] - mesh_np[:, 0]),
                axis=1,
            )
            bary = (ctri * area[:, None]).sum(axis=0) / area.sum()
            pts = jnp.asarray(bary, dtype=float).reshape(1, 3)
            moments = (jnp.asarray(volume, dtype=float) * magnetization).reshape(1, 3)
            return pts, moments, "magnet"
        from magpylib_jax.core.kernels import _mask_inside_trimesh_jax

        spacing = (volume / meshing) ** (1 / 3)
        min_c = mesh_np.reshape(-1, 3).min(axis=0)
        max_c = mesh_np.reshape(-1, 3).max(axis=0)
        pad = spacing * 0.5
        grid = _create_regular_grid(min_c - pad, max_c + pad, spacing)
        inside = np.asarray(
            _mask_inside_trimesh_jax(
                jnp.asarray(grid, dtype=float), jnp.asarray(mesh_np, dtype=float)
            )
        )
        pts_np = grid[inside]
        if len(pts_np) == 0:  # degenerate fallback: barycenter
            ctri = mesh_np.mean(axis=1)
            area = 0.5 * np.linalg.norm(
                np.cross(mesh_np[:, 1] - mesh_np[:, 0], mesh_np[:, 2] - mesh_np[:, 0]),
                axis=1,
            )
            pts_np = ((ctri * area[:, None]).sum(axis=0) / area.sum()).reshape(1, 3)
        pts = jnp.asarray(pts_np, dtype=float)
        cell_vol = volume / pts.shape[0]
        moments = (jnp.asarray(cell_vol, dtype=float) * magnetization)
        moments = jnp.broadcast_to(moments, (pts.shape[0], 3))
        return pts, moments, "magnet"

    if isinstance(target, TriangleStrip):
        meshing = int(getattr(target, "meshing", None) or 1)
        verts_np = np.asarray(target.vertices, dtype=float)
        i0 = jnp.asarray(target.current, dtype=float)
        triangles = np.array([verts_np[i : i + 3] for i in range(len(verts_np) - 2)])
        side_a = triangles[:, 1] - triangles[:, 0]
        side_b = triangles[:, 2] - triangles[:, 0]
        side_aa = np.sum(side_a * side_a, axis=1)
        side_bb = np.sum(side_b * side_b, axis=1)
        side_ab = np.sum(side_a * side_b, axis=1)
        area2 = side_aa * side_bb - side_ab * side_ab
        area2_rel = area2 / np.sum(area2)
        mask_good = area2_rel >= 1e-19
        triangles = triangles[mask_good]
        base_length = np.linalg.norm(side_b[mask_good], axis=1)
        height = np.sqrt(area2[mask_good]) / base_length
        # current-density geometry factor (per triangle); excitation applied below
        cds_geom = side_b[mask_good] / base_length[:, None] / height[:, None]
        centroids, splits, surfaces = _triangle_current_cells(triangles, meshing)
        cds_rep = np.repeat(cds_geom, 2**splits, axis=0) * surfaces[:, None]
        pts = jnp.asarray(centroids, dtype=float)
        cvecs = jnp.asarray(cds_rep, dtype=float) * i0
        return pts, cvecs, "current"

    if isinstance(target, TriangleSheet):
        meshing = int(getattr(target, "meshing", None) or 1)
        verts_np = np.asarray(target.vertices, dtype=float)
        faces_np = np.asarray(target.faces, dtype=int)
        cds = jnp.asarray(target.current_densities, dtype=float)
        triangles = verts_np[faces_np]
        centroids, splits, surfaces = _triangle_current_cells(triangles, meshing)
        reps = jnp.asarray(np.repeat(np.arange(len(faces_np)), 2**splits))
        pts = jnp.asarray(centroids, dtype=float)
        cvecs = cds[reps] * jnp.asarray(surfaces, dtype=float)[:, None]
        return pts, cvecs, "current"

    raise NotImplementedError(
        f"getFT does not support target type {type(target).__name__!r}. "
        "Supported targets: Dipole, Sphere, Cuboid, Cylinder, CylinderSegment, "
        "Tetrahedron, TriangularMesh, Circle, Polyline, TriangleStrip, TriangleSheet."
    )


# ---------------------------------------------------------------------------
# Input formatting helpers.
# ---------------------------------------------------------------------------
def _as_source_list(sources: Any) -> tuple[list, list[int]]:
    """Flatten sources to leaf objects, tracking top-level source groups.

    Returns ``(flat_leaves, group_starts)`` where ``group_starts[g]`` is the index
    in ``flat_leaves`` where top-level source ``g`` begins. A Collection source is
    expanded into its leaf sources but kept as one output group (magpylib sums a
    collection's members into a single force on the target).
    """
    from magpylib_jax.collection import Collection

    raw = list(sources) if isinstance(sources, (list, tuple)) else [sources]
    flat: list = []
    group_starts: list[int] = []
    for s in raw:
        group_starts.append(len(flat))
        if isinstance(s, Collection):
            flat.extend(s.sources)
        else:
            flat.append(s)
    return flat, group_starts


def _flatten_targets(targets: Any) -> tuple[list, list[int]]:
    """Flatten a Collection into member sources; return (flat, coll_start_idx)."""
    from magpylib_jax.collection import Collection

    if not isinstance(targets, (list, tuple)):
        targets = [targets]

    flat: list = []
    coll_idx: list[int] = []
    idx = 0
    for t in targets:
        if isinstance(t, Collection):
            members = t.sources
            if not members:
                raise ValueError(f"Target Collection {t!r} has no member sources.")
            flat.extend(members)
            coll_idx.append(idx)
            idx += len(members)
        else:
            flat.append(t)
            coll_idx.append(idx)
            idx += 1
    return flat, coll_idx


def _pad_edge(arr: Any, n: int) -> jnp.ndarray:
    """Edge-pad axis 0 of a JAX array to length ``n`` (keeps gradients)."""
    arr = jnp.asarray(arr, dtype=float)
    if arr.shape[0] == n:
        return arr
    pad = jnp.broadcast_to(arr[-1:], (n - arr.shape[0],) + arr.shape[1:])
    return jnp.concatenate([arr, pad], axis=0)


def _target_centroid(target: Any, n_path: int) -> jnp.ndarray:
    """Per-path centroid, shape (n_path, 3)."""
    cent = getattr(target, "centroid", None)
    if cent is None:
        cent = target.position
    cent = jnp.asarray(cent, dtype=float)
    if cent.ndim == 1:
        cent = jnp.broadcast_to(cent, (n_path, 3))
    else:
        cent = _pad_edge(cent, n_path)
    return cent


def _format_pivot(pivot: Any, targets: list, n_path: int) -> jnp.ndarray | None:
    """Return pivots with shape (n_tgt, n_path, 3) or None."""
    if pivot is None:
        return None
    n_tgt = len(targets)
    if isinstance(pivot, str):
        if pivot != "centroid":
            raise ValueError(
                f"Input pivot string must be 'centroid'; received {pivot!r}."
            )
        return jnp.stack([_target_centroid(t, n_path) for t in targets], axis=0)

    piv = jnp.asarray(pivot, dtype=float)
    if piv.shape == (3,):
        return jnp.broadcast_to(piv, (n_tgt, n_path, 3))
    if piv.shape == (n_tgt, 3):
        return jnp.broadcast_to(piv[:, None, :], (n_tgt, n_path, 3))
    if piv.shape == (n_tgt, n_path, 3):
        return piv
    raise ValueError(
        "Input pivot must be 'centroid', None, or array-like with shape (3,), "
        f"(t, 3), or (t, p, 3); received array with shape {piv.shape}."
    )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def getFT(
    sources: Any,
    targets: Any,
    pivot: Any = "centroid",
    eps: float = 1e-5,  # noqa: ARG001 - kept for API parity; result is eps-free
    squeeze: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable magnetic force and torque on targets from sources.

    Computes the force ``F`` (N) and torque ``T`` (N.m) exerted on each target by
    each source. Magnet targets use the exact field gradient from
    :func:`jax.jacfwd` (``F = grad(m . B)``, ``T = m x B``); current targets use
    the Laplace force (``F = (I dL) x B``, ``T = 0``). SI units throughout.

    Parameters
    ----------
    sources : Source | Collection | list[Source | Collection]
        One or more magpylib_jax sources generating the field. A ``Collection`` is
        treated as a single source: its member fields are summed into one force on
        each target (matching magpylib).
    targets : Target | list[Target] | Collection
        Objects the field acts on. Supported target types:

        - ``Dipole``          (magnet, single cell)
        - ``Sphere``          (magnet, single cell)
        - ``Cuboid``          (magnet, grid meshing via ``target.meshing``; default 1 cell)
        - ``Cylinder``        (magnet, cylindrical meshing via ``target.meshing``)
        - ``CylinderSegment`` (magnet, cylindrical-segment meshing via ``target.meshing``)
        - ``Tetrahedron``     (magnet, barycentric meshing via ``target.meshing``)
        - ``TriangularMesh``  (magnet, inside-mask grid meshing via ``target.meshing``)
        - ``Circle``          (current, polygon meshing via ``target.meshing``; default 100)
        - ``Polyline``        (current; ``target.meshing`` points, default 1 per segment)
        - ``TriangleStrip``   (current, triangle-bisection meshing via ``target.meshing``)
        - ``TriangleSheet``   (current, triangle-bisection meshing via ``target.meshing``)

        Cell centers and per-cell moments/current-vectors replicate magpylib's
        ``target_meshing`` exactly, so results match ``magpy.getFT`` for a matching
        ``meshing``. Any other target type raises ``NotImplementedError``.
        ``Collection`` targets are flattened and their members summed into one
        target column.
    pivot : 'centroid' | None | array-like, default 'centroid'
        Point about which the force contributes to the torque via
        ``(x_cell - pivot) x F``. ``'centroid'`` uses each target's ``.centroid``
        (falling back to ``.position``). ``None`` omits the term. Arrays of shape
        ``(3,)``, ``(t, 3)`` or ``(t, p, 3)`` set explicit pivots.
    eps : float, default 1e-5
        Accepted for API compatibility with ``magpy.getFT`` but unused: the magnet
        gradient is computed by autodiff, so the result is independent of ``eps``
        and exact (an advantage over magpylib's finite differences).
    squeeze : bool, default True
        If ``True``, remove length-1 dimensions from the outputs.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(F, T)`` with shape ``(s, p, t, 3)`` before squeezing, where ``s`` is
        the number of sources, ``p`` the path length and ``t`` the number of
        targets.
    """
    src_list, src_group_starts = _as_source_list(sources)
    flat_targets, coll_idx = _flatten_targets(targets)

    n_src = len(src_list)
    n_src_out = len(src_group_starts)
    n_tgt_out = len(coll_idx)

    # Path length: max over source and target position paths (static shapes).
    tgt_path_lengths = [t._position.shape[0] for t in flat_targets]
    src_path_lengths = [s._position.shape[0] for s in src_list]
    n_path = max(tgt_path_lengths + src_path_lengths)

    piv = _format_pivot(pivot, flat_targets, n_path)

    # Pre-pad source paths and cache their field kwargs / type (kept in JAX).
    src_pos = [_pad_edge(s._position, n_path) for s in src_list]
    src_quat = [_pad_edge(s._orientation.as_quat(), n_path) for s in src_list]
    src_type = [s._source_type for s in src_list]
    src_kwargs = [s._field_kwargs() for s in src_list]

    # Pre-compute per-target meshes and padded target motion.
    meshes = [_generate_ft_mesh(t) for t in flat_targets]
    tgt_pos = [_pad_edge(t._position, n_path) for t in flat_targets]
    tgt_mat = [_pad_edge(t._orientation.as_matrix(), n_path) for t in flat_targets]

    n_tgt_flat = len(flat_targets)
    F_flat = [[[None] * n_tgt_flat for _ in range(n_path)] for _ in range(n_src)]
    T_flat = [[[None] * n_tgt_flat for _ in range(n_path)] for _ in range(n_src)]

    for si in range(n_src):
        s_type = src_type[si]
        s_kw = src_kwargs[si]
        for pi in range(n_path):
            pos_p = jnp.asarray(src_pos[si][pi])
            rot_p = R.from_quat(jnp.asarray(src_quat[si][pi]))

            def _bfun(x, s_type=s_type, pos_p=pos_p, rot_p=rot_p, s_kw=s_kw):
                return getB(
                    s_type,
                    x,
                    position=pos_p,
                    orientation=rot_p,
                    squeeze=True,
                    **s_kw,
                )

            b_points = jax.vmap(_bfun)
            jac_points = jax.vmap(jax.jacfwd(_bfun))

            for ti in range(n_tgt_flat):
                pts_local, weights, kind = meshes[ti]
                mat = jnp.asarray(tgt_mat[ti][pi])  # (3,3)
                pos = jnp.asarray(tgt_pos[ti][pi])  # (3,)

                x_c = pts_local @ mat.T + pos  # (n,3)
                w_g = weights @ mat.T  # (n,3)

                b_c = b_points(x_c)  # (n,3)

                if kind == "magnet":
                    jac = jac_points(x_c)  # (n,3,3): J[n,k,i]=dB_k/dx_i
                    f_cell = jnp.einsum("nk,nki->ni", w_g, jac)
                    t_cell = jnp.cross(w_g, b_c)
                else:  # current
                    f_cell = jnp.cross(w_g, b_c)
                    t_cell = jnp.zeros_like(f_cell)

                if piv is not None:
                    t_cell = t_cell + jnp.cross(x_c - piv[ti, pi], f_cell)

                F_flat[si][pi][ti] = jnp.sum(f_cell, axis=0)
                T_flat[si][pi][ti] = jnp.sum(t_cell, axis=0)

    # Assemble (s, p, t_flat, 3), then sum collection members -> (s, p, t_out, 3).
    F_arr = jnp.stack(
        [jnp.stack([jnp.stack(F_flat[si][pi], axis=0) for pi in range(n_path)], axis=0)
         for si in range(n_src)],
        axis=0,
    )
    T_arr = jnp.stack(
        [jnp.stack([jnp.stack(T_flat[si][pi], axis=0) for pi in range(n_path)], axis=0)
         for si in range(n_src)],
        axis=0,
    )

    if n_tgt_out != n_tgt_flat:
        ends = coll_idx[1:] + [n_tgt_flat]
        F_arr = jnp.stack(
            [jnp.sum(F_arr[:, :, coll_idx[g]:ends[g], :], axis=2) for g in range(n_tgt_out)],
            axis=2,
        )
        T_arr = jnp.stack(
            [jnp.sum(T_arr[:, :, coll_idx[g]:ends[g], :], axis=2) for g in range(n_tgt_out)],
            axis=2,
        )

    # Sum leaf sources within each top-level source group (Collection -> 1 source).
    if n_src_out != n_src:
        src_ends = src_group_starts[1:] + [n_src]
        F_arr = jnp.stack(
            [jnp.sum(F_arr[src_group_starts[g]:src_ends[g]], axis=0) for g in range(n_src_out)],
            axis=0,
        )
        T_arr = jnp.stack(
            [jnp.sum(T_arr[src_group_starts[g]:src_ends[g]], axis=0) for g in range(n_src_out)],
            axis=0,
        )

    if squeeze:
        return jnp.squeeze(F_arr), jnp.squeeze(T_arr)
    return F_arr, T_arr
