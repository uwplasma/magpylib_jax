"""Tetrahedron magnet field kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import ensure_observers
from magpylib_jax.core.kernels._common import (
    _IN_OUT_FLAGS,
    _broadcast_vec3,
    _in_out_flag,
    _jit_kernel,
)
from magpylib_jax.core.kernels.triangle import (
    _triangle_bfield_const_precomp,
    _triangle_geom_terms,
    triangle_bfield,
)

_TETRA_FACES = jnp.array(
    [
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [0, 3, 2],
    ],
    dtype=jnp.int32,
)


def _check_tetra_chirality(vertices: jnp.ndarray) -> jnp.ndarray:
    vecs = jnp.stack(
        (
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ),
        axis=-1,
    )
    dets = jnp.linalg.det(vecs)
    swap = dets < 0
    v = vertices
    v_swapped = v.at[:, 2:4].set(v[:, 3:1:-1])
    return jnp.where(swap[:, None, None], v_swapped, v)


def _points_inside_tetra(points: jnp.ndarray, vertices: jnp.ndarray) -> jnp.ndarray:
    mat = jnp.transpose(vertices[:, 1:] - vertices[:, 0][:, None, :], (0, 2, 1))
    inv = jnp.linalg.inv(mat)
    delta = (points - vertices[:, 0])[:, :, None]
    newp = jnp.matmul(inv, delta).squeeze(-1)
    return (
        jnp.all(newp >= 0.0, axis=1) & jnp.all(newp <= 1.0, axis=1) & (jnp.sum(newp, axis=1) <= 1.0)
    )


def _points_inside_tetra_single(points: jnp.ndarray, vertices: jnp.ndarray) -> jnp.ndarray:
    mat = (vertices[1:] - vertices[0]).T
    inv = jnp.linalg.inv(mat)
    delta = points - vertices[0]
    newp = jnp.matmul(delta, inv.T)
    return (
        jnp.all(newp >= 0.0, axis=1) & jnp.all(newp <= 1.0, axis=1) & (jnp.sum(newp, axis=1) <= 1.0)
    )


def _tetrahedron_bfield_const_impl(
    obs: jnp.ndarray,
    tet_const: jnp.ndarray,
    pol: jnp.ndarray,
    *,
    in_out_flag: int,
) -> jnp.ndarray:
    tet_const = _check_tetra_chirality(tet_const[None, :, :])[0]
    faces = tet_const[_TETRA_FACES]
    nvec, L, l1, l2 = _triangle_geom_terms(faces)
    b_faces = jax.vmap(
        _triangle_bfield_const_precomp,
        in_axes=(None, 0, None, 0, 0, 0, 0),
    )(obs, faces, pol, nvec, L, l1, l2)
    b = jnp.sum(b_faces, axis=0)

    if in_out_flag == _IN_OUT_FLAGS["outside"]:
        inside = jnp.zeros((obs.shape[0],), dtype=bool)
    elif in_out_flag == _IN_OUT_FLAGS["inside"]:
        inside = jnp.ones((obs.shape[0],), dtype=bool)
    else:
        inside = _points_inside_tetra_single(obs, tet_const)
    return b + jnp.where(inside[:, None], pol, 0.0)


def tetrahedron_bfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    n = obs.shape[0]

    tet = jnp.asarray(vertices, dtype=jnp.float64)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)
    if tet.ndim == 2 or (tet.ndim == 3 and tet.shape[0] == 1):
        tet_const = tet if tet.ndim == 2 else tet[0]
        tet_const = _check_tetra_chirality(tet_const[None, :, :])[0]
        faces = tet_const[_TETRA_FACES]
        nvec, L, l1, l2 = _triangle_geom_terms(faces)
        b_faces = jax.vmap(
            _triangle_bfield_const_precomp,
            in_axes=(None, 0, None, 0, 0, 0, 0),
        )(obs, faces, pol, nvec, L, l1, l2)
        b = jnp.sum(b_faces, axis=0)

        if in_out == "inside":
            inside = jnp.ones((n,), dtype=bool)
        elif in_out == "outside":
            inside = jnp.zeros((n,), dtype=bool)
        else:
            inside = _points_inside_tetra_single(obs, tet_const)
        return b + jnp.where(inside[:, None], pol, 0.0)

    tet = jnp.broadcast_to(tet, (n, 4, 3))
    tet = _check_tetra_chirality(tet)
    faces = tet[:, _TETRA_FACES, :]
    b = jnp.sum(
        jax.vmap(lambda tri: triangle_bfield(obs, tri, pol))(faces.swapaxes(0, 1)),
        axis=0,
    )

    if in_out == "inside":
        inside = jnp.ones((n,), dtype=bool)
    elif in_out == "outside":
        inside = jnp.zeros((n,), dtype=bool)
    else:
        inside = _points_inside_tetra(obs, tet)
    return b + jnp.where(inside[:, None], pol, 0.0)


def tetrahedron_bfield_jit(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """JIT-specialized tetrahedron B-field for fixed observer counts."""
    obs = ensure_observers(observers)
    tet = jnp.asarray(vertices, dtype=jnp.float64)
    if tet.ndim == 3 and tet.shape[0] == 1:
        tet = tet[0]
    if tet.ndim != 2:
        return tetrahedron_bfield(obs, tet, polarizations, in_out=in_out)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    flag = _in_out_flag(in_out)
    jit_fn = _jit_kernel(
        "tetrahedron_bfield",
        _tetrahedron_bfield_const_impl,
        obs.shape[0],
        flag,
    )
    return jit_fn(obs, tet, pol, in_out_flag=flag)


def tetrahedron_hfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    b = tetrahedron_bfield(observers, vertices, polarizations, in_out=in_out)
    j = tetrahedron_jfield(observers, vertices, polarizations, in_out=in_out)
    return (b - j) / MU0


def tetrahedron_jfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    n = obs.shape[0]

    tet = jnp.asarray(vertices, dtype=jnp.float64)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)
    if tet.ndim == 2 or (tet.ndim == 3 and tet.shape[0] == 1):
        tet_const = tet if tet.ndim == 2 else tet[0]
        tet_const = _check_tetra_chirality(tet_const[None, :, :])[0]
        if in_out == "inside":
            inside = jnp.ones((n,), dtype=bool)
        elif in_out == "outside":
            inside = jnp.zeros((n,), dtype=bool)
        else:
            inside = _points_inside_tetra_single(obs, tet_const)
        return jnp.where(inside[:, None], pol, 0.0)

    tet = jnp.broadcast_to(tet, (n, 4, 3))
    if in_out == "inside":
        inside = jnp.ones((n,), dtype=bool)
    elif in_out == "outside":
        inside = jnp.zeros((n,), dtype=bool)
    else:
        inside = _points_inside_tetra(obs, tet)
    return jnp.where(inside[:, None], pol, 0.0)


def tetrahedron_mfield(
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    return tetrahedron_jfield(observers, vertices, polarizations, in_out=in_out) / MU0
