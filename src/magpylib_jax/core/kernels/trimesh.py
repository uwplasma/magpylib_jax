"""Triangular mesh magnet field kernels."""

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
    _jit_kernel_mesh,
)
from magpylib_jax.core.kernels._raycast import _inside_mask_mesh, _inside_mask_mesh_masked
from magpylib_jax.core.kernels.triangle import (
    _triangle_bfield_const_precomp,
    _triangle_geom_terms,
    triangle_bfield,
)


def _broadcast_mesh(mesh: jnp.ndarray, n: int) -> jnp.ndarray:
    if mesh.ndim == 3:
        return jnp.broadcast_to(mesh[None, :, :, :], (n, *mesh.shape))
    if mesh.ndim == 4:
        return jnp.broadcast_to(mesh, (n, mesh.shape[1], 3, 3))
    raise ValueError(f"Expected mesh shape (t,3,3) or (n,t,3,3), got {mesh.shape}.")


def magnet_trimesh_bfield(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """B-field of uniformly polarized closed triangular meshes."""
    obs = ensure_observers(observers)
    n = obs.shape[0]
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    if mesh_arr.ndim == 4:
        mesh_arr = _broadcast_mesh(mesh_arr, n)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)

    # Evaluate each face as a batched triangle field and reduce over faces.
    # This avoids flatten+repeat expansions and lowers peak memory pressure.
    if mesh_arr.ndim == 3:
        flag = _in_out_flag(in_out)
        return _magnet_trimesh_bfield_const_impl(obs, mesh_arr, pol, in_out_flag=flag)

    mesh_by_face = jnp.swapaxes(mesh_arr, 0, 1)  # (n_faces, n_obs, 3, 3)
    b_faces = jax.vmap(lambda face_vertices: triangle_bfield(obs, face_vertices, pol))(mesh_by_face)
    b = jnp.sum(b_faces, axis=0)

    if in_out == "outside":
        inside = jnp.zeros((n,), dtype=bool)
    elif in_out == "inside":
        inside = jnp.ones((n,), dtype=bool)
    else:
        inside = _inside_mask_mesh(obs, mesh_arr)

    return b + jnp.where(inside[:, None], pol, 0.0)


def _magnet_trimesh_bfield_const_impl(
    obs: jnp.ndarray,
    mesh_arr: jnp.ndarray,
    pol: jnp.ndarray,
    *,
    in_out_flag: int,
) -> jnp.ndarray:
    nvec, L, l1, l2 = _triangle_geom_terms(mesh_arr)

    def _accumulate_faces() -> jnp.ndarray:
        def body(i: int, acc: jnp.ndarray) -> jnp.ndarray:
            return acc + _triangle_bfield_const_precomp(
                obs, mesh_arr[i], pol, nvec[i], L[i], l1[i], l2[i]
            )

        init = jnp.zeros((obs.shape[0], 3), dtype=jnp.float64)
        return jax.lax.fori_loop(0, mesh_arr.shape[0], body, init)

    if mesh_arr.shape[0] <= 64:
        b_faces = jax.vmap(
            _triangle_bfield_const_precomp,
            in_axes=(None, 0, None, 0, 0, 0, 0),
        )(obs, mesh_arr, pol, nvec, L, l1, l2)
        b = jnp.sum(b_faces, axis=0)
    else:
        b = _accumulate_faces()

    if in_out_flag == _IN_OUT_FLAGS["outside"]:
        inside = jnp.zeros((obs.shape[0],), dtype=bool)
    elif in_out_flag == _IN_OUT_FLAGS["inside"]:
        inside = jnp.ones((obs.shape[0],), dtype=bool)
    else:
        inside = _inside_mask_mesh(obs, mesh_arr)
    return b + jnp.where(inside[:, None], pol, 0.0)


def precompute_trimesh_geometry(
    mesh: ArrayLike,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute triangle mesh geometry terms for reuse."""
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    if mesh_arr.ndim != 3 or mesh_arr.shape[1:] != (3, 3):
        raise ValueError("Mesh must have shape (n_faces,3,3).")
    nvec, L, l1, l2 = _triangle_geom_terms(mesh_arr)
    return mesh_arr, nvec, L, l1, l2


def _magnet_trimesh_bfield_precomp_impl(
    obs: jnp.ndarray,
    mesh_arr: jnp.ndarray,
    pol: jnp.ndarray,
    nvec: jnp.ndarray,
    L: jnp.ndarray,
    l1: jnp.ndarray,
    l2: jnp.ndarray,
    *,
    in_out_flag: int,
    n_faces: int,
) -> jnp.ndarray:
    def _accumulate_faces() -> jnp.ndarray:
        def body(i: int, acc: jnp.ndarray) -> jnp.ndarray:
            return acc + _triangle_bfield_const_precomp(
                obs, mesh_arr[i], pol, nvec[i], L[i], l1[i], l2[i]
            )

        init = jnp.zeros((obs.shape[0], 3), dtype=jnp.float64)
        return jax.lax.fori_loop(0, n_faces, body, init)

    if n_faces <= 64:
        b_faces = jax.vmap(
            _triangle_bfield_const_precomp,
            in_axes=(None, 0, None, 0, 0, 0, 0),
        )(obs, mesh_arr, pol, nvec, L, l1, l2)
        b = jnp.sum(b_faces, axis=0)
    else:
        b = _accumulate_faces()

    if in_out_flag == _IN_OUT_FLAGS["outside"]:
        inside = jnp.zeros((obs.shape[0],), dtype=bool)
    elif in_out_flag == _IN_OUT_FLAGS["inside"]:
        inside = jnp.ones((obs.shape[0],), dtype=bool)
    else:
        inside = _inside_mask_mesh(obs, mesh_arr)
    return b + jnp.where(inside[:, None], pol, 0.0)


def magnet_trimesh_bfield_precomp_masked(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    nvec: ArrayLike,
    L: ArrayLike,
    l1: ArrayLike,
    l2: ArrayLike,
    face_mask: ArrayLike,
    in_out_flag: int,
) -> jnp.ndarray:
    """B-field of triangular mesh using precomputed geometry with face masking."""
    obs = ensure_observers(observers)
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    nvec_arr = jnp.asarray(nvec, dtype=jnp.float64)
    L_arr = jnp.asarray(L, dtype=jnp.float64)
    l1_arr = jnp.asarray(l1, dtype=jnp.float64)
    l2_arr = jnp.asarray(l2, dtype=jnp.float64)
    mask = jnp.asarray(face_mask, dtype=bool).reshape((-1,))
    n_faces = mesh_arr.shape[0]

    def _accumulate_faces() -> jnp.ndarray:
        def body(i: int, acc: jnp.ndarray) -> jnp.ndarray:
            term = _triangle_bfield_const_precomp(
                obs, mesh_arr[i], pol, nvec_arr[i], L_arr[i], l1_arr[i], l2_arr[i]
            )
            term = jnp.where(mask[i], term, 0.0)
            return acc + term

        init = jnp.zeros((obs.shape[0], 3), dtype=jnp.float64)
        return jax.lax.fori_loop(0, n_faces, body, init)

    if n_faces <= 64:
        b_faces = jax.vmap(
            _triangle_bfield_const_precomp,
            in_axes=(None, 0, None, 0, 0, 0, 0),
        )(obs, mesh_arr, pol, nvec_arr, L_arr, l1_arr, l2_arr)
        b_faces = jnp.where(mask[:, None, None], b_faces, 0.0)
        b = jnp.sum(b_faces, axis=0)
    else:
        b = _accumulate_faces()

    inside = jax.lax.switch(
        in_out_flag,
        (
            lambda: _inside_mask_mesh_masked(obs, mesh_arr, mask),
            lambda: jnp.ones((obs.shape[0],), dtype=bool),
            lambda: jnp.zeros((obs.shape[0],), dtype=bool),
        ),
    )
    return b + jnp.where(inside[:, None], pol, 0.0)


def _magnet_trimesh_bfield_faces_impl(
    obs: jnp.ndarray,
    mesh_arr: jnp.ndarray,
    pol: jnp.ndarray,
    *,
    in_out_flag: int,
    n_faces: int,
) -> jnp.ndarray:
    return _magnet_trimesh_bfield_const_impl(obs, mesh_arr, pol, in_out_flag=in_out_flag)


def magnet_trimesh_bfield_jit(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """JIT-specialized triangular mesh B-field for fixed observer counts."""
    obs = ensure_observers(observers)
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    if mesh_arr.ndim == 3:
        return magnet_trimesh_bfield_jit_faces(obs, mesh_arr, pol, in_out=in_out)
    flag = _in_out_flag(in_out)
    jit_fn = _jit_kernel(
        "triangularmesh_bfield",
        _magnet_trimesh_bfield_const_impl,
        obs.shape[0],
        flag,
    )
    return jit_fn(obs, mesh_arr, pol, in_out_flag=flag)


def magnet_trimesh_bfield_jit_faces(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """JIT-specialized triangular mesh B-field for fixed observer + face counts."""
    obs = ensure_observers(observers)
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    if mesh_arr.ndim != 3:
        raise ValueError("TriangularMesh JIT expects mesh with shape (n_faces,3,3).")
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    flag = _in_out_flag(in_out)
    n_faces = int(mesh_arr.shape[0])
    jit_fn = _jit_kernel_mesh(
        "triangularmesh_bfield_faces",
        _magnet_trimesh_bfield_faces_impl,
        obs.shape[0],
        n_faces,
        flag,
    )
    return jit_fn(obs, mesh_arr, pol, in_out_flag=flag, n_faces=n_faces)


def magnet_trimesh_bfield_jit_faces_precomp(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    nvec: ArrayLike,
    L: ArrayLike,
    l1: ArrayLike,
    l2: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    """JIT-specialized triangular mesh B-field using precomputed geometry."""
    obs = ensure_observers(observers)
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    if mesh_arr.ndim != 3:
        raise ValueError("TriangularMesh JIT expects mesh with shape (n_faces,3,3).")
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape[0])
    n_faces = int(mesh_arr.shape[0])
    flag = _in_out_flag(in_out)
    jit_fn = _jit_kernel_mesh(
        "triangularmesh_bfield_precomp",
        _magnet_trimesh_bfield_precomp_impl,
        obs.shape[0],
        n_faces,
        flag,
    )
    return jit_fn(
        obs,
        mesh_arr,
        pol,
        jnp.asarray(nvec, dtype=jnp.float64),
        jnp.asarray(L, dtype=jnp.float64),
        jnp.asarray(l1, dtype=jnp.float64),
        jnp.asarray(l2, dtype=jnp.float64),
        in_out_flag=flag,
        n_faces=n_faces,
    )


def magnet_trimesh_hfield(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    b = magnet_trimesh_bfield(observers, mesh, polarizations, in_out=in_out)
    j = magnet_trimesh_jfield(observers, mesh, polarizations, in_out=in_out)
    return (b - j) / MU0


def magnet_trimesh_jfield(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    obs = ensure_observers(observers)
    n = obs.shape[0]
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    if mesh_arr.ndim == 4:
        mesh_arr = _broadcast_mesh(mesh_arr, n)
    pol = _broadcast_vec3(jnp.asarray(polarizations, dtype=jnp.float64), n)

    if in_out == "outside":
        inside = jnp.zeros((n,), dtype=bool)
    elif in_out == "inside":
        inside = jnp.ones((n,), dtype=bool)
    else:
        inside = _inside_mask_mesh(obs, mesh_arr)

    return jnp.where(inside[:, None], pol, 0.0)


def magnet_trimesh_mfield(
    observers: ArrayLike,
    mesh: ArrayLike,
    polarizations: ArrayLike,
    in_out: str = "auto",
) -> jnp.ndarray:
    return magnet_trimesh_jfield(observers, mesh, polarizations, in_out=in_out) / MU0
