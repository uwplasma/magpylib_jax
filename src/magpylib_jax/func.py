"""High-level functional interface mirroring :mod:`magpylib.func`.

This module provides thin, magpylib-compatible wrappers around the string-based
functional API (:func:`magpylib_jax.getB` / :func:`magpylib_jax.getH`). Each
``*_field(field, observers, ...)`` function accepts ``i`` independent source
instances (broadcasting scalars/vectors up to the largest instance count) and
returns the field with shape ``(3,)`` for a single instance or ``(i, 3)`` for
several, matching :mod:`magpylib.func` exactly.

Signatures follow magpylib (``field`` in ``{'B', 'H'}``). Values are returned as
JAX arrays whose dtype follows the active JAX configuration (``float32`` unless
x64 is enabled).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib_jax._types import ArrayLike
from magpylib_jax.fields.api import getB, getH

__all__ = [
    "circle_field",
    "polyline_field",
    "cuboid_field",
    "cylinder_field",
    "cylinder_segment_field",
    "sphere_field",
    "tetrahedron_field",
    "dipole_field",
    "triangle_charge_field",
    "triangle_current_field",
]

_SCALAR: tuple[int, ...] = ()
_VEC: tuple[int, ...] = (3,)


def _stack_instances(
    name: str, value: ArrayLike, core_shape: tuple[int, ...]
) -> tuple[jnp.ndarray, int]:
    """Return ``value`` with a leading instance axis plus the instance count."""
    arr = jnp.asarray(value, dtype=float)
    core_ndim = len(core_shape)
    if arr.ndim == core_ndim:
        if core_ndim and arr.shape != core_shape:
            raise ValueError(
                f"Input {name} must have shape {core_shape} for a single instance; "
                f"instead received shape {arr.shape}."
            )
        return arr[None, ...], 1
    if arr.ndim == core_ndim + 1:
        if core_ndim and arr.shape[1:] != core_shape:
            raise ValueError(
                f"Input {name} must have shape {core_shape} for ndim > 0; "
                f"instead received shape {arr.shape[1:]}."
            )
        return arr, int(arr.shape[0])
    raise ValueError(
        f"Input {name} must have at most {core_ndim + 1} dimensions; "
        f"instead received ndim {arr.ndim}."
    )


def _getbh_func(
    source_type: str,
    field: str,
    observers: ArrayLike,
    positions: ArrayLike,
    orientations: R | None,
    squeeze: bool,
    params: Sequence[tuple[str, ArrayLike, tuple[int, ...]]],
    *,
    trianglesheet: bool = False,
) -> jnp.ndarray:
    """Shared broadcasting/dispatch for the ``func`` wrappers.

    ``params`` maps each source-kernel keyword to ``(value, core_shape)``. All
    instance counts must be ``1`` or the common maximum ``nmax`` (magpylib rule).
    """
    if field not in ("B", "H"):
        raise ValueError(f"Input field must be one of ('B', 'H'); instead received {field!r}.")
    getfield = getB if field == "B" else getH

    stacks: list[tuple[str, jnp.ndarray, int]] = []
    obs_stack, n_obs = _stack_instances("observers", observers, _VEC)
    pos_stack, n_pos = _stack_instances("positions", positions, _VEC)
    counts = [n_obs, n_pos]
    for name, value, core in params:
        arr, n = _stack_instances(name, value, core)
        stacks.append((name, arr, n))
        counts.append(n)
    nmax = max(counts)

    for name, n in [("observers", n_obs), ("positions", n_pos), *[(s[0], s[2]) for s in stacks]]:
        if n not in (1, nmax):
            raise ValueError(
                f"Input {name} must have 1 or {nmax} instances; instead received {n}."
            )

    if orientations is None:
        ori_mats: np.ndarray | None = None
        n_ori = 1
    elif isinstance(orientations, R):
        quat = np.atleast_2d(np.asarray(orientations.as_quat(), dtype=float))
        n_ori = int(quat.shape[0])
        if n_ori not in (1, nmax):
            raise ValueError(
                f"Input orientation must have 1 or {nmax} instances; instead received {n_ori}."
            )
        ori_mats = R.from_quat(quat).as_matrix()
    else:
        raise TypeError(
            "Input orientation must be a SciPy Rotation instance or None; "
            f"instead received type {type(orientations).__name__}."
        )

    results = []
    for i in range(nmax):
        kwargs: dict[str, ArrayLike] = {}
        for name, arr, n in stacks:
            val = arr[0] if n == 1 else arr[i]
            if trianglesheet and name == "current_densities":
                kwargs[name] = val[None, :]
            else:
                kwargs[name] = val
        if trianglesheet:
            kwargs["faces"] = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
        obs_i = obs_stack[0] if n_obs == 1 else obs_stack[i]
        pos_i = pos_stack[0] if n_pos == 1 else pos_stack[i]
        ori_i = None if ori_mats is None else ori_mats[0 if n_ori == 1 else i]
        results.append(
            getfield(source_type, obs_i, position=pos_i, orientation=ori_i, squeeze=True, **kwargs)
        )

    out = jnp.stack(results, axis=0)
    if squeeze:
        return jnp.squeeze(out)
    return out


def circle_field(
    field: str,
    observers: ArrayLike,
    diameters: ArrayLike,
    currents: ArrayLike,
    positions: ArrayLike = (0, 0, 0),
    orientations: R | None = None,
    squeeze: bool = True,
) -> jnp.ndarray:
    """Return the B- or H-field of circular current loops for ``i`` instances."""
    return _getbh_func(
        "circle",
        field,
        observers,
        positions,
        orientations,
        squeeze,
        [("diameter", diameters, _SCALAR), ("current", currents, _SCALAR)],
    )


def polyline_field(
    field: str,
    observers: ArrayLike,
    segments_start: ArrayLike,
    segments_end: ArrayLike,
    currents: ArrayLike,
    positions: ArrayLike = (0, 0, 0),
    orientations: R | None = None,
    squeeze: bool = True,
) -> jnp.ndarray:
    """Return the B- or H-field of straight current segments for ``i`` instances."""
    return _getbh_func(
        "polyline",
        field,
        observers,
        positions,
        orientations,
        squeeze,
        [
            ("segment_start", segments_start, _VEC),
            ("segment_end", segments_end, _VEC),
            ("current", currents, _SCALAR),
        ],
    )


def cuboid_field(
    field: str,
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
    positions: ArrayLike = (0, 0, 0),
    orientations: R | None = None,
    squeeze: bool = True,
) -> jnp.ndarray:
    """Return the B- or H-field of cuboid magnets for ``i`` instances."""
    return _getbh_func(
        "cuboid",
        field,
        observers,
        positions,
        orientations,
        squeeze,
        [("dimension", dimensions, (3,)), ("polarization", polarizations, _VEC)],
    )


def cylinder_field(
    field: str,
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
    positions: ArrayLike = (0, 0, 0),
    orientations: R | None = None,
    squeeze: bool = True,
) -> jnp.ndarray:
    """Return the B- or H-field of solid cylinder magnets for ``i`` instances."""
    return _getbh_func(
        "cylinder",
        field,
        observers,
        positions,
        orientations,
        squeeze,
        [("dimension", dimensions, (2,)), ("polarization", polarizations, _VEC)],
    )


def cylinder_segment_field(
    field: str,
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
    positions: ArrayLike = (0, 0, 0),
    orientations: R | None = None,
    squeeze: bool = True,
) -> jnp.ndarray:
    """Return the B- or H-field of cylinder-segment magnets for ``i`` instances."""
    return _getbh_func(
        "cylindersegment",
        field,
        observers,
        positions,
        orientations,
        squeeze,
        [("dimension", dimensions, (5,)), ("polarization", polarizations, _VEC)],
    )


def sphere_field(
    field: str,
    observers: ArrayLike,
    diameters: ArrayLike,
    polarizations: ArrayLike,
    positions: ArrayLike = (0, 0, 0),
    orientations: R | None = None,
    squeeze: bool = True,
) -> jnp.ndarray:
    """Return the B- or H-field of sphere magnets for ``i`` instances."""
    return _getbh_func(
        "sphere",
        field,
        observers,
        positions,
        orientations,
        squeeze,
        [("diameter", diameters, _SCALAR), ("polarization", polarizations, _VEC)],
    )


def tetrahedron_field(
    field: str,
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    positions: ArrayLike = (0, 0, 0),
    orientations: R | None = None,
    squeeze: bool = True,
) -> jnp.ndarray:
    """Return the B- or H-field of tetrahedron magnets for ``i`` instances."""
    return _getbh_func(
        "tetrahedron",
        field,
        observers,
        positions,
        orientations,
        squeeze,
        [("vertices", vertices, (4, 3)), ("polarization", polarizations, _VEC)],
    )


def dipole_field(
    field: str,
    observers: ArrayLike,
    moments: ArrayLike,
    positions: ArrayLike = (0, 0, 0),
    orientations: R | None = None,
    squeeze: bool = True,
) -> jnp.ndarray:
    """Return the B- or H-field of magnetic dipoles for ``i`` instances."""
    return _getbh_func(
        "dipole",
        field,
        observers,
        positions,
        orientations,
        squeeze,
        [("moment", moments, _VEC)],
    )


def triangle_charge_field(
    field: str,
    observers: ArrayLike,
    vertices: ArrayLike,
    polarizations: ArrayLike,
    positions: ArrayLike = (0, 0, 0),
    orientations: R | None = None,
    squeeze: bool = True,
) -> jnp.ndarray:
    """Return the B- or H-field of magnetically charged triangles for ``i`` instances."""
    return _getbh_func(
        "triangle",
        field,
        observers,
        positions,
        orientations,
        squeeze,
        [("vertices", vertices, (3, 3)), ("polarization", polarizations, _VEC)],
    )


def triangle_current_field(
    field: str,
    observers: ArrayLike,
    vertices: ArrayLike,
    current_densities: ArrayLike,
    positions: ArrayLike = (0, 0, 0),
    orientations: R | None = None,
    squeeze: bool = True,
) -> jnp.ndarray:
    """Return the B- or H-field of triangular current sheets for ``i`` instances."""
    return _getbh_func(
        "trianglesheet",
        field,
        observers,
        positions,
        orientations,
        squeeze,
        [("vertices", vertices, (3, 3)), ("current_densities", current_densities, _VEC)],
        trianglesheet=True,
    )
