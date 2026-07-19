"""Public field API, config constants and low-level helpers.

This module is a leaf of the ``fields`` package: it depends only on the
standard library and JAX so that every other ``fields`` submodule can import
its constants and helpers without creating an import cycle. The public
``getB``/``getH``/``getJ``/``getM`` entry points and the ``_compute_field``
router lazy-import the heavier engine/eager submodules inside their bodies.
"""

from __future__ import annotations

from math import prod

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike

_ALIASES = {
    "dipole": "dipole",
    "circle": "circle",
    "cuboid": "cuboid",
    "box": "cuboid",
    "cylinder": "cylinder",
    "cylindersegment": "cylindersegment",
    "cylinder_segment": "cylindersegment",
    "sphere": "sphere",
    "polyline": "polyline",
    "triangle": "triangle",
    "trianglesheet": "trianglesheet",
    "triangle_sheet": "trianglesheet",
    "trianglestrip": "trianglestrip",
    "triangle_strip": "trianglestrip",
    "triangularmesh": "triangularmesh",
    "triangular_mesh": "triangularmesh",
    "tetrahedron": "tetrahedron",
}

_SOURCE_TYPE_ORDER = (
    "dipole",
    "circle",
    "cuboid",
    "cylinder",
    "cylindersegment",
    "sphere",
    "triangle",
    "polyline",
    "trianglesheet",
    "trianglestrip",
    "triangularmesh",
    "tetrahedron",
)
_SOURCE_TYPE_IDS = {name: idx for idx, name in enumerate(_SOURCE_TYPE_ORDER)}
_SUPPORTED_PIXEL_AGGS = {"mean", "sum", "min", "max"}


def _check_getbh_output_type(output: str) -> str:
    acceptable = ("ndarray", "dataframe")
    if output not in acceptable:
        msg = f"Input output must be one of {acceptable}; instead received {output!r}."
        raise ValueError(msg)
    if output == "dataframe":
        try:  # pragma: no cover - import check
            import pandas as _  # noqa: F401
        except Exception as err:  # pragma: no cover
            msg = (
                "Input output='dataframe' requires Pandas installation, "
                "see https://pandas.pydata.org/docs/getting_started/install.html"
            )
            raise ModuleNotFoundError(msg) from err
    return output


def _check_pixel_agg(pixel_agg: str | None):
    if pixel_agg is None:
        return None
    if callable(pixel_agg):
        return pixel_agg
    if not isinstance(pixel_agg, str):
        raise AttributeError(
            "Input pixel_agg must be a reference to a NumPy callable that reduces "
            "an array shape like 'mean', 'std', 'median', 'min', ...; "
            f"instead received {pixel_agg!r}."
        )
    if not hasattr(jnp, pixel_agg):
        raise AttributeError(
            "Input pixel_agg must be a reference to a NumPy callable that reduces "
            "an array shape like 'mean', 'std', 'median', 'min', ...; "
            f"instead received {pixel_agg!r}."
        )
    return getattr(jnp, pixel_agg)


def _is_array_like(obj: object) -> bool:
    if isinstance(obj, (jnp.ndarray, jax.Array)):
        return True
    if isinstance(obj, jax.core.Tracer):
        return True
    return hasattr(obj, "shape") and hasattr(obj, "dtype")


def _has_tracer(obj: object) -> bool:
    if isinstance(obj, jax.core.Tracer):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_has_tracer(item) for item in obj)
    if isinstance(obj, dict):
        return any(_has_tracer(val) for val in obj.values())
    return False


def _pad_path(arr: ArrayLike, target_len: int) -> jnp.ndarray:
    arr_jnp = jnp.asarray(arr, dtype=float)
    if arr_jnp.shape[0] == target_len:
        return arr_jnp
    if arr_jnp.shape[0] == 1:
        return jnp.broadcast_to(arr_jnp, (target_len,) + arr_jnp.shape[1:])
    pad_len = target_len - arr_jnp.shape[0]
    pad = jnp.broadcast_to(arr_jnp[-1:], (pad_len,) + arr_jnp.shape[1:])
    return jnp.concatenate((arr_jnp, pad), axis=0)


def _pad_axis0(arr: ArrayLike, target_len: int, pad_value: float = 0.0) -> jnp.ndarray:
    arr_jnp = jnp.asarray(arr, dtype=float)
    if arr_jnp.shape[0] == target_len:
        return arr_jnp
    pad_shape = (target_len - arr_jnp.shape[0],) + arr_jnp.shape[1:]
    pad = jnp.full(pad_shape, pad_value, dtype=arr_jnp.dtype)
    return jnp.concatenate((arr_jnp, pad), axis=0)


def _normalize_source_type(source_type: str) -> str:
    key = source_type.lower()
    if key not in _ALIASES:
        valid = sorted(_ALIASES)
        raise ValueError(f"Unsupported source_type {source_type!r}. Expected one of {valid}.")
    return _ALIASES[key]


def _apply_squeeze(
    field: jnp.ndarray,
    observer_input: jnp.ndarray,
    *,
    squeeze: bool,
    sumup: bool,
    n_sources: int,
) -> jnp.ndarray:
    obs_prefix = observer_input.shape[:-1] if observer_input.ndim > 1 else ()
    n_obs = prod(obs_prefix) if obs_prefix else 1

    def _unflatten_obs(arr: jnp.ndarray) -> jnp.ndarray:
        if arr.ndim >= 2 and arr.shape[-1] == 3 and arr.shape[-2] == n_obs:
            return arr.reshape((*arr.shape[:-2], *obs_prefix, 3))
        return arr

    field = _unflatten_obs(field)

    if squeeze:
        axes = tuple(i for i, size in enumerate(field.shape[:-1]) if size == 1)
        return jnp.squeeze(field, axis=axes) if axes else field

    obs_has_path = observer_input.ndim >= 3
    if sumup:
        if field.ndim == len(obs_prefix) + 2:
            path_len = field.shape[0]
            pix = field.shape[1:-1]
        elif obs_has_path:
            path_len = field.shape[0]
            pix = field.shape[1:-1]
        else:
            path_len = 1
            pix = field.shape[:-1]
        arr = field.reshape((1, path_len, 1, *pix, 3))
    else:
        if n_sources == 1 and (field.ndim == observer_input.ndim or field.shape[0] != 1):
            field = field[None, ...]
        rest = field.shape[1:-1]
        if len(rest) > len(obs_prefix):
            path_len = rest[0]
            pix = rest[1:]
        elif obs_has_path and len(rest) >= 1:
            path_len = rest[0]
            pix = rest[1:]
        else:
            path_len = 1
            pix = rest
        arr = field.reshape((n_sources, path_len, 1, *pix, 3))
    return arr


# Cache of jitted field evaluations, keyed on the static call structure. Passing
# params/pose/observers as *traced* arguments lets the Python-side preparation run
# once per (source type, shapes); repeated eager calls and parameter-varying loops
# then reuse the compiled function instead of re-preparing every call. Bounded so a
# workload that sweeps many distinct shapes cannot grow the cache without limit.
_FIELD_JIT_CACHE: dict = {}
_FIELD_JIT_CACHE_MAX = 256


def _orientation_to_matrix(orientation: object) -> jnp.ndarray:
    if orientation is None:
        return jnp.eye(3)
    if hasattr(orientation, "as_matrix"):
        return jnp.asarray(orientation.as_matrix(), dtype=float)
    return jnp.asarray(orientation, dtype=float)


def _cached_field(
    source: str,
    observers: object,
    field: str,
    *,
    position: ArrayLike,
    orientation: object,
    squeeze: bool,
    sumup: bool,
    pixel_agg: str | None,
    in_out: str,
    kwargs: dict,
) -> jnp.ndarray | None:
    """Fast path: a cached, jitted wrapper around ``_compute_field_jit``.

    Returns ``None`` when the fast path does not apply (the caller then uses the
    standard path). Numerically identical to the standard path — it is the same
    computation under ``jax.jit`` — but the preparation is traced once per shape.
    """
    from magpylib_jax.fields.engine import _compute_field_jit

    obs = jnp.asarray(observers, dtype=float)
    if obs.ndim < 1 or obs.shape[-1] != 3:
        return None
    pos = jnp.asarray(position, dtype=float)
    ori = _orientation_to_matrix(orientation)
    names = tuple(sorted(kwargs))
    vals = tuple(jnp.asarray(kwargs[n]) for n in names)

    key = (
        source, field, squeeze, sumup, pixel_agg, in_out,
        obs.shape, pos.shape, ori.shape, names,
        tuple(v.shape for v in vals), tuple(str(v.dtype) for v in vals),
    )
    fn = _FIELD_JIT_CACHE.get(key)
    if fn is None:

        def _core(obs, pos, ori, *pvals, _n=names, _s=source, _f=field,
                  _sq=squeeze, _su=sumup, _pa=pixel_agg, _io=in_out):
            return _compute_field_jit(
                _s, obs, _f, position=pos, orientation=ori,
                squeeze=_sq, sumup=_su, pixel_agg=_pa, output="ndarray", in_out=_io,
                **dict(zip(_n, pvals, strict=True)),
            )

        fn = jax.jit(_core)
        if len(_FIELD_JIT_CACHE) >= _FIELD_JIT_CACHE_MAX:
            _FIELD_JIT_CACHE.pop(next(iter(_FIELD_JIT_CACHE)))
        _FIELD_JIT_CACHE[key] = fn
    return fn(obs, pos, ori, *vals)


def _compute_field(
    source: str | object,
    observers: object,
    field: str,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    pixel_agg: str | None = None,
    output: str = "ndarray",
    in_out: str = "auto",
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    from magpylib_jax.fields.eager import _compute_field_legacy
    from magpylib_jax.fields.engine import _compute_field_jit

    if isinstance(source, str):
        src_type = _normalize_source_type(source)
        if src_type == "triangularmesh":
            mesh = kwargs.get("mesh")
            if mesh is not None:
                mesh_arr = jnp.asarray(mesh)
                if mesh_arr.ndim == 4:
                    return _compute_field_legacy(
                        source,
                        observers,
                        field,
                        position=position,
                        orientation=orientation,
                        squeeze=squeeze,
                        sumup=sumup,
                        pixel_agg=pixel_agg,
                        output=output,
                        in_out=in_out,
                        **kwargs,
                    )
    if pixel_agg is not None and not isinstance(pixel_agg, str):
        return _compute_field_legacy(
            source,
            observers,
            field,
            position=position,
            orientation=orientation,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
            **kwargs,
        )
    if isinstance(pixel_agg, str) and pixel_agg not in _SUPPORTED_PIXEL_AGGS:
        return _compute_field_legacy(
            source,
            observers,
            field,
            position=position,
            orientation=orientation,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
            **kwargs,
        )

    # Fast path: a string source with array observers and no tracers can reuse a
    # cached jitted evaluation (params/pose traced), avoiding per-call preparation.
    if (
        isinstance(source, str)
        and output == "ndarray"
        and not _has_tracer(observers)
        and not _has_tracer(position)
        and not _has_tracer(orientation)
        and not _has_tracer(kwargs)
    ):
        try:
            cached = _cached_field(
                _normalize_source_type(source),
                observers,
                field,
                position=position,
                orientation=orientation,
                squeeze=squeeze,
                sumup=sumup,
                pixel_agg=pixel_agg,
                in_out=in_out,
                kwargs=kwargs,
            )
        except (ValueError, TypeError):
            cached = None
        if cached is not None:
            return cached

    return _compute_field_jit(
        source,
        observers,
        field,
        position=position,
        orientation=orientation,
        squeeze=squeeze,
        sumup=sumup,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


def getB(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    pixel_agg: str | None = None,
    output: str = "ndarray",
    in_out: str = "auto",
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return B-field in Tesla from source type strings or source objects."""
    return _compute_field(
        source,
        observers,
        "B",
        position=position,
        orientation=orientation,
        squeeze=squeeze,
        sumup=sumup,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


def getH(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    pixel_agg: str | None = None,
    output: str = "ndarray",
    in_out: str = "auto",
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return H-field in A/m from source type strings or source objects."""
    return _compute_field(
        source,
        observers,
        "H",
        position=position,
        orientation=orientation,
        squeeze=squeeze,
        sumup=sumup,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


def getJ(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    pixel_agg: str | None = None,
    output: str = "ndarray",
    in_out: str = "auto",
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return J-field from source type strings or source objects."""
    return _compute_field(
        source,
        observers,
        "J",
        position=position,
        orientation=orientation,
        squeeze=squeeze,
        sumup=sumup,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )


def getM(
    source: str | object,
    observers: ArrayLike | None = None,
    *,
    position: ArrayLike = (0.0, 0.0, 0.0),
    orientation: ArrayLike | None = None,
    squeeze: bool = True,
    sumup: bool = False,
    pixel_agg: str | None = None,
    output: str = "ndarray",
    in_out: str = "auto",
    **kwargs: ArrayLike,
) -> jnp.ndarray:
    """Return M-field from source type strings or source objects."""
    return _compute_field(
        source,
        observers,
        "M",
        position=position,
        orientation=orientation,
        squeeze=squeeze,
        sumup=sumup,
        pixel_agg=pixel_agg,
        output=output,
        in_out=in_out,
        **kwargs,
    )
