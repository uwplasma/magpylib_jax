"""LRU caches and cache-key builders for source/sensor JIT preparation."""

from __future__ import annotations

from collections import OrderedDict

import jax.numpy as jnp

from magpylib_jax.fields.api import _has_tracer, _is_array_like

_SOURCE_PREP_CACHE_MAX = 8
_SOURCE_PREP_CACHE: OrderedDict[
    tuple[object, ...],
    tuple[dict[str, jnp.ndarray], dict[str, object]],
] = OrderedDict()
_SENSOR_PREP_CACHE_MAX = 16
_SENSOR_PREP_CACHE: OrderedDict[
    tuple[object, ...],
    tuple[dict[str, jnp.ndarray], dict[str, object]],
] = OrderedDict()


def _lru_get(
    cache: OrderedDict[tuple[object, ...], object],
    key: tuple[object, ...],
) -> object | None:
    val = cache.get(key)
    if val is not None:
        cache.move_to_end(key)
    return val


def _lru_put(
    cache: OrderedDict[tuple[object, ...], object],
    key: tuple[object, ...],
    value: object,
    *,
    max_items: int,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def _source_prep_cache_key(source: object, *, in_out: str) -> tuple[object, ...] | None:
    # Lazy import avoids a cache <-> prepare module import cycle.
    from magpylib_jax.fields.prepare import _format_source_groups

    if isinstance(source, str):
        return None
    if _has_tracer(source):
        return None
    try:
        groups = _format_source_groups(source)
    except Exception:
        return None

    key_parts: list[object] = ["source-prep", in_out]
    for group in groups:
        group_label = group.get("label")
        group_sources = group.get("sources")
        if not isinstance(group_sources, list):
            return None
        key_parts.append(("group", group_label))
        for src in group_sources:
            key_parts.append(
                (
                    type(src).__name__,
                    getattr(src, "cache_token", (id(src), 0)),
                )
            )
    return tuple(key_parts)


def _sensor_prep_cache_key(
    observers: object,
    *,
    pixel_agg: str | None,
) -> tuple[object, ...] | None:
    if observers is None or _has_tracer(observers):
        return None
    if (
        _is_array_like(observers)
        and not isinstance(observers, (list, tuple))
        and not getattr(observers, "_is_sensor", False)
        and not getattr(observers, "_is_collection", False)
    ):
        return None

    if getattr(observers, "_is_collection", False) or getattr(observers, "_is_sensor", False):
        seq = (observers,)
    else:
        if not isinstance(observers, (list, tuple)):
            return None
        seq = observers

    key_parts: list[object] = ["sensor-prep", pixel_agg]
    for obj in seq:
        if getattr(obj, "_is_sensor", False):
            key_parts.append(
                ("sensor", type(obj).__name__, getattr(obj, "cache_token", (id(obj), 0)))
            )
        elif getattr(obj, "_is_collection", False):
            sensors = getattr(obj, "sensors", None)
            if sensors is None:
                return None
            key_parts.append(
                (
                    "sensor-collection",
                    getattr(obj, "cache_token", (id(obj), 0)),
                    tuple(
                        (
                            type(sens).__name__,
                            getattr(sens, "cache_token", (id(sens), 0)),
                        )
                        for sens in sensors
                    ),
                )
            )
        else:
            return None
    return tuple(key_parts)
