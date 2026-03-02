"""Collection compatibility layer for mixed source/sensor containers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.base import BaseGeo, BaseSource, MagpylibBadUserInput
from magpylib_jax.functional import getB, getH, getJ, getM
from magpylib_jax.sensor import Sensor


def _format_star_args(args: tuple[Any, ...]) -> Any:
    if len(args) == 1:
        return args[0]
    return list(args)


class Collection(BaseGeo):
    """Container for source and sensor objects with Magpylib-like behavior."""

    _is_collection = True

    def __init__(
        self,
        *children: object,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        style_label: str | None = None,
        **_kwargs,
    ) -> None:
        super().__init__(position=position, orientation=orientation, style_label=style_label)
        self.children = []
        if children:
            self.add(*children)

    def add(self, *objects: object, override_parent: bool = False) -> Collection:
        if (
            len(objects) == 1
            and isinstance(objects[0], Iterable)
            and not isinstance(objects[0], (BaseGeo, str, bytes))
        ):
            objects = tuple(objects[0])

        for obj in objects:
            if isinstance(obj, Iterable) and not isinstance(obj, (BaseGeo, str, bytes)):
                self.add(*obj, override_parent=override_parent)
                continue
            if obj is None:
                continue
            if not isinstance(obj, BaseGeo):
                raise MagpylibBadUserInput(
                    f"Cannot add object of type {type(obj).__name__!r} to Collection."
                )
            if isinstance(obj, Collection) and (obj is self or self in obj._flatten_children()):
                msg = f"Cannot add {obj!r} because a Collection must not reference itself."
                raise MagpylibBadUserInput(msg)

            current_parent = getattr(obj, "_parent", None)
            if current_parent is None:
                obj._parent = self
            elif override_parent:
                if hasattr(current_parent, "remove"):
                    current_parent.remove(obj, errors="ignore")
                obj._parent = self
            else:
                msg = (
                    f"Cannot add {obj!r} to {self!r} because it already has a parent. "
                    "Consider using override_parent=True."
                )
                raise MagpylibBadUserInput(msg)

            self.children.append(obj)
        return self

    def remove(
        self,
        *objects: object,
        recursive: bool = True,
        errors: str = "raise",
    ) -> Collection:
        if (
            len(objects) == 1
            and isinstance(objects[0], Iterable)
            and not isinstance(objects[0], (BaseGeo, str, bytes))
        ):
            objects = tuple(objects[0])

        def _remove_from(node: Collection, target: BaseGeo) -> bool:
            if target in node.children:
                node.children.remove(target)
                return True
            if recursive:
                for child in node.children:
                    if isinstance(child, Collection) and _remove_from(child, target):
                        return True
            return False

        for obj in objects:
            if obj is None:
                continue
            if not isinstance(obj, BaseGeo):
                if errors == "raise":
                    raise MagpylibBadUserInput(f"Cannot find and remove {obj!r} from {self!r}.")
                if errors != "ignore":
                    raise MagpylibBadUserInput(
                        "Input errors must be one of {'raise', 'ignore'}; "
                        f"instead received {errors!r}."
                    )
                continue

            found = _remove_from(self, obj)
            if found:
                if getattr(obj, "_parent", None) is self:
                    obj._parent = None
            else:
                if errors == "raise":
                    raise MagpylibBadUserInput(f"Cannot find and remove {obj!r} from {self!r}.")
                if errors != "ignore":
                    raise MagpylibBadUserInput(
                        "Input errors must be one of {'raise', 'ignore'}; "
                        f"instead received {errors!r}."
                    )

        return self

    def _flatten_children(self) -> list[object]:
        out: list[object] = []
        for child in self.children:
            out.append(child)
            if isinstance(child, Collection):
                out.extend(child._flatten_children())
        return out

    @property
    def sources(self) -> list[BaseSource]:
        return [obj for obj in self._flatten_children() if isinstance(obj, BaseSource)]

    @property
    def sensors(self) -> list[Sensor]:
        return [obj for obj in self._flatten_children() if isinstance(obj, Sensor)]

    @property
    def volume(self) -> float:
        return float(sum(getattr(obj, "volume", 0.0) for obj in self.sources))

    @property
    def centroid(self) -> jnp.ndarray:
        vols = []
        cents = []
        for obj in self.sources:
            vol = float(getattr(obj, "volume", 0.0))
            if vol > 0:
                vols.append(vol)
                cents.append(jnp.asarray(getattr(obj, "centroid", obj.position), dtype=jnp.float64))
        if not vols:
            return jnp.asarray(self.position, dtype=jnp.float64)
        vols_arr = jnp.asarray(vols, dtype=jnp.float64)
        cents_arr = jnp.stack(cents, axis=0)
        return jnp.sum(cents_arr * vols_arr[:, None], axis=0) / jnp.sum(vols_arr)

    def reset_path(self) -> Collection:
        for child in self.children:
            if hasattr(child, "reset_path"):
                child.reset_path()
        super().reset_path()
        return self

    def _validate_getBH_inputs(self, *inputs: object):
        current_sources = self.sources
        current_sensors = self.sensors

        if current_sensors and current_sources:
            sources, sensors = self, self
            if inputs:
                msg = (
                    "Collections with sensors and sources do not allow collection.getB() inputs."
                    "Consider using magpy.getB() instead."
                )
                raise MagpylibBadUserInput(msg)
        elif not current_sources:
            sources, sensors = inputs, self
        else:
            if len(inputs) == 1:
                sources, sensors = self, inputs[0]
            else:
                sources, sensors = self, inputs
        return sources, sensors

    def getB(
        self,
        *inputs: object,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
        pixel_agg: str | None = None,
        output: str = "ndarray",
    ) -> jnp.ndarray:
        sources, sensors = self._validate_getBH_inputs(*inputs)
        return getB(
            sources,
            sensors,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
        )

    def getH(
        self,
        *inputs: object,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
        pixel_agg: str | None = None,
        output: str = "ndarray",
    ) -> jnp.ndarray:
        sources, sensors = self._validate_getBH_inputs(*inputs)
        return getH(
            sources,
            sensors,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
        )

    def getJ(
        self,
        *inputs: object,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
        pixel_agg: str | None = None,
        output: str = "ndarray",
    ) -> jnp.ndarray:
        sources, sensors = self._validate_getBH_inputs(*inputs)
        return getJ(
            sources,
            sensors,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
        )

    def getM(
        self,
        *inputs: object,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
        pixel_agg: str | None = None,
        output: str = "ndarray",
    ) -> jnp.ndarray:
        sources, sensors = self._validate_getBH_inputs(*inputs)
        return getM(
            sources,
            sensors,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
        )

    def set_children_styles(self, **kwargs) -> None:
        allowed = {"magnetization_show"}
        invalid = [k for k in kwargs if k not in allowed]
        if invalid:
            raise ValueError("The following style properties are invalid: " + ", ".join(invalid))
        if "magnetization_show" in kwargs:
            val = kwargs["magnetization_show"]
            for obj in self._flatten_children():
                if hasattr(obj, "style") and hasattr(obj.style, "magnetization"):
                    obj.style.magnetization.show = bool(val)

    def _describe_label(self, obj: object, parts: list[str]) -> str:
        label = getattr(obj, "style_label", None) or getattr(obj, "style", None) and obj.style.label
        label = label or None
        type_name = obj.__class__.__name__
        want_label = "label" in parts
        want_type = "type" in parts
        if want_label and want_type:
            return f"{type_name} {label or 'nolabel'}"
        if want_label and not want_type:
            return f"{label or type_name}"
        return type_name

    def _describe_properties(self, obj: object) -> list[str]:
        from magpylib_jax.constants import MU0

        def fmt_vec(val) -> str:
            return str(jax.device_get(jnp.asarray(val, dtype=jnp.float64)))

        props: list[str] = []
        props.append(f"position: {fmt_vec(getattr(obj, 'position', (0, 0, 0)))} m")
        ori = getattr(obj, "orientation", None)
        if hasattr(ori, "as_rotvec"):
            rotvec = jnp.asarray(ori.as_rotvec(), dtype=jnp.float64)
        else:
            rotvec = jnp.zeros(3, dtype=jnp.float64)
        props.append(f"orientation: {fmt_vec(jnp.rad2deg(rotvec))} deg")

        dip = None
        if hasattr(obj, "dipole_moment"):
            dip = obj.dipole_moment
        else:
            pol = getattr(obj, "polarization", None)
            mag = getattr(obj, "magnetization", None)
            if mag is None and pol is not None:
                mag = jnp.asarray(pol, dtype=jnp.float64) / MU0
            if mag is not None:
                dip = jnp.asarray(mag, dtype=jnp.float64) * float(getattr(obj, "volume", 0.0))
        if dip is None:
            dip = jnp.zeros(3, dtype=jnp.float64)
        centroid = getattr(obj, "centroid", getattr(obj, "position", (0, 0, 0)))
        props.append(f"centroid: {fmt_vec(centroid)}")
        props.append(f"dipole_moment: {fmt_vec(dip)}")

        if hasattr(obj, "polarization") or hasattr(obj, "magnetization"):
            dim = getattr(obj, "dimension", None)
            props.insert(2, f"dimension: {dim if dim is None else fmt_vec(dim)} m")
            mag = getattr(obj, "magnetization", None)
            if mag is None and getattr(obj, "polarization", None) is not None:
                mag = jnp.asarray(obj.polarization, dtype=jnp.float64) / MU0
            props.insert(
                3,
                f"magnetization: {mag if mag is None else fmt_vec(mag)} A/m",
            )
            pol = getattr(obj, "polarization", None)
            props.insert(4, f"polarization: {pol if pol is None else fmt_vec(pol)} T")
            meshing = getattr(obj, "meshing", None)
            props.append(f"meshing: {meshing}")
        props.append(f"volume: {float(getattr(obj, 'volume', 0.0))}")
        return props

    def describe(self, format: str = "label,type,id", return_string: bool = False):
        fmt = format.replace(" ", "")
        if fmt == "type+label":
            counts = {}
            for obj in self._flatten_children():
                key = obj.__class__.__name__
                counts[key] = counts.get(key, 0) + 1
            lines = [self._describe_label(self, ["type", "label"])]
            for idx, (key, count) in enumerate(counts.items()):
                suffix = "s" if count != 1 else ""
                lines.append(f"{'└──' if idx == len(counts) - 1 else '├──'} {count}x {key}{suffix}")
            out = "\n".join(lines)
            if return_string:
                return out
            print(out)
            return None

        parts = [p for p in fmt.split(",") if p]
        include_properties = "properties" in parts
        parts = [p for p in parts if p != "properties"]
        if not parts:
            parts = ["type"]

        lines: list[str] = []

        root_label = self._describe_label(self, parts)
        if "id" in parts:
            root_label += f" (id={id(self)})"
        lines.append(root_label)

        if include_properties:
            prop_prefix = "│   " if self.children else "    "
            for prop in self._describe_properties(self):
                lines.append(f"{prop_prefix}• {prop}")

        def walk(node: Collection, prefix: str = "") -> None:
            total = len(node.children)
            for idx, child in enumerate(node.children):
                is_last = idx == total - 1
                branch = "└── " if is_last else "├── "
                label = self._describe_label(child, parts)
                if "id" in parts:
                    label += f" (id={id(child)})"
                lines.append(f"{prefix}{branch}{label}")

                child_prefix = prefix + ("    " if is_last else "│   ")
                if include_properties:
                    for prop in self._describe_properties(child):
                        lines.append(f"{child_prefix}    • {prop}")
                if isinstance(child, Collection):
                    walk(child, child_prefix)

        walk(self, "")

        out = "\n".join(lines)
        if return_string:
            return out
        print(out)
        return None

    def _repr_html_(self) -> str:
        desc = self.describe(format="label,type,id", return_string=True)
        return f"<pre>{desc.replace(chr(10), '<br>')}</pre>"

    def __iter__(self):
        return iter(self.children)

    def __len__(self) -> int:
        return len(self.children)

    def __getitem__(self, idx: int) -> object:
        return self.children[idx]

    def __add__(self, other: object) -> Collection:
        return Collection(self, other)

    def __repr__(self) -> str:
        return super().__repr__()
