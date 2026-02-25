"""Base classes and input checks for path-based motion."""

from __future__ import annotations

import numbers
import re
from copy import deepcopy
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib_jax.constants import MU0
from magpylib_jax.core.style import BaseStyle

class MagpylibMissingInput(ValueError):
    """Raised when required source inputs are missing."""


class MagpylibBadUserInput(ValueError):
    """Raised when invalid or unsupported user inputs are provided."""


def _as_array(x: Any) -> np.ndarray:
    return np.array(x, dtype=float)


def check_format_input_vector(
    value: Any,
    *,
    name: str | None = None,
    dims: tuple[int, ...] = (1, 2),
    shape_m1: int = 3,
    sig_type: str | None = None,
    reshape: tuple[int, ...] | None = None,
    allow_None: bool = False,
) -> np.ndarray | None:
    if value is None:
        if allow_None:
            return None
        sig_name = name or "input"
        sig_type = sig_type or f"array-like with shape ({shape_m1},) or (n, {shape_m1})"
        raise MagpylibBadUserInput(f"Input {sig_name} must be {sig_type}.")

    arr = _as_array(value)
    if arr.ndim not in dims:
        sig_name = name or "input"
        sig_type = sig_type or f"array-like with shape ({shape_m1},) or (n, {shape_m1})"
        raise MagpylibBadUserInput(
            f"Input {sig_name} must be {sig_type}; got array with shape {arr.shape}."
        )
    if arr.ndim == 1:
        if arr.shape[0] != shape_m1:
            sig_name = name or "input"
            sig_type = sig_type or f"array-like with shape ({shape_m1},)"
            raise MagpylibBadUserInput(
                f"Input {sig_name} must be {sig_type}; got array with shape {arr.shape}."
            )
    else:
        if arr.shape[-1] != shape_m1:
            sig_name = name or "input"
            sig_type = sig_type or f"array-like with shape (n, {shape_m1})"
            raise MagpylibBadUserInput(
                f"Input {sig_name} must be {sig_type}; got array with shape {arr.shape}."
            )
    if reshape is not None:
        arr = np.reshape(arr, reshape)
    return arr


def check_format_input_orientation(orientation: Any | None, *, init_format: bool = False):
    if orientation is None:
        quat = np.array([[0.0, 0.0, 0.0, 1.0]])
        rot = R.from_quat(quat)
        return quat if init_format else (rot, rot.as_quat())

    if hasattr(orientation, "as_quat"):
        rot = orientation
        quat = rot.as_quat()
        if quat.ndim == 1:
            quat = quat[None, :]
        return quat if init_format else (rot, rot.as_quat())

    arr = _as_array(orientation)
    if arr.ndim == 2 and arr.shape == (3, 3):
        rot = R.from_matrix(arr)
        return rot.as_quat()[None, :] if init_format else (rot, rot.as_quat())
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        rot = R.from_matrix(arr)
        return rot.as_quat() if init_format else (rot, rot.as_quat())
    if arr.ndim == 1 and arr.shape[0] == 3:
        rot = R.from_rotvec(arr)
        return rot.as_quat()[None, :] if init_format else (rot, rot.as_quat())
    if arr.ndim == 2 and arr.shape[1] == 3:
        rot = R.from_rotvec(arr)
        return rot.as_quat() if init_format else (rot, rot.as_quat())

    raise MagpylibBadUserInput(
        "Input orientation must be a scipy Rotation or array-like in rotvec or matrix form."
    )


def check_format_input_anchor(anchor: Any | None) -> np.ndarray | None:
    if anchor is None:
        return None
    if isinstance(anchor, numbers.Number):
        if anchor == 0:
            return np.array([0.0, 0.0, 0.0])
        raise MagpylibBadUserInput(
            "Input anchor must be None, 0, or array-like with shape (3,) or (n,3)."
        )
    return check_format_input_vector(
        anchor,
        name="anchor",
        allow_None=True,
        sig_type="None or 0 or array-like with shape (3,) or (n, 3)",
    )


def check_format_input_angle(angle: Any) -> np.ndarray:
    if isinstance(angle, numbers.Number):
        return float(angle)
    arr = _as_array(angle)
    if arr.ndim != 1:
        raise MagpylibBadUserInput(
            "Input angle must be int, float, or array-like with shape (n,)."
        )
    return arr


def check_format_input_axis(axis: Any) -> np.ndarray:
    if isinstance(axis, str):
        axis = axis.lower()
        if axis == "x":
            return np.array([1.0, 0.0, 0.0])
        if axis == "y":
            return np.array([0.0, 1.0, 0.0])
        if axis == "z":
            return np.array([0.0, 0.0, 1.0])
        raise MagpylibBadUserInput(f"Unsupported axis {axis!r}.")
    vec = check_format_input_vector(
        axis,
        name="axis",
        dims=(1,),
        sig_type="array-like with shape (3,) or one of {'x', 'y', 'z'}",
    )
    if vec is not None and np.all(np.asarray(vec) == 0):
        raise MagpylibBadUserInput(
            "Input axis must be a non-zero vector; instead received (0, 0, 0)."
        )
    return vec


def check_start_type(start: int | str) -> None:
    if start == "auto":
        return
    if isinstance(start, numbers.Integral):
        return
    raise MagpylibBadUserInput("start must be an int or 'auto'.")


def check_degree_type(degrees: Any) -> None:
    if isinstance(degrees, (bool, np.bool_)):
        return
    raise MagpylibBadUserInput("degrees must be a boolean.")


def _pad_slice_path(path1: np.ndarray, path2: np.ndarray) -> np.ndarray:
    delta_path = len(path1) - len(path2)
    if delta_path > 0:
        return np.pad(path2, ((0, delta_path), (0, 0)), "edge")
    if delta_path < 0:
        return path2[-delta_path:]
    return path2


def _multi_anchor_behavior(anchor: np.ndarray, inrotQ: np.ndarray, rotation: R):
    len_inrotQ = 0 if inrotQ.ndim == 1 else inrotQ.shape[0]
    len_anchor = 0 if anchor.ndim == 1 else anchor.shape[0]

    if len_inrotQ > len_anchor:
        if len_anchor == 0:
            anchor = np.reshape(anchor, (1, 3))
            len_anchor = 1
        anchor = np.pad(anchor, ((0, len_inrotQ - len_anchor), (0, 0)), "edge")
    elif len_inrotQ < len_anchor:
        if len_inrotQ == 0:
            inrotQ = np.reshape(inrotQ, (1, 4))
            len_inrotQ = 1
        inrotQ = np.pad(inrotQ, ((0, len_anchor - len_inrotQ), (0, 0)), "edge")
        rotation = R.from_quat(inrotQ)
    return anchor, inrotQ, rotation


def _path_padding_param(scalar_input: bool, lenop: int, lenip: int, start: int | str):
    pad_before = 0
    pad_behind = 0

    if start == "auto":
        start = 0 if scalar_input else lenop

    if isinstance(start, numbers.Integral) and start < 0:
        start = lenop + start
        if start < 0:
            pad_before = -start
            start = 0

    if isinstance(start, numbers.Integral) and start + lenip > lenop + pad_before:
        pad_behind = start + lenip - (lenop + pad_before)

    if pad_before + pad_behind > 0:
        return (pad_before, pad_behind), int(start)
    return [], int(start)


def _path_padding(inpath: np.ndarray, start: int | str, target_object):
    scalar_input = inpath.ndim == 1

    ppath = target_object._position
    opath = target_object._orientation.as_quat()

    lenip = 1 if scalar_input else len(inpath)

    padding, start = _path_padding_param(scalar_input, len(ppath), lenip, start)
    if padding:
        ppath = np.pad(ppath, (padding, (0, 0)), "edge")
        opath = np.pad(opath, (padding, (0, 0)), "edge")

    end = len(ppath) if scalar_input else start + lenip
    return ppath, opath, start, end, bool(padding)


def _apply_move(target_object, displacement, start: int | str = "auto"):
    inpath = check_format_input_vector(displacement, name="displacement")
    check_start_type(start)

    ppath, opath, start, end, padded = _path_padding(inpath, start, target_object)
    if padded:
        target_object._orientation = R.from_quat(opath)

    ppath[start:end] += inpath
    target_object._position = ppath
    return target_object


def _apply_rotation(target_object, rotation: R, anchor=None, start: int | str = "auto", parent_path=None):
    rotation, inrotQ = check_format_input_orientation(rotation)
    anchor = check_format_input_anchor(anchor)
    check_start_type(start)

    if anchor is not None:
        anchor, inrotQ, rotation = _multi_anchor_behavior(anchor, inrotQ, rotation)

    ppath, opath, newstart, end, _ = _path_padding(inrotQ, start, target_object)

    if anchor is None and parent_path is not None:
        len_anchor = end - newstart
        padding, start = _path_padding_param(inrotQ.ndim == 1, parent_path.shape[0], len_anchor, start)
        if padding:
            parent_path = np.pad(parent_path, (padding, (0, 0)), "edge")
        anchor = parent_path[start : start + len_anchor]

    if anchor is not None:
        ppath[newstart:end] -= anchor
        ppath[newstart:end] = rotation.apply(ppath[newstart:end])
        ppath[newstart:end] += anchor

    oldrot = R.from_quat(opath[newstart:end])
    opath[newstart:end] = (rotation * oldrot).as_quat()

    target_object._orientation = R.from_quat(opath)
    target_object._position = ppath
    return target_object


_UNITS = {
    "parent": None,
    "position": "m",
    "orientation": "deg",
    "dimension": "m",
    "diameter": "m",
    "current": "A",
    "magnetization": "A/m",
    "polarization": "T",
    "moment": "A·m²",
}


def add_iteration_suffix(name: str) -> str:
    m = re.search(r"\d+$", name)
    n = "00"
    endstr = None
    midchar = "_" if name and name[-1] != "_" else ""
    if m is not None:
        midchar = ""
        n = m.group()
        endstr = -len(n)
    return f"{name[:endstr]}{midchar}{int(n) + 1:0{len(n)}}"


class BaseDisplayRepr:
    """Provide minimal describe and repr helpers."""

    def _get_description(self, exclude=None):
        if exclude is None:
            exclude = ()
        elif isinstance(exclude, str):
            exclude = (exclude,)

        lines = [f"{self!r}"]
        extra_keys = [
            "barycenter",
            "centroid",
            "dipole_moment",
            "faces",
            "mesh",
            "meshing",
            "status_disconnected",
            "status_disconnected_data",
            "status_open",
            "status_open_data",
            "status_reoriented",
            "status_selfintersecting",
            "status_selfintersecting_data",
            "vertices",
            "volume",
            "handedness",
            "pixel",
            "style",
        ]
        key_order = list(_UNITS) + sorted(extra_keys)
        for key in key_order:
            if key in exclude:
                continue
            if key not in ("position", "orientation") and not hasattr(self, key):
                continue

            unit = _UNITS.get(key)
            unit_str = f" {unit}" if unit else ""
            k = key
            val: object = ""

            if key == "position":
                val = getattr(self, "_position", None)
                if hasattr(val, "shape"):
                    arr = np.asarray(val)
                    if arr.shape[0] != 1:
                        lines.append(f"  • path length: {arr.shape[0]}")
                        k = f"{k} (last)"
                    val = f"{arr[-1]}"
            elif key == "orientation":
                val = getattr(self, "_orientation", None)
                if isinstance(val, R):
                    rotvec = val.as_rotvec(degrees=True)
                    if len(rotvec) != 1:
                        k = f"{k} (last)"
                    val = f"{rotvec[-1]}"
            elif key == "pixel":
                val = getattr(self, "pixel", None)
                if hasattr(val, "shape"):
                    px_shape = np.asarray(val).shape[:-1]
                    val_str = f"{int(np.prod(px_shape))}"
                    if np.asarray(val).ndim > 2:
                        val_str += f" ({'x'.join(str(p) for p in px_shape)})"
                    val = val_str
            elif key == "status_disconnected_data":
                val = getattr(self, key)
                if val is not None and isinstance(val, (list, tuple)):
                    val = f"{len(val)} part{'s'[: len(val) ^ 1]}"
            elif key == "magnetization":
                mag = getattr(self, "magnetization", None)
                if mag is None and getattr(self, "polarization", None) is not None:
                    mag = np.asarray(getattr(self, "polarization"), dtype=float) / MU0
                val = mag
            elif key == "dipole_moment":
                if hasattr(self, "dipole_moment"):
                    val = getattr(self, "dipole_moment")
                else:
                    mag = getattr(self, "magnetization", None)
                    if mag is None and getattr(self, "polarization", None) is not None:
                        mag = np.asarray(getattr(self, "polarization"), dtype=float) / MU0
                    if mag is not None:
                        val = np.asarray(mag, dtype=float) * float(
                            getattr(self, "volume", 0.0)
                        )
                    else:
                        val = None
            else:
                val = getattr(self, key)

            if isinstance(val, (list, tuple, np.ndarray)) or hasattr(val, "shape"):
                arr = np.asarray(val, dtype=float)
                if np.prod(arr.shape) > 4:
                    val = f"shape{arr.shape}"
                else:
                    val = f"{arr}"
            lines.append(f"  • {k}: {val}{unit_str}")
        return lines

    def describe(self, *, exclude=("style", "field_func"), return_string=False):
        lines = self._get_description(exclude=exclude)
        output = "\n".join(lines)
        if return_string:
            return output
        print(output)  # noqa: T201
        return None

    def _repr_html_(self):
        lines = self._get_description(exclude=("style", "field_func"))
        return f"""<pre>{"<br>".join(lines)}</pre>"""

    def __repr__(self) -> str:
        name = getattr(self, "name", None)
        if name is None:
            style = getattr(self, "style", None)
            name = getattr(style, "label", None)
        name_str = "" if name is None else f", label={name!r}"
        return f"{type(self).__name__}(id={id(self)!r}{name_str})"


class BaseTransform:
    def move(self, displacement, start: int | str = "auto"):
        for child in getattr(self, "children", []):
            child.move(displacement, start=start)
        _apply_move(self, displacement, start=start)
        return self

    def _rotate(self, rotation: R, anchor=None, start: int | str = "auto", parent_path=None):
        for child in getattr(self, "children", []):
            ppth = self._position if parent_path is None else parent_path
            child._rotate(rotation, anchor=anchor, start=start, parent_path=ppth)
        _apply_rotation(self, rotation, anchor=anchor, start=start, parent_path=parent_path)
        return self

    def rotate(self, rotation: R, anchor=None, start: int | str = "auto"):
        return self._rotate(rotation=rotation, anchor=anchor, start=start)

    def rotate_from_angax(self, angle, axis, anchor=None, start: int | str = "auto", degrees: bool = True):
        angle = check_format_input_angle(angle)
        axis = check_format_input_axis(axis)
        check_start_type(start)
        check_degree_type(degrees)

        if degrees:
            angle = angle / 180.0 * np.pi

        if isinstance(angle, numbers.Number):
            angle = np.ones(3) * angle
        else:
            angle = np.tile(angle, (3, 1)).T
        axis = axis / np.linalg.norm(axis) * angle

        rot = R.from_rotvec(axis)
        return self.rotate(rot, anchor, start)

    def rotate_from_rotvec(self, rotvec, anchor=None, start: int | str = "auto", degrees: bool = True):
        rot = R.from_rotvec(rotvec, degrees=degrees)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_euler(self, angle, seq, anchor=None, start: int | str = "auto", degrees: bool = True):
        rot = R.from_euler(seq, angle, degrees=degrees)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_matrix(self, matrix, anchor=None, start: int | str = "auto"):
        rot = R.from_matrix(matrix)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_mrp(self, mrp, anchor=None, start: int | str = "auto"):
        rot = R.from_mrp(mrp)
        rot = rot.as_quat()
        rot = R.from_quat(rot)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_quat(self, quat, anchor=None, start: int | str = "auto"):
        rot = R.from_quat(quat)
        return self.rotate(rot, anchor=anchor, start=start)


class BaseGeo(BaseTransform, BaseDisplayRepr):
    _style_class = BaseStyle

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        orientation=None,
        style=None,
        style_label: str | None = None,
        **kwargs,
    ):
        self._style_kwargs: dict[str, Any] = {}
        self._style = None
        self._style_label = style_label
        self._parent = None
        self.children: list[Any] = []
        self._init_position_orientation(position, orientation)

        if style is not None or kwargs:
            style_kwargs = self._process_style_kwargs(style=style, **kwargs)
            if isinstance(style_kwargs, BaseStyle):
                self._style = style_kwargs
            elif style_kwargs:
                self._style_kwargs = style_kwargs

        if style_label is not None:
            if self._style is not None:
                self._style.label = style_label
            else:
                if not self._style_kwargs:
                    self._style_kwargs = {}
                self._style_kwargs["label"] = style_label

    @staticmethod
    def _process_style_kwargs(style=None, **kwargs):
        if kwargs:
            if style is None:
                style = {}
            style_kwargs = {}
            for k, v in kwargs.items():
                if k.startswith("style_"):
                    style_kwargs[k[6:]] = v
                else:
                    msg = f"__init__() got an unexpected keyword argument {k!r}"
                    raise TypeError(msg)
            if isinstance(style, BaseStyle):
                style.update(style_kwargs)
            elif isinstance(style, dict):
                style.update(style_kwargs)
            else:
                style = style_kwargs
        return style

    def _init_position_orientation(self, position, orientation):
        pos = check_format_input_vector(
            position,
            name="position",
            dims=(1, 2),
            shape_m1=3,
            sig_type="array-like with shape (3,) or (n, 3)",
            reshape=(-1, 3),
        )
        oriQ = check_format_input_orientation(orientation, init_format=True)

        len_pos = pos.shape[0]
        len_ori = oriQ.shape[0]

        if len_pos > len_ori:
            oriQ = np.pad(oriQ, ((0, len_pos - len_ori), (0, 0)), "edge")
        elif len_pos < len_ori:
            pos = np.pad(pos, ((0, len_ori - len_pos), (0, 0)), "edge")

        self._position = pos
        self._orientation = R.from_quat(oriQ)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        from magpylib_jax.collection import Collection

        if isinstance(parent, Collection):
            parent.add(self, override_parent=True)
        elif parent is None:
            if self._parent is not None:
                self._parent.remove(self, errors="ignore")
            self._parent = None
        else:
            msg = (
                "Input parent must be None or a Collection instance; "
                f"instead received type {type(parent).__name__}."
            )
            raise MagpylibBadUserInput(msg)

    @property
    def style_label(self) -> str | None:
        if getattr(self, "_style", None) is not None:
            return self._style.label
        return self._style_label

    @style_label.setter
    def style_label(self, val: str | None) -> None:
        self._style_label = val
        if getattr(self, "_style", None) is not None:
            self._style.label = val

    @property
    def position(self):
        return np.squeeze(self._position)

    @position.setter
    def position(self, position):
        old_pos = self._position
        self._position = check_format_input_vector(
            position,
            name="position",
            dims=(1, 2),
            shape_m1=3,
            sig_type="array-like with shape (3,) or (n, 3)",
            reshape=(-1, 3),
        )
        oriQ = self._orientation.as_quat()
        self._orientation = R.from_quat(_pad_slice_path(self._position, oriQ))

        for child in getattr(self, "children", []):
            old_pos = _pad_slice_path(self._position, old_pos)
            child_pos = _pad_slice_path(self._position, child._position)
            rel_child_pos = child_pos - old_pos
            child.position = self._position + rel_child_pos

    @property
    def orientation(self):
        if len(self._orientation) == 1:
            return self._orientation[0]
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        old_oriQ = self._orientation.as_quat()
        oriQ = check_format_input_orientation(orientation, init_format=True)
        self._orientation = R.from_quat(oriQ)
        self._position = _pad_slice_path(oriQ, self._position)

        for child in getattr(self, "children", []):
            child.position = _pad_slice_path(self._position, child._position)
            old_ori_pad = R.from_quat(np.squeeze(_pad_slice_path(oriQ, old_oriQ)))
            child.rotate(self.orientation * old_ori_pad.inv(), anchor=self._position, start=0)

    def reset_path(self):
        self.position = (0, 0, 0)
        self.orientation = None
        return self

    def _validate_style(self, val=None):
        val = {} if val is None else val
        style = self.style
        if isinstance(val, dict):
            style.update(val)
        elif not isinstance(val, self._style_class):
            msg = (
                f"Input style must be an instance of {self._style_class.__name__}; "
                f"instead received type {type(val).__name__}."
            )
            raise ValueError(msg)
        return style

    @property
    def style(self):
        if getattr(self, "_style", None) is None:
            self._style = self._style_class()
        if self._style_kwargs:
            style_kwargs = self._style_kwargs.copy()
            self._style_kwargs = {}
            try:
                self._style.update(style_kwargs)
            except (AttributeError, ValueError) as e:
                e.args = (
                    f"{self!r} has been initialized with some invalid style arguments."
                    + str(e),
                )
                raise
        if self._style_label is not None and self._style.label is None:
            self._style.label = self._style_label
        return self._style

    @style.setter
    def style(self, style):
        self._style = self._validate_style(style)

    def copy(self, **kwargs):
        if self.parent is not None:
            parent = self._parent
            self._parent = None
            obj_copy = deepcopy(self)
            self._parent = parent
        else:
            obj_copy = deepcopy(self)

        if getattr(self, "_style", None) is not None or bool(getattr(self, "_style_kwargs", False)):
            label = self.style.label
            if label is None:
                label = f"{type(self).__name__}_01"
            else:
                label = add_iteration_suffix(label)
            obj_copy.style.label = label

        style_kwargs: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k.startswith("style"):
                style_kwargs[k] = v
            else:
                setattr(obj_copy, k, v)

        if style_kwargs:
            style_kwargs = self._process_style_kwargs(**style_kwargs)
            if isinstance(style_kwargs, BaseStyle):
                obj_copy._style = style_kwargs
            else:
                obj_copy.style.update(style_kwargs)
        return obj_copy

    def __add__(self, obj):
        from magpylib_jax.collection import Collection

        return Collection(self, obj)


class BaseSource(BaseGeo):
    """Marker base class for source objects."""

    _is_source = True

    @property
    def dipole_moment(self) -> np.ndarray:
        pol = getattr(self, "polarization", None)
        mag = getattr(self, "magnetization", None)
        if mag is None and pol is not None:
            mag = np.asarray(pol, dtype=float) / MU0
        if mag is None:
            return np.zeros(3, dtype=float)
        return np.asarray(mag, dtype=float) * float(getattr(self, "volume", 0.0))
