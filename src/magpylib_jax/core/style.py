"""Style helpers for compatibility with Magpylib APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MagnetizationStyle:
    """Magnetization style placeholder."""

    def __init__(self, show: bool = True) -> None:
        self.show = bool(show)


class BaseStyle:
    """Minimal style container with label, color, and magnetization state."""

    def __init__(
        self,
        label: str | None = None,
        color: str | None = None,
        magnetization_show: bool = True,
        size: float | None = None,
        sizemode: str | None = None,
    ) -> None:
        self.label = label if label is None else str(label)
        self.color = color
        self.size = size
        self.sizemode = sizemode
        self.magnetization = MagnetizationStyle(show=magnetization_show)

    def update(self, mapping: dict[str, Any] | None = None, **kwargs: Any) -> None:
        data: dict[str, Any] = {}
        if mapping:
            data.update(mapping)
        if kwargs:
            data.update(kwargs)
        invalid: list[str] = []
        for key, val in data.items():
            if key == "magnetization_show":
                self.magnetization.show = bool(val)
                continue
            if hasattr(self, key):
                setattr(self, key, val)
                continue
            invalid.append(key)
        if invalid:
            raise ValueError("The following style properties are invalid: " + ", ".join(invalid))

    def copy(self) -> BaseStyle:
        return BaseStyle(
            label=self.label,
            color=self.color,
            magnetization_show=self.magnetization.show,
            size=self.size,
            sizemode=self.sizemode,
        )


@dataclass
class ArrowSingle:
    color: Any = None
    show: bool = True


@dataclass
class ArrowCS:
    x: ArrowSingle = field(default_factory=ArrowSingle)
    y: ArrowSingle = field(default_factory=ArrowSingle)
    z: ArrowSingle = field(default_factory=ArrowSingle)


@dataclass
class Description:
    show: Any = None
    text: Any = None


@dataclass
class Legend:
    show: Any = None
    text: Any = None


@dataclass
class Model3d:
    data: list[Any] = field(default_factory=list)
    showdefault: bool = True


@dataclass
class Line:
    color: Any = None
    style: Any = None
    width: Any = None


@dataclass
class Marker:
    color: Any = None
    size: Any = None
    symbol: Any = None


@dataclass
class Path:
    frames: Any = None
    line: Line = field(default_factory=Line)
    marker: Marker = field(default_factory=Marker)
    numbering: Any = None
    show: Any = None


@dataclass
class PixelField:
    colormap: Any = None
    colorscaling: Any = None
    shownull: Any = None
    sizemin: Any = None
    sizescaling: Any = None
    source: Any = None
    symbol: Any = None


@dataclass
class Pixel:
    color: Any = None
    field: PixelField = field(default_factory=PixelField)
    size: int = 1
    sizemode: Any = None
    symbol: Any = None


class SensorStyle(BaseStyle):
    """Sensor style placeholder with Magpylib-like repr."""

    def __init__(
        self,
        label: str | None = None,
        color: str | None = None,
        opacity: float | None = None,
        size: float | None = None,
        sizemode: str | None = None,
    ) -> None:
        super().__init__(
            label=label,
            color=color,
            magnetization_show=True,
            size=size,
            sizemode=sizemode,
        )
        self.arrows = ArrowCS()
        self.description = Description()
        self.legend = Legend()
        self.model3d = Model3d()
        self.opacity = opacity
        self.path = Path()
        self.pixel = Pixel()

    def __repr__(self) -> str:
        return (
            "SensorStyle("
            f"arrows={self.arrows}, "
            f"color={self.color}, "
            f"description={self.description}, "
            f"label={self.label}, "
            f"legend={self.legend}, "
            f"model3d={self.model3d}, "
            f"opacity={self.opacity}, "
            f"path={self.path}, "
            f"pixel={self.pixel}, "
            f"size={self.size}, "
            f"sizemode={self.sizemode})"
        )


Style = BaseStyle
