"""Sensor compatibility layer."""

from __future__ import annotations

import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.core.base import (
    BaseGeo,
    MagpylibBadUserInput,
    check_format_input_vector,
)
from magpylib_jax.core.style import SensorStyle
from magpylib_jax.functional import getB, getH, getJ, getM


class Sensor(BaseGeo):
    """Sensor with one or multiple pixel locations."""

    _is_sensor = True
    _style_class = SensorStyle

    def __init__(
        self,
        pixel: ArrayLike | None = None,
        position: ArrayLike = (0.0, 0.0, 0.0),
        orientation: ArrayLike | None = None,
        handedness: str = "right",
        style=None,
        style_label: str | None = None,
        **kwargs,
    ) -> None:
        self._pixel = None
        self.pixel = pixel
        self.handedness = handedness
        super().__init__(
            position=position,
            orientation=orientation,
            style=style,
            style_label=style_label,
            **kwargs,
        )

    @property
    def pixel(self):
        return self._pixel

    @pixel.setter
    def pixel(self, pix):
        self._pixel = check_format_input_vector(
            pix,
            name="pixel",
            dims=tuple(range(1, 20)),
            shape_m1=3,
            sig_type="array-like with shape (o1, o2, ..., 3) or None",
            allow_None=True,
        )

    @property
    def handedness(self) -> str:
        return self._handedness

    @handedness.setter
    def handedness(self, val: str) -> None:
        if val not in ("right", "left"):
            msg = (
                f"Input handedness of {self} must be either 'right' or 'left'; "
                f"instead received {val!r}."
            )
            raise MagpylibBadUserInput(msg)
        self._handedness = val

    @property
    def observers(self) -> jnp.ndarray:
        pix = self._pixel
        if pix is None:
            pix = jnp.zeros((1, 3), dtype=jnp.float64)
            pix_shape = (1, 3)
        else:
            pix = jnp.asarray(pix, dtype=jnp.float64)
            if pix.shape == (3,):
                pix = pix[None, :]
            pix_shape = pix.shape
        pix_flat = pix.reshape((-1, 3))

        pos_path = jnp.asarray(self._position, dtype=jnp.float64)
        rot_mats = jnp.asarray(self._orientation_matrix, dtype=jnp.float64)
        obs_path = []
        for idx in range(pos_path.shape[0]):
            rot = rot_mats[min(idx, rot_mats.shape[0] - 1)]
            obs = pix_flat @ rot.T + pos_path[idx]
            obs_path.append(obs)
        obs_path = jnp.stack(obs_path, axis=0)

        if pos_path.shape[0] == 1:
            return obs_path[0].reshape(pix_shape)
        return obs_path.reshape((pos_path.shape[0],) + pix_shape[:-1] + (3,))

    @property
    def centroid(self) -> jnp.ndarray:
        if self._pixel is None:
            return jnp.asarray(self.position, dtype=jnp.float64)
        pix_mean = jnp.mean(jnp.asarray(self._pixel, dtype=jnp.float64).reshape(-1, 3), axis=0)
        centroid = jnp.asarray(self._position, dtype=jnp.float64) + pix_mean
        if centroid.shape[0] == 1:
            centroid = centroid[0]
        return centroid

    def getB(
        self,
        *sources: object,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
        pixel_agg: str | None = None,
        output: str = "ndarray",
    ) -> jnp.ndarray:
        srcs = sources[0] if len(sources) == 1 else list(sources)
        return getB(
            srcs,
            self,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
        )

    def getH(
        self,
        *sources: object,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
        pixel_agg: str | None = None,
        output: str = "ndarray",
    ) -> jnp.ndarray:
        srcs = sources[0] if len(sources) == 1 else list(sources)
        return getH(
            srcs,
            self,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
        )

    def getJ(
        self,
        *sources: object,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
        pixel_agg: str | None = None,
        output: str = "ndarray",
    ) -> jnp.ndarray:
        srcs = sources[0] if len(sources) == 1 else list(sources)
        return getJ(
            srcs,
            self,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
        )

    def getM(
        self,
        *sources: object,
        in_out: str = "auto",
        squeeze: bool = True,
        sumup: bool = False,
        pixel_agg: str | None = None,
        output: str = "ndarray",
    ) -> jnp.ndarray:
        srcs = sources[0] if len(sources) == 1 else list(sources)
        return getM(
            srcs,
            self,
            in_out=in_out,
            squeeze=squeeze,
            sumup=sumup,
            pixel_agg=pixel_agg,
            output=output,
        )

    def copy(self, **kwargs) -> Sensor:
        return super().copy(**kwargs)
