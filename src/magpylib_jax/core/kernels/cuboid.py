"""Cuboid magnet field kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.geometry import ensure_observers
from magpylib_jax.core.kernels._common import _broadcast_vector


@jax.jit
def magnet_cuboid_bfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """B-field of homogeneously polarized cuboids centered at the origin."""
    obs = ensure_observers(observers)
    dim = _broadcast_vector(jnp.asarray(dimensions, dtype=jnp.float64), obs.shape)
    pol = _broadcast_vector(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape)

    pol_x, pol_y, pol_z = pol.T
    a, b, c = (dim / 2.0).T
    x, y, z = obs.T

    maskx = x < 0.0
    masky = y > 0.0
    maskz = z > 0.0

    x = jnp.where(maskx, -x, x)
    y = jnp.where(masky, -y, y)
    z = jnp.where(maskz, -z, z)

    qsigns = jnp.ones((obs.shape[0], 3, 3), dtype=jnp.float64)
    qs_flipx = jnp.array([[1, -1, -1], [-1, 1, 1], [-1, 1, 1]], dtype=jnp.float64)
    qs_flipy = jnp.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=jnp.float64)
    qs_flipz = jnp.array([[1, 1, -1], [1, 1, -1], [-1, -1, 1]], dtype=jnp.float64)

    qsigns = qsigns * jnp.where(maskx[:, None, None], qs_flipx, 1.0)
    qsigns = qsigns * jnp.where(masky[:, None, None], qs_flipy, 1.0)
    qsigns = qsigns * jnp.where(maskz[:, None, None], qs_flipz, 1.0)

    xma, xpa = x - a, x + a
    ymb, ypb = y - b, y + b
    zmc, zpc = z - c, z + c

    xma2, xpa2 = xma * xma, xpa * xpa
    ymb2, ypb2 = ymb * ymb, ypb * ypb
    zmc2, zpc2 = zmc * zmc, zpc * zpc

    mmm = jnp.sqrt(xma2 + ymb2 + zmc2)
    pmp = jnp.sqrt(xpa2 + ymb2 + zpc2)
    pmm = jnp.sqrt(xpa2 + ymb2 + zmc2)
    mmp = jnp.sqrt(xma2 + ymb2 + zpc2)
    mpm = jnp.sqrt(xma2 + ypb2 + zmc2)
    ppp = jnp.sqrt(xpa2 + ypb2 + zpc2)
    ppm = jnp.sqrt(xpa2 + ypb2 + zmc2)
    mpp = jnp.sqrt(xma2 + ypb2 + zpc2)

    ff2x = jnp.log((xma + mmm) * (xpa + ppm) * (xpa + pmp) * (xma + mpp))
    ff2x = ff2x - jnp.log((xpa + pmm) * (xma + mpm) * (xma + mmp) * (xpa + ppp))

    ff2y = jnp.log((-ymb + mmm) * (-ypb + ppm) * (-ymb + pmp) * (-ypb + mpp))
    ff2y = ff2y - jnp.log((-ymb + pmm) * (-ypb + mpm) * (ymb - mmp) * (ypb - ppp))

    ff2z = jnp.log((-zmc + mmm) * (-zmc + ppm) * (-zpc + pmp) * (-zpc + mpp))
    ff2z = ff2z - jnp.log((-zmc + pmm) * (zmc - mpm) * (-zpc + mmp) * (zpc - ppp))

    ff1x = (
        jnp.arctan2(ymb * zmc, xma * mmm)
        - jnp.arctan2(ymb * zmc, xpa * pmm)
        - jnp.arctan2(ypb * zmc, xma * mpm)
        + jnp.arctan2(ypb * zmc, xpa * ppm)
        - jnp.arctan2(ymb * zpc, xma * mmp)
        + jnp.arctan2(ymb * zpc, xpa * pmp)
        + jnp.arctan2(ypb * zpc, xma * mpp)
        - jnp.arctan2(ypb * zpc, xpa * ppp)
    )

    ff1y = (
        jnp.arctan2(xma * zmc, ymb * mmm)
        - jnp.arctan2(xpa * zmc, ymb * pmm)
        - jnp.arctan2(xma * zmc, ypb * mpm)
        + jnp.arctan2(xpa * zmc, ypb * ppm)
        - jnp.arctan2(xma * zpc, ymb * mmp)
        + jnp.arctan2(xpa * zpc, ymb * pmp)
        + jnp.arctan2(xma * zpc, ypb * mpp)
        - jnp.arctan2(xpa * zpc, ypb * ppp)
    )

    ff1z = (
        jnp.arctan2(xma * ymb, zmc * mmm)
        - jnp.arctan2(xpa * ymb, zmc * pmm)
        - jnp.arctan2(xma * ypb, zmc * mpm)
        + jnp.arctan2(xpa * ypb, zmc * ppm)
        - jnp.arctan2(xma * ymb, zpc * mmp)
        + jnp.arctan2(xpa * ymb, zpc * pmp)
        + jnp.arctan2(xma * ypb, zpc * mpp)
        - jnp.arctan2(xpa * ypb, zpc * ppp)
    )

    bx_pol_x = pol_x * ff1x * qsigns[:, 0, 0]
    by_pol_x = pol_x * ff2z * qsigns[:, 0, 1]
    bz_pol_x = pol_x * ff2y * qsigns[:, 0, 2]

    bx_pol_y = pol_y * ff2z * qsigns[:, 1, 0]
    by_pol_y = pol_y * ff1y * qsigns[:, 1, 1]
    bz_pol_y = -pol_y * ff2x * qsigns[:, 1, 2]

    bx_pol_z = pol_z * ff2y * qsigns[:, 2, 0]
    by_pol_z = -pol_z * ff2x * qsigns[:, 2, 1]
    bz_pol_z = pol_z * ff1z * qsigns[:, 2, 2]

    bx_tot = bx_pol_x + bx_pol_y + bx_pol_z
    by_tot = by_pol_x + by_pol_y + by_pol_z
    bz_tot = bz_pol_x + bz_pol_y + bz_pol_z

    return jnp.stack((bx_tot, by_tot, bz_tot), axis=-1) / (4.0 * jnp.pi)


@jax.jit
def _cuboid_masks(
    observers: jnp.ndarray,
    dimensions: jnp.ndarray,
    polarizations: jnp.ndarray,
    rtol_surface: float = 1e-15,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    x, y, z = observers.T
    a, b, c = jnp.abs(dimensions.T) / 2.0
    pol_x, pol_y, pol_z = polarizations.T

    mask_pol_not_null = ~((pol_x == 0.0) & (pol_y == 0.0) & (pol_z == 0.0))
    mask_dim_not_null = (a * b * c) != 0.0

    x_dist = jnp.abs(x) - a
    y_dist = jnp.abs(y) - b
    z_dist = jnp.abs(z) - c

    mask_surf_x = jnp.abs(x_dist) < rtol_surface * a
    mask_surf_y = jnp.abs(y_dist) < rtol_surface * b
    mask_surf_z = jnp.abs(z_dist) < rtol_surface * c

    mask_inside_x = x_dist < rtol_surface * a
    mask_inside_y = y_dist < rtol_surface * b
    mask_inside_z = z_dist < rtol_surface * c
    mask_inside = mask_inside_x & mask_inside_y & mask_inside_z

    mask_xedge = mask_surf_y & mask_surf_z & mask_inside_x
    mask_yedge = mask_surf_x & mask_surf_z & mask_inside_y
    mask_zedge = mask_surf_x & mask_surf_y & mask_inside_z
    mask_not_edge = ~(mask_xedge | mask_yedge | mask_zedge)

    mask_gen = mask_pol_not_null & mask_dim_not_null & mask_not_edge
    return mask_inside, mask_gen


@jax.jit
def magnet_cuboid_jfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """J-field for homogeneously polarized cuboids."""
    obs = ensure_observers(observers)
    dim = _broadcast_vector(jnp.asarray(dimensions, dtype=jnp.float64), obs.shape)
    pol = _broadcast_vector(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape)
    mask_inside, _ = _cuboid_masks(obs, dim, pol)
    return jnp.where(mask_inside[:, None], pol, 0.0)


@jax.jit
def magnet_cuboid_mfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """M-field for homogeneously polarized cuboids."""
    return magnet_cuboid_jfield(observers, dimensions, polarizations) / MU0


@jax.jit
def magnet_cuboid_hfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """H-field for homogeneously polarized cuboids."""
    obs = ensure_observers(observers)
    dim = _broadcast_vector(jnp.asarray(dimensions, dtype=jnp.float64), obs.shape)
    pol = _broadcast_vector(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape)

    mask_inside, mask_gen = _cuboid_masks(obs, dim, pol)
    b_all = magnet_cuboid_bfield(obs, dim, pol)
    b_out = jnp.where(mask_gen[:, None], b_all, 0.0)
    h = b_out - jnp.where(mask_inside[:, None], pol, 0.0)
    return h / MU0
