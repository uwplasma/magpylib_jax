"""Cylinder magnet field kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.elliptic import cel, ellipe, ellipk, ellippi
from magpylib_jax.core.geometry import cart_to_cyl, cyl_field_to_cart, ensure_observers
from magpylib_jax.core.kernels._common import _broadcast_vector


@jax.jit
def magnet_cylinder_axial_bfield(z0: jnp.ndarray, r: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
    """B-field in cylindrical coordinates for axially polarized cylinders."""
    zph = z + z0
    zmh = z - z0
    dpr = 1.0 + r
    dmr = 1.0 - r

    sq0 = jnp.sqrt(zmh * zmh + dpr * dpr)
    sq1 = jnp.sqrt(zph * zph + dpr * dpr)

    k1 = jnp.sqrt((zph * zph + dmr * dmr) / (zph * zph + dpr * dpr))
    k0 = jnp.sqrt((zmh * zmh + dmr * dmr) / (zmh * zmh + dpr * dpr))
    gamma = dmr / dpr
    one = jnp.ones_like(z0)

    br = (cel(k1, one, one, -one) / sq1 - cel(k0, one, one, -one) / sq0) / jnp.pi

    bz = (zph * cel(k1, gamma * gamma, one, gamma) / sq1) - (
        zmh * cel(k0, gamma * gamma, one, gamma) / sq0
    )
    bz = bz / (dpr * jnp.pi)

    return jnp.stack((br, jnp.zeros_like(br), bz), axis=-1)


@jax.jit
def magnet_cylinder_diametral_hfield(
    z0: jnp.ndarray,
    r: jnp.ndarray,
    z: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    """H-field in cylindrical coordinates for diametral polarization."""
    zp = z + z0
    zm = z - z0

    zp2 = zp * zp
    zm2 = zm * zm
    r2 = r * r

    mask_small_r = r < 0.05

    zpp = zp2 + 1.0
    zmm = zm2 + 1.0
    sqrt_p = jnp.sqrt(zpp)
    sqrt_m = jnp.sqrt(zmm)

    frac1 = zp / sqrt_p
    frac2 = zm / sqrt_m

    r3 = r2 * r
    r4 = r3 * r
    r5 = r4 * r

    term1 = frac1 - frac2
    term2 = (frac1 / zpp**2 - frac2 / zmm**2) * r2 / 8.0
    term3 = (3.0 - 4.0 * zp2) * frac1 / zpp**4 - (3.0 - 4.0 * zm2) * frac2 / zmm**4
    term3 = term3 * r4 / 64.0

    hr_small = -jnp.cos(phi) / 4.0 * (term1 + 9.0 * term2 + 25.0 * term3)
    hphi_small = jnp.sin(phi) / 4.0 * (term1 + 3.0 * term2 + 5.0 * term3)

    hz_small = r * (1.0 / zpp / sqrt_p - 1.0 / zmm / sqrt_m)
    hz_small = hz_small + (3.0 / 8.0) * r3 * (
        (1.0 - 4.0 * zp2) / zpp**3 / sqrt_p - (1.0 - 4.0 * zm2) / zmm**3 / sqrt_m
    )
    hz_small = hz_small + (15.0 / 64.0) * r5 * (
        (1.0 - 12.0 * zp2 + 8.0 * zp2 * zp2) / zpp**5 / sqrt_p
        - (1.0 - 12.0 * zm2 + 8.0 * zm2 * zm2) / zmm**5 / sqrt_m
    )
    hz_small = -jnp.cos(phi) / 4.0 * hz_small

    rp = r + 1.0
    rm = r - 1.0
    rp2 = rp * rp
    rm2 = rm * rm

    ap2 = zp2 + rm2
    am2 = zm2 + rm2
    ap = jnp.sqrt(ap2)
    am = jnp.sqrt(am2)

    argp = -4.0 * r / ap2
    argm = -4.0 * r / am2

    mask_special = rm == 0.0
    argc = jnp.where(mask_special, 1e16, -4.0 * r / rm2)
    one_over_rm = jnp.where(mask_special, 0.0, 1.0 / rm)

    elle_p = ellipe(argp)
    elle_m = ellipe(argm)
    ellk_p = ellipk(argp)
    ellk_m = ellipk(argm)
    ellpi_p = ellippi(argc, argp)
    ellpi_m = ellippi(argc, argm)

    safe_r = jnp.where(r == 0.0, 1.0, r)
    safe_r2 = safe_r * safe_r

    hr_general = (
        -jnp.cos(phi)
        / (4.0 * jnp.pi * safe_r2)
        * (
            -zm * am * elle_m
            + zp * ap * elle_p
            + zm / am * (2.0 + zm2) * ellk_m
            - zp / ap * (2.0 + zp2) * ellk_p
            + (zm / am * ellpi_m - zp / ap * ellpi_p) * rp * (r2 + 1.0) * one_over_rm
        )
    )

    hphi_general = (
        jnp.sin(phi)
        / (4.0 * jnp.pi * safe_r2)
        * (
            +zm * am * elle_m
            - zp * ap * elle_p
            - zm / am * (2.0 + zm2 + 2.0 * r2) * ellk_m
            + zp / ap * (2.0 + zp2 + 2.0 * r2) * ellk_p
            + zm / am * rp2 * ellpi_m
            - zp / ap * rp2 * ellpi_p
        )
    )

    hz_general = (
        -jnp.cos(phi)
        / (2.0 * jnp.pi * safe_r)
        * (
            +am * elle_m
            - ap * elle_p
            - (1.0 + zm2 + r2) / am * ellk_m
            + (1.0 + zp2 + r2) / ap * ellk_p
        )
    )

    hr = jnp.where(mask_small_r, hr_small, hr_general)
    hphi = jnp.where(mask_small_r, hphi_small, hphi_general)
    hz = jnp.where(mask_small_r, hz_small, hz_general)

    return jnp.stack((hr, hphi, hz), axis=-1)


@jax.jit
def _cylinder_masks(
    r: jnp.ndarray,
    z: jnp.ndarray,
    z0: jnp.ndarray,
    r0: jnp.ndarray,
    polarization: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    pol_x, pol_y, pol_z = polarization.T

    mask_dim_not_null = (r0 != 0.0) & (z0 != 0.0)
    mask_between_bases = jnp.abs(z) <= z0
    mask_inside_hull = r <= 1.0
    mask_inside = mask_between_bases & mask_inside_hull & mask_dim_not_null

    mask_on_hull = jnp.isclose(r, 1.0, rtol=1e-15, atol=0.0)
    mask_on_bases = jnp.isclose(jnp.abs(z), z0, rtol=1e-15, atol=0.0)
    mask_not_on_edge = ~(mask_on_hull & mask_on_bases)

    mask_pol_not_null = ~((pol_x == 0.0) & (pol_y == 0.0) & (pol_z == 0.0))
    mask_gen = mask_pol_not_null & mask_not_on_edge & mask_dim_not_null

    mask_pol_tv = ((pol_x != 0.0) | (pol_y != 0.0)) & mask_gen
    mask_pol_ax = (pol_z != 0.0) & mask_gen
    mask_inside_gen = mask_inside & mask_gen

    return mask_pol_tv, mask_pol_ax, mask_inside_gen, mask_dim_not_null


@jax.jit
def magnet_cylinder_jfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """J-field for homogeneously polarized cylinders."""
    obs = ensure_observers(observers)
    dim = _broadcast_vector(jnp.asarray(dimensions, dtype=float), (obs.shape[0], 2))
    pol = _broadcast_vector(jnp.asarray(polarizations, dtype=float), obs.shape)

    r, _, z = cart_to_cyl(obs)
    r0, z0 = (dim / 2.0).T
    safe_r0 = jnp.where(r0 == 0.0, 1.0, r0)
    rs = r / safe_r0
    zs = z / safe_r0
    z0s = z0 / safe_r0

    _, _, mask_inside, mask_dim_not_null = _cylinder_masks(rs, zs, z0s, r0, pol)
    mask_inside = mask_inside & mask_dim_not_null
    return jnp.where(mask_inside[:, None], pol, 0.0)


@jax.jit
def magnet_cylinder_mfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """M-field for homogeneously polarized cylinders."""
    return magnet_cylinder_jfield(observers, dimensions, polarizations) / MU0


@jax.jit
def magnet_cylinder_bfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """B-field of homogeneously polarized cylinders centered at the origin."""
    obs = ensure_observers(observers)
    dim = _broadcast_vector(jnp.asarray(dimensions, dtype=float), (obs.shape[0], 2))
    pol = _broadcast_vector(jnp.asarray(polarizations, dtype=float), obs.shape)

    r, phi, z = cart_to_cyl(obs)
    r0, z0 = (dim / 2.0).T

    safe_r0 = jnp.where(r0 == 0.0, 1.0, r0)
    rs = r / safe_r0
    zs = z / safe_r0
    z0s = z0 / safe_r0

    mask_pol_tv, mask_pol_ax, mask_inside, _ = _cylinder_masks(rs, zs, z0s, r0, pol)

    pol_x, pol_y, pol_z = pol.T
    pol_xy = jnp.sqrt(pol_x * pol_x + pol_y * pol_y)
    theta = jnp.arctan2(pol_y, pol_x)

    tv_cyl = magnet_cylinder_diametral_hfield(z0s, rs, zs, phi - theta) * pol_xy[:, None]
    tv_cyl = jnp.where(mask_pol_tv[:, None], tv_cyl, 0.0)

    ax_cyl = magnet_cylinder_axial_bfield(z0s, rs, zs) * pol_z[:, None]
    ax_cyl = jnp.where(mask_pol_ax[:, None], ax_cyl, 0.0)

    bh_cyl = tv_cyl + ax_cyl
    b_cart = cyl_field_to_cart(phi, bh_cyl[:, 0], bh_cyl[:, 1], bh_cyl[:, 2])

    mask_tv_inside = mask_pol_tv & mask_inside
    bx = b_cart[:, 0] + jnp.where(mask_tv_inside, pol_x, 0.0)
    by = b_cart[:, 1] + jnp.where(mask_tv_inside, pol_y, 0.0)
    bz = b_cart[:, 2]
    return jnp.stack((bx, by, bz), axis=-1)


@jax.jit
def magnet_cylinder_hfield(
    observers: ArrayLike,
    dimensions: ArrayLike,
    polarizations: ArrayLike,
) -> jnp.ndarray:
    """H-field of homogeneously polarized cylinders centered at the origin."""
    obs = ensure_observers(observers)
    dim = _broadcast_vector(jnp.asarray(dimensions, dtype=float), (obs.shape[0], 2))
    pol = _broadcast_vector(jnp.asarray(polarizations, dtype=float), obs.shape)

    r, phi, z = cart_to_cyl(obs)
    r0, z0 = (dim / 2.0).T

    safe_r0 = jnp.where(r0 == 0.0, 1.0, r0)
    rs = r / safe_r0
    zs = z / safe_r0
    z0s = z0 / safe_r0

    mask_pol_tv, mask_pol_ax, mask_inside, _ = _cylinder_masks(rs, zs, z0s, r0, pol)

    pol_x, pol_y, pol_z = pol.T
    pol_xy = jnp.sqrt(pol_x * pol_x + pol_y * pol_y)
    theta = jnp.arctan2(pol_y, pol_x)

    tv_cyl = magnet_cylinder_diametral_hfield(z0s, rs, zs, phi - theta) * pol_xy[:, None]
    tv_cyl = jnp.where(mask_pol_tv[:, None], tv_cyl, 0.0)

    ax_cyl = magnet_cylinder_axial_bfield(z0s, rs, zs) * pol_z[:, None]
    ax_cyl = jnp.where(mask_pol_ax[:, None], ax_cyl, 0.0)

    bh_cyl = tv_cyl + ax_cyl
    h_cart = cyl_field_to_cart(phi, bh_cyl[:, 0], bh_cyl[:, 1], bh_cyl[:, 2])

    mask_ax_inside = mask_pol_ax & mask_inside
    hz = h_cart[:, 2] - jnp.where(mask_ax_inside, pol_z, 0.0)
    return jnp.stack((h_cart[:, 0], h_cart[:, 1], hz), axis=-1) / MU0
