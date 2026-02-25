"""JAX-native differentiable magnetic field kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from magpylib_jax._types import ArrayLike
from magpylib_jax.constants import MU0
from magpylib_jax.core.elliptic import cel, ellipe, ellipk, ellippi
from magpylib_jax.core.geometry import cart_to_cyl, cyl_field_to_cart, ensure_observers

_FOUR_PI = 4.0 * jnp.pi


def _broadcast_vector(vector: jnp.ndarray, target_shape: tuple[int, ...]) -> jnp.ndarray:
    if vector.ndim == 1:
        return jnp.broadcast_to(vector[None, :], target_shape)
    return jnp.broadcast_to(vector, target_shape)


def _cel_iter(
    qc: jnp.ndarray,
    p: jnp.ndarray,
    g: jnp.ndarray,
    cc: jnp.ndarray,
    ss: jnp.ndarray,
    em: jnp.ndarray,
    kk: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorized Bulirsch CEL iteration in JAX."""

    def body_fn(_: int, state: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
        qc_, p_, g_, cc_, ss_, em_, kk_ = state
        mask = jnp.abs(g_ - qc_) >= qc_ * 1e-8

        qc_new = 2.0 * jnp.sqrt(kk_)
        kk_new = qc_new * em_
        f = cc_
        cc_new = cc_ + ss_ / p_
        g_new = kk_new / p_
        ss_new = 2.0 * (ss_ + f * g_new)
        p_new = p_ + g_new
        g_store = em_
        em_new = em_ + qc_new

        qc_out = jnp.where(mask, qc_new, qc_)
        p_out = jnp.where(mask, p_new, p_)
        g_out = jnp.where(mask, g_store, g_)
        cc_out = jnp.where(mask, cc_new, cc_)
        ss_out = jnp.where(mask, ss_new, ss_)
        em_out = jnp.where(mask, em_new, em_)
        kk_out = jnp.where(mask, kk_new, kk_)
        return qc_out, p_out, g_out, cc_out, ss_out, em_out, kk_out

    qc, p, _, cc, ss, em, _ = lax.fori_loop(0, 32, body_fn, (qc, p, g, cc, ss, em, kk))
    return 0.5 * jnp.pi * (ss + cc * em) / (em * (em + p))


@jax.jit
def dipole_hfield(observers: ArrayLike, moments: ArrayLike) -> jnp.ndarray:
    """H-field of dipole moments located at the origin."""
    obs = ensure_observers(observers)
    mom = _broadcast_vector(jnp.asarray(moments, dtype=jnp.float64), obs.shape)

    r2 = jnp.sum(obs * obs, axis=-1)
    inv_r3 = jnp.where(r2 > 0.0, r2 ** (-1.5), jnp.inf)
    inv_r5 = jnp.where(r2 > 0.0, r2 ** (-2.5), jnp.inf)
    mdotr = jnp.sum(mom * obs, axis=-1)

    h = (3.0 * mdotr[:, None] * obs * inv_r5[:, None] - mom * inv_r3[:, None]) / _FOUR_PI

    origin_mask = r2 == 0.0
    h_origin = jnp.where(mom == 0.0, 0.0, jnp.sign(mom) * jnp.inf)
    return jnp.where(origin_mask[:, None], h_origin, h)


@jax.jit
def current_circle_hfield(
    observers: ArrayLike,
    diameter: ArrayLike,
    current: ArrayLike,
    *,
    singular_tol: float = 1e-15,
) -> jnp.ndarray:
    """H-field of circular current loops centered at the origin in the xy plane."""
    obs = ensure_observers(observers)
    r, phi, z = cart_to_cyl(obs)

    radius = jnp.abs(jnp.asarray(diameter, dtype=jnp.float64) / 2.0)
    cur = jnp.asarray(current, dtype=jnp.float64)
    radius = jnp.broadcast_to(radius, r.shape)
    cur = jnp.broadcast_to(cur, r.shape)

    mask_zero_radius = radius == 0.0
    mask_singular = jnp.logical_and(jnp.abs(r - radius) < singular_tol * radius, z == 0.0)
    mask_general = jnp.logical_not(jnp.logical_or(mask_zero_radius, mask_singular))

    safe_radius = jnp.where(mask_general, radius, 1.0)
    rr = r / safe_radius
    zz = z / safe_radius

    z2 = zz * zz
    x0 = z2 + (rr + 1.0) ** 2
    k2 = 4.0 * rr / x0
    q2 = (z2 + (rr - 1.0) ** 2) / x0

    q2 = jnp.where(mask_general, q2, 1.0)
    q = jnp.sqrt(q2)
    p = 1.0 + q
    pf = cur / (_FOUR_PI * safe_radius * jnp.sqrt(x0) * q2)

    cc = k2 * 4.0 * zz / x0
    ss = 2.0 * cc * q / p
    hr = pf * _cel_iter(q, p, jnp.ones_like(q), cc, ss, p, q)

    k4 = k2 * k2
    cc = k4 - (q2 + 1.0) * (4.0 / x0)
    ss = 2.0 * q * (k4 / p - (4.0 / x0) * p)
    hz = -pf * _cel_iter(q, p, jnp.ones_like(q), cc, ss, p, q)

    hr = jnp.where(mask_general, hr, 0.0)
    hz = jnp.where(mask_general, hz, 0.0)

    return cyl_field_to_cart(phi, hr, hz)


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

    bz = (
        (zph * cel(k1, gamma * gamma, one, gamma) / sq1)
        - (zmh * cel(k0, gamma * gamma, one, gamma) / sq0)
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
    term3 = ((3.0 - 4.0 * zp2) * frac1 / zpp**4 - (3.0 - 4.0 * zm2) * frac2 / zmm**4)
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
    dim = _broadcast_vector(jnp.asarray(dimensions, dtype=jnp.float64), (obs.shape[0], 2))
    pol = _broadcast_vector(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape)

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
    dim = _broadcast_vector(jnp.asarray(dimensions, dtype=jnp.float64), (obs.shape[0], 2))
    pol = _broadcast_vector(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape)

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
    dim = _broadcast_vector(jnp.asarray(dimensions, dtype=jnp.float64), (obs.shape[0], 2))
    pol = _broadcast_vector(jnp.asarray(polarizations, dtype=jnp.float64), obs.shape)

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


def dipole_bfield(observers: ArrayLike, moments: ArrayLike) -> jnp.ndarray:
    """B-field of a dipole (Tesla)."""
    return jnp.asarray(MU0 * dipole_hfield(observers, moments), dtype=jnp.float64)


def current_circle_bfield(
    observers: ArrayLike,
    diameter: ArrayLike,
    current: ArrayLike,
) -> jnp.ndarray:
    """B-field of a current circle (Tesla)."""
    return jnp.asarray(MU0 * current_circle_hfield(observers, diameter, current), dtype=jnp.float64)
