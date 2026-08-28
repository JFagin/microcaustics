"""Observer azimuth and coordinate-delay integrals for primary Kerr rays."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .elliptic import carlson_rd, carlson_rf, carlson_rj
from .geodesics import (
    invert_radial_motion,
    kerr_radial_root_parts,
    polar_mino_time,
)
from .primary import PrimaryKerrTrace
from .transfer import ObserverScreen, ObserverTransfer

_G = 6.67430e-11
_C = 299_792_458.0
_M_SUN = 1.988409870698051e30
_QUADRATURE_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class ObserverCoordinateTrace:
    """A primary transfer augmented with azimuth and relative delay."""

    transfer: ObserverTransfer
    observer_mino_time: torch.Tensor
    polar_consistency_error: torch.Tensor
    failed_pixels: int
    repaired_pixels: int = 0


def _gauss_legendre(
    order: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(order) < 8:
        raise ValueError("quadrature_order must be at least 8")
    key = (int(order), str(device), dtype)
    cached = _QUADRATURE_CACHE.get(key)
    if cached is None:
        nodes, weights = np.polynomial.legendre.leggauss(int(order))
        cached = (
            torch.as_tensor(0.5 * (nodes + 1.0), device=device, dtype=dtype),
            torch.as_tensor(0.5 * weights, device=device, dtype=dtype),
        )
        _QUADRATURE_CACHE[key] = cached
    return cached


def _safe_denominator(value: torch.Tensor) -> torch.Tensor:
    epsilon = math.sqrt(torch.finfo(value.dtype).tiny)
    return torch.where(
        value >= 0,
        value.clamp_min(epsilon),
        value.clamp_max(-epsilon),
    )


def _polar_integrals_analytic(
    spin: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor,
    beta: torch.Tensor,
    inclination_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = eta.dtype
    epsilon = torch.finfo(dtype).eps
    tiny = torch.finfo(dtype).tiny
    spin2 = spin.square()
    coefficient = spin2 - eta - lam.square()
    discriminant = torch.sqrt(
        (coefficient.square() + 4.0 * spin2 * eta).clamp_min(0.0)
    )
    uminus_spin2 = 0.5 * (coefficient - discriminant)
    u_plus_raw = 2.0 * eta / _safe_denominator(discriminant - coefficient)
    u_plus = u_plus_raw.clamp(0.0, 1.0)
    inverse_scale = torch.rsqrt(uminus_spin2.abs().clamp_min(tiny))
    parameter = spin2 * u_plus / _safe_denominator(uminus_spin2)
    sine_amplitude_raw = torch.cos(inclination_rad) / torch.sqrt(
        u_plus.clamp_min(tiny)
    )
    sine_amplitude = sine_amplitude_raw.clamp(
        -1.0 + 4.0 * epsilon,
        1.0 - 4.0 * epsilon,
    )
    sine2 = sine_amplitude.square()
    x = (1.0 - sine2).clamp_min(0.0)
    y = (1.0 - parameter * sine2).clamp_min(0.0)
    one = torch.ones_like(x)
    cubic_third = sine_amplitude.pow(3) / 3.0
    rf_observer = carlson_rf(x, y, one)
    f_observer = sine_amplitude * rf_observer
    p_observer = 1.0 - u_plus * sine2
    pi_observer = f_observer + u_plus * cubic_third * carlson_rj(
        x,
        y,
        one,
        p_observer,
        iterations=6 if dtype == torch.float32 else 12,
    )
    time_observer = u_plus * cubic_third * carlson_rd(x, y, one)

    complete_x = torch.zeros_like(x)
    complete_y = (1.0 - parameter).clamp_min(0.0)
    rf_complete = carlson_rf(complete_x, complete_y, one)
    p_complete = 1.0 - u_plus
    pi_complete = rf_complete + (u_plus / 3.0) * carlson_rj(
        complete_x,
        complete_y,
        one,
        p_complete,
        iterations=6 if dtype == torch.float32 else 12,
    )
    time_complete = (u_plus / 3.0) * carlson_rd(
        complete_x,
        complete_y,
        one,
    )
    turning = beta > 0.0
    gtheta = torch.where(
        turning,
        inverse_scale * (2.0 * rf_complete - f_observer),
        inverse_scale * f_observer,
    )
    gphi = torch.where(
        turning,
        inverse_scale * (2.0 * pi_complete - pi_observer),
        inverse_scale * pi_observer,
    )
    gtime = torch.where(
        turning,
        inverse_scale * (2.0 * time_complete - time_observer),
        inverse_scale * time_observer,
    )
    conditioning_floor = 1.0e-4 if dtype == torch.float32 else 64.0 * epsilon
    valid = (
        (eta >= 0.0)
        & (u_plus_raw > 0.0)
        & (u_plus_raw <= 1.0 + 64.0 * epsilon)
        & (sine_amplitude_raw.abs() <= 1.0 + 64.0 * epsilon)
        & (parameter < 1.0)
        & (p_observer > conditioning_floor)
        & (p_complete > conditioning_floor)
        & torch.isfinite(gtheta)
        & torch.isfinite(gphi)
        & torch.isfinite(gtime)
        & (gtheta >= 0.0)
        & (gphi >= 0.0)
        & (gtime >= 0.0)
    )
    return gtheta, gphi, gtime, valid


def _polar_integrals_quadrature(
    spin: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor,
    beta: torch.Tensor,
    inclination_rad: torch.Tensor,
    *,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes, weights = _gauss_legendre(order, device=eta.device, dtype=eta.dtype)
    spin2 = spin.square()
    coefficient = spin2 - eta - lam.square()
    discriminant = torch.sqrt(
        (coefficient.square() + 4.0 * spin2 * eta).clamp_min(0.0)
    )
    u_plus = (
        2.0 * eta / _safe_denominator(discriminant - coefficient)
    ).clamp(0.0, 1.0)
    turning_theta = torch.acos(torch.sqrt(u_plus))
    equator = torch.full_like(eta, 0.5 * math.pi)
    observer = torch.ones_like(eta) * inclination_rad

    def segment(start, end, segment_eta, segment_spin2, segment_lam):
        span = (end - start).clamp_min(0.0)
        sample = nodes.unsqueeze(0)
        theta = start.unsqueeze(-1) + span.unsqueeze(-1) * sample.square()
        jacobian = 2.0 * span.unsqueeze(-1) * sample
        sine2 = torch.sin(theta).square().clamp_min(torch.finfo(eta.dtype).tiny)
        cosine2 = torch.cos(theta).square()
        potential = (
            segment_eta.unsqueeze(-1)
            + segment_spin2.unsqueeze(-1) * cosine2
            - segment_lam.unsqueeze(-1).square() * cosine2 / sine2
        )
        floor = 16.0 * torch.finfo(eta.dtype).eps * (
            segment_eta.abs()
            + segment_spin2.abs()
            + segment_lam.square()
            + 1.0
        ).unsqueeze(-1)
        measure = (
            weights.unsqueeze(0)
            * jacobian
            * torch.rsqrt(torch.maximum(potential, floor))
        )
        return (
            measure.sum(-1),
            (measure / sine2).sum(-1),
            (measure * cosine2).sum(-1),
        )

    direct = segment(observer, equator, eta, spin2, lam)
    turning = beta > 0.0
    first = segment(
        turning_theta[turning],
        observer[turning],
        eta[turning],
        spin2[turning],
        lam[turning],
    )
    second = segment(
        turning_theta[turning],
        equator[turning],
        eta[turning],
        spin2[turning],
        lam[turning],
    )
    result = []
    for direct_value, first_value, second_value in zip(
        direct,
        first,
        second,
        strict=True,
    ):
        combined = direct_value.clone()
        combined[turning] = first_value + second_value
        result.append(combined)
    return result[0], result[1], result[2]


def _polar_integrals(
    spin: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor,
    beta: torch.Tensor,
    inclination_rad: torch.Tensor,
    *,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spin, eta, lam, beta = torch.broadcast_tensors(spin, eta, lam, beta)
    gtheta, gphi, gtime, valid = _polar_integrals_analytic(
        spin,
        eta,
        lam,
        beta,
        inclination_rad,
    )
    return _repair_polar_integrals(
        spin,
        eta,
        lam,
        beta,
        inclination_rad,
        gtheta,
        gphi,
        gtime,
        valid,
        order=order,
    )


@torch.compiler.disable
def _repair_polar_integrals(
    spin: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor,
    beta: torch.Tensor,
    inclination_rad: torch.Tensor,
    gtheta: torch.Tensor,
    gphi: torch.Tensor,
    gtime: torch.Tensor,
    valid: torch.Tensor,
    *,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repair the uncommon ill-conditioned analytic polar integrals.

    The compact Boolean-indexed repair queue is intentionally eager. Its
    cardinality is data dependent, while the surrounding analytic calculation
    remains compiled. Marking this boundary explicitly avoids a noisy Dynamo
    ``Tensor.item()`` graph-break warning without changing either result or the
    amount of quadrature work.
    """

    invalid = ~valid
    if not bool(torch.any(invalid)):
        return gtheta, gphi, gtime
    fallback = _polar_integrals_quadrature(
        spin[invalid],
        eta[invalid],
        lam[invalid],
        beta[invalid],
        inclination_rad,
        order=order,
    )
    gtheta = gtheta.clone()
    gphi = gphi.clone()
    gtime = gtime.clone()
    gtheta[invalid], gphi[invalid], gtime[invalid] = fallback
    return gtheta, gphi, gtime


def _observer_mino_time(
    spin: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor,
    observer_radius_rg: float,
) -> torch.Tensor:
    nodes, weights = _gauss_legendre(8, device=eta.device, dtype=eta.dtype)
    inverse_observer = torch.full_like(eta, 1.0 / observer_radius_rg)
    inverse_radius = inverse_observer.unsqueeze(-1) * nodes.unsqueeze(0)
    coefficient2 = spin.square() - eta - lam.square()
    coefficient3 = 2.0 * (eta + (lam - spin).square())
    coefficient4 = -spin.square() * eta
    potential = (
        1.0
        + coefficient2.unsqueeze(-1) * inverse_radius.square()
        + coefficient3.unsqueeze(-1) * inverse_radius.pow(3)
        + coefficient4.unsqueeze(-1) * inverse_radius.pow(4)
    )
    return inverse_observer * (
        weights.unsqueeze(0)
        * torch.rsqrt(potential.clamp_min(torch.finfo(eta.dtype).tiny))
    ).sum(-1)


def _radial_integrals(
    spin: torch.Tensor,
    root_real: torch.Tensor,
    root_imag: torch.Tensor,
    disk_mino_time: torch.Tensor,
    lam: torch.Tensor,
    eta: torch.Tensor,
    *,
    observer_radius_rg: float,
    order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observer_mino = _observer_mino_time(spin, eta, lam, observer_radius_rg)
    nodes, weights = _gauss_legendre(
        order,
        device=eta.device,
        dtype=eta.dtype,
    )
    ratio = (disk_mino_time / observer_mino.clamp_min(torch.finfo(eta.dtype).tiny))
    log_ratio = torch.log(ratio.clamp_min(1.0))
    tau = observer_mino.unsqueeze(-1) * torch.exp(
        log_ratio.unsqueeze(-1) * nodes.unsqueeze(0)
    )
    measure = weights.unsqueeze(0) * tau * log_ratio.unsqueeze(-1)
    radius, _, _ = invert_radial_motion(
        spin,
        root_real.unsqueeze(-2),
        root_imag.unsqueeze(-2),
        tau,
    )
    local_spin = spin.unsqueeze(-1)
    delta = _safe_denominator(
        radius.square() - 2.0 * radius + local_spin.square()
    )
    radial_momentum = (
        radius.square() + local_spin.square() - local_spin * lam.unsqueeze(-1)
    )
    phi_integrand = local_spin * (
        2.0 * radius - local_spin * lam.unsqueeze(-1)
    ) / delta
    time_integrand = radius.square() + 2.0 * radius * radial_momentum / delta
    return (
        (measure * phi_integrand).sum(-1),
        (measure * time_integrand).sum(-1),
        observer_mino,
    )


def _coordinate_values(
    spin: torch.Tensor,
    inclination_rad: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor,
    beta: torch.Tensor,
    disk_mino: torch.Tensor,
    root_real: torch.Tensor,
    root_imag: torch.Tensor,
    *,
    observer_radius_rg: float,
    quadrature_order: int,
) -> tuple[torch.Tensor, ...]:
    """Evaluate observer coordinates for one compact ray queue."""

    gtheta, gphi, gtime = _polar_integrals(
        spin,
        eta,
        lam,
        beta,
        inclination_rad,
        order=quadrature_order,
    )
    radial_phi, radial_time, observer_mino = _radial_integrals(
        spin,
        root_real,
        root_imag,
        disk_mino,
        lam,
        eta,
        observer_radius_rg=observer_radius_rg,
        order=quadrature_order,
    )
    finite_polar_time = (
        gtime - observer_mino * torch.cos(inclination_rad).square()
    )
    azimuth = radial_phi + lam * gphi
    travel = radial_time + spin.square() * finite_polar_time
    error = (gtheta - disk_mino).abs()
    valid = (
        torch.isfinite(azimuth)
        & torch.isfinite(travel)
        & torch.isfinite(observer_mino)
        & torch.isfinite(error)
        & (observer_mino > 0.0)
        & (observer_mino < disk_mino)
    )
    return azimuth, travel, observer_mino, error, valid


def _constants_from_screen(
    screen: ObserverScreen,
    index: torch.Tensor,
    spin: torch.Tensor,
    inclination_rad: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Recompute photon constants at ``dtype`` from original screen points."""

    alpha = -screen.x_rg.reshape(-1)[index].to(dtype)
    beta = -screen.y_rg.reshape(-1)[index].to(dtype)
    photon_lambda = -alpha * torch.sin(inclination_rad)
    carter_eta = (
        (alpha.square() - spin.square())
        * torch.cos(inclination_rad).square()
        + beta.square()
    )
    disk_mino, polar_valid = polar_mino_time(
        spin,
        carter_eta,
        photon_lambda,
        beta,
        inclination_rad,
    )
    root_real, root_imag = kerr_radial_root_parts(
        spin,
        carter_eta,
        photon_lambda,
    )
    return (
        carter_eta,
        photon_lambda,
        beta,
        disk_mino,
        root_real,
        root_imag,
        polar_valid,
    )


def add_observer_coordinates(
    primary: PrimaryKerrTrace,
    screen: ObserverScreen,
    *,
    black_hole_mass_solar: float,
    spin: float,
    inclination_deg: float,
    source_redshift: float = 0.0,
    observer_radius_rg: float = 3000.0,
    quadrature_order: int = 24,
    chunk_size: int = 65_536,
    coordinate_dtype: torch.dtype | None = None,
    repair_float32: bool = True,
    float32_repair_tolerance: float = 1.0e-6,
    repair_chunk_size: int = 16_384,
) -> ObserverCoordinateTrace:
    """Add emission azimuth and finite-observer relative delay.

    Delays are returned in observer-frame days. By default, float32 images use
    a dense float32 calculation followed by a compact float64 repair queue for
    poorly conditioned rays. Passing ``coordinate_dtype=torch.float64`` runs
    every hit ray in float64. Only hit rays are materialized and processed in
    bounded chunks.
    """

    if primary.transfer.shape != screen.shape:
        raise ValueError("primary transfer and observer screen shapes must match")
    if not math.isfinite(observer_radius_rg) or observer_radius_rg <= 0.0:
        raise ValueError("observer_radius_rg must be finite and positive")
    if not math.isfinite(black_hole_mass_solar) or black_hole_mass_solar <= 0.0:
        raise ValueError("black_hole_mass_solar must be finite and positive")
    if not math.isfinite(source_redshift) or source_redshift < 0.0:
        raise ValueError("source_redshift must be finite and non-negative")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if int(repair_chunk_size) < 1:
        raise ValueError("repair_chunk_size must be positive")
    if (
        not math.isfinite(float(float32_repair_tolerance))
        or float(float32_repair_tolerance) <= 0.0
    ):
        raise ValueError("float32_repair_tolerance must be finite and positive")
    device = primary.transfer.radius_rg.device
    output_dtype = primary.transfer.radius_rg.dtype
    work_dtype = output_dtype if coordinate_dtype is None else coordinate_dtype
    if work_dtype not in (torch.float32, torch.float64):
        raise TypeError("coordinate_dtype must be float32 or float64")
    primary_spin = float(primary.transfer.metadata.get("spin", float("nan")))
    primary_inclination = float(
        primary.transfer.metadata.get("inclination_deg", float("nan"))
    )
    if not math.isclose(primary_spin, float(spin), rel_tol=0.0, abs_tol=1.0e-7):
        raise ValueError("spin must match the primary Kerr trace")
    if not math.isclose(
        primary_inclination,
        float(inclination_deg),
        rel_tol=0.0,
        abs_tol=1.0e-7,
    ):
        raise ValueError("inclination_deg must match the primary Kerr trace")
    hit = primary.transfer.hit
    interpolated_mask = primary.interpolated_mask
    if interpolated_mask is None:
        interpolated_mask = torch.zeros_like(hit)
    else:
        interpolated_mask = torch.as_tensor(
            interpolated_mask,
            device=hit.device,
            dtype=torch.bool,
        )
        if interpolated_mask.shape != hit.shape:
            raise ValueError("primary interpolated_mask must match the screen")
    traced_hit = hit & ~interpolated_mask
    indices = torch.nonzero(traced_hit.reshape(-1), as_tuple=False).reshape(-1)
    azimuth_flat = torch.full(
        (hit.numel(),),
        float("nan"),
        device=device,
        dtype=output_dtype,
    )
    travel_flat = torch.full_like(azimuth_flat, float("nan"))
    observer_mino_flat = torch.full_like(azimuth_flat, float("nan"))
    error_flat = torch.full_like(azimuth_flat, float("nan"))
    spin_tensor = torch.as_tensor(spin, device=device, dtype=work_dtype)
    inclination = torch.as_tensor(
        math.radians(inclination_deg),
        device=device,
        dtype=work_dtype,
    )
    repaired = 0
    failed = 0
    for start in range(0, indices.numel(), int(chunk_size)):
        index = indices[start : start + int(chunk_size)]
        if work_dtype == output_dtype:
            eta = primary.carter_eta.reshape(-1)[index].to(work_dtype)
            lam = primary.photon_lambda.reshape(-1)[index].to(work_dtype)
            beta = -screen.y_rg.reshape(-1)[index].to(work_dtype)
            disk_mino = primary.mino_time.reshape(-1)[index].to(work_dtype)
            root_real = primary.radial_root_real.reshape(-1, 4)[index].to(
                work_dtype
            )
            root_imag = primary.radial_root_imag.reshape(-1, 4)[index].to(
                work_dtype
            )
            polar_valid = torch.ones_like(disk_mino, dtype=torch.bool)
        else:
            (
                eta,
                lam,
                beta,
                disk_mino,
                root_real,
                root_imag,
                polar_valid,
            ) = _constants_from_screen(
                screen,
                index,
                spin_tensor,
                inclination,
                work_dtype,
            )
        azimuth, travel, observer_mino, error, valid = _coordinate_values(
            spin_tensor,
            inclination,
            eta,
            lam,
            beta,
            disk_mino,
            root_real,
            root_imag,
            observer_radius_rg=observer_radius_rg,
            quadrature_order=quadrature_order,
        )
        valid &= polar_valid

        if work_dtype == torch.float32 and repair_float32:
            tolerance = float(float32_repair_tolerance) * torch.maximum(
                torch.ones_like(disk_mino),
                disk_mino.abs(),
            )
            # Two-real-root radial branches are rare but substantially less
            # well conditioned in float32 than the usual four-real-root
            # branch. Repair the complete rare branch rather than attempting
            # to infer its conditioning from a posteriori polar error alone.
            root_count = primary.radial_root_count.reshape(-1)[index]
            repair_local = torch.nonzero(
                (~valid) | (error > tolerance) | (root_count != 4),
                as_tuple=False,
            ).reshape(-1)
            repaired += int(repair_local.numel())
            for repair_start in range(
                0,
                repair_local.numel(),
                int(repair_chunk_size),
            ):
                local = repair_local[
                    repair_start : repair_start + int(repair_chunk_size)
                ]
                repair_index = index[local]
                repair_spin = spin_tensor.to(torch.float64)
                repair_inclination = inclination.to(torch.float64)
                (
                    eta_r,
                    lam_r,
                    beta_r,
                    disk_mino_r,
                    root_real_r,
                    root_imag_r,
                    polar_valid_r,
                ) = _constants_from_screen(
                    screen,
                    repair_index,
                    repair_spin,
                    repair_inclination,
                    torch.float64,
                )
                repaired_values = _coordinate_values(
                    repair_spin,
                    repair_inclination,
                    eta_r,
                    lam_r,
                    beta_r,
                    disk_mino_r,
                    root_real_r,
                    root_imag_r,
                    observer_radius_rg=observer_radius_rg,
                    quadrature_order=quadrature_order,
                )
                azimuth_r, travel_r, observer_mino_r, error_r, valid_r = (
                    repaired_values
                )
                valid_r &= polar_valid_r
                azimuth[local] = azimuth_r.to(work_dtype)
                travel[local] = travel_r.to(work_dtype)
                observer_mino[local] = observer_mino_r.to(work_dtype)
                error[local] = error_r.to(work_dtype)
                valid[local] = valid_r

        failed += int((~valid).sum().detach().cpu())
        azimuth_flat[index] = azimuth.to(output_dtype)
        travel_flat[index] = travel.to(output_dtype)
        observer_mino_flat[index] = observer_mino.to(output_dtype)
        error_flat[index] = error.to(output_dtype)

    if failed:
        raise RuntimeError(
            f"observer coordinates failed for {failed} hit pixels. Use float64 "
            "coordinates or a larger quadrature order"
        )
    shape = hit.shape
    radius_flat = primary.transfer.radius_rg.reshape(-1).to(output_dtype)
    traced_flat = traced_hit.reshape(-1)
    finite_traced = traced_flat & torch.isfinite(travel_flat) & torch.isfinite(
        azimuth_flat
    )
    if not bool(torch.any(finite_traced)):
        raise RuntimeError("observer-coordinate calculation produced no valid rays")

    # Coordinate time is a smooth image-plane quantity. Rare, ill-conditioned
    # radial branches can nevertheless return the wrong asymptotic branch while
    # satisfying the local elliptic-integral checks. Detect only the unphysical
    # low branch after removing the leading flat-space projection term. The
    # robust residual is extremely narrow for valid rays, so this compact repair
    # preserves the Kerr structure while preventing a handful of pixels from
    # setting a spurious time zero hundreds of days early.
    expected_flat = -radius_flat * torch.sin(inclination) * torch.cos(azimuth_flat)
    branch_repairs = 0
    residual_center = torch.zeros((), device=device, dtype=output_dtype)
    # A single repair pass can expose the next-lowest rare branch as the new
    # outlier. Iterate this deliberately conservative lower-tail test until
    # stable. Four passes cover the nested float32 failure layers seen in the
    # full 1024^2 production screens while the 16-MAD/16-r_g threshold is far
    # wider than the physical Kerr residual variation.
    for _ in range(4):
        residual = travel_flat[finite_traced] - expected_flat[finite_traced]
        residual_center = residual.median()
        residual_mad = (residual - residual_center).abs().median()
        residual_scale = torch.maximum(
            16.0 * residual_mad,
            torch.as_tensor(16.0, device=device, dtype=output_dtype),
        )
        wrong_branch = finite_traced & (
            travel_flat - expected_flat < residual_center - residual_scale
        )
        repaired_this_pass = int(wrong_branch.sum().detach().cpu())
        if repaired_this_pass == 0:
            break
        branch_repairs += repaired_this_pass
        travel_flat = torch.where(
            wrong_branch,
            expected_flat + residual_center,
            travel_flat,
        )

    def fill_from_neighbors(values_flat: torch.Tensor, targets: torch.Tensor):
        """Fill compact pinholes from adjacent traced coordinates."""

        values = values_flat.reshape(shape)
        known = traced_hit & torch.isfinite(values)
        pending = targets.clone()
        for _ in range(4):
            if not bool(torch.any(pending)):
                break
            weights = known.to(values.dtype)
            sums = torch.nn.functional.avg_pool2d(
                torch.where(known, values, torch.zeros_like(values))[None, None],
                kernel_size=3,
                stride=1,
                padding=1,
                count_include_pad=True,
            )[0, 0] * 9.0
            counts = torch.nn.functional.avg_pool2d(
                weights[None, None],
                kernel_size=3,
                stride=1,
                padding=1,
                count_include_pad=True,
            )[0, 0] * 9.0
            fill = pending & (counts > 0.0)
            values = torch.where(fill, sums / counts.clamp_min(1.0), values)
            known = known | fill
            pending = pending & ~fill
        return values.reshape(-1), pending

    pinhole_repairs = int(interpolated_mask.sum().detach().cpu())
    if pinhole_repairs:
        target = interpolated_mask.clone()
        cos_flat, pending_cos = fill_from_neighbors(
            torch.cos(azimuth_flat), target
        )
        sin_flat, pending_sin = fill_from_neighbors(
            torch.sin(azimuth_flat), target
        )
        pending = pending_cos | pending_sin
        if bool(torch.any(pending)):
            raise RuntimeError(
                "observer-coordinate pinhole repair has no traced neighbors"
            )
        repaired_azimuth = torch.atan2(sin_flat, cos_flat)
        azimuth_flat = torch.where(
            interpolated_mask.reshape(-1), repaired_azimuth, azimuth_flat
        )
        # Coordinate time is smooth across these compact primary-image
        # pinholes.  Fill it from adjacent traced rays, just as we do for the
        # angular coordinates.  Reconstructing it from the leading flat-space
        # term plus one global residual can put a repaired outer-disk pixel on
        # an artificially early branch. That pixel then corrupts the common
        # reverberation-delay zero even though its brightness is negligible.
        travel_flat, pending_travel = fill_from_neighbors(travel_flat, target)
        if bool(torch.any(pending_travel)):
            raise RuntimeError(
                "observer-coordinate time pinhole repair did not converge"
            )
        observer_mino_flat, pending = fill_from_neighbors(
            observer_mino_flat, target
        )
        error_flat, pending_error = fill_from_neighbors(error_flat, target)
        if bool(torch.any(pending | pending_error)):
            raise RuntimeError(
                "observer-coordinate scalar pinhole repair did not converge"
            )

    # Coordinate time is locally smooth on the primary image.  Repair rare
    # nested branch failures using only adjacent valid rays before choosing a
    # global zero.  A 16-r_g jump over one screen pixel is deliberately much
    # larger than the resolved Kerr delay gradient, including at the ISCO.
    # Requiring three neighbours also protects the physical image boundary.
    local_mask = hit & torch.isfinite(travel_flat.reshape(shape))
    for _ in range(6):
        travel_image = travel_flat.reshape(shape)
        weights = local_mask.to(travel_image.dtype)
        finite_values = torch.where(
            local_mask,
            travel_image,
            torch.zeros_like(travel_image),
        )
        sums = torch.nn.functional.avg_pool2d(
            finite_values[None, None],
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=True,
        )[0, 0] * 9.0 - finite_values
        counts = torch.nn.functional.avg_pool2d(
            weights[None, None],
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=True,
        )[0, 0] * 9.0 - weights
        neighbor_mean = sums / counts.clamp_min(1.0)
        local_outlier = (
            local_mask
            & (counts >= 3.0)
            & ((travel_image - neighbor_mean).abs() > 16.0)
        )
        repaired_this_pass = int(local_outlier.sum().detach().cpu())
        if repaired_this_pass == 0:
            break
        branch_repairs += repaired_this_pass
        travel_flat = torch.where(
            local_outlier.reshape(-1),
            neighbor_mean.reshape(-1),
            travel_flat,
        )

    finite_travel = travel_flat[hit.reshape(-1) & torch.isfinite(travel_flat)]
    if finite_travel.numel() == 0:
        raise RuntimeError("observer-coordinate calculation produced no valid rays")
    relative_time = travel_flat - finite_travel.min()
    seconds_per_rg = _G * _M_SUN * black_hole_mass_solar / _C**3
    relative_days = (
        relative_time
        * (seconds_per_rg / 86_400.0)
        * (1.0 + float(source_redshift))
    )
    transfer = ObserverTransfer(
        primary.transfer.radius_rg,
        primary.transfer.gfactor,
        primary.transfer.solid_angle_sr,
        hit,
        relative_days.reshape(shape),
        azimuth_flat.reshape(shape),
        {
            **dict(primary.transfer.metadata),
            "observer_coordinates": True,
            "observer_radius_rg": float(observer_radius_rg),
            "quadrature_order": int(quadrature_order),
            "source_redshift": float(source_redshift),
            "coordinate_dtype": str(work_dtype),
            "coordinate_failed_pixels": failed,
            "coordinate_repaired_pixels": repaired,
            "coordinate_branch_repaired_pixels": branch_repairs,
            "primary_pinhole_coordinate_repairs": pinhole_repairs,
            "float32_repair": bool(repair_float32),
            "float32_repair_tolerance": float(float32_repair_tolerance),
        },
    )
    return ObserverCoordinateTrace(
        transfer,
        observer_mino_flat.reshape(shape),
        error_flat.reshape(shape),
        failed,
        repaired,
    )
