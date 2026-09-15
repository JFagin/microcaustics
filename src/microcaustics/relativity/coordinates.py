"""Observer azimuth and coordinate-delay integrals for primary Kerr rays."""

from __future__ import annotations

import math
import threading
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from ..runtime import RuntimeCapabilities, _torch_compile_supported, warn_compilation
from .elliptic import _elliptic_f_principal, carlson_rd, carlson_rf, carlson_rj
from .geodesics import (
    _radial_four_real,
    invert_radial_motion,
    kerr_radial_root_parts,
    polar_mino_time,
    radial_root_real_count,
)
from .primary import PrimaryKerrTrace
from .transfer import ObserverScreen, ObserverTransfer

_G = 6.67430e-11
_C = 299_792_458.0
_M_SUN = 1.988409870698051e30
_QUADRATURE_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}
_COMPILED_COORDINATES: dict[tuple[object, ...], object] = {}
_COMPILED_COORDINATES_LOCK = threading.Lock()
_SAFE_COORDINATE_CHUNKS: dict[tuple[object, ...], int] = {}


@dataclass(frozen=True)
class ObserverCoordinateTrace:
    """A primary transfer augmented with azimuth and relative delay."""

    transfer: ObserverTransfer
    observer_mino_time: torch.Tensor
    polar_consistency_error: torch.Tensor
    failed_pixels: int
    repaired_pixels: int = 0


@dataclass(frozen=True)
class _PreparedCoordinateArrays:
    """Dense traced-ray arrays prepared by the cross-disk scheduler."""

    azimuth: torch.Tensor
    travel: torch.Tensor
    observer_mino: torch.Tensor
    error: torch.Tensor
    repaired_pixels: int
    failed_pixels: int
    execution: str
    effective_chunk_size: int
    oom_retries: int = 0
    metadata: dict[str, object] | None = None


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


def _polar_delay_integrals_analytic(
    spin: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor,
    beta: torch.Tensor,
    inclination_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return only the polar integrals required by an axisymmetric disk."""

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
    f_observer = sine_amplitude * carlson_rf(x, y, one)
    time_observer = u_plus * cubic_third * carlson_rd(x, y, one)

    complete_x = torch.zeros_like(x)
    complete_y = (1.0 - parameter).clamp_min(0.0)
    f_complete = carlson_rf(complete_x, complete_y, one)
    time_complete = (u_plus / 3.0) * carlson_rd(
        complete_x,
        complete_y,
        one,
    )
    turning = beta > 0.0
    gtheta = torch.where(
        turning,
        inverse_scale * (2.0 * f_complete - f_observer),
        inverse_scale * f_observer,
    )
    gtime = torch.where(
        turning,
        inverse_scale * (2.0 * time_complete - time_observer),
        inverse_scale * time_observer,
    )
    valid = (
        (eta >= 0.0)
        & (u_plus_raw > 0.0)
        & (u_plus_raw <= 1.0 + 64.0 * epsilon)
        & (sine_amplitude_raw.abs() <= 1.0 + 64.0 * epsilon)
        & (parameter < 1.0)
        & torch.isfinite(gtheta)
        & torch.isfinite(gtime)
        & (gtheta >= 0.0)
        & (gtime >= 0.0)
    )
    return gtheta, gtime, valid


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
    observer_nodes: torch.Tensor | None = None,
    observer_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if observer_nodes is None or observer_weights is None:
        nodes, weights = _gauss_legendre(8, device=eta.device, dtype=eta.dtype)
    else:
        nodes, weights = observer_nodes, observer_weights
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


def _radial_integrals_four_real_analytic(
    spin: torch.Tensor,
    root_real: torch.Tensor,
    disk_mino_time: torch.Tensor,
    lam: torch.Tensor,
    eta: torch.Tensor,
    *,
    observer_radius_rg: float,
    observer_nodes: torch.Tensor | None = None,
    observer_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate exterior radial azimuth and time from elliptic endpoints."""

    spin, disk_mino_time, lam, eta = torch.broadcast_tensors(
        spin,
        disk_mino_time,
        lam,
        eta,
    )
    dtype = disk_mino_time.dtype
    eps = torch.finfo(dtype).eps
    tiny = torch.finfo(dtype).tiny
    r1, r2, r3, r4 = root_real.unbind(dim=-1)
    r31 = r3 - r1
    r32 = r3 - r2
    r41 = r4 - r1
    r42 = r4 - r2
    r43 = r4 - r3
    scale = torch.sqrt((r31 * r42).clamp_min(tiny))
    parameter = r32 * r41 / _safe_denominator(r31 * r42)
    coefficient = 2.0 / scale

    x_inf = torch.sqrt(
        (r31 / _safe_denominator(r41)).clamp(0.0, 1.0)
    )
    i_inf = coefficient * _elliptic_f_principal(
        torch.asin(x_inf),
        parameter,
    )
    horizon = 1.0 + torch.sqrt((1.0 - spin.square()).clamp_min(0.0))
    disk_radius, disk_radius_valid = _radial_four_real(
        root_real,
        disk_mino_time,
        horizon,
    )
    observer_radius = torch.full_like(disk_radius, float(observer_radius_rg))
    observer_mino = _observer_mino_time(
        spin,
        eta,
        lam,
        observer_radius_rg,
        observer_nodes,
        observer_weights,
    )

    rplus = horizon
    rminus = 1.0 - torch.sqrt((1.0 - spin.square()).clamp_min(0.0))
    n1 = r41 / _safe_denominator(r31)

    def endpoint(radius: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x2 = (
            (radius - r4)
            / _safe_denominator(radius - r3)
            * r31
            / _safe_denominator(r41)
        )
        x = torch.sqrt(x2.clamp(0.0, 1.0))
        sin2 = x.square()
        carlson_x = (1.0 - sin2).clamp_min(0.0)
        carlson_y = (1.0 - parameter * sin2).clamp_min(0.0)
        one = torch.ones_like(x)
        rf_value = carlson_rf(carlson_x, carlson_y, one)
        base_value = x * rf_value
        cubic_third = x.pow(3) / 3.0
        f_value = coefficient * base_value
        e_value = scale * (
            base_value
            - parameter
            * cubic_third
            * carlson_rd(carlson_x, carlson_y, one)
        )
        p1 = 1.0 - n1 * sin2
        pi1_value = coefficient * (
            base_value
            + n1
            * cubic_third
            * carlson_rj(
                carlson_x,
                carlson_y,
                one,
                p1,
                iterations=6 if dtype == torch.float32 else 12,
            )
        )

        def horizon_pi(
            horizon_radius: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            hp3 = horizon_radius - r3
            hp4 = horizon_radius - r4
            characteristic = hp3 * r41 / _safe_denominator(hp4 * r31)
            pole_distance = 1.0 - characteristic * sin2
            value = (
                coefficient
                * r43
                / _safe_denominator(hp3 * hp4)
                * (
                    base_value
                    + characteristic
                    * cubic_third
                    * carlson_rj(
                        carlson_x,
                        carlson_y,
                        one,
                        pole_distance,
                        iterations=6 if dtype == torch.float32 else 12,
                    )
                )
            )
            return value, pole_distance

        pi_plus, p_plus = horizon_pi(rplus)
        pi_minus, p_minus = horizon_pi(rminus)
        radial_potential = (
            (radius - r1)
            * (radius - r2)
            * (radius - r3)
            * (radius - r4)
        )
        h_value = torch.sqrt(radial_potential.clamp_min(0.0)) / (
            radius - r3
        ).clamp_min(tiny)
        valid = (
            torch.isfinite(f_value)
            & torch.isfinite(e_value)
            & torch.isfinite(pi1_value)
            & torch.isfinite(pi_plus)
            & torch.isfinite(pi_minus)
            & (x2 >= -64.0 * eps)
            & (x2 <= 1.0 + 64.0 * eps)
            & (p1 > 0.0)
            & (p_plus > 0.0)
            & (p_minus > 0.0)
        )
        return (
            f_value,
            e_value,
            pi1_value,
            pi_plus,
            pi_minus,
            h_value,
            valid,
        )

    observer_values = endpoint(observer_radius)
    disk_values = endpoint(disk_radius)
    endpoint_sign = torch.where(
        disk_mino_time <= i_inf,
        torch.ones_like(disk_mino_time),
        -torch.ones_like(disk_mino_time),
    )

    def path_value(observer_value: torch.Tensor, disk_value: torch.Tensor):
        return observer_value - endpoint_sign * disk_value

    mino_span = disk_mino_time - observer_mino
    analytic_mino_span = path_value(observer_values[0], disk_values[0])
    e_path = path_value(observer_values[1], disk_values[1])
    pi1_path = path_value(observer_values[2], disk_values[2])
    pi_plus_path = path_value(observer_values[3], disk_values[3])
    pi_minus_path = path_value(observer_values[4], disk_values[4])
    h_path = path_value(observer_values[5], disk_values[5])

    i0 = mino_span
    i1 = r3 * i0 + r43 * pi1_path
    i2 = h_path - 0.5 * (r1 * r4 + r2 * r3) * i0 - e_path
    i_plus = -pi_plus_path - i0 / _safe_denominator(rplus - r3)
    i_minus = -pi_minus_path - i0 / _safe_denominator(rminus - r3)
    horizon_span = _safe_denominator(rplus - rminus)
    radial_phi = (
        2.0
        * spin
        / horizon_span
        * (
            (rplus - 0.5 * spin * lam) * i_plus
            - (rminus - 0.5 * spin * lam) * i_minus
        )
    )
    radial_time = (
        4.0
        / horizon_span
        * (
            rplus * (rplus - 0.5 * spin * lam) * i_plus
            - rminus * (rminus - 0.5 * spin * lam) * i_minus
        )
        + 4.0 * i0
        + 2.0 * i1
        + i2
    )
    mino_tolerance = (
        (2.0e-4 if dtype == torch.float32 else 2.0e-10)
        * torch.maximum(torch.ones_like(mino_span), mino_span.abs())
    )

    def separated_from_horizon(
        root: torch.Tensor,
        horizon_radius: torch.Tensor,
    ) -> torch.Tensor:
        local_scale = torch.maximum(
            torch.ones_like(root),
            torch.maximum(root.abs(), horizon_radius.abs()),
        )
        return (root - horizon_radius).abs() > eps * local_scale

    valid = (
        disk_radius_valid
        & observer_values[6]
        & disk_values[6]
        & (r4 > horizon)
        & separated_from_horizon(r3, rplus)
        & separated_from_horizon(r4, rplus)
        & separated_from_horizon(r3, rminus)
        & separated_from_horizon(r4, rminus)
        & (
            disk_radius
            >= r4
            - 64.0
            * eps
            * torch.maximum(r4.abs(), torch.ones_like(r4))
        )
        & (observer_mino > 0.0)
        & (mino_span > 0.0)
        & ((analytic_mino_span - mino_span).abs() <= mino_tolerance)
        & torch.isfinite(radial_phi)
        & torch.isfinite(radial_time)
    )
    return radial_phi, radial_time, observer_mino, valid


def _coordinate_delay_analytic(
    spin: torch.Tensor,
    inclination_rad: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor,
    beta: torch.Tensor,
    disk_mino: torch.Tensor,
    root_real: torch.Tensor,
    *,
    observer_radius_rg: float,
    observer_nodes: torch.Tensor | None = None,
    observer_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Compiled dense kernel for the axisymmetric observer-delay path."""

    gtheta, gtime, polar_valid = _polar_delay_integrals_analytic(
        spin,
        eta,
        lam,
        beta,
        inclination_rad,
    )
    _, radial_time, observer_mino, radial_valid = (
        _radial_integrals_four_real_analytic(
            spin,
            root_real,
            disk_mino,
            lam,
            eta,
            observer_radius_rg=observer_radius_rg,
            observer_nodes=observer_nodes,
            observer_weights=observer_weights,
        )
    )
    travel = radial_time + spin.square() * (
        gtime - observer_mino * torch.cos(inclination_rad).square()
    )
    error = (gtheta - disk_mino).abs()
    valid = (
        polar_valid
        & radial_valid
        & torch.isfinite(travel)
        & torch.isfinite(observer_mino)
        & torch.isfinite(error)
        & (observer_mino > 0.0)
        & (observer_mino < disk_mino)
    )
    return travel, observer_mino, error, valid


def _coordinate_azimuth_analytic(
    spin: torch.Tensor,
    inclination_rad: torch.Tensor,
    eta: torch.Tensor,
    lam: torch.Tensor,
    beta: torch.Tensor,
    disk_mino: torch.Tensor,
    root_real: torch.Tensor,
    *,
    observer_radius_rg: float,
    observer_nodes: torch.Tensor | None = None,
    observer_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Compiled dense kernel for observer delay plus emission azimuth."""

    gtheta, gphi, gtime, polar_valid = _polar_integrals_analytic(
        spin,
        eta,
        lam,
        beta,
        inclination_rad,
    )
    radial_phi, radial_time, observer_mino, radial_valid = (
        _radial_integrals_four_real_analytic(
            spin,
            root_real,
            disk_mino,
            lam,
            eta,
            observer_radius_rg=observer_radius_rg,
            observer_nodes=observer_nodes,
            observer_weights=observer_weights,
        )
    )
    azimuth = radial_phi + lam * gphi
    travel = radial_time + spin.square() * (
        gtime - observer_mino * torch.cos(inclination_rad).square()
    )
    error = (gtheta - disk_mino).abs()
    valid = (
        polar_valid
        & radial_valid
        & torch.isfinite(azimuth)
        & torch.isfinite(travel)
        & torch.isfinite(observer_mino)
        & torch.isfinite(error)
        & (observer_mino > 0.0)
        & (observer_mino < disk_mino)
    )
    return azimuth, travel, observer_mino, error, valid


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
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, ...]:
    """Recompute photon constants at ``dtype`` from original screen points."""

    target = spin.device if device is None else torch.device(device)
    local_index = index.to(screen.x_rg.device)
    alpha = -screen.x_rg.reshape(-1)[local_index].to(device=target, dtype=dtype)
    beta = -screen.y_rg.reshape(-1)[local_index].to(device=target, dtype=dtype)
    spin = spin.to(device=target, dtype=dtype)
    inclination_rad = inclination_rad.to(device=target, dtype=dtype)
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


def clear_compiled_coordinate_cache() -> None:
    """Forget observer-coordinate callables and learned CUDA chunk ceilings."""

    with _COMPILED_COORDINATES_LOCK:
        _COMPILED_COORDINATES.clear()
        _SAFE_COORDINATE_CHUNKS.clear()


def _coordinate_solver(
    kernel,
    *,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
    observer_radius_rg: float,
    compile_solver: bool,
    compile_mode: str,
    fallback_to_eager: bool,
    warn_on_compile: bool,
):
    if not compile_solver:
        return kernel, "torch eager"
    if not _torch_compile_supported(device, RuntimeCapabilities.detect()):
        if not fallback_to_eager:
            raise RuntimeError(
                "torch.compile is unavailable or its required native compiler "
                "toolchain could not be found"
            )
        return kernel, "torch eager (torch.compile toolchain unavailable)"
    key = (
        kernel,
        str(device),
        dtype,
        int(chunk_size),
        float(observer_radius_rg),
        str(compile_mode),
    )
    # Keep NumPy construction and cached quadrature tensors outside Dynamo;
    # the analytic radial endpoint uses only the fixed order-8 observer tail.
    _gauss_legendre(8, device=device, dtype=dtype)
    with _COMPILED_COORDINATES_LOCK:
        calculation = _COMPILED_COORDINATES.get(key)
        cache_hit = calculation is not None
        if calculation is None:
            warn_compilation(
                "Kerr observer coordinates",
                backend="torch.compile",
                device=device,
                dtype=dtype,
                enabled=warn_on_compile,
            )
            calculation = torch.compile(
                kernel,
                mode=compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            _COMPILED_COORDINATES[key] = calculation
    return calculation, (
        f"torch.compile({compile_mode}, cache={'hit' if cache_hit else 'miss'})"
    )


def _evict_coordinate_solver(
    kernel,
    *,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
    observer_radius_rg: float,
    compile_mode: str,
) -> None:
    """Drop one failed compiled specialization without disturbing warm peers."""

    key = (
        kernel,
        str(device),
        dtype,
        int(chunk_size),
        float(observer_radius_rg),
        str(compile_mode),
    )
    with _COMPILED_COORDINATES_LOCK:
        _COMPILED_COORDINATES.pop(key, None)


def _is_cuda_oom(error: Exception, device: torch.device) -> bool:
    return device.type == "cuda" and (
        isinstance(error, torch.OutOfMemoryError)
        or "out of memory" in str(error).lower()
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
    chunk_size: int = 524_288,
    coordinate_dtype: torch.dtype | None = None,
    compute_emission_azimuth: bool = False,
    compile_solver: bool = False,
    compile_mode: str = "reduce-overhead",
    fallback_to_eager: bool = True,
    warn_on_compile: bool = True,
    repair_float32: bool = True,
    float32_repair_tolerance: float = 1.0e-6,
    repair_chunk_size: int = 16_384,
    repair_quadrature_order: int = 32,
    repair_device: str = "cpu",
    _prepared_arrays: _PreparedCoordinateArrays | None = None,
) -> ObserverCoordinateTrace:
    """Add finite-observer delay and optionally emission azimuth.

    Delays are returned in observer-frame days. By default, float32 images use
    a dense float32 calculation followed by a compact float64 repair queue for
    poorly conditioned rays. The compiled float32 path uses analytic radial
    and polar endpoint expressions in fixed, padded chunks. Passing
    ``coordinate_dtype=torch.float64`` runs the eager quadrature reference.
    Only hit rays are materialized.
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
    if int(repair_quadrature_order) < 8:
        raise ValueError("repair_quadrature_order must be at least 8")
    if repair_device not in {"cpu", "same"}:
        raise ValueError("repair_device must be 'cpu' or 'same'")
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
    if _prepared_arrays is not None:
        expected = (hit.numel(),)
        prepared_values = (
            _prepared_arrays.azimuth,
            _prepared_arrays.travel,
            _prepared_arrays.observer_mino,
            _prepared_arrays.error,
        )
        if any(value.shape != expected for value in prepared_values):
            raise ValueError("prepared observer-coordinate arrays have wrong shape")
        azimuth_flat, travel_flat, observer_mino_flat, error_flat = prepared_values
        indices = indices[:0]
    spin_tensor = torch.as_tensor(spin, device=device, dtype=work_dtype)
    inclination = torch.as_tensor(
        math.radians(inclination_deg),
        device=device,
        dtype=work_dtype,
    )
    analytic_path = work_dtype == torch.float32 and device.type == "cuda"
    kernel = (
        _coordinate_azimuth_analytic
        if compute_emission_azimuth
        else _coordinate_delay_analytic
    )
    safe_key = (str(device), work_dtype, kernel)
    effective_chunk_size = int(chunk_size)
    if device.type == "cuda":
        cached_safe = _SAFE_COORDINATE_CHUNKS.get(safe_key)
        if cached_safe is not None:
            effective_chunk_size = min(effective_chunk_size, cached_safe)
    else:
        # Quadrature materializes a node axis, so retain a conservative host
        # and MPS ceiling even when the public CUDA launch size is larger.
        effective_chunk_size = min(effective_chunk_size, 65_536)
    compile_enabled = bool(
        compile_solver and analytic_path and _prepared_arrays is None
    )
    execution = "torch eager quadrature"
    oom_retries = 0
    compile_fallback_used = False
    observer_nodes = observer_weights = None
    if analytic_path:
        observer_nodes, observer_weights = _gauss_legendre(
            8, device=device, dtype=work_dtype
        )

    while True:
        repaired = 0
        failed = 0
        try:
            solver, execution = _coordinate_solver(
                kernel,
                device=device,
                dtype=work_dtype,
                chunk_size=effective_chunk_size,
                observer_radius_rg=observer_radius_rg,
                compile_solver=compile_enabled,
                compile_mode=compile_mode,
                fallback_to_eager=fallback_to_eager,
                warn_on_compile=warn_on_compile,
            )
            for start in range(0, indices.numel(), effective_chunk_size):
                actual_size = min(
                    effective_chunk_size,
                    int(indices.numel()) - start,
                )
                index = indices[start : start + actual_size]
                padded_index = index
                if analytic_path and actual_size < effective_chunk_size:
                    padded_index = torch.cat(
                        (
                            index,
                            index[-1:].expand(effective_chunk_size - actual_size),
                        )
                    )
                if work_dtype == output_dtype:
                    eta = primary.carter_eta.reshape(-1)[padded_index]
                    lam = primary.photon_lambda.reshape(-1)[padded_index]
                    beta = -screen.y_rg.reshape(-1)[padded_index]
                    disk_mino = primary.mino_time.reshape(-1)[padded_index]
                    root_real = primary.radial_root_real.reshape(-1, 4)[padded_index]
                    root_imag = primary.radial_root_imag.reshape(-1, 4)[padded_index]
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
                        padded_index,
                        spin_tensor,
                        inclination,
                        work_dtype,
                    )

                if analytic_path:
                    if compile_enabled and device.type == "cuda":
                        mark_step = getattr(
                            getattr(torch, "compiler", None),
                            "cudagraph_mark_step_begin",
                            None,
                        )
                        if callable(mark_step):
                            mark_step()
                    values = solver(
                        spin_tensor.expand_as(eta),
                        inclination.expand_as(eta),
                        eta,
                        lam,
                        beta,
                        disk_mino,
                        root_real,
                        observer_radius_rg=observer_radius_rg,
                        observer_nodes=observer_nodes,
                        observer_weights=observer_weights,
                    )
                    if compute_emission_azimuth:
                        azimuth, travel, observer_mino, error, valid = values
                    else:
                        travel, observer_mino, error, valid = values
                        azimuth = torch.full_like(travel, float("nan"))
                    root_count = radial_root_real_count(root_real, root_imag)
                    valid &= root_count == 4
                else:
                    azimuth, travel, observer_mino, error, valid = (
                        _coordinate_values(
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
                    )
                valid &= polar_valid
                active = torch.arange(
                    padded_index.numel(), device=device
                ) < actual_size
                tolerance = float(float32_repair_tolerance) * torch.maximum(
                    torch.ones_like(disk_mino),
                    disk_mino.abs(),
                )
                root_count = radial_root_real_count(root_real, root_imag)
                repair_local = torch.nonzero(
                    active
                    & ((~valid) | (error > tolerance) | (root_count != 4)),
                    as_tuple=False,
                ).reshape(-1)
                if work_dtype == torch.float32 and repair_float32:
                    repaired += int(repair_local.numel())
                    for repair_start in range(
                        0,
                        repair_local.numel(),
                        int(repair_chunk_size),
                    ):
                        local = repair_local[
                            repair_start : repair_start + int(repair_chunk_size)
                        ]
                        repair_index = padded_index[local]
                        repair_target = (
                            torch.device("cpu")
                            if repair_device == "cpu"
                            else device
                        )
                        repair_spin = spin_tensor.to(
                            device=repair_target, dtype=torch.float64
                        )
                        repair_inclination = inclination.to(
                            device=repair_target, dtype=torch.float64
                        )
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
                            device=repair_target,
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
                            quadrature_order=int(repair_quadrature_order),
                        )
                        azimuth_r, travel_r, observer_mino_r, error_r, valid_r = (
                            repaired_values
                        )
                        valid_r &= polar_valid_r
                        if compute_emission_azimuth:
                            azimuth[local] = azimuth_r.to(device=device, dtype=work_dtype)
                        travel[local] = travel_r.to(device=device, dtype=work_dtype)
                        observer_mino[local] = observer_mino_r.to(
                            device=device, dtype=work_dtype
                        )
                        error[local] = error_r.to(device=device, dtype=work_dtype)
                        valid[local] = valid_r.to(device)

                failed += int((active & ~valid).sum().detach().cpu())
                output_index = index
                if compute_emission_azimuth:
                    azimuth_flat[output_index] = azimuth[:actual_size].to(output_dtype)
                travel_flat[output_index] = travel[:actual_size].to(output_dtype)
                observer_mino_flat[output_index] = observer_mino[:actual_size].to(
                    output_dtype
                )
                error_flat[output_index] = error[:actual_size].to(output_dtype)
            break
        except Exception as error:
            if _is_cuda_oom(error, device) and effective_chunk_size > 1:
                oom_retries += 1
                effective_chunk_size = max(1, effective_chunk_size // 2)
                _SAFE_COORDINATE_CHUNKS[safe_key] = effective_chunk_size
                torch.cuda.empty_cache()
                warnings.warn(
                    "CUDA OOM in Kerr observer coordinates; retrying with "
                    f"chunk_size={effective_chunk_size}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if compile_enabled and fallback_to_eager and not compile_fallback_used:
                _evict_coordinate_solver(
                    kernel,
                    device=device,
                    dtype=work_dtype,
                    chunk_size=effective_chunk_size,
                    observer_radius_rg=observer_radius_rg,
                    compile_mode=compile_mode,
                )
                compile_enabled = False
                compile_fallback_used = True
                warnings.warn(
                    "compiled Kerr observer coordinates failed; retrying with "
                    f"eager Torch ({type(error).__name__}: {error})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            raise

    prepared_metadata: dict[str, object] = {}
    if _prepared_arrays is not None:
        repaired = int(_prepared_arrays.repaired_pixels)
        failed = int(_prepared_arrays.failed_pixels)
        execution = _prepared_arrays.execution
        effective_chunk_size = int(_prepared_arrays.effective_chunk_size)
        oom_retries = int(_prepared_arrays.oom_retries)
        prepared_metadata = dict(_prepared_arrays.metadata or {})

    if failed:
        raise RuntimeError(
            f"observer coordinates failed for {failed} hit pixels. Use float64 "
            "coordinates or a larger quadrature order"
        )
    shape = hit.shape
    radius_flat = primary.transfer.radius_rg.reshape(-1).to(output_dtype)
    traced_flat = traced_hit.reshape(-1)
    finite_traced = traced_flat & torch.isfinite(travel_flat)
    if compute_emission_azimuth:
        finite_traced &= torch.isfinite(azimuth_flat)
    if not bool(torch.any(finite_traced)):
        raise RuntimeError("observer-coordinate calculation produced no valid rays")

    # Coordinate time is a smooth image-plane quantity. Rare, ill-conditioned
    # radial branches can nevertheless return the wrong asymptotic branch while
    # satisfying the local elliptic-integral checks. Detect only the unphysical
    # low branch after removing the leading flat-space projection term. The
    # robust residual is extremely narrow for valid rays, so this compact repair
    # preserves the Kerr structure while preventing a handful of pixels from
    # setting a spurious time zero hundreds of days early.
    branch_repairs = 0
    residual_center = torch.zeros((), device=device, dtype=output_dtype)
    # A single repair pass can expose the next-lowest rare branch as the new
    # outlier. Iterate this deliberately conservative lower-tail test until
    # stable. Four passes cover the nested float32 failure layers seen in the
    # full 1024^2 production screens while the 16-MAD/16-r_g threshold is far
    # wider than the physical Kerr residual variation.
    if compute_emission_azimuth:
        expected_flat = (
            -radius_flat * torch.sin(inclination) * torch.cos(azimuth_flat)
        )
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
        if compute_emission_azimuth:
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
        azimuth_flat.reshape(shape) if compute_emission_azimuth else None,
        {
            **dict(primary.transfer.metadata),
            "observer_coordinates": True,
            "observer_radius_rg": float(observer_radius_rg),
            "quadrature_order": int(quadrature_order),
            "source_redshift": float(source_redshift),
            "coordinate_dtype": str(work_dtype),
            "coordinate_execution": execution,
            "coordinate_compiled": execution.startswith("torch.compile"),
            "coordinate_chunk_size_requested": int(chunk_size),
            "coordinate_chunk_size": int(effective_chunk_size),
            "coordinate_oom_retries": int(oom_retries),
            "compute_emission_azimuth": bool(compute_emission_azimuth),
            "coordinate_repair_device": str(repair_device),
            "coordinate_failed_pixels": failed,
            "coordinate_repaired_pixels": repaired,
            "coordinate_branch_repaired_pixels": branch_repairs,
            "primary_pinhole_coordinate_repairs": pinhole_repairs,
            "float32_repair": bool(repair_float32),
            "float32_repair_tolerance": float(float32_repair_tolerance),
            "repair_quadrature_order": int(repair_quadrature_order),
            **prepared_metadata,
        },
    )
    return ObserverCoordinateTrace(
        transfer,
        observer_mino_flat.reshape(shape),
        error_flat.reshape(shape),
        failed,
        repaired,
    )


def add_observer_coordinates_batch(
    primaries: tuple[PrimaryKerrTrace, ...],
    screens: tuple[ObserverScreen, ...],
    *,
    black_hole_masses_solar: tuple[float, ...],
    spins: tuple[float, ...],
    inclinations_deg: tuple[float, ...],
    source_redshifts: tuple[float, ...],
    observer_radius_rg: float = 3000.0,
    quadrature_order: int = 24,
    chunk_size: int = 524_288,
    compile_solver: bool = True,
    compile_mode: str = "reduce-overhead",
    fallback_to_eager: bool = True,
    warn_on_compile: bool = True,
    repair_float32: bool = True,
    float32_repair_tolerance: float = 1.0e-6,
    repair_chunk_size: int = 16_384,
    repair_quadrature_order: int = 32,
    repair_device: str = "cpu",
) -> tuple[ObserverCoordinateTrace, ...]:
    """Pool hit rays from compatible Kerr disks through one compiled shape.

    This changes scheduling only: each returned transfer has the same shape,
    metadata, sparse repair, and post-processing contract as an independent
    :func:`add_observer_coordinates` call. Pooling removes partially empty
    per-disk tail chunks and never keys compilation on physical parameters or
    the number of disks in this call.
    """

    count = len(primaries)
    values = (
        screens,
        black_hole_masses_solar,
        spins,
        inclinations_deg,
        source_redshifts,
    )
    if count == 0:
        return ()
    if any(len(value) != count for value in values):
        raise ValueError("batched observer-coordinate inputs must have equal length")
    first = primaries[0]
    device = first.transfer.radius_rg.device
    dtype = first.transfer.radius_rg.dtype
    compatible = (
        device.type == "cuda"
        and dtype == torch.float32
        and all(primary.transfer.shape == first.transfer.shape for primary in primaries)
        and all(primary.transfer.radius_rg.device == device for primary in primaries)
        and all(primary.transfer.radius_rg.dtype == dtype for primary in primaries)
        and all(screen.shape == first.transfer.shape for screen in screens)
        and all(screen.x_rg.device == device for screen in screens)
        and all(screen.x_rg.dtype == dtype for screen in screens)
    )
    common = dict(
        observer_radius_rg=observer_radius_rg,
        quadrature_order=quadrature_order,
        chunk_size=chunk_size,
        compute_emission_azimuth=False,
        compile_solver=compile_solver,
        compile_mode=compile_mode,
        fallback_to_eager=fallback_to_eager,
        warn_on_compile=warn_on_compile,
        repair_float32=repair_float32,
        float32_repair_tolerance=float32_repair_tolerance,
        repair_chunk_size=repair_chunk_size,
        repair_quadrature_order=repair_quadrature_order,
        repair_device=repair_device,
    )
    if count == 1 or not compatible:
        return tuple(
            add_observer_coordinates(
                primary,
                screen,
                black_hole_mass_solar=mass,
                spin=spin,
                inclination_deg=inclination,
                source_redshift=redshift,
                **common,
            )
            for primary, screen, mass, spin, inclination, redshift in zip(
                primaries,
                screens,
                black_hole_masses_solar,
                spins,
                inclinations_deg,
                source_redshifts,
                strict=True,
            )
        )

    per_disk_indices = []
    per_disk_fields = []
    offsets = [0]
    for primary, screen, spin, inclination_deg in zip(
        primaries, screens, spins, inclinations_deg, strict=True
    ):
        interpolated = primary.interpolated_mask
        if interpolated is None:
            interpolated = torch.zeros_like(primary.transfer.hit)
        traced = primary.transfer.hit & ~interpolated
        index = torch.nonzero(traced.reshape(-1), as_tuple=False).reshape(-1)
        if index.numel() == 0:
            raise RuntimeError("a batched Kerr disk contains no directly traced rays")
        eta = primary.carter_eta.reshape(-1)[index]
        per_disk_indices.append(index)
        per_disk_fields.append(
            (
                torch.full_like(eta, float(spin)),
                torch.full_like(eta, math.radians(float(inclination_deg))),
                eta,
                primary.photon_lambda.reshape(-1)[index],
                -screen.y_rg.reshape(-1)[index],
                primary.mino_time.reshape(-1)[index],
                primary.radial_root_real.reshape(-1, 4)[index],
                primary.radial_root_imag.reshape(-1, 4)[index],
                screen.x_rg.reshape(-1)[index],
                screen.y_rg.reshape(-1)[index],
            )
        )
        offsets.append(offsets[-1] + int(index.numel()))
    pooled = tuple(torch.cat(items) for items in zip(*per_disk_fields, strict=True))
    (
        spin_p,
        inclination_p,
        eta_p,
        lam_p,
        beta_p,
        disk_mino_p,
        root_real_p,
        root_imag_p,
        screen_x_p,
        screen_y_p,
    ) = pooled
    total = int(eta_p.numel())
    travel_p = torch.full_like(eta_p, float("nan"))
    mino_p = torch.full_like(eta_p, float("nan"))
    error_p = torch.full_like(eta_p, float("nan"))
    valid_p = torch.zeros_like(eta_p, dtype=torch.bool)
    effective_chunk = int(chunk_size)
    safe_key = (str(device), dtype, _coordinate_delay_analytic)
    cached_safe = _SAFE_COORDINATE_CHUNKS.get(safe_key)
    if cached_safe is not None:
        effective_chunk = min(effective_chunk, cached_safe)
    observer_nodes, observer_weights = _gauss_legendre(8, device=device, dtype=dtype)
    oom_retries = 0
    compile_enabled = bool(compile_solver)
    compile_fallback_used = False
    while True:
        try:
            solver, execution = _coordinate_solver(
                _coordinate_delay_analytic,
                device=device,
                dtype=dtype,
                chunk_size=effective_chunk,
                observer_radius_rg=observer_radius_rg,
                compile_solver=compile_enabled,
                compile_mode=compile_mode,
                fallback_to_eager=fallback_to_eager,
                warn_on_compile=warn_on_compile,
            )
            for start in range(0, total, effective_chunk):
                actual = min(effective_chunk, total - start)
                selection = slice(start, start + actual)
                fields = [value[selection] for value in pooled[:7]]
                if actual < effective_chunk:
                    padding = effective_chunk - actual
                    fields = [
                        torch.cat(
                            (value, value[-1:].expand((padding,) + value.shape[1:]))
                        )
                        for value in fields
                    ]
                if compile_enabled:
                    mark_step = getattr(
                        getattr(torch, "compiler", None),
                        "cudagraph_mark_step_begin",
                        None,
                    )
                    if callable(mark_step):
                        mark_step()
                travel, observer_mino, error, valid = solver(
                    *fields[:-1],
                    fields[-1],
                    observer_radius_rg=observer_radius_rg,
                    observer_nodes=observer_nodes,
                    observer_weights=observer_weights,
                )
                roots_imag = root_imag_p[selection]
                if actual < effective_chunk:
                    roots_imag = torch.cat(
                        (
                            roots_imag,
                            roots_imag[-1:].expand(
                                (effective_chunk - actual,) + roots_imag.shape[1:]
                            ),
                        )
                    )
                valid &= radial_root_real_count(fields[6], roots_imag) == 4
                travel_p[selection] = travel[:actual]
                mino_p[selection] = observer_mino[:actual]
                error_p[selection] = error[:actual]
                valid_p[selection] = valid[:actual]
            break
        except Exception as caught:
            if _is_cuda_oom(caught, device) and effective_chunk > 1:
                oom_retries += 1
                effective_chunk = max(1, effective_chunk // 2)
                _SAFE_COORDINATE_CHUNKS[safe_key] = effective_chunk
                torch.cuda.empty_cache()
                warnings.warn(
                    "CUDA OOM in pooled Kerr observer coordinates; retrying with "
                    f"chunk_size={effective_chunk}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if compile_enabled and fallback_to_eager and not compile_fallback_used:
                _evict_coordinate_solver(
                    _coordinate_delay_analytic,
                    device=device,
                    dtype=dtype,
                    chunk_size=effective_chunk,
                    observer_radius_rg=observer_radius_rg,
                    compile_mode=compile_mode,
                )
                compile_enabled = False
                compile_fallback_used = True
                continue
            raise

    tolerance = float(float32_repair_tolerance) * torch.maximum(
        torch.ones_like(disk_mino_p), disk_mino_p.abs()
    )
    repair = (~valid_p) | (error_p > tolerance)
    repair_index = torch.nonzero(repair, as_tuple=False).reshape(-1)
    if repair_index.numel() and repair_float32:
        target = torch.device("cpu") if repair_device == "cpu" else device
        # The general quadrature repair expects one inclination scalar. Keep
        # its tiny queues separated by disk while the dense GPU work remains
        # pooled across all physical parameters.
        for disk_index, (spin, inclination) in enumerate(
            zip(spins, inclinations_deg, strict=True)
        ):
            begin, end = offsets[disk_index : disk_index + 2]
            disk_repairs = torch.nonzero(
                repair[begin:end], as_tuple=False
            ).reshape(-1) + begin
            for start in range(0, disk_repairs.numel(), int(repair_chunk_size)):
                local = disk_repairs[start : start + int(repair_chunk_size)]
                spin_r = torch.as_tensor(
                    spin, device=target, dtype=torch.float64
                )
                inclination_r = torch.as_tensor(
                    math.radians(inclination), device=target, dtype=torch.float64
                )
                alpha_r = -screen_x_p[local].to(device=target, dtype=torch.float64)
                beta_r = -screen_y_p[local].to(device=target, dtype=torch.float64)
                lam_r = -alpha_r * torch.sin(inclination_r)
                eta_r = (
                    (alpha_r.square() - spin_r.square())
                    * torch.cos(inclination_r).square()
                    + beta_r.square()
                )
                disk_mino_r, polar_valid_r = polar_mino_time(
                    spin_r, eta_r, lam_r, beta_r, inclination_r
                )
                root_real_r, root_imag_r = kerr_radial_root_parts(
                    spin_r, eta_r, lam_r
                )
                repaired = _coordinate_values(
                    spin_r,
                    inclination_r,
                    eta_r,
                    lam_r,
                    beta_r,
                    disk_mino_r,
                    root_real_r,
                    root_imag_r,
                    observer_radius_rg=observer_radius_rg,
                    quadrature_order=int(repair_quadrature_order),
                )
                _, travel_r, mino_r, error_r, valid_r = repaired
                valid_r &= polar_valid_r
                travel_p[local] = travel_r.to(device=device, dtype=dtype)
                mino_p[local] = mino_r.to(device=device, dtype=dtype)
                error_p[local] = error_r.to(device=device, dtype=dtype)
                valid_p[local] = valid_r.to(device)
    failed_total = int((~valid_p).sum().detach().cpu())
    if failed_total:
        raise RuntimeError(f"observer coordinates failed for {failed_total} hit pixels")

    traces = []
    for disk_index, (primary, screen, mass, spin, inclination, redshift, index) in enumerate(
        zip(
            primaries,
            screens,
            black_hole_masses_solar,
            spins,
            inclinations_deg,
            source_redshifts,
            per_disk_indices,
            strict=True,
        )
    ):
        begin, end = offsets[disk_index : disk_index + 2]
        size = primary.transfer.hit.numel()
        azimuth = torch.full((size,), float("nan"), device=device, dtype=dtype)
        travel = torch.full_like(azimuth, float("nan"))
        observer_mino = torch.full_like(azimuth, float("nan"))
        error = torch.full_like(azimuth, float("nan"))
        travel[index] = travel_p[begin:end]
        observer_mino[index] = mino_p[begin:end]
        error[index] = error_p[begin:end]
        repaired_count = int(repair[begin:end].sum().detach().cpu())
        prepared = _PreparedCoordinateArrays(
            azimuth,
            travel,
            observer_mino,
            error,
            repaired_count,
            0,
            execution,
            effective_chunk,
            oom_retries,
            {
                "coordinate_cross_disk_pooled": True,
                "coordinate_cross_disk_batch_size": count,
                "coordinate_cross_disk_pooled_rays": total,
                "coordinate_cross_disk_pooled_chunks": (
                    total + effective_chunk - 1
                )
                // effective_chunk,
            },
        )
        traces.append(
            add_observer_coordinates(
                primary,
                screen,
                black_hole_mass_solar=mass,
                spin=spin,
                inclination_deg=inclination,
                source_redshift=redshift,
                **common,
                _prepared_arrays=prepared,
            )
        )
    return tuple(traces)
