"""Separated Kerr null-geodesic algebra."""

from __future__ import annotations

import math

import torch

from .elliptic import _elliptic_f_principal, elliptic_f, elliptic_k, jacobi_sn_cn


def _signed_safe_denominator(value: torch.Tensor) -> torch.Tensor:
    epsilon = math.sqrt(torch.finfo(value.dtype).tiny)
    return torch.where(
        value >= 0,
        value.clamp_min(epsilon),
        value.clamp_max(-epsilon),
    )


def _complex_multiply(
    left_real: torch.Tensor,
    left_imag: torch.Tensor,
    right_real: torch.Tensor,
    right_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        left_real * right_real - left_imag * right_imag,
        left_real * right_imag + left_imag * right_real,
    )


def _complex_divide(
    numerator_real: torch.Tensor,
    numerator_imag: torch.Tensor,
    denominator_real: torch.Tensor,
    denominator_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = _signed_safe_denominator(
        denominator_real.square() + denominator_imag.square()
    )
    return (
        (
            numerator_real * denominator_real
            + numerator_imag * denominator_imag
        )
        / scale,
        (
            numerator_imag * denominator_real
            - numerator_real * denominator_imag
        )
        / scale,
    )


def _complex_sqrt(
    real: torch.Tensor,
    imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    magnitude = torch.hypot(real, imag)
    root_real = torch.sqrt((0.5 * (magnitude + real)).clamp_min(0.0))
    root_imag = torch.copysign(
        torch.sqrt((0.5 * (magnitude - real)).clamp_min(0.0)),
        imag,
    )
    return root_real, root_imag


def _complex_cbrt(
    real: torch.Tensor,
    imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    magnitude = torch.hypot(real, imag).pow(1.0 / 3.0)
    phase = torch.atan2(imag, real) / 3.0
    return magnitude * torch.cos(phase), magnitude * torch.sin(phase)


def kerr_radial_root_parts(
    spin: torch.Tensor,
    carter_eta: torch.Tensor,
    photon_lambda: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return four Kerr radial-potential roots as real/imaginary arrays.

    The two outputs have shape ``broadcast_shape + (4,)``. The root ordering
    is stable across the four-, two-, and zero-real-root branches used by the
    manifestly real geodesic inversion. Only real tensors are constructed, so
    this function is suitable for ``torch.compile`` and CUDA graph capture.
    """

    spin, eta, lam = torch.broadcast_tensors(spin, carter_eta, photon_lambda)
    if spin.is_complex() or eta.is_complex() or lam.is_complex():
        raise TypeError("Kerr constants of motion must be real")
    if eta.dtype not in (torch.float32, torch.float64):
        raise TypeError("Kerr geodesics require float32 or float64 tensors")
    spin2 = spin.square()
    coefficient_a = spin2 - eta - lam.square()
    coefficient_b = 2.0 * (eta + (lam - spin).square())
    coefficient_c = -spin2 * eta
    p = -coefficient_a.square() / 12.0 - coefficient_c
    q = (
        -(coefficient_a / 3.0)
        * (coefficient_a.square() / 36.0 - coefficient_c)
        - coefficient_b.square() / 8.0
    )
    negative_discriminant3 = 4.0 * p.pow(3) + 27.0 * q.square()
    sqrt_real, sqrt_imag = _complex_sqrt(
        negative_discriminant3 / 108.0,
        torch.zeros_like(negative_discriminant3),
    )
    omega_base_real = -0.5 * q + sqrt_real
    omega_base_imag = sqrt_imag
    tiny = torch.finfo(eta.dtype).tiny
    omega_small = torch.hypot(omega_base_real, omega_base_imag) <= tiny
    omega_base_real = torch.where(
        omega_small,
        omega_base_real + tiny,
        omega_base_real,
    )
    omega_real, omega_imag = _complex_cbrt(
        omega_base_real,
        omega_base_imag,
    )

    unity_real = torch.tensor(
        [-0.5, -0.5, 1.0],
        dtype=eta.dtype,
        device=eta.device,
    )
    unity_imag = torch.tensor(
        [0.5 * math.sqrt(3.0), -0.5 * math.sqrt(3.0), 0.0],
        dtype=eta.dtype,
        device=eta.device,
    )
    candidate_real, candidate_imag = _complex_multiply(
        omega_real.unsqueeze(-1),
        omega_imag.unsqueeze(-1),
        unity_real,
        unity_imag,
    )
    candidate_small = torch.hypot(candidate_real, candidate_imag) <= tiny
    candidate_real = torch.where(
        candidate_small,
        candidate_real + tiny,
        candidate_real,
    )
    quotient_real, quotient_imag = _complex_divide(
        p.unsqueeze(-1),
        torch.zeros_like(candidate_real),
        3.0 * candidate_real,
        3.0 * candidate_imag,
    )
    resolvent_real = candidate_real - quotient_real
    resolvent_imag = candidate_imag - quotient_imag
    choice = torch.argmax(resolvent_real, dim=-1, keepdim=True)
    xi_real = (
        torch.gather(resolvent_real, -1, choice).squeeze(-1)
        - coefficient_a / 3.0
    )
    xi_imag = torch.gather(resolvent_imag, -1, choice).squeeze(-1)

    sqrt_2xi_real, sqrt_2xi_imag = _complex_sqrt(
        2.0 * xi_real,
        2.0 * xi_imag,
    )
    sqrt_xi_real, sqrt_xi_imag = _complex_sqrt(xi_real, xi_imag)
    sqrt_xi_small = torch.hypot(sqrt_xi_real, sqrt_xi_imag) <= tiny
    sqrt_xi_real = torch.where(
        sqrt_xi_small,
        sqrt_xi_real + tiny,
        sqrt_xi_real,
    )
    predeterminant_real, predeterminant_imag = _complex_divide(
        math.sqrt(2.0) * coefficient_b,
        torch.zeros_like(coefficient_b),
        sqrt_xi_real,
        sqrt_xi_imag,
    )
    common_real = 2.0 * (coefficient_a + xi_real)
    common_imag = 2.0 * xi_imag
    det1_real, det1_imag = _complex_sqrt(
        -common_real + predeterminant_real,
        -common_imag + predeterminant_imag,
    )
    det2_real, det2_imag = _complex_sqrt(
        -common_real - predeterminant_real,
        -common_imag - predeterminant_imag,
    )
    root_real = torch.stack(
        (
            0.5 * (-sqrt_2xi_real - det1_real),
            0.5 * (-sqrt_2xi_real + det1_real),
            0.5 * (sqrt_2xi_real - det2_real),
            0.5 * (sqrt_2xi_real + det2_real),
        ),
        dim=-1,
    )
    root_imag = torch.stack(
        (
            0.5 * (-sqrt_2xi_imag - det1_imag),
            0.5 * (-sqrt_2xi_imag + det1_imag),
            0.5 * (sqrt_2xi_imag - det2_imag),
            0.5 * (sqrt_2xi_imag + det2_imag),
        ),
        dim=-1,
    )

    tolerance = math.sqrt(torch.finfo(eta.dtype).eps)
    root_scale = torch.maximum(
        torch.hypot(root_real, root_imag),
        torch.ones_like(root_real),
    )
    realish = root_imag.abs() <= tolerance * root_scale
    reorder = (realish.sum(dim=-1) == 2) & realish[..., 3]
    reordered_real = root_real[..., [0, 3, 1, 2]]
    reordered_imag = root_imag[..., [0, 3, 1, 2]]
    return (
        torch.where(reorder.unsqueeze(-1), reordered_real, root_real),
        torch.where(reorder.unsqueeze(-1), reordered_imag, root_imag),
    )


def radial_root_real_count(
    root_real: torch.Tensor,
    root_imag: torch.Tensor,
) -> torch.Tensor:
    """Count numerically real roots using the tracer's scale-aware rule."""

    if root_real.shape != root_imag.shape or root_real.shape[-1] != 4:
        raise ValueError("root arrays must share shape [..., 4]")
    tolerance = math.sqrt(torch.finfo(root_real.dtype).eps)
    scale = torch.maximum(
        torch.hypot(root_real, root_imag),
        torch.ones_like(root_real),
    )
    return (root_imag.abs() <= tolerance * scale).sum(dim=-1)


def polar_mino_time(
    spin: torch.Tensor,
    carter_eta: torch.Tensor,
    photon_lambda: torch.Tensor,
    screen_beta: torch.Tensor,
    inclination_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Mino time to the first equatorial crossing and validity mask."""

    spin, eta, lam, beta, inclination = torch.broadcast_tensors(
        spin,
        carter_eta,
        photon_lambda,
        screen_beta,
        inclination_rad,
    )
    spin2 = spin.square()
    coefficient = spin2 - eta - lam.square()
    discriminant = torch.sqrt(
        (coefficient.square() + 4.0 * spin2 * eta).clamp_min(0.0)
    )
    u_plus = 2.0 * eta / _signed_safe_denominator(
        discriminant - coefficient
    )
    uminus_spin2 = 0.5 * (coefficient - discriminant)
    scale = torch.rsqrt(
        uminus_spin2.abs().clamp_min(torch.finfo(eta.dtype).tiny)
    )
    parameter = spin2 * u_plus / _signed_safe_denominator(uminus_spin2)
    observer_argument = torch.cos(inclination) / torch.sqrt(
        u_plus.clamp_min(torch.finfo(eta.dtype).tiny)
    )
    valid = (
        (eta >= 0.0)
        & (u_plus > 0.0)
        & (observer_argument.abs() <= 1.0 + 64.0 * torch.finfo(eta.dtype).eps)
        & torch.isfinite(parameter)
    )
    safe_argument = observer_argument.clamp(
        -1.0 + 4.0 * torch.finfo(eta.dtype).eps,
        1.0 - 4.0 * torch.finfo(eta.dtype).eps,
    )
    observer_time = scale * _elliptic_f_principal(
        torch.asin(safe_argument),
        parameter,
    )
    half_period = 2.0 * scale * elliptic_k(parameter)
    mino_time = torch.where(beta > 0.0, half_period - observer_time, observer_time)
    valid = valid & torch.isfinite(mino_time) & (mino_time >= 0.0)
    return mino_time, valid


def _radial_four_real(
    root_real: torch.Tensor,
    mino_time: torch.Tensor,
    horizon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r1, r2, r3, r4 = root_real.unbind(dim=-1)
    r31, r32 = r3 - r1, r3 - r2
    r41, r42 = r4 - r1, r4 - r2
    parameter = r32 * r41 / _signed_safe_denominator(r31 * r42)
    coefficient = 2.0 * torch.rsqrt(
        (r31 * r42).clamp_min(torch.finfo(mino_time.dtype).tiny)
    )
    infinity_argument = torch.sqrt(
        (r31 / _signed_safe_denominator(r41)).clamp(0.0, 1.0)
    )
    infinity_time = coefficient * _elliptic_f_principal(
        torch.asin(infinity_argument),
        parameter,
    )
    horizon_argument = torch.sqrt(
        (
            (horizon - r4)
            / _signed_safe_denominator(horizon - r3)
            * r31
            / _signed_safe_denominator(r41)
        ).abs()
    )
    horizon_time = torch.where(
        horizon_argument < 1.0,
        coefficient
        * _elliptic_f_principal(
            torch.asin(horizon_argument.clamp(0.0, 1.0)),
            parameter,
        ),
        torch.zeros_like(horizon_argument),
    )
    valid = (mino_time <= 2.0 * infinity_time) & ~(
        (r4 < horizon) & (mino_time > infinity_time - horizon_time)
    )
    argument = 0.5 * torch.sqrt((r31 * r42).clamp_min(0.0)) * (
        infinity_time - mino_time
    )
    sn, _ = jacobi_sn_cn(argument, parameter)
    sn_term = r41 * sn.square()
    radius = (r31 * r4 - r3 * sn_term) / _signed_safe_denominator(
        r31 - sn_term
    )
    return radius, valid & torch.isfinite(radius) & (radius > horizon)


def _complex_product_abs(
    left_real: torch.Tensor,
    left_imag: torch.Tensor,
    right_real: torch.Tensor,
    right_imag: torch.Tensor,
) -> torch.Tensor:
    real = left_real * right_real - left_imag * right_imag
    imag = left_real * right_imag + left_imag * right_real
    return torch.hypot(real, imag)


def _radial_two_real(
    root_real: torch.Tensor,
    root_imag: torch.Tensor,
    mino_time: torch.Tensor,
    horizon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r1, r2, r3, r4 = root_real.unbind(dim=-1)
    i1, i2, i3, i4 = root_imag.unbind(dim=-1)
    amp_a = torch.sqrt(
        _complex_product_abs(r3 - r2, i3 - i2, r4 - r2, i4 - i2)
    )
    amp_b = torch.sqrt(
        _complex_product_abs(r3 - r1, i3 - i1, r4 - r1, i4 - i1)
    )
    parameter = ((amp_a + amp_b).square() - (r2 - r1).square()) / (
        _signed_safe_denominator(4.0 * amp_a * amp_b)
    )
    coefficient = torch.rsqrt(
        (amp_a * amp_b).clamp_min(torch.finfo(mino_time.dtype).tiny)
    )
    infinity_argument = (
        (amp_a - amp_b) / _signed_safe_denominator(amp_a + amp_b)
    ).clamp(-1.0, 1.0)
    infinity_time = coefficient * elliptic_f(
        torch.acos(infinity_argument),
        parameter,
    )
    ratio = amp_b * (horizon - r2) / _signed_safe_denominator(
        amp_a * (horizon - r1)
    )
    horizon_argument = (
        (1.0 - ratio) / _signed_safe_denominator(1.0 + ratio)
    ).clamp(-1.0, 1.0)
    horizon_time = coefficient * elliptic_f(
        torch.acos(horizon_argument),
        parameter,
    )
    valid = mino_time <= infinity_time - horizon_time
    argument = torch.sqrt((amp_a * amp_b).clamp_min(0.0)) * (
        infinity_time - mino_time
    )
    _, cn = jacobi_sn_cn(argument, parameter)
    numerator = (
        -amp_a * r1
        + amp_b * r2
        + (amp_a * r1 + amp_b * r2) * cn
    )
    denominator = -amp_a + amp_b + (amp_a + amp_b) * cn
    radius = numerator / _signed_safe_denominator(denominator)
    return radius, valid & torch.isfinite(radius) & (radius > horizon)


def _radial_no_real(
    root_real: torch.Tensor,
    root_imag: torch.Tensor,
    mino_time: torch.Tensor,
    horizon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, r2, _, r4 = root_real.unbind(dim=-1)
    _, i2, _, i4 = root_imag.unbind(dim=-1)
    amp1 = i4.abs()
    amp2 = i2.abs().clamp_min(torch.finfo(mino_time.dtype).tiny)
    cvalue = torch.sqrt((amp1 - amp2).square() + (r4 - r2).square())
    dvalue = torch.sqrt((amp1 + amp2).square() + (r4 - r2).square())
    sum_cd = (cvalue + dvalue).clamp_min(torch.finfo(mino_time.dtype).tiny)
    parameter = 4.0 * cvalue * dvalue / sum_cd.square()
    tangent_numerator = (
        4.0 * amp2.square() - (cvalue - dvalue).square()
    ).clamp_min(0.0)
    tangent_denominator = _signed_safe_denominator(
        sum_cd.square() - 4.0 * amp2.square()
    )
    tangent = torch.sqrt(
        (tangent_numerator / tangent_denominator).clamp_min(0.0)
    )
    coefficient = 2.0 / sum_cd
    infinity_time = coefficient * elliptic_f(
        0.5 * math.pi + torch.atan(tangent),
        parameter,
    )
    horizon_time = coefficient * elliptic_f(
        torch.atan((horizon + r4) / amp2) + torch.atan(tangent),
        parameter,
    )
    valid = mino_time <= infinity_time - horizon_time
    argument = 0.5 * sum_cd * (infinity_time - mino_time)
    sn, cn = jacobi_sn_cn(argument, parameter)
    sc = sn / _signed_safe_denominator(cn)
    radius = -(
        amp2 * (tangent - sc) / _signed_safe_denominator(1.0 + tangent * sc)
        + r4
    )
    return radius, valid & torch.isfinite(radius) & (radius > horizon)


def invert_radial_motion(
    spin: torch.Tensor,
    root_real: torch.Tensor,
    root_imag: torch.Tensor,
    mino_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert separated radial motion using manifestly real root branches."""

    horizon = torch.ones_like(mino_time) * (
        1.0 + torch.sqrt((1.0 - spin.square()).clamp_min(0.0))
    )
    counts = radial_root_real_count(root_real, root_imag)
    radius4, valid4 = _radial_four_real(root_real, mino_time, horizon)
    radius2, valid2 = _radial_two_real(
        root_real,
        root_imag,
        mino_time,
        horizon,
    )
    radius0, valid0 = _radial_no_real(
        root_real,
        root_imag,
        mino_time,
        horizon,
    )
    is4, is2, is0 = counts == 4, counts == 2, counts == 0
    radius = torch.where(is4, radius4, torch.where(is2, radius2, radius0))
    valid = (is4 & valid4) | (is2 & valid2) | (is0 & valid0)
    return radius, valid, counts


def incoming_radial_mino_time(
    spin: torch.Tensor,
    root_real: torch.Tensor,
    root_imag: torch.Tensor,
    target_radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locate a finite radius on the incoming infinity-connected branch."""

    counts = radial_root_real_count(root_real, root_imag)
    r1, r2, r3, r4 = root_real.unbind(dim=-1)
    i1, i2, i3, i4 = root_imag.unbind(dim=-1)
    tiny = torch.finfo(target_radius.dtype).tiny

    r31, r32 = r3 - r1, r3 - r2
    r41, r42 = r4 - r1, r4 - r2
    parameter4 = r32 * r41 / _signed_safe_denominator(r31 * r42)
    coefficient4 = 2.0 * torch.rsqrt((r31 * r42).clamp_min(tiny))
    infinity_argument4 = torch.sqrt(
        (r31 / _signed_safe_denominator(r41)).clamp(0.0, 1.0)
    )
    infinity_time4 = coefficient4 * _elliptic_f_principal(
        torch.asin(infinity_argument4),
        parameter4,
    )
    radius_argument4 = torch.sqrt(
        (
            r31
            * (target_radius - r4)
            / _signed_safe_denominator(r41 * (target_radius - r3))
        ).clamp(0.0, 1.0)
    )
    time4 = infinity_time4 - coefficient4 * _elliptic_f_principal(
        torch.asin(radius_argument4),
        parameter4,
    )

    amp_a = torch.sqrt(
        _complex_product_abs(r3 - r2, i3 - i2, r4 - r2, i4 - i2)
    )
    amp_b = torch.sqrt(
        _complex_product_abs(r3 - r1, i3 - i1, r4 - r1, i4 - i1)
    )
    parameter2 = (
        (amp_a + amp_b).square() - (r2 - r1).square()
    ) / _signed_safe_denominator(4.0 * amp_a * amp_b)
    coefficient2 = torch.rsqrt((amp_a * amp_b).clamp_min(tiny))
    infinity_argument2 = (
        (amp_a - amp_b) / _signed_safe_denominator(amp_a + amp_b)
    ).clamp(-1.0, 1.0)
    infinity_time2 = coefficient2 * elliptic_f(
        torch.acos(infinity_argument2),
        parameter2,
    )
    radius_argument2 = (
        ((-amp_a * r1 + amp_b * r2) - target_radius * (-amp_a + amp_b))
        / _signed_safe_denominator(
            target_radius * (amp_a + amp_b) - (amp_a * r1 + amp_b * r2)
        )
    ).clamp(-1.0, 1.0)
    time2 = infinity_time2 - coefficient2 * elliptic_f(
        torch.acos(radius_argument2),
        parameter2,
    )

    amp1 = i4.abs()
    amp2 = i2.abs().clamp_min(tiny)
    cvalue = torch.sqrt((amp1 - amp2).square() + (r4 - r2).square())
    dvalue = torch.sqrt((amp1 + amp2).square() + (r4 - r2).square())
    sum_cd = (cvalue + dvalue).clamp_min(tiny)
    parameter0 = 4.0 * cvalue * dvalue / sum_cd.square()
    tangent = torch.sqrt(
        (
            (4.0 * amp2.square() - (cvalue - dvalue).square()).clamp_min(0.0)
            / _signed_safe_denominator(sum_cd.square() - 4.0 * amp2.square())
        ).clamp_min(0.0)
    )
    coefficient0 = 2.0 / sum_cd
    infinity_time0 = coefficient0 * elliptic_f(
        0.5 * math.pi + torch.atan(tangent),
        parameter0,
    )
    radius_value0 = -(target_radius + r4) / amp2
    sc_radius0 = (tangent - radius_value0) / _signed_safe_denominator(
        1.0 + radius_value0 * tangent
    )
    amplitude0 = torch.atan(sc_radius0)
    amplitude0 = torch.where(
        amplitude0 < 0.0,
        amplitude0 + math.pi,
        amplitude0,
    )
    time0 = infinity_time0 - coefficient0 * elliptic_f(
        amplitude0,
        parameter0,
    )

    time = torch.where(counts == 4, time4, torch.where(counts == 2, time2, time0))
    time = time.clamp_min(0.0)
    radius_check, _, _ = invert_radial_motion(
        spin,
        root_real,
        root_imag,
        time,
    )
    tolerance = 3.0e-3 if target_radius.dtype == torch.float32 else 3.0e-8
    valid = (
        torch.isfinite(time)
        & torch.isfinite(radius_check)
        & (
            (radius_check - target_radius).abs()
            <= tolerance * torch.maximum(torch.ones_like(target_radius), target_radius)
        )
    )
    return time, valid


def inner_four_root_phase(
    root_real: torch.Tensor,
    radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return phase on the bounded interval of a four-real-root ray."""

    q1, q2, q3, q4 = root_real.unbind(dim=-1)
    scale = torch.sqrt(
        ((q4 - q2) * (q3 - q1)).clamp_min(torch.finfo(radius.dtype).tiny)
    )
    parameter = (q3 - q2) * (q4 - q1) / _signed_safe_denominator(
        (q3 - q1) * (q4 - q2)
    )
    argument2 = (q4 - q2) * (q3 - radius) / _signed_safe_denominator(
        (q3 - q2) * (q4 - radius)
    )
    phase = 2.0 / scale * _elliptic_f_principal(
        torch.asin(torch.sqrt(argument2.clamp(0.0, 1.0))),
        parameter,
    )
    valid = (
        torch.isfinite(phase)
        & (radius >= q2)
        & (radius <= q3)
        & (parameter >= 0.0)
        & (parameter < 1.0)
    )
    return phase, valid


def lamppost_radial_radius(
    spin: torch.Tensor,
    root_real: torch.Tensor,
    root_imag: torch.Tensor,
    phase: torch.Tensor,
    inner_four_real: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate either radial branch used by an axial lamppost ray."""

    outer_radius, outer_valid, _ = invert_radial_motion(
        spin,
        root_real,
        root_imag,
        phase,
    )
    q1, q2, q3, q4 = root_real.unbind(dim=-1)
    scale = torch.sqrt(
        ((q4 - q2) * (q3 - q1)).clamp_min(torch.finfo(phase.dtype).tiny)
    )
    parameter = (q3 - q2) * (q4 - q1) / _signed_safe_denominator(
        (q3 - q1) * (q4 - q2)
    )
    sn, _ = jacobi_sn_cn(0.5 * scale * phase, parameter)
    sn2 = sn.square()
    inner_radius = (
        (q4 - q2) * q3 - (q3 - q2) * q4 * sn2
    ) / _signed_safe_denominator((q4 - q2) - (q3 - q2) * sn2)
    horizon = 1.0 + torch.sqrt((1.0 - spin.square()).clamp_min(0.0))
    horizon_in_interval = (horizon > q2) & (horizon < q3)
    horizon_phase, horizon_phase_valid = inner_four_root_phase(root_real, horizon)
    inner_valid = (
        torch.isfinite(inner_radius)
        & torch.isfinite(parameter)
        & (parameter >= 0.0)
        & (parameter < 1.0)
        & (inner_radius > horizon)
        & (
            (~horizon_in_interval)
            | (horizon_phase_valid & (phase < horizon_phase))
        )
    )
    return (
        torch.where(inner_four_real, inner_radius, outer_radius),
        torch.where(inner_four_real, inner_valid, outer_valid),
    )
