"""Differentiable circular-orbit Kerr thin-disk functions."""

from __future__ import annotations

import math

import torch


def _floating_tensor(value) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.is_floating_point():
        return tensor
    return tensor.to(torch.get_default_dtype())


def _signed_safe_denominator(value: torch.Tensor) -> torch.Tensor:
    tiny = math.sqrt(torch.finfo(value.dtype).tiny)
    positive = torch.full_like(value, tiny)
    negative = torch.full_like(value, -tiny)
    return torch.where(
        value >= 0,
        torch.maximum(value, positive),
        torch.minimum(value, negative),
    )


def kerr_isco_radius(spin) -> torch.Tensor:
    """Return the equatorial ISCO radius in units of ``GM/c^2``.

    Positive spin denotes a prograde disk. The small-spin expansion removes
    the derivative singularity introduced by the absolute value and explicit
    branch sign in the usual closed form.
    """

    spin = _floating_tensor(spin)
    if bool(torch.any(~torch.isfinite(spin))) or bool(torch.any(spin.abs() > 0.998)):
        raise ValueError("spin must be finite and lie in [-0.998, 0.998]")
    central_width = 1.0e-4
    exact_spin = torch.where(
        spin.abs() < central_width,
        torch.full_like(spin, central_width),
        spin,
    )
    absolute = exact_spin.abs()
    z1 = 1.0 + (1.0 - absolute.square()).clamp_min(0.0).pow(1.0 / 3.0) * (
        (1.0 + absolute).pow(1.0 / 3.0)
        + (1.0 - absolute).clamp_min(0.0).pow(1.0 / 3.0)
    )
    z2 = torch.sqrt(3.0 * absolute.square() + z1.square())
    sign = torch.where(exact_spin >= 0, 1.0, -1.0)
    exact = 3.0 + z2 - sign * torch.sqrt(
        ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).clamp_min(0.0)
    )
    central = (
        6.0
        - 4.0 * math.sqrt(2.0 / 3.0) * spin
        - (7.0 / 18.0) * spin.square()
    )
    return torch.where(spin.abs() < central_width, central, exact)


def kerr_radiative_efficiency(spin) -> torch.Tensor:
    """Return the Novikov--Thorne efficiency at the equatorial Kerr ISCO.

    The expression ``1 - sqrt(1 - 2 / (3 r_ISCO))`` is evaluated with the
    same differentiable ISCO convention as :func:`kerr_isco_radius`.
    """

    radius = kerr_isco_radius(spin)
    return 1.0 - torch.sqrt(
        (1.0 - 2.0 / (3.0 * radius)).clamp_min(0.0)
    )


def lamppost_source_height_rg(spin, height_above_isco_rg) -> torch.Tensor:
    """Convert a lamppost height above the ISCO to a Boyer--Lindquist radius."""

    radius = kerr_isco_radius(spin)
    height = torch.as_tensor(
        height_above_isco_rg,
        device=radius.device,
        dtype=radius.dtype,
    )
    if bool(torch.any(~torch.isfinite(height))) or bool(torch.any(height < 0.0)):
        raise ValueError("height_above_isco_rg must be finite and non-negative")
    return radius + height


def circular_disk_gfactor(radius_rg, spin, photon_lambda) -> torch.Tensor:
    """Frequency shift for a circular equatorial emitter.

    ``photon_lambda`` is the conserved axial angular momentum divided by the
    photon energy. The result is ``nu_observed / nu_emitted``.
    """

    radius = _floating_tensor(radius_rg)
    spin = torch.as_tensor(spin, device=radius.device, dtype=radius.dtype)
    photon_lambda = torch.as_tensor(
        photon_lambda,
        device=radius.device,
        dtype=radius.dtype,
    )
    radius, spin, photon_lambda = torch.broadcast_tensors(
        radius,
        spin,
        photon_lambda,
    )
    r = radius.clamp_min(torch.finfo(radius.dtype).tiny)
    omega = torch.reciprocal(r.pow(1.5) + spin)
    metric_tt = -(1.0 - 2.0 / r)
    metric_tphi = -2.0 * spin / r
    metric_phiphi = r.square() + spin.square() + 2.0 * spin.square() / r
    inverse_ut = torch.sqrt(
        (
            -metric_tt
            - 2.0 * omega * metric_tphi
            - omega.square() * metric_phiphi
        ).clamp_min(0.0)
    )
    return inverse_ut / _signed_safe_denominator(1.0 - omega * photon_lambda)


def circular_disk_zamo_lorentz_factor(radius_rg, spin) -> torch.Tensor:
    """Lorentz factor of a circular disk relative to the local ZAMO."""

    radius = _floating_tensor(radius_rg)
    spin = torch.as_tensor(spin, device=radius.device, dtype=radius.dtype)
    radius, spin = torch.broadcast_tensors(radius, spin)
    r = radius.clamp_min(torch.finfo(radius.dtype).tiny)
    omega = torch.reciprocal(r.pow(1.5) + spin)
    metric_tt = -(1.0 - 2.0 / r)
    metric_tphi = -2.0 * spin / r
    metric_phiphi = r.square() + spin.square() + 2.0 * spin.square() / r
    inverse_ut = torch.sqrt(
        (
            -metric_tt
            - 2.0 * omega * metric_tphi
            - omega.square() * metric_phiphi
        ).clamp_min(torch.finfo(radius.dtype).tiny)
    )
    delta = r.square() - 2.0 * r + spin.square()
    lapse = torch.sqrt(
        (delta / metric_phiphi).clamp_min(torch.finfo(radius.dtype).tiny)
    )
    return lapse / inverse_ut


def approximate_circular_disk_gfactor(
    radius_rg,
    azimuth_rad,
    inclination_rad,
    spin,
) -> torch.Tensor:
    """Straight-screen approximation to the circular-disk frequency shift.

    This retains gravitational and orbital Doppler shifts but does not bend
    photon trajectories. It matches the legacy package's ``gr_mode='approx'``
    convention and is explicitly distinct from full Kerr ray tracing.
    """

    radius = _floating_tensor(radius_rg)
    azimuth = torch.as_tensor(
        azimuth_rad,
        device=radius.device,
        dtype=radius.dtype,
    )
    inclination = torch.as_tensor(
        inclination_rad,
        device=radius.device,
        dtype=radius.dtype,
    )
    spin = torch.as_tensor(spin, device=radius.device, dtype=radius.dtype)
    radius, azimuth, inclination, spin = torch.broadcast_tensors(
        radius,
        azimuth,
        inclination,
        spin,
    )
    r = radius.clamp_min(torch.finfo(radius.dtype).tiny)
    omega = torch.reciprocal(r.pow(1.5) + spin)
    metric_tt = -(1.0 - 2.0 / r)
    metric_tphi = -2.0 * spin / r
    metric_phiphi = r.square() + spin.square() + 2.0 * spin.square() / r
    inverse_ut = torch.sqrt(
        (
            -metric_tt
            - 2.0 * omega * metric_tphi
            - omega.square() * metric_phiphi
        ).clamp_min(torch.finfo(radius.dtype).tiny)
    )
    impact = r * torch.sin(inclination) * torch.sin(azimuth)
    one_plus_redshift = (1.0 + omega * impact) / inverse_ut
    return torch.reciprocal(
        one_plus_redshift.clamp_min(torch.finfo(radius.dtype).tiny)
    )


def page_thorne_flux_factor(radius_rg, spin, isco_rg=None) -> torch.Tensor:
    """Return the dimensionless Page--Thorne surface-flux factor.

    The Schwarzschild limit is evaluated analytically. A centered continuation
    through a narrow interval around zero spin preserves finite derivatives.
    Values at and inside the ISCO are exactly zero.
    """

    radius = _floating_tensor(radius_rg)
    spin = torch.as_tensor(spin, device=radius.device, dtype=radius.dtype)
    radius, spin = torch.broadcast_tensors(radius, spin)
    if isco_rg is None:
        isco = kerr_isco_radius(spin)
    else:
        isco = torch.as_tensor(isco_rg, device=radius.device, dtype=radius.dtype)
        isco = torch.broadcast_to(isco, radius.shape)
    x_raw = torch.sqrt(radius.clamp_min(torch.finfo(radius.dtype).tiny))
    x0 = torch.sqrt(isco.clamp_min(torch.finfo(radius.dtype).tiny))
    x = torch.maximum(x_raw, x0)

    def generic(spin_value, inner_x):
        root1 = 2.0 * torch.cos(torch.acos(spin_value) / 3.0 - math.pi / 3.0)
        root2 = 2.0 * torch.cos(torch.acos(spin_value) / 3.0 + math.pi / 3.0)
        root3 = -2.0 * torch.cos(torch.acos(spin_value) / 3.0)
        tiny = torch.finfo(radius.dtype).tiny
        bracket = x - inner_x - 1.5 * spin_value * torch.log(
            (x / inner_x).clamp_min(tiny)
        )
        roots = ((root1, root2, root3), (root2, root1, root3), (root3, root1, root2))
        for current, other1, other2 in roots:
            coefficient = 3.0 * (current - spin_value).square() / (
                current * (current - other1) * (current - other2)
            )
            ratio = (x - current) / (inner_x - current)
            bracket = bracket - coefficient * torch.log(ratio.clamp_min(tiny))
        denominator = (
            x.pow(7) - 3.0 * x.pow(5) + 2.0 * spin_value * x.pow(4)
        )
        return bracket / denominator.clamp_min(tiny)

    sqrt_three = math.sqrt(3.0)
    schwarzschild_bracket = x - math.sqrt(6.0) - 0.5 * sqrt_three * torch.log(
        (
            (x - sqrt_three)
            * (math.sqrt(6.0) + sqrt_three)
            / ((x + sqrt_three) * (math.sqrt(6.0) - sqrt_three))
        ).clamp_min(torch.finfo(radius.dtype).tiny)
    )
    schwarzschild = schwarzschild_bracket / (
        x.pow(5) * (x.square() - 3.0)
    ).clamp_min(torch.finfo(radius.dtype).tiny)
    central_width = 1.0e-3 if radius.dtype == torch.float32 else 1.0e-6
    width = torch.as_tensor(central_width, device=radius.device, dtype=radius.dtype)
    plus = torch.full_like(spin, central_width)
    minus = -plus
    slope = (
        generic(plus, torch.sqrt(kerr_isco_radius(plus)))
        - generic(minus, torch.sqrt(kerr_isco_radius(minus)))
    ) / (2.0 * width)
    central = schwarzschild + spin * slope
    safe_spin = torch.where(spin.abs() < width, plus, spin)
    result = torch.where(spin.abs() < width, central, generic(safe_spin, x0))
    return torch.where(
        radius > isco,
        result.clamp_min(0.0),
        torch.zeros_like(result),
    )
