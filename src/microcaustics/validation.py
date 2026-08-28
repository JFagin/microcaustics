"""Unambiguous numerical comparisons for maps and light curves."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MapComparison:
    """Map-error metrics evaluated over common valid positive pixels.

    ``linear_nrmse`` is the global linear residual RMS divided by the
    reference-map RMS. ``fractional_nrmse`` is the RMS of the pixelwise
    fractional residual, matching the paper's Zheng-style convention.
    Magnitude residuals are candidate minus reference.
    """

    linear_nrmse: float
    fractional_nrmse: float
    rmse_mmag: float
    bias_mmag: float
    p95_absolute_mmag: float
    valid_pixels: int


@dataclass(frozen=True)
class LightCurveComparison:
    """Flux-ratio magnitude errors for a time-by-band light curve."""

    rmse_mmag: float
    bias_mmag: float
    p95_absolute_mmag: float
    maximum_absolute_mmag: float
    valid_samples: int
    constant_offset_removed: bool = False


def compare_magnification_maps(candidate, reference) -> MapComparison:
    """Compare two magnification arrays without fitting an offset or scale."""

    candidate_tensor = torch.as_tensor(candidate, dtype=torch.float64)
    reference_tensor = torch.as_tensor(
        reference,
        device=candidate_tensor.device,
        dtype=torch.float64,
    )
    if candidate_tensor.shape != reference_tensor.shape:
        raise ValueError("candidate and reference maps must have the same shape")
    valid = (
        torch.isfinite(candidate_tensor)
        & torch.isfinite(reference_tensor)
        & (candidate_tensor > 0)
        & (reference_tensor > 0)
    )
    count = int(valid.sum().detach().cpu())
    if count == 0:
        raise ValueError("maps have no common finite positive pixels")
    candidate_valid = candidate_tensor[valid]
    reference_valid = reference_tensor[valid]
    residual = candidate_valid - reference_valid
    reference_rms = torch.sqrt(torch.mean(reference_valid.square()))
    linear_nrmse = torch.sqrt(torch.mean(residual.square())) / reference_rms
    fractional = residual / reference_valid
    magnitude_mmag = -2500.0 * torch.log10(candidate_valid / reference_valid)
    absolute = magnitude_mmag.abs()
    return MapComparison(
        linear_nrmse=float(linear_nrmse.detach().cpu()),
        fractional_nrmse=float(torch.sqrt(torch.mean(fractional.square())).detach().cpu()),
        rmse_mmag=float(torch.sqrt(torch.mean(magnitude_mmag.square())).detach().cpu()),
        bias_mmag=float(torch.mean(magnitude_mmag).detach().cpu()),
        p95_absolute_mmag=float(torch.quantile(absolute, 0.95).detach().cpu()),
        valid_pixels=count,
    )


def compare_light_curves(
    candidate_flux,
    reference_flux,
    *,
    remove_constant_offset: bool = False,
) -> LightCurveComparison:
    """Compare positive fluxes in millimagnitudes.

    By default a constant magnification offset is retained. Set
    ``remove_constant_offset=True`` only for an explicitly shape-only audit;
    the median magnitude residual is then removed before calculating errors.
    """

    candidate = torch.as_tensor(candidate_flux, dtype=torch.float64)
    reference = torch.as_tensor(
        reference_flux,
        device=candidate.device,
        dtype=torch.float64,
    )
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference fluxes must have the same shape")
    valid = (
        torch.isfinite(candidate)
        & torch.isfinite(reference)
        & (candidate > 0)
        & (reference > 0)
    )
    count = int(valid.sum().detach().cpu())
    if count == 0:
        raise ValueError("light curves have no common finite positive samples")
    residual_mmag = -2500.0 * torch.log10(candidate[valid] / reference[valid])
    if remove_constant_offset:
        residual_mmag = residual_mmag - torch.median(residual_mmag)
    absolute = residual_mmag.abs()
    return LightCurveComparison(
        rmse_mmag=float(torch.sqrt(torch.mean(residual_mmag.square())).detach().cpu()),
        bias_mmag=float(torch.mean(residual_mmag).detach().cpu()),
        p95_absolute_mmag=float(torch.quantile(absolute, 0.95).detach().cpu()),
        maximum_absolute_mmag=float(torch.max(absolute).detach().cpu()),
        valid_samples=count,
        constant_offset_removed=bool(remove_constant_offset),
    )
