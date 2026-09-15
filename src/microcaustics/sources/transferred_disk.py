"""Thin-disk emission evaluated through a precomputed observer transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import torch

from ..relativity import ObserverTransfer, kerr_isco_radius
from .base import SourceGeometry, _as_times
from .thin_disk import (
    _C,
    _H,
    _K_B,
    _NOVIKOV_THORNE,
    RadiativeEfficiency,
    ViscousFluxProfile,
    _prescription_metadata,
    _prescription_name,
    _radiative_efficiency,
    _validate_viscous_prescriptions,
    thin_disk_temperature4,
)


def _transferred_brightness_from_temperature4(
    temperature4: torch.Tensor,
    *,
    geometry: SourceGeometry,
    transfer: ObserverTransfer,
    source_redshift: float | torch.Tensor,
    color_correction: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate redshifted Planck brightness through an observer transfer."""

    temperature4 = torch.as_tensor(temperature4)
    device, dtype = temperature4.device, temperature4.dtype
    local_transfer = transfer.to(device=device, dtype=dtype)
    temperature = temperature4.clamp_min(0.0).pow(0.25)
    redshift = torch.as_tensor(source_redshift, device=device, dtype=dtype)
    color = torch.as_tensor(color_correction, device=device, dtype=dtype)
    wavelengths = torch.as_tensor(
        geometry.wavelengths_angstrom,
        device=device,
        dtype=dtype,
    )
    rest_wavelength_m = wavelengths * 1.0e-10 / (1.0 + redshift)
    exponent = (
        _H
        * _C
        / (
            rest_wavelength_m
            * local_transfer.gfactor[..., None]
            * _K_B
            * color
            * temperature[..., None].clamp_min(1.0e-12)
        )
    )
    intensity_nu = (
        2.0
        * _H
        * _C
        / rest_wavelength_m.pow(3)
        / torch.expm1(exponent.clamp(max=85.0))
        / color.pow(4)
    )
    flux_per_pixel_jy = (
        intensity_nu
        * local_transfer.solid_angle_sr[..., None]
        / (1.0 + redshift).pow(3)
        * 1.0e26
    )
    pixel_area = geometry.pixel_scale_m[0] * geometry.pixel_scale_m[1]
    brightness = flux_per_pixel_jy / pixel_area
    return torch.where(local_transfer.hit[..., None], brightness, 0.0)


@dataclass(frozen=True)
class TransferredThinDiskSource:
    """A physical thin disk seen through a supplied observer transfer.

    The transfer may come from the package's analytic Kerr tracer, SIM5, or a
    user backend, as long as it obeys the one-pixel observer-transfer contract.
    Brightness uses ``Jy m^-2`` of projected source grid, matching
    :class:`ThinDiskSource` and the finite-source photometry interface.
    """

    geometry: SourceGeometry
    transfer: ObserverTransfer
    black_hole_mass_solar: float | torch.Tensor
    eddington_ratio: float | torch.Tensor
    spin: float | torch.Tensor
    source_redshift: float | torch.Tensor
    color_correction: float | torch.Tensor = 1.0
    temperature_slope_beta: float | torch.Tensor = 0.75
    viscous_flux_profile: ViscousFluxProfile = _NOVIKOV_THORNE
    radiative_efficiency: RadiativeEfficiency = None
    name: str = "transferred_thin_disk"
    is_time_static: bool = True

    def __post_init__(self) -> None:
        _validate_viscous_prescriptions(
            self.viscous_flux_profile,
            self.radiative_efficiency,
        )
        if self.transfer.shape != self.geometry.shape:
            raise ValueError("observer transfer and source geometry shapes must match")
        for name in (
            "black_hole_mass_solar",
            "eddington_ratio",
            "spin",
            "source_redshift",
            "color_correction",
            "temperature_slope_beta",
        ):
            value = torch.as_tensor(getattr(self, name))
            if value.numel() != 1 or not bool(torch.isfinite(value)):
                raise ValueError(f"{name} must be one finite scalar")
        scalar = {
            name: float(torch.as_tensor(getattr(self, name)).detach().cpu())
            for name in (
                "black_hole_mass_solar",
                "eddington_ratio",
                "spin",
                "source_redshift",
                "color_correction",
                "temperature_slope_beta",
            )
        }
        if scalar["black_hole_mass_solar"] <= 0:
            raise ValueError("black_hole_mass_solar must be positive")
        if scalar["eddington_ratio"] <= 0:
            raise ValueError("eddington_ratio must be positive")
        if not -0.998 <= scalar["spin"] <= 0.998:
            raise ValueError("spin must lie in [-0.998, 0.998]")
        if scalar["source_redshift"] < 0:
            raise ValueError("source_redshift must be non-negative")
        if scalar["color_correction"] <= 0:
            raise ValueError("color_correction must be positive")
        if scalar["temperature_slope_beta"] <= 0:
            raise ValueError("temperature_slope_beta must be positive")

    def _frame(self, *, device, dtype) -> torch.Tensor:
        transfer = self.transfer.to(device=device, dtype=dtype)
        temperature4, _ = thin_disk_temperature4(
            transfer.radius_rg,
            black_hole_mass_solar=self.black_hole_mass_solar,
            eddington_ratio=self.eddington_ratio,
            spin=self.spin,
            temperature_slope_beta=self.temperature_slope_beta,
            viscous_flux_profile=self.viscous_flux_profile,
            radiative_efficiency=self.radiative_efficiency,
        )
        return _transferred_brightness_from_temperature4(
            temperature4,
            geometry=self.geometry,
            transfer=transfer,
            source_redshift=self.source_redshift,
            color_correction=self.color_correction,
        )

    def support_radius_m(self, distances=None) -> float:
        """Return the circular disk support represented by the observer grid.

        ``distances`` is accepted for compatibility with physical source
        models.  The pixel geometry is already stored in metres, so no new
        cosmological conversion is required.
        """

        height_m = self.geometry.shape[0] * self.geometry.pixel_scale_m[0]
        width_m = self.geometry.shape[1] * self.geometry.pixel_scale_m[1]
        return 0.5 * min(float(height_m), float(width_m))

    def with_bands(
        self, bands_angstrom: Mapping[str, float]
    ) -> TransferredThinDiskSource:
        """Reuse the achromatic observer transfer at new observed wavelengths."""

        return replace(self, geometry=self.geometry.with_bands(bands_angstrom))

    def brightness(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return transferred static brightness for every requested time."""

        times = _as_times(times_days)
        device = times.device if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        frame = self._frame(device=device, dtype=dtype)
        return frame.unsqueeze(0).expand(times.numel(), -1, -1, -1)

    def metadata(self) -> Mapping[str, object]:
        """Return disk and observer-transfer provenance."""

        def scalar(value):
            return float(torch.as_tensor(value).detach().cpu())

        return {
            "type": "transferred_thin_disk",
            "name": self.name,
            "black_hole_mass_solar": scalar(self.black_hole_mass_solar),
            "eddington_ratio": scalar(self.eddington_ratio),
            "spin": scalar(self.spin),
            "source_redshift": scalar(self.source_redshift),
            "color_correction": scalar(self.color_correction),
            "temperature_slope_beta": scalar(self.temperature_slope_beta),
            "viscous_flux_profile": _prescription_name(self.viscous_flux_profile),
            "viscous_flux_profile_metadata": _prescription_metadata(
                self.viscous_flux_profile
            ),
            "radiative_efficiency": float(
                _radiative_efficiency(
                    self.radiative_efficiency,
                    self.viscous_flux_profile,
                    torch.as_tensor(self.spin, dtype=torch.float64),
                    kerr_isco_radius(torch.as_tensor(self.spin, dtype=torch.float64)),
                )
            ),
            "radiative_efficiency_prescription": _prescription_name(
                self.radiative_efficiency
            ),
            "brightness_units": "Jy m^-2 projected source plane",
            "integrated_flux_units": "Jy",
            "observer_transfer": dict(self.transfer.metadata),
            "is_time_static": True,
        }
