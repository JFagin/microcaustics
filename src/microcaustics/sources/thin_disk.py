"""Physical axisymmetric thin-disk source profiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..relativity import (
    approximate_circular_disk_gfactor,
    kerr_isco_radius,
    kerr_radiative_efficiency,
    page_thorne_flux_factor,
)
from .base import SourceGeometry, _as_times

_C = 299_792_458.0
_G = 6.67430e-11
_H = 6.62607015e-34
_K_B = 1.380649e-23
_M_PROTON = 1.67262192369e-27
_M_SUN = 1.988409870698051e30
_SIGMA_SB = 5.670374419e-8
_SIGMA_T = 6.6524587321e-29


def _scalar_tensor(value, *, name: str, device, dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.numel() != 1 or not bool(torch.isfinite(tensor)):
        raise ValueError(f"{name} must be one finite scalar")
    return tensor.reshape(())


def _thin_disk_temperature4_coefficient(
    black_hole_mass_solar,
    eddington_ratio,
    spin,
    *,
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the common Page--Thorne and lamp-heating temperature scale."""

    mass_solar = _scalar_tensor(
        black_hole_mass_solar,
        name="black_hole_mass_solar",
        device=device,
        dtype=dtype,
    )
    ratio = _scalar_tensor(
        eddington_ratio,
        name="eddington_ratio",
        device=device,
        dtype=dtype,
    )
    spin = _scalar_tensor(spin, name="spin", device=device, dtype=dtype)
    isco = kerr_isco_radius(spin)
    efficiency = kerr_radiative_efficiency(spin)
    gravitational_radius = (_G * _M_SUN / _C**2) * mass_solar
    eddington_rate = (
        4.0
        * math.pi
        * (_G * _M_SUN)
        * mass_solar
        * _M_PROTON
        / (_SIGMA_T * _C * efficiency)
    )
    accretion_rate = ratio * eddington_rate
    coefficient = (
        3.0
        / (8.0 * math.pi * _SIGMA_SB)
        * (_C**2 / gravitational_radius)
        * (accretion_rate / gravitational_radius)
    )
    return coefficient, isco


def thin_disk_temperature4(
    radius_rg: torch.Tensor,
    *,
    black_hole_mass_solar,
    eddington_ratio,
    spin,
    temperature_slope_beta=0.75,
    normalization_samples: int = 20_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return effective temperature to the fourth power and ISCO radius.

    The radial tilt ``r**(3 - 4 beta)`` is normalized to preserve the
    coordinate-area luminosity ``integral F(r) r dr``. The normalization uses
    the same logarithmic radial domain as the validated paper model and is
    therefore independent of an observer transfer's pixel sampling.
    """

    radius = torch.as_tensor(radius_rg)
    if not radius.is_floating_point():
        radius = radius.to(torch.get_default_dtype())
    device, dtype = radius.device, radius.dtype
    spin = _scalar_tensor(spin, name="spin", device=device, dtype=dtype)
    beta = _scalar_tensor(
        temperature_slope_beta,
        name="temperature_slope_beta",
        device=device,
        dtype=dtype,
    )
    if int(normalization_samples) < 16:
        raise ValueError("normalization_samples must be at least 16")
    coefficient, isco = _thin_disk_temperature4_coefficient(
        black_hole_mass_solar,
        eddington_ratio,
        spin,
        device=device,
        dtype=dtype,
    )
    temperature4 = coefficient * page_thorne_flux_factor(radius, spin, isco)
    radial_tilt = 3.0 - 4.0 * beta
    modifier = radius.clamp(1.0e-6, 1.0e6).pow(radial_tilt)
    physical = radius > isco

    normalization_radius = torch.logspace(
        -4.0,
        6.0,
        int(normalization_samples),
        device=device,
        dtype=dtype,
    )
    widths = torch.empty_like(normalization_radius)
    widths[1:-1] = 0.5 * (
        normalization_radius[2:] - normalization_radius[:-2]
    )
    widths[0] = normalization_radius[1] - normalization_radius[0]
    widths[-1] = normalization_radius[-1] - normalization_radius[-2]
    normalization_flux = page_thorne_flux_factor(
        normalization_radius,
        spin,
        isco,
    )
    normalization_physical = normalization_radius >= isco
    area_weight = normalization_radius * widths
    baseline_power = torch.where(
        normalization_physical,
        normalization_flux * area_weight,
        0.0,
    ).sum()
    modified_power = torch.where(
        normalization_physical,
        normalization_flux
        * normalization_radius.pow(radial_tilt)
        * area_weight,
        0.0,
    ).sum()
    normalization = baseline_power / modified_power.clamp_min(
        torch.finfo(dtype).tiny
    )
    value = torch.where(
        physical,
        temperature4 * modifier * normalization,
        torch.zeros_like(temperature4),
    )
    return value, isco


def thin_disk_flux_radius_rg(
    *,
    black_hole_mass_solar,
    eddington_ratio,
    spin,
    observed_wavelength_angstrom: float,
    source_redshift: float,
    temperature_slope_beta: float = 0.75,
    color_correction: float = 1.0,
    lamp_fraction: float = 0.1,
    corona_height_above_isco_rg: float = 20.0,
    flux_fraction: float = 0.999,
    safety_factor: float = 1.05,
    radial_samples: int = 20_000,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> float:
    """Return a conservative monochromatic outer disk radius in ``r_g``.

    The radius encloses ``flux_fraction`` of the face-on monochromatic flux at
    the supplied observed wavelength and then applies ``safety_factor``. The
    calculation uses the same Page--Thorne dissipation, optional axis-lamp
    heating, logarithmic radial domain, and annular integration convention as
    the validated production disk model. The reddest requested band is usually
    the appropriate input when one common field must contain every band.

    ``corona_height_above_isco_rg`` follows the public lamppost convention. The
    Boyer--Lindquist source height used by the heating profile is this value
    plus the Kerr ISCO radius. Set ``lamp_fraction=0`` for a stationary
    Novikov--Thorne disk without external heating.
    """

    if not 0.0 < float(flux_fraction) < 1.0:
        raise ValueError("flux_fraction must lie strictly between zero and one")
    if float(safety_factor) < 1.0:
        raise ValueError("safety_factor must be at least one")
    if int(radial_samples) < 128:
        raise ValueError("radial_samples must be at least 128")
    if float(observed_wavelength_angstrom) <= 0.0:
        raise ValueError("observed_wavelength_angstrom must be positive")
    if float(source_redshift) < 0.0:
        raise ValueError("source_redshift must be non-negative")
    if float(color_correction) <= 0.0:
        raise ValueError("color_correction must be positive")
    if float(lamp_fraction) < 0.0:
        raise ValueError("lamp_fraction must be non-negative")
    if float(corona_height_above_isco_rg) < 0.0:
        raise ValueError("corona_height_above_isco_rg must be non-negative")

    radius = torch.logspace(
        -4.0,
        6.0,
        int(radial_samples),
        device=device,
        dtype=dtype,
    )
    viscous_temperature4, isco = thin_disk_temperature4(
        radius,
        black_hole_mass_solar=black_hole_mass_solar,
        eddington_ratio=eddington_ratio,
        spin=spin,
        temperature_slope_beta=temperature_slope_beta,
        normalization_samples=radial_samples,
    )
    coefficient, _ = _thin_disk_temperature4_coefficient(
        black_hole_mass_solar,
        eddington_ratio,
        spin,
        device=device,
        dtype=dtype,
    )
    ratio = _scalar_tensor(
        eddington_ratio,
        name="eddington_ratio",
        device=device,
        dtype=dtype,
    )
    efficiency = kerr_radiative_efficiency(spin)
    height = isco + float(corona_height_above_isco_rg)
    lamp_temperature4 = (
        float(lamp_fraction)
        * efficiency
        / ratio
        * coefficient
        * (4.0 / 3.0)
        * height
        * (radius.square() + height.square()).pow(-1.5)
    )
    lamp_temperature4 = torch.where(
        radius >= isco,
        lamp_temperature4,
        torch.zeros_like(lamp_temperature4),
    )
    temperature = (
        viscous_temperature4 + lamp_temperature4
    ).clamp_min(torch.finfo(dtype).tiny).pow(0.25)
    rest_wavelength_m = (
        float(observed_wavelength_angstrom)
        * 1.0e-10
        / (1.0 + float(source_redshift))
    )
    hardening = float(color_correction)
    exponent = (
        _H
        * _C
        / (rest_wavelength_m * _K_B * hardening * temperature)
    ).clamp(max=85.0)
    intensity = torch.reciprocal(torch.expm1(exponent)) / hardening**4
    widths = torch.empty_like(radius)
    widths[1:-1] = 0.5 * (radius[2:] - radius[:-2])
    widths[0] = radius[1] - radius[0]
    widths[-1] = radius[-1] - radius[-2]
    annular_flux = torch.where(
        radius >= isco,
        intensity * radius * widths,
        torch.zeros_like(radius),
    )
    cumulative = torch.cumsum(annular_flux, dim=0)
    cumulative = cumulative / cumulative[-1].clamp_min(torch.finfo(dtype).tiny)
    target = torch.as_tensor(
        float(flux_fraction), device=device, dtype=dtype
    )
    index = torch.searchsorted(cumulative, target).clamp(max=radius.numel() - 1)
    return float((float(safety_factor) * radius[index]).detach().cpu())


@dataclass(frozen=True)
class ThinDiskSource:
    """A static Novikov--Thorne/Page--Thorne continuum disk.

    Pixels describe the projected source plane. The disk radius is deprojected
    using ``inclination_deg`` and rotated by ``position_angle_deg``. Returned
    brightness is observed specific flux density per square meter of projected
    source plane in ``Jy m^-2``. The finite-source photometry integrator
    multiplies by each physical pixel area and therefore returns Jy.

    ``relativity='none'`` uses no photon frequency shift while retaining the
    relativistic Page--Thorne radial dissipation profile. ``'approximate'``
    adds the legacy straight-screen circular-orbit shift. Full light bending
    is represented by a separate Kerr observer-transfer source and is not
    silently approximated by this class.
    """

    geometry: SourceGeometry
    black_hole_mass_solar: float | torch.Tensor
    eddington_ratio: float | torch.Tensor
    spin: float | torch.Tensor = 0.0
    inclination_deg: float | torch.Tensor = 30.0
    position_angle_deg: float | torch.Tensor = 0.0
    source_redshift: float | torch.Tensor = 1.0
    luminosity_distance_m: float | torch.Tensor = 1.0e26
    color_correction: float | torch.Tensor = 1.0
    temperature_slope_beta: float | torch.Tensor = 0.75
    outer_radius_m: float | torch.Tensor | None = None
    relativity: str = "none"
    name: str = "thin_disk"
    is_time_static: bool = True

    def __post_init__(self) -> None:
        if self.relativity not in {"none", "approximate"}:
            raise ValueError("relativity must be 'none' or 'approximate'")
        checks = {
            "black_hole_mass_solar": (self.black_hole_mass_solar, 0.0, None),
            "eddington_ratio": (self.eddington_ratio, 0.0, None),
            "spin": (self.spin, -0.998, 0.998),
            "inclination_deg": (self.inclination_deg, 0.0, 90.0),
            "source_redshift": (self.source_redshift, 0.0, None),
            "luminosity_distance_m": (self.luminosity_distance_m, 0.0, None),
            "color_correction": (self.color_correction, 0.0, None),
        }
        for name, (value, lower, upper) in checks.items():
            tensor = torch.as_tensor(value)
            if tensor.numel() != 1 or not bool(torch.isfinite(tensor)):
                raise ValueError(f"{name} must be one finite scalar")
            scalar = float(tensor.detach().cpu())
            if name == "spin":
                valid = lower <= scalar <= upper
            elif name == "source_redshift":
                valid = scalar >= lower
            elif name == "inclination_deg":
                valid = lower <= scalar < upper
            else:
                valid = scalar > lower
            if not valid:
                raise ValueError(f"{name} lies outside its physical range")
        if self.outer_radius_m is not None and float(self.outer_radius_m) <= 0:
            raise ValueError("outer_radius_m must be positive")

    @classmethod
    def from_lensing_distances(
        cls,
        geometry: SourceGeometry,
        black_hole_mass_solar,
        eddington_ratio,
        distances,
        *,
        source_redshift,
        **kwargs,
    ) -> ThinDiskSource:
        """Construct a disk using Etherington distance duality.

        ``LensingDistances.source_m`` is an angular-diameter distance. The
        luminosity distance is ``(1 + z)^2 D_A``. The method accepts the
        lightweight distance protocol by attribute, avoiding a hard dependency
        on a particular cosmology package.
        """

        redshift = float(torch.as_tensor(source_redshift).detach().cpu())
        luminosity_distance = (1.0 + redshift) ** 2 * float(distances.source_m)
        return cls(
            geometry,
            black_hole_mass_solar,
            eddington_ratio,
            source_redshift=source_redshift,
            luminosity_distance_m=luminosity_distance,
            **kwargs,
        )

    def _frame(self, *, device, dtype) -> torch.Tensor:
        mass_solar = _scalar_tensor(
            self.black_hole_mass_solar,
            name="black_hole_mass_solar",
            device=device,
            dtype=dtype,
        )
        eddington_ratio = _scalar_tensor(
            self.eddington_ratio,
            name="eddington_ratio",
            device=device,
            dtype=dtype,
        )
        spin = _scalar_tensor(self.spin, name="spin", device=device, dtype=dtype)
        inclination = torch.deg2rad(
            _scalar_tensor(
                self.inclination_deg,
                name="inclination_deg",
                device=device,
                dtype=dtype,
            )
        )
        position_angle = torch.deg2rad(
            _scalar_tensor(
                self.position_angle_deg,
                name="position_angle_deg",
                device=device,
                dtype=dtype,
            )
        )
        redshift = _scalar_tensor(
            self.source_redshift,
            name="source_redshift",
            device=device,
            dtype=dtype,
        )
        luminosity_distance = _scalar_tensor(
            self.luminosity_distance_m,
            name="luminosity_distance_m",
            device=device,
            dtype=dtype,
        )
        color_correction = _scalar_tensor(
            self.color_correction,
            name="color_correction",
            device=device,
            dtype=dtype,
        )
        beta = _scalar_tensor(
            self.temperature_slope_beta,
            name="temperature_slope_beta",
            device=device,
            dtype=dtype,
        )

        ny, nx = self.geometry.shape
        dy, dx = self.geometry.pixel_scale_m
        x = (torch.arange(nx, device=device, dtype=dtype) + 0.5 - 0.5 * nx) * dx
        y = (torch.arange(ny, device=device, dtype=dtype) + 0.5 - 0.5 * ny) * dy
        projected_y, projected_x = torch.meshgrid(y, x, indexing="ij")
        cosine = torch.cos(position_angle)
        sine = torch.sin(position_angle)
        disk_x = cosine * projected_x + sine * projected_y
        disk_y = -sine * projected_x + cosine * projected_y
        disk_x = disk_x / torch.cos(inclination).clamp_min(
            torch.finfo(dtype).tiny
        )
        radius_m = torch.sqrt(disk_x.square() + disk_y.square())
        azimuth = torch.atan2(disk_y, disk_x)

        # Multiply physical constants before the solar-mass parameter. A
        # billion-solar-mass black hole exceeds float32 if materialized in kg,
        # although GM, r_g, and every required observable are representable.
        gravitational_radius = (_G * _M_SUN / _C**2) * mass_solar
        radius_rg = radius_m / gravitational_radius
        temperature4, isco_rg = thin_disk_temperature4(
            radius_rg,
            black_hole_mass_solar=mass_solar,
            eddington_ratio=eddington_ratio,
            spin=spin,
            temperature_slope_beta=beta,
        )
        temperature = temperature4.clamp_min(0.0).pow(0.25)

        if self.outer_radius_m is None:
            outer_radius = 0.5 * min(ny * dy, nx * dx)
        else:
            outer_radius = self.outer_radius_m
        outer_radius = _scalar_tensor(
            outer_radius,
            name="outer_radius_m",
            device=device,
            dtype=dtype,
        )
        masked = (radius_rg <= isco_rg) | (radius_m > outer_radius)
        if self.relativity == "none":
            gfactor = torch.ones_like(radius_rg)
        else:
            gfactor = approximate_circular_disk_gfactor(
                radius_rg,
                azimuth,
                inclination,
                spin,
            )
        wavelengths = torch.as_tensor(
            self.geometry.wavelengths_angstrom,
            device=device,
            dtype=dtype,
        )
        rest_wavelength_m = wavelengths * 1.0e-10 / (1.0 + redshift)
        exponent = _H * _C / (
            rest_wavelength_m[None, None, :]
            * gfactor[..., None]
            * _K_B
            * color_correction
            * temperature[..., None].clamp_min(1.0e-12)
        )
        exponent = exponent.clamp(max=85.0)
        intensity_nu = (
            2.0
            * _H
            * _C
            / rest_wavelength_m.pow(3)[None, None, :]
            / torch.expm1(exponent)
            / color_correction.pow(4)
        )
        # Evaluate the distance dilution in scaled units. Squaring a typical
        # luminosity distance (~1e26 m) overflows float32 before the final Jy
        # conversion, even though the physical answer is representable.
        scaled_distance = luminosity_distance / 1.0e20
        brightness = (
            intensity_nu
            * (1.0 + redshift)
            / scaled_distance.square()
            * 1.0e-14
        )
        return torch.where(masked[..., None], 0.0, brightness)

    def brightness(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the static observed disk brightness for requested times."""

        times = _as_times(times_days)
        device = times.device if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        frame = self._frame(device=device, dtype=dtype)
        return frame.unsqueeze(0).expand(times.numel(), -1, -1, -1)

    def metadata(self) -> Mapping[str, object]:
        """Return physical disk parameters and approximation provenance."""

        def scalar(value):
            return float(torch.as_tensor(value).detach().cpu())

        return {
            "type": "thin_disk",
            "name": self.name,
            "black_hole_mass_solar": scalar(self.black_hole_mass_solar),
            "eddington_ratio": scalar(self.eddington_ratio),
            "spin": scalar(self.spin),
            "inclination_deg": scalar(self.inclination_deg),
            "position_angle_deg": scalar(self.position_angle_deg),
            "source_redshift": scalar(self.source_redshift),
            "luminosity_distance_m": scalar(self.luminosity_distance_m),
            "color_correction": scalar(self.color_correction),
            "temperature_slope_beta": scalar(self.temperature_slope_beta),
            "relativity": self.relativity,
            "radial_flux_profile": "Page-Thorne",
            "brightness_units": "Jy m^-2 projected source plane",
            "integrated_flux_units": "Jy",
            "is_time_static": True,
        }
