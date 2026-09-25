"""Thin-disk emission evaluated through a precomputed observer transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import torch

from ..compile import run_tensor_kernel
from ..config import Backend
from ..relativity import ObserverTransfer, kerr_isco_radius
from ..runtime import ResolvedRuntime
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
from .triton_thermal_flux import triton_thermal_flux


def _planck_brightness(
    temperature4: torch.Tensor,
    gfactor: torch.Tensor,
    solid_angle_sr: torch.Tensor,
    hit: torch.Tensor,
    rest_wavelength_m: torch.Tensor,
    color: torch.Tensor,
    redshift_dimming: torch.Tensor,
    inverse_pixel_area: torch.Tensor,
) -> torch.Tensor:
    """Planck emission for explicitly broadcast spatial/spectral dimensions."""

    temperature = temperature4.clamp_min(0.0).pow(0.25)
    exponent = (
        _H
        * _C
        / (rest_wavelength_m * gfactor * _K_B * color * temperature.clamp_min(1.0e-12))
    )
    intensity_nu = (
        2.0
        * _H
        * _C
        / rest_wavelength_m.pow(3)
        / torch.expm1(exponent.clamp(max=85.0))
        / color.pow(4)
    )
    brightness = (
        intensity_nu * solid_angle_sr * redshift_dimming * 1.0e26 * inverse_pixel_area
    )
    return torch.where(hit, brightness, 0.0)


def _transferred_brightness_kernel(
    temperature4,
    gfactor,
    solid_angle_sr,
    hit,
    rest_wavelength_m,
    color,
    redshift_dimming,
    inverse_pixel_area,
):
    """Materialized brightness retains the public [time,y,x,band] layout."""
    return _planck_brightness(
        temperature4[..., None],
        gfactor[..., None],
        solid_angle_sr[..., None],
        hit[..., None],
        rest_wavelength_m,
        color,
        redshift_dimming,
        inverse_pixel_area,
    )


def _transferred_arguments(
    temperature4: torch.Tensor,
    *,
    geometry: SourceGeometry,
    transfer: ObserverTransfer,
    source_redshift: float | torch.Tensor,
    color_correction: float | torch.Tensor,
    runtime: ResolvedRuntime | None = None,
    spectral_cache: dict | None = None,
) -> tuple[tuple[torch.Tensor, ...], ResolvedRuntime | None]:
    """Resolve the shared tensor inputs for brightness and fused photometry."""

    temperature4 = torch.as_tensor(temperature4)
    device, dtype = temperature4.device, temperature4.dtype
    # The prepared thermal state already owns a validated transfer on this
    # device. Reconstructing it per wavelength chunk would revalidate every
    # observer pixel and synchronize CUDA several times for no change.
    local_transfer = (
        transfer
        if transfer.radius_rg.device == device and transfer.radius_rg.dtype == dtype
        else transfer.to(device=device, dtype=dtype)
    )
    cacheable = (
        spectral_cache is not None
        and not torch.is_grad_enabled()
        and not isinstance(source_redshift, torch.Tensor)
        and not isinstance(color_correction, torch.Tensor)
    )
    key = (device, dtype) if cacheable else None
    cached = spectral_cache.get(key) if cacheable else None
    if cached is None:
        redshift = torch.as_tensor(source_redshift, device=device, dtype=dtype)
        color = torch.as_tensor(color_correction, device=device, dtype=dtype)
        wavelengths = torch.as_tensor(
            geometry.wavelengths_angstrom, device=device, dtype=dtype
        )
        pixel_area = geometry.pixel_scale_m[0] * geometry.pixel_scale_m[1]
        spectral_inputs = (
            wavelengths * 1.0e-10 / (1.0 + redshift),
            color,
            (1.0 + redshift).pow(-3),
            torch.as_tensor(1.0 / pixel_area, device=device, dtype=dtype),
        )
        if cacheable:
            ready = None
            if device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(device))
            spectral_cache[key] = (spectral_inputs, ready)
    else:
        spectral_inputs, ready = cached
        if ready is not None:
            torch.cuda.current_stream(device).wait_event(ready)
    arguments = (
        temperature4,
        local_transfer.gfactor,
        local_transfer.solid_angle_sr,
        local_transfer.hit,
        *spectral_inputs,
    )
    compatible_runtime = (
        runtime
        if runtime is not None
        and runtime.device.type == device.type
        and (runtime.device.index is None or runtime.device.index == device.index)
        and runtime.dtype == dtype
        else None
    )
    return arguments, compatible_runtime


def _transferred_brightness_from_temperature4(temperature4, **kwargs):
    """Evaluate Planck brightness with the physical source's compiled runtime."""

    arguments, compatible_runtime = _transferred_arguments(temperature4, **kwargs)
    if compatible_runtime is None:
        return _transferred_brightness_kernel(*arguments)
    brightness, _ = run_tensor_kernel(
        compatible_runtime,
        "relativistic disk brightness",
        _transferred_brightness_kernel,
        *arguments,
    )
    return brightness


def _transferred_flux_kernel(left, right, fraction, *arguments):
    """Fuse Planck emission, map interpolation and both spatial reductions.

    Compiling the entire contraction avoids writing a [time,y,x,wavelength]
    brightness cube and a second equally large magnified cube to GPU memory.
    """

    temperature4, gfactor, solid_angle, hit, wavelength, color, dimming, area = (
        arguments
    )
    # Make pixels the innermost axis and reduce small spatial tiles first.
    # A single reduction of [time,y,x,band] makes Inductor materialize the
    # entire brightness cube before summing it. This layout lets each tile
    # evaluate Planck emission in registers and write only partial fluxes.
    pixels = gfactor.numel()
    tile = 64 if pixels % 64 == 0 else 1
    brightness = _planck_brightness(
        temperature4.reshape(temperature4.shape[0], 1, -1, tile),
        gfactor.reshape(1, 1, -1, tile),
        solid_angle.reshape(1, 1, -1, tile),
        hit.reshape(1, 1, -1, tile),
        wavelength[None, :, None, None],
        color,
        dimming,
        area,
    )
    left = left.expand(-1, *gfactor.shape).reshape(left.shape[0], 1, -1, tile)
    right = right.expand(-1, *gfactor.shape).reshape(right.shape[0], 1, -1, tile)
    magnification = left + fraction[:, None, None, None] * (right - left)
    return (brightness * magnification).sum(-1).sum(-1), brightness.sum(-1).sum(-1)


def _transferred_flux_from_temperature4(temperature4, left, right, fraction, **kwargs):
    """Integrate a thermal state without retaining its spatial brightness cube."""

    arguments, runtime = _transferred_arguments(temperature4, **kwargs)
    if runtime is not None and runtime.backend is Backend.TRITON:
        fast_result = triton_thermal_flux(left, right, fraction, arguments, runtime)
        if fast_result is not None:
            return fast_result
    if runtime is None:
        return _transferred_flux_kernel(left, right, fraction, *arguments)
    result, _ = run_tensor_kernel(
        runtime,
        "thermal spectral photometry",
        _transferred_flux_kernel,
        left,
        right,
        fraction,
        *arguments,
    )
    return result


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
    _brightness_cache: dict = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

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

    def _uncached_frame(self, *, device, dtype) -> torch.Tensor:
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

    def _frame(self, *, device, dtype) -> torch.Tensor:
        device = torch.device(device)
        key = (device.type, device.index, dtype)
        if not torch.is_grad_enabled():
            entry = self._brightness_cache.get(key)
            if entry is not None:
                frame, ready = entry
                if ready is not None:
                    torch.cuda.current_stream(device).wait_event(ready)
                return frame
        frame = self._uncached_frame(device=device, dtype=dtype)
        if not torch.is_grad_enabled():
            ready = None
            if device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(device))
            self._brightness_cache[key] = (frame, ready)
        return frame

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
