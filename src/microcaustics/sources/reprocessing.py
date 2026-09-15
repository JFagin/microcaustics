"""Thermal reverberation through a precomputed observer transfer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import torch

from ..relativity import ObserverTransfer, kerr_isco_radius
from ..relativity.lamppost import AxisLamppostProfile
from .base import SourceGeometry, _as_times
from .thin_disk import (
    _C,
    _G,
    _H,
    _K_B,
    _M_SUN,
    _NOVIKOV_THORNE,
    RadiativeEfficiency,
    ViscousFluxProfile,
    _prescription_metadata,
    _prescription_name,
    _radiative_efficiency,
    _thin_disk_temperature4_coefficient,
    _validate_viscous_prescriptions,
    thin_disk_temperature4,
)
from .transferred_disk import _transferred_brightness_from_temperature4
from .variability import DrivingSignal


def lamppost_irradiation_efficiency(
    lamp_fraction: float,
    eddington_ratio: float,
    spin,
    *,
    viscous_flux_profile: ViscousFluxProfile = _NOVIKOV_THORNE,
    radiative_efficiency: RadiativeEfficiency = None,
) -> torch.Tensor:
    """Return the dimensionless lamppost heating normalization.

    ``lamp_fraction`` is the fraction of the bolometric accretion luminosity
    assigned to the lamppost. Dividing by the Eddington ratio converts that
    fraction to the temperature-scale normalization used by
    :meth:`ThermalReprocessingSource.from_axis_lamppost`. The radiative
    efficiency follows the selected viscous profile unless it is overridden.
    """

    if not math.isfinite(float(lamp_fraction)) or float(lamp_fraction) < 0.0:
        raise ValueError("lamp_fraction must be finite and non-negative")
    if not math.isfinite(float(eddington_ratio)) or float(eddington_ratio) <= 0.0:
        raise ValueError("eddington_ratio must be finite and positive")
    spin_tensor = torch.as_tensor(spin)
    if not spin_tensor.is_floating_point():
        spin_tensor = spin_tensor.to(torch.get_default_dtype())
    isco = kerr_isco_radius(spin_tensor)
    efficiency = _radiative_efficiency(
        radiative_efficiency,
        viscous_flux_profile,
        spin_tensor,
        isco,
    )
    return float(lamp_fraction) * efficiency / float(eddington_ratio)


@dataclass(frozen=True)
class ThermalReprocessingSource:
    """A thin disk with additive, delayed heating in ``T^4``.

    ``response_temperature4`` is the local additive temperature-to-the-fourth
    map for unit driving amplitude. ``delay_days`` includes every delay the
    chosen physical model requires. This separation lets an analytic Kerr
    lamppost, a Newtonian prescription, or a user-provided illumination model
    share the same differentiable source and transfer-function calculation.
    """

    geometry: SourceGeometry
    transfer: ObserverTransfer
    signal: DrivingSignal
    response_temperature4: torch.Tensor
    delay_days: torch.Tensor
    black_hole_mass_solar: float | torch.Tensor
    eddington_ratio: float | torch.Tensor
    spin: float | torch.Tensor
    source_redshift: float | torch.Tensor
    color_correction: float | torch.Tensor = 1.0
    temperature_slope_beta: float | torch.Tensor = 0.75
    viscous_flux_profile: ViscousFluxProfile = _NOVIKOV_THORNE
    radiative_efficiency: RadiativeEfficiency = None
    name: str = "thermal_reprocessing"
    heating_metadata: Mapping[str, object] | None = None
    is_time_static: bool = False
    _linear_response_cache: dict = field(
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
        response = torch.as_tensor(self.response_temperature4)
        delay = torch.as_tensor(self.delay_days, device=response.device)
        if response.shape != self.geometry.shape or not response.is_floating_point():
            raise ValueError("response_temperature4 must be one floating [y, x] map")
        if delay.shape != self.geometry.shape or not delay.is_floating_point():
            raise ValueError("delay_days must be one floating [y, x] map")
        hit = self.transfer.hit.to(response.device)
        if bool(torch.any(~torch.isfinite(response[hit]))) or bool(
            torch.any(response[hit] < 0.0)
        ):
            raise ValueError(
                "response_temperature4 must be finite and non-negative on hits"
            )
        if bool(torch.any(~torch.isfinite(delay[hit]))):
            raise ValueError("delay_days must be finite on transfer hits")
        scalar_names = (
            "black_hole_mass_solar",
            "eddington_ratio",
            "spin",
            "source_redshift",
            "color_correction",
            "temperature_slope_beta",
        )
        scalars = {}
        for field_name in scalar_names:
            value = torch.as_tensor(getattr(self, field_name))
            if value.numel() != 1 or not bool(torch.isfinite(value)):
                raise ValueError(f"{field_name} must be one finite scalar")
            scalars[field_name] = float(value.detach().cpu())
        if scalars["black_hole_mass_solar"] <= 0.0:
            raise ValueError("black_hole_mass_solar must be positive")
        if scalars["eddington_ratio"] <= 0.0:
            raise ValueError("eddington_ratio must be positive")
        if not -0.998 <= scalars["spin"] <= 0.998:
            raise ValueError("spin must lie in [-0.998, 0.998]")
        if scalars["source_redshift"] < 0.0:
            raise ValueError("source_redshift must be non-negative")
        if scalars["color_correction"] <= 0.0:
            raise ValueError("color_correction must be positive")
        if scalars["temperature_slope_beta"] <= 0.0:
            raise ValueError("temperature_slope_beta must be positive")
        object.__setattr__(self, "response_temperature4", response)
        object.__setattr__(self, "delay_days", delay)

    def support_radius_m(self, distances=None) -> float:
        """Return the circular disk support represented by the observer grid."""

        height_m = self.geometry.shape[0] * self.geometry.pixel_scale_m[0]
        width_m = self.geometry.shape[1] * self.geometry.pixel_scale_m[1]
        return 0.5 * min(float(height_m), float(width_m))

    @classmethod
    def from_axis_lamppost(
        cls,
        geometry: SourceGeometry,
        transfer: ObserverTransfer,
        signal: DrivingSignal,
        profile: AxisLamppostProfile,
        *,
        black_hole_mass_solar,
        eddington_ratio,
        spin,
        source_redshift,
        irradiation_efficiency: float | None = None,
        lamp_fraction: float | None = None,
        viscous_flux_profile: ViscousFluxProfile = _NOVIKOV_THORNE,
        radiative_efficiency: RadiativeEfficiency = None,
        **kwargs,
    ) -> ThermalReprocessingSource:
        """Construct a disk from a validated axial Kerr lamppost profile.

        ``irradiation_efficiency`` multiplies the common accretion temperature
        scale and the profile's conservative proper-area illumination. As a
        physical convenience, ``lamp_fraction`` computes that coefficient
        from the selected disk radiative efficiency and ``eddington_ratio``.
        Supply at most one of the two. If neither is supplied, the coefficient
        is one. No hidden rescaling to a requested variability amplitude is
        performed.
        """

        if transfer.relative_delay_days is None:
            raise ValueError("observer transfer must contain a relative delay map")
        if irradiation_efficiency is not None and lamp_fraction is not None:
            raise ValueError("supply irradiation_efficiency or lamp_fraction, not both")
        if lamp_fraction is not None:
            efficiency = lamppost_irradiation_efficiency(
                lamp_fraction,
                float(torch.as_tensor(eddington_ratio).detach().cpu()),
                spin,
                viscous_flux_profile=viscous_flux_profile,
                radiative_efficiency=radiative_efficiency,
            )
        else:
            efficiency = (
                1.0 if irradiation_efficiency is None else irradiation_efficiency
            )
        efficiency_tensor = torch.as_tensor(
            efficiency,
            device=transfer.radius_rg.device,
            dtype=transfer.radius_rg.dtype,
        )
        if (
            efficiency_tensor.numel() != 1
            or not bool(torch.isfinite(efficiency_tensor))
            or bool(efficiency_tensor < 0)
        ):
            raise ValueError("irradiation_efficiency must be finite and non-negative")
        if not math.isclose(
            float(profile.rays.spin),
            float(spin),
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise ValueError("lamppost profile spin must match the disk spin")
        transfer_redshift = transfer.metadata.get("source_redshift")
        if transfer_redshift is not None and not math.isclose(
            float(transfer_redshift),
            float(source_redshift),
            rel_tol=0.0,
            abs_tol=1.0e-7,
        ):
            raise ValueError("observer-coordinate and disk source redshifts must match")
        radius = transfer.radius_rg
        lamp_delay_rg, illumination, _ = profile.interpolate(radius)
        coefficient, isco = _thin_disk_temperature4_coefficient(
            black_hole_mass_solar,
            eddington_ratio,
            spin,
            viscous_flux_profile=viscous_flux_profile,
            radiative_efficiency=radiative_efficiency,
            device=radius.device,
            dtype=radius.dtype,
        )
        response_temperature4 = efficiency_tensor * coefficient * illumination
        mass = torch.as_tensor(
            black_hole_mass_solar,
            device=radius.device,
            dtype=radius.dtype,
        )
        redshift = torch.as_tensor(
            source_redshift,
            device=radius.device,
            dtype=radius.dtype,
        )
        lamp_delay_days = (
            lamp_delay_rg * (_G * _M_SUN / _C**3) * mass / 86_400.0 * (1.0 + redshift)
        )
        observer_delay = transfer.relative_delay_days.to(
            device=radius.device,
            dtype=radius.dtype,
        )
        total_delay = observer_delay + lamp_delay_days
        # Define zero using the physical disk, not rays landing inside the
        # ISCO.  The latter carry no thin-disk emission or response but may
        # have formally finite coordinate times. Allowing one to set the zero
        # adds a large, resolution-dependent constant to every transfer
        # function.  This convention matches the first physically responsive
        # disk element and leaves all inter-band delay differences unchanged.
        delay_support = (
            transfer.hit
            & torch.isfinite(total_delay)
            & torch.isfinite(response_temperature4)
            & (radius >= isco)
            & (response_temperature4 > 0.0)
        )
        if not bool(torch.any(delay_support)):
            raise RuntimeError("lamppost profile has no responsive disk pixels")
        minimum = total_delay[delay_support].min()
        total_delay = torch.where(
            transfer.hit,
            total_delay - minimum,
            torch.full_like(total_delay, float("nan")),
        )
        return cls(
            geometry,
            transfer,
            signal,
            response_temperature4,
            total_delay,
            black_hole_mass_solar,
            eddington_ratio,
            spin,
            source_redshift,
            viscous_flux_profile=viscous_flux_profile,
            radiative_efficiency=radiative_efficiency,
            heating_metadata={
                "model": "axis_kerr_lamppost",
                "source_height_rg": profile.rays.source_height_rg,
                "lamp_g_power": profile.lamp_g_power,
                "irradiation_efficiency": float(efficiency_tensor.detach().cpu()),
                "lamp_fraction": (
                    None if lamp_fraction is None else float(lamp_fraction)
                ),
                "lamppost_hit_fraction": float(profile.hit_fraction.detach().cpu()),
            },
            **kwargs,
        )

    def _static_temperature4(self, *, device, dtype) -> torch.Tensor:
        temperature4, _ = thin_disk_temperature4(
            self.transfer.radius_rg.to(device=device, dtype=dtype),
            black_hole_mass_solar=self.black_hole_mass_solar,
            eddington_ratio=self.eddington_ratio,
            spin=self.spin,
            temperature_slope_beta=self.temperature_slope_beta,
            viscous_flux_profile=self.viscous_flux_profile,
            radiative_efficiency=self.radiative_efficiency,
        )
        return temperature4

    def with_bands(
        self, bands_angstrom: Mapping[str, float]
    ) -> ThermalReprocessingSource:
        """Reuse the disk and observer transfer at new observed wavelengths."""

        return replace(self, geometry=self.geometry.with_bands(bands_angstrom))

    def at_driver_mean(self) -> ThermalReprocessingSource:
        """Return the same heated disk with stochastic driver fluctuations disabled."""

        from .variability import _source_at_driver_mean

        return _source_at_driver_mean(self)

    def brightness(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Evaluate the nonlinear thermal response at retarded source times."""

        times = _as_times(times_days)
        device = times.device if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        times = times.to(device=device, dtype=dtype)
        output_count = times.numel()
        evaluation_times = times[:1] if self.is_time_static else times
        transfer = self.transfer.to(device=device, dtype=dtype)
        delay = self.delay_days.to(device=device, dtype=dtype)
        safe_delay = torch.where(transfer.hit, delay, torch.zeros_like(delay))
        query = evaluation_times[:, None, None] - safe_delay[None]
        driving = self.signal.amplitudes(
            query.reshape(-1),
            bands=1,
            dtype=dtype,
            device=device,
        ).reshape(evaluation_times.numel(), *self.geometry.shape)
        static_temperature4 = self._static_temperature4(
            device=device,
            dtype=dtype,
        )
        response_temperature4 = self.response_temperature4.to(
            device=device,
            dtype=dtype,
        )
        temperature4 = static_temperature4[None] + response_temperature4[None] * driving
        brightness = _transferred_brightness_from_temperature4(
            temperature4,
            geometry=self.geometry,
            transfer=transfer,
            source_redshift=self.source_redshift,
            color_correction=self.color_correction,
        )
        if self.is_time_static and output_count > 1:
            brightness = brightness.expand(output_count, *brightness.shape[1:])
        return brightness

    def linear_response_weights(
        self,
        *,
        driver_amplitude: float = 1.0,
        normalize: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return ``d(brightness)/d(driver)`` for every pixel and band."""

        if not math.isfinite(driver_amplitude) or driver_amplitude < 0.0:
            raise ValueError("driver_amplitude must be finite and non-negative")
        device = self.transfer.radius_rg.device if device is None else device
        dtype = self.transfer.radius_rg.dtype if dtype is None else dtype
        device = torch.device(device)
        if normalize:
            weights = self.linear_response_weights(
                driver_amplitude=driver_amplitude,
                normalize=False,
                dtype=dtype,
                device=device,
            )
            return weights / weights.sum(dim=(0, 1), keepdim=True).clamp_min(1.0e-30)
        cache_key = (float(driver_amplitude), device, dtype)
        if not torch.is_grad_enabled():
            entry = self._linear_response_cache.get(cache_key)
            if entry is not None:
                cached, ready = entry
                if ready is not None:
                    torch.cuda.current_stream(device).wait_event(ready)
                return cached
        transfer = self.transfer.to(device=device, dtype=dtype)
        static = self._static_temperature4(device=device, dtype=dtype)
        response = self.response_temperature4.to(device=device, dtype=dtype)
        total4 = (static + float(driver_amplitude) * response).clamp_min(1.0e-30)
        temperature = total4.pow(0.25)
        derivative_temperature = response / (4.0 * temperature.pow(3))
        redshift = torch.as_tensor(
            self.source_redshift,
            device=device,
            dtype=dtype,
        )
        color = torch.as_tensor(
            self.color_correction,
            device=device,
            dtype=dtype,
        )
        wavelength = (
            torch.as_tensor(
                self.geometry.wavelengths_angstrom,
                device=device,
                dtype=dtype,
            )
            * 1.0e-10
            / (1.0 + redshift)
        )
        exponent = (
            _H
            * _C
            / (
                wavelength
                * transfer.gfactor[..., None]
                * _K_B
                * color
                * temperature[..., None]
            )
        )
        exponent = exponent.clamp(max=85.0)
        exponential = torch.exp(exponent)
        prefactor = 2.0 * _H * _C / wavelength.pow(3) / color.pow(4)
        derivative_intensity = (
            prefactor
            * exponential
            * exponent
            / temperature[..., None]
            / torch.expm1(exponent).square().clamp_min(1.0e-30)
        )
        pixel_area = self.geometry.pixel_scale_m[0] * self.geometry.pixel_scale_m[1]
        weights = (
            derivative_intensity
            * derivative_temperature[..., None]
            * transfer.solid_angle_sr[..., None]
            / (1.0 + redshift).pow(3)
            * 1.0e26
            / pixel_area
        )
        weights = torch.where(
            transfer.hit[..., None] & torch.isfinite(weights),
            weights,
            torch.zeros_like(weights),
        )
        if not torch.is_grad_enabled():
            ready = None
            if device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(device))
            self._linear_response_cache[cache_key] = (weights, ready)
        return weights

    def mean_response_delays(
        self,
        *,
        magnification: torch.Tensor | None = None,
        driver_amplitude: float = 1.0,
    ) -> torch.Tensor:
        """Return the exact response-weighted mean delay in each band.

        This first-moment calculation avoids constructing a binned transfer
        function. Repeated no-gradient calls reuse the source's invariant
        linear-response weights.
        """

        weights = self.linear_response_weights(
            driver_amplitude=driver_amplitude,
            normalize=False,
        )
        if magnification is not None:
            magnification = torch.as_tensor(
                magnification,
                device=weights.device,
                dtype=weights.dtype,
            )
            if magnification.shape != self.geometry.shape:
                raise ValueError("magnification must match the source geometry")
            weights = weights * magnification[..., None]
        delay = self.delay_days.to(device=weights.device, dtype=weights.dtype)
        valid = self.transfer.hit.to(weights.device) & torch.isfinite(delay)
        safe_delay = torch.where(valid, delay, torch.zeros_like(delay))
        weights = torch.where(valid[..., None], weights, torch.zeros_like(weights))
        normalizer = weights.sum(dim=(0, 1)).clamp_min(1.0e-30)
        return (weights * safe_delay[..., None]).sum(dim=(0, 1)) / normalizer

    def batched_mean_response_delays(
        self,
        *,
        magnification: torch.Tensor,
        driver_amplitude: float = 1.0,
        spatial_chunk_size: int = 262_144,
    ) -> torch.Tensor:
        """Return exact mean delays for ``[batch, y, x]`` magnifications."""

        if spatial_chunk_size < 1:
            raise ValueError("spatial_chunk_size must be positive")
        weights = self.linear_response_weights(
            driver_amplitude=driver_amplitude,
            normalize=False,
        )
        magnification = torch.as_tensor(
            magnification,
            device=weights.device,
            dtype=weights.dtype,
        )
        if magnification.ndim != 3 or magnification.shape[1:] != self.geometry.shape:
            raise ValueError("magnification must have shape [batch, y, x]")
        delay = self.delay_days.to(device=weights.device, dtype=weights.dtype)
        valid = self.transfer.hit.to(weights.device) & torch.isfinite(delay)
        positions = torch.nonzero(valid.reshape(-1)).reshape(-1)
        delay = delay.reshape(-1)[positions]
        weights = weights.reshape(-1, weights.shape[-1])[positions]
        magnification = magnification.reshape(magnification.shape[0], -1)
        numerator = torch.zeros(
            (magnification.shape[0], weights.shape[-1]),
            device=weights.device,
            dtype=weights.dtype,
        )
        denominator = torch.zeros_like(numerator)
        for start in range(0, positions.numel(), spatial_chunk_size):
            stop = min(positions.numel(), start + spatial_chunk_size)
            contribution = (
                magnification[:, positions[start:stop], None]
                * weights[None, start:stop]
            )
            denominator.add_(contribution.sum(dim=1))
            numerator.add_((contribution * delay[None, start:stop, None]).sum(dim=1))
        return numerator / denominator.clamp_min(1.0e-30)

    def transfer_function(
        self,
        delay_edges_days: torch.Tensor | Sequence[float],
        *,
        magnification: torch.Tensor | None = None,
        driver_amplitude: float = 1.0,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Bin linear response mass by delay, returning ``[delay, band]``."""

        weights = self.linear_response_weights(
            driver_amplitude=driver_amplitude,
            normalize=False,
        )
        if magnification is not None:
            magnification = torch.as_tensor(
                magnification,
                device=weights.device,
                dtype=weights.dtype,
            )
            if magnification.shape != self.geometry.shape:
                raise ValueError("magnification must match the source geometry")
            weights = weights * magnification[..., None]
        edges = torch.as_tensor(
            delay_edges_days,
            device=weights.device,
            dtype=weights.dtype,
        )
        if (
            edges.ndim != 1
            or edges.numel() < 2
            or not bool(torch.all(edges[1:] > edges[:-1]))
        ):
            raise ValueError("delay_edges_days must be strictly increasing")
        delay = self.delay_days.to(device=weights.device, dtype=weights.dtype)
        valid = self.transfer.hit.to(weights.device) & torch.isfinite(delay)
        indices = torch.bucketize(delay[valid], edges, right=True) - 1
        in_range = (indices >= 0) & (indices < edges.numel() - 1)
        output = torch.zeros(
            (edges.numel() - 1, weights.shape[-1]),
            device=weights.device,
            dtype=weights.dtype,
        )
        output.index_add_(0, indices[in_range], weights[valid][in_range])
        if normalize:
            output = output / output.sum(dim=0, keepdim=True).clamp_min(1.0e-30)
        return output

    def batched_transfer_function(
        self,
        delay_edges_days: torch.Tensor | Sequence[float],
        *,
        magnification: torch.Tensor,
        driver_amplitude: float = 1.0,
        normalize: bool = True,
        spatial_chunk_size: int = 262_144,
    ) -> torch.Tensor:
        """Bin several magnification-weighted responses in bounded chunks.

        ``magnification`` has shape ``[batch, y, x]``. Source-dependent
        response weights and delay-bin assignments are constructed once, and
        the spatial chunk bounds the temporary ``[batch, pixel, band]``
        product independently of temporal batch size.
        """

        if spatial_chunk_size < 1:
            raise ValueError("spatial_chunk_size must be positive")
        weights = self.linear_response_weights(
            driver_amplitude=driver_amplitude,
            normalize=False,
        )
        magnification = torch.as_tensor(
            magnification,
            device=weights.device,
            dtype=weights.dtype,
        )
        if magnification.ndim != 3 or magnification.shape[1:] != self.geometry.shape:
            raise ValueError("magnification must have shape [batch, y, x]")
        edges = torch.as_tensor(
            delay_edges_days,
            device=weights.device,
            dtype=weights.dtype,
        )
        if (
            edges.ndim != 1
            or edges.numel() < 2
            or not bool(torch.all(edges[1:] > edges[:-1]))
        ):
            raise ValueError("delay_edges_days must be strictly increasing")
        delay = self.delay_days.to(device=weights.device, dtype=weights.dtype)
        valid = self.transfer.hit.to(weights.device) & torch.isfinite(delay)
        indices = torch.bucketize(delay.reshape(-1), edges, right=True) - 1
        in_range = valid.reshape(-1) & (indices >= 0) & (indices < edges.numel() - 1)
        positions = torch.nonzero(in_range).reshape(-1)
        indices = indices[positions]
        weights = weights.reshape(-1, weights.shape[-1])[positions]
        magnification = magnification.reshape(magnification.shape[0], -1)
        output = torch.zeros(
            (magnification.shape[0], edges.numel() - 1, weights.shape[-1]),
            device=weights.device,
            dtype=weights.dtype,
        )
        for start in range(0, positions.numel(), spatial_chunk_size):
            stop = min(positions.numel(), start + spatial_chunk_size)
            chunk_positions = positions[start:stop]
            contribution = (
                magnification[:, chunk_positions, None] * weights[None, start:stop]
            )
            scatter_indices = indices[None, start:stop, None].expand(
                magnification.shape[0],
                stop - start,
                weights.shape[-1],
            )
            output.scatter_add_(1, scatter_indices, contribution)
        if normalize:
            output = output / output.sum(dim=1, keepdim=True).clamp_min(1.0e-30)
        return output

    def metadata(self) -> Mapping[str, object]:
        """Return source, driver, and transfer provenance without map arrays."""

        def scalar(value):
            return float(torch.as_tensor(value).detach().cpu())

        return {
            "type": "thermal_reprocessing",
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
            "signal": dict(self.signal.metadata()),
            "observer_transfer": dict(self.transfer.metadata),
            "brightness_units": "Jy m^-2 projected source plane",
            "response_semantics": "additive T^4 per unit driver",
            "heating": dict(self.heating_metadata or {}),
            "is_time_static": bool(self.is_time_static),
        }
