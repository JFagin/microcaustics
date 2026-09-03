"""Configurable expanding-photosphere supernova sources."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from .base import SourceGeometry, _as_times, _geometry_grid
from .physical import _resolve_source_distances

_C = 299_792_458.0
_H = 6.626_070_15e-34
_K_B = 1.380_649e-23
_SIGMA_SB = 5.670_374_419e-8
_SECONDS_PER_DAY = 86_400.0
_ANGSTROM_M = 1.0e-10


@runtime_checkable
class PhotosphereEvolution(Protocol):
    """Temporal physics required by :class:`ExpandingPhotosphereSource`.

    Implementations operate on rest-frame times in days and must preserve the
    input tensor's shape, device, and floating dtype. This small protocol lets
    users substitute tabulated radiation-hydrodynamic models or arbitrary
    differentiable Torch functions without changing the microlensing code.
    """

    def luminosity(self, rest_times_days: torch.Tensor) -> torch.Tensor:
        """Return bolometric luminosity in watts."""

        ...

    def radius(self, rest_times_days: torch.Tensor) -> torch.Tensor:
        """Return photospheric radius in meters."""

        ...

    def temperature(self, rest_times_days: torch.Tensor) -> torch.Tensor:
        """Return color temperature in Kelvin."""

        ...

    def metadata(self) -> Mapping[str, object]:
        """Return serializable model provenance."""

        ...


@dataclass(frozen=True)
class PowerLawExponentialPhotosphere:
    """Simple rise/decline and homologous-expansion photosphere.

    Every scientific parameter is explicit. The class deliberately has no
    paper-specific numerical defaults.
    """

    peak_time_rest_days: float
    peak_luminosity_watts: float
    rise_power: float
    decline_time_rest_days: float
    photosphere_velocity_km_s: float
    initial_radius_m: float
    temperature_floor_k: float
    temperature_ceiling_k: float

    def __post_init__(self) -> None:
        positive = {
            "peak_time_rest_days": self.peak_time_rest_days,
            "peak_luminosity_watts": self.peak_luminosity_watts,
            "rise_power": self.rise_power,
            "decline_time_rest_days": self.decline_time_rest_days,
            "photosphere_velocity_km_s": self.photosphere_velocity_km_s,
            "initial_radius_m": self.initial_radius_m,
            "temperature_floor_k": self.temperature_floor_k,
            "temperature_ceiling_k": self.temperature_ceiling_k,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.temperature_floor_k > self.temperature_ceiling_k:
            raise ValueError("temperature floor must not exceed the ceiling")

    def luminosity(self, rest_times_days: torch.Tensor) -> torch.Tensor:
        """Return an exact-zero rise followed by exponential decline."""

        times = torch.as_tensor(rest_times_days)
        peak = times.new_tensor(self.peak_luminosity_watts)
        phase = (times.clamp_min(0.0) / self.peak_time_rest_days).clamp_max(1.0)
        rise = peak * phase.pow(self.rise_power)
        decline = peak * torch.exp(
            -(times - self.peak_time_rest_days).clamp_min(0.0)
            / self.decline_time_rest_days
        )
        value = torch.where(times <= self.peak_time_rest_days, rise, decline)
        return torch.where(times > 0, value, torch.zeros_like(value))

    def radius(self, rest_times_days: torch.Tensor) -> torch.Tensor:
        """Return the homologously expanding radius in meters."""

        times = torch.as_tensor(rest_times_days).clamp_min(0.0)
        velocity_m_s = self.photosphere_velocity_km_s * 1_000.0
        return self.initial_radius_m + velocity_m_s * times * _SECONDS_PER_DAY

    def temperature(self, rest_times_days: torch.Tensor) -> torch.Tensor:
        """Return the luminosity-consistent bounded color temperature."""

        luminosity = self.luminosity(rest_times_days)
        radius = self.radius(rest_times_days)
        denominator = 4.0 * math.pi * _SIGMA_SB * radius.square()
        temperature = (
            (luminosity / denominator.clamp_min(1e-30)).clamp_min(1e-30).pow(0.25)
        )
        return temperature.clamp(
            min=self.temperature_floor_k,
            max=self.temperature_ceiling_k,
        )

    def metadata(self) -> Mapping[str, object]:
        """Return the explicit evolution parameters."""

        return {
            "type": "power_law_exponential_photosphere",
            "peak_time_rest_days": self.peak_time_rest_days,
            "peak_luminosity_watts": self.peak_luminosity_watts,
            "rise_power": self.rise_power,
            "decline_time_rest_days": self.decline_time_rest_days,
            "photosphere_velocity_km_s": self.photosphere_velocity_km_s,
            "initial_radius_m": self.initial_radius_m,
            "temperature_floor_k": self.temperature_floor_k,
            "temperature_ceiling_k": self.temperature_ceiling_k,
        }


@dataclass(frozen=True)
class PhotosphereAppearance:
    """Built-in projected shape and phenomenological spectral corrections.

    Defaults describe a circular gray limb-darkened blackbody without UV
    blanketing. They are generic numerical behavior, not the paper preset.
    """

    axis_ratio: float = 1.0
    position_angle_deg: float = 0.0
    limb_darkening: float = 0.5
    achromatic_phase_rest_days: float = 0.0
    chromatic_transition_rest_days: float = 1.0
    chromatic_limb_slope: float = 0.0
    uv_blanketing_wavelength_angstrom: float = 4_000.0
    uv_blanketing_tau_early: float = 0.0
    uv_blanketing_tau_late: float = 0.0

    def __post_init__(self) -> None:
        finite = {
            "axis_ratio": self.axis_ratio,
            "position_angle_deg": self.position_angle_deg,
            "limb_darkening": self.limb_darkening,
            "achromatic_phase_rest_days": self.achromatic_phase_rest_days,
            "chromatic_transition_rest_days": (self.chromatic_transition_rest_days),
            "chromatic_limb_slope": self.chromatic_limb_slope,
            "uv_blanketing_wavelength_angstrom": (
                self.uv_blanketing_wavelength_angstrom
            ),
            "uv_blanketing_tau_early": self.uv_blanketing_tau_early,
            "uv_blanketing_tau_late": self.uv_blanketing_tau_late,
        }
        if any(not math.isfinite(value) for value in finite.values()):
            raise ValueError("appearance parameters must be finite")
        if not 0 < self.axis_ratio <= 1:
            raise ValueError("axis_ratio must lie in (0, 1]")
        if not 0 <= self.limb_darkening <= 1:
            raise ValueError("limb_darkening must lie in [0, 1]")
        if self.achromatic_phase_rest_days < 0:
            raise ValueError("achromatic phase must be non-negative")
        if self.chromatic_transition_rest_days <= 0:
            raise ValueError("chromatic transition must be positive")
        if self.uv_blanketing_wavelength_angstrom <= 0:
            raise ValueError("UV blanketing wavelength must be positive")
        if self.uv_blanketing_tau_early < 0 or self.uv_blanketing_tau_late < 0:
            raise ValueError("UV blanketing optical depths must be non-negative")

    def metadata(self) -> Mapping[str, object]:
        """Return the projected-appearance parameters."""

        return {
            "axis_ratio": self.axis_ratio,
            "position_angle_deg": self.position_angle_deg,
            "limb_darkening": self.limb_darkening,
            "achromatic_phase_rest_days": self.achromatic_phase_rest_days,
            "chromatic_transition_rest_days": (self.chromatic_transition_rest_days),
            "chromatic_limb_slope": self.chromatic_limb_slope,
            "uv_blanketing_wavelength_angstrom": (
                self.uv_blanketing_wavelength_angstrom
            ),
            "uv_blanketing_tau_early": self.uv_blanketing_tau_early,
            "uv_blanketing_tau_late": self.uv_blanketing_tau_late,
        }


SpatialProfile = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]
SpectralModifier = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ExpandingPhotosphereSource:
    """A multiband, time-dependent expanding source on a fixed pixel grid.

    ``evolution`` may be any :class:`PhotosphereEvolution`. An optional
    ``spatial_profile`` receives ``(x, y, radius, rest_time, wavelength)`` and
    returns values broadcastable to ``[time, y, x, band]``. An optional
    ``spectral_modifier`` receives ``(rest_time, rest_wavelength)`` and returns
    factors broadcastable to ``[time, band]``. Brightness is Jy per square
    meter of projected source plane. Omit ``source_redshift`` when passing the
    model to a system. The system supplies its cosmological geometry. For
    standalone use, pass it here or to ``pixelate(source_redshift=...)``.
    ``maximum_observer_time_days`` fixes the source field over a declared
    evolution interval rather than changing it for every light-curve request.
    """

    is_time_static = False

    def __init__(
        self,
        *,
        source_redshift: float | None = None,
        wavelengths_angstrom: Sequence[float] | None = None,
        bands_angstrom: Mapping[str, float] | None = None,
        maximum_observer_time_days: float,
        evolution: PhotosphereEvolution,
        band_names: Sequence[str] | None = None,
        source_grid_shape: int = 256,
        explosion_time_days: float = 0.0,
        source_margin: float = 1.05,
        appearance: PhotosphereAppearance | None = None,
        spatial_profile: SpatialProfile | None = None,
        spectral_modifier: SpectralModifier | None = None,
        maximum_photosphere_radius_m: float | None = None,
        luminosity_distance_m: float | None = None,
        H0: float | None = None,
        Om0: float | None = None,
        name: str = "expanding_photosphere",
    ) -> None:
        self.source_redshift = (
            None
            if source_redshift is None
            else self._positive(source_redshift, "source_redshift")
        )
        self._H0, self._Om0 = H0, Om0
        self.maximum_observer_time_days = self._finite(
            maximum_observer_time_days,
            "maximum_observer_time_days",
        )
        self.explosion_time_days = self._finite(
            explosion_time_days,
            "explosion_time_days",
        )
        if self.maximum_observer_time_days <= self.explosion_time_days:
            raise ValueError("maximum time must be later than the explosion")
        if not isinstance(evolution, PhotosphereEvolution):
            raise TypeError("evolution must implement PhotosphereEvolution")
        self.evolution = evolution
        self.appearance = appearance or PhotosphereAppearance()
        self.spatial_profile = spatial_profile
        self.spectral_modifier = spectral_modifier
        if not isinstance(source_grid_shape, int) or source_grid_shape < 2:
            raise ValueError("source_grid_shape must be an integer of at least two")
        self.source_grid_shape = source_grid_shape
        self.source_margin = self._positive(
            source_margin,
            "source_margin",
        )
        if self.source_margin <= 1:
            raise ValueError("source_margin must be greater than one")
        self.name = str(name)
        if not self.name:
            raise ValueError("name must be non-empty")

        if bands_angstrom is not None:
            if wavelengths_angstrom is not None or band_names is not None:
                raise ValueError(
                    "supply bands_angstrom or wavelengths_angstrom/band_names, not both"
                )
            band_names = tuple(str(name) for name in bands_angstrom)
            wavelengths_angstrom = tuple(
                float(value) for value in bands_angstrom.values()
            )
        if wavelengths_angstrom is None:
            raise ValueError("bands must be supplied")
        wavelengths = tuple(float(value) for value in wavelengths_angstrom)
        if not wavelengths or any(
            not math.isfinite(value) or value <= 0 for value in wavelengths
        ):
            raise ValueError("wavelengths_angstrom must be finite and positive")
        if band_names is None:
            band_names = tuple(f"band_{index}" for index in range(len(wavelengths)))
        bands = tuple(str(value) for value in band_names)
        if len(bands) != len(wavelengths):
            raise ValueError("band_names must match wavelengths_angstrom")
        if any(not band for band in bands) or len(set(bands)) != len(bands):
            raise ValueError("band names must be non-empty and unique")
        self.wavelengths_angstrom = wavelengths
        self.band_names = bands

        self._luminosity_distance_override = (
            None
            if luminosity_distance_m is None
            else self._positive(luminosity_distance_m, "luminosity_distance_m")
        )
        self._radius_override = (
            None
            if maximum_photosphere_radius_m is None
            else self._positive(
                maximum_photosphere_radius_m, "maximum_photosphere_radius_m"
            )
        )
        self.geometry = None
        self.maximum_photosphere_radius_m = self._radius_override
        self.maximum_radius_source = (
            "user" if self._radius_override is not None else "sampled_evolution"
        )
        self.luminosity_distance_m = self._luminosity_distance_override
        self.distance_source = (
            "user" if self._luminosity_distance_override is not None else "unresolved"
        )
        self._coordinate_cache = {}
        if self.source_redshift is not None:
            self._initialize_geometry(
                _resolve_source_distances(
                    None,
                    source_redshift=self.source_redshift,
                    H0=H0,
                    Om0=Om0,
                )
            )

    @property
    def redshift(self) -> float:
        """Resolved source redshift used for rest-frame evolution."""
        if self.source_redshift is None:
            raise ValueError(
                "resolve this supernova through a system or pixelate(source_redshift=...) first"
            )
        return self.source_redshift

    def _initialize_geometry(self, distances):
        if distances.source_redshift is None:
            raise ValueError("supernova evolution requires a source redshift")
        self.source_redshift = float(distances.source_redshift)
        self.luminosity_distance_m = (
            self._luminosity_distance_override
            if self._luminosity_distance_override is not None
            else distances.source_m * (1 + self.redshift) ** 2
        )
        self.distance_source = (
            "user"
            if self._luminosity_distance_override is not None
            else "source_geometry"
        )
        self.maximum_photosphere_radius_m = (
            self._sample_maximum_radius()
            if self._radius_override is None
            else self._radius_override
        )
        self.maximum_radius_source = (
            "sampled_evolution" if self._radius_override is None else "user"
        )
        pixel_scale = (
            2
            * self.maximum_photosphere_radius_m
            * self.source_margin
            / self.source_grid_shape
        )
        self.geometry = SourceGeometry(
            self.source_grid_shape,
            (pixel_scale, pixel_scale),
            self.wavelengths_angstrom,
            self.band_names,
        )
        self._coordinate_cache = {}

    def pixelate(
        self,
        distances=None,
        *,
        source_redshift=None,
        H0=None,
        Om0=None,
        grid=None,
        policy=None,
        runtime=None,
    ):
        """Resolve a supernova using system distances or source-only cosmology.

        The evolution horizon fixes the field size, independently of a later
        light-curve duration. An explicit luminosity distance remains supported.
        The original source specification is never mutated by a system.
        """
        if policy is not None:
            raise ValueError("set source_grid_shape and source_margin on the supernova")
        resolved = _resolve_source_distances(
            distances,
            source_redshift=source_redshift,
            model_redshift=self.source_redshift,
            H0=(self._H0 if H0 is None else H0) if distances is None else H0,
            Om0=(self._Om0 if Om0 is None else Om0) if distances is None else Om0,
            runtime=runtime,
        )
        result = copy(self)
        result._initialize_geometry(resolved)
        native = _geometry_grid(result.geometry, resolved)
        if grid is not None and any(
            a > b * (1 + 1e-6)
            for a, b in zip(
                native.field_of_view_uas, grid.field_of_view_uas, strict=True
            )
        ):
            raise ValueError("source_grid does not enclose the supernova source field")
        return result

    def recommended_grid(self, distances, policy=None):
        """Return the fixed angular field that encloses the expanding source."""
        source = self.pixelate(distances, policy=policy)
        return _geometry_grid(source.geometry, distances)

    @staticmethod
    def _finite(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        result = cls._finite(value, name)
        if result <= 0:
            raise ValueError(f"{name} must be positive")
        return result

    def observer_to_rest_time(self, times_days) -> torch.Tensor:
        """Convert observer epochs to elapsed rest-frame days."""

        times = _as_times(times_days)
        return (times - self.explosion_time_days) / (1.0 + self.redshift)

    def support_radius_m(self, distances=None) -> float:
        """Return the largest emitting photospheric radius in meters.

        ``distances`` is accepted for compatibility with physical source
        models. The expanding source already stores its source-plane radius.
        The separate ``source_margin`` remains a numerical image margin
        and is therefore not part of the emitting support.
        """

        source = self if distances is None else self.pixelate(distances)
        if source.geometry is None:
            raise ValueError("resolve the source geometry before querying its support")
        return float(source.maximum_photosphere_radius_m)

    def _sample_maximum_radius(self) -> float:
        rest_maximum = (self.maximum_observer_time_days - self.explosion_time_days) / (
            1.0 + self.redshift
        )
        rest_times = torch.linspace(0.0, rest_maximum, 2049, dtype=torch.float64)
        radii = torch.as_tensor(self.evolution.radius(rest_times))
        if radii.shape != rest_times.shape:
            raise ValueError("evolution radius must preserve the time shape")
        if not torch.isfinite(radii).all() or torch.any(radii <= 0):
            raise ValueError("evolution radii must be finite and positive")
        return float(radii.max())

    def _coordinates(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), dtype)
        cached = self._coordinate_cache.get(key)
        if cached is not None:
            return cached
        ny, nx = self.geometry.shape
        dy, dx = self.geometry.pixel_scale_m
        x = (torch.arange(nx, device=device, dtype=dtype) + 0.5 - nx / 2) * dx
        y = (torch.arange(ny, device=device, dtype=dtype) + 0.5 - ny / 2) * dy
        y, x = torch.meshgrid(y, x, indexing="ij")
        cosine = math.cos(math.radians(self.appearance.position_angle_deg))
        sine = math.sin(math.radians(self.appearance.position_angle_deg))
        rotated_x = cosine * x + sine * y
        rotated_y = -sine * x + cosine * y
        self._coordinate_cache[key] = (rotated_x, rotated_y)
        return rotated_x, rotated_y

    def _built_in_profile(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        radius: torch.Tensor,
        rest_times: torch.Tensor,
        wavelengths_rest: torch.Tensor,
    ) -> torch.Tensor:
        radius_grid = radius[:, None, None].clamp_min(1e-30)
        rho2 = (x[None] / radius_grid).square()
        rho2 += (y[None] / (self.appearance.axis_ratio * radius_grid)).square()
        inside = rho2 <= 1.0
        mu = torch.sqrt((1.0 - rho2).clamp(0.0, 1.0))
        elapsed = (rest_times - self.appearance.achromatic_phase_rest_days).clamp_min(
            0.0
        )
        phase = 1.0 - torch.exp(
            -elapsed / self.appearance.chromatic_transition_rest_days
        )
        wavelength_lever = torch.log(5_500.0 / wavelengths_rest.clamp_min(1.0)).clamp(
            -1.0, 2.0
        )
        limb = (
            self.appearance.limb_darkening
            + self.appearance.chromatic_limb_slope
            * phase[:, None]
            * wavelength_lever[None]
        ).clamp(0.0, 0.95)
        profile = 1.0 - limb[:, None, None] + limb[:, None, None] * mu[..., None]
        return profile.clamp_min(0.0) * inside[..., None]

    def _built_in_spectral_modifier(
        self,
        rest_times: torch.Tensor,
        wavelengths_rest: torch.Tensor,
    ) -> torch.Tensor:
        elapsed = (rest_times - self.appearance.achromatic_phase_rest_days).clamp_min(
            0.0
        )
        phase = 1.0 - torch.exp(
            -elapsed / self.appearance.chromatic_transition_rest_days
        )
        lever = (
            self.appearance.uv_blanketing_wavelength_angstrom / wavelengths_rest - 1.0
        ).clamp_min(0.0)
        tau = self.appearance.uv_blanketing_tau_early + phase * (
            self.appearance.uv_blanketing_tau_late
            - self.appearance.uv_blanketing_tau_early
        )
        return torch.exp(-tau[:, None] * lever[None].square())

    @staticmethod
    def _normalized_profile(profile: torch.Tensor) -> torch.Tensor:
        support = profile > 0
        count = support.sum(dim=(1, 2), keepdim=True).clamp_min(1)
        mean = profile.sum(dim=(1, 2), keepdim=True) / count
        return torch.where(support, profile / mean.clamp_min(1e-30), 0.0)

    def brightness(
        self,
        times_days,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return ``[time, y, x, band]`` brightness in Jy per square meter."""

        if self.geometry is None:
            raise ValueError(
                "resolve this source through a system or pixelate(source_redshift=...) before brightness"
            )
        times = _as_times(times_days)
        if bool(torch.any(times > self.maximum_observer_time_days)):
            raise ValueError(
                "source times exceed maximum_observer_time_days; increase the supernova horizon to size a field that covers them"
            )
        resolved_device = times.device if device is None else torch.device(device)
        resolved_dtype = dtype or (
            times.dtype if times.is_floating_point() else torch.get_default_dtype()
        )
        times = times.to(device=resolved_device, dtype=resolved_dtype)
        rest_times = self.observer_to_rest_time(times).to(
            device=resolved_device,
            dtype=resolved_dtype,
        )
        luminosity = torch.as_tensor(
            self.evolution.luminosity(rest_times),
            device=resolved_device,
            dtype=resolved_dtype,
        )
        radius = torch.as_tensor(
            self.evolution.radius(rest_times),
            device=resolved_device,
            dtype=resolved_dtype,
        )
        temperature = torch.as_tensor(
            self.evolution.temperature(rest_times),
            device=resolved_device,
            dtype=resolved_dtype,
        )
        for name, value in (
            ("luminosity", luminosity),
            ("radius", radius),
            ("temperature", temperature),
        ):
            if value.shape != rest_times.shape:
                raise ValueError(f"evolution {name} must preserve the time shape")
            if not torch.isfinite(value).all():
                raise ValueError(f"evolution {name} must be finite")
        if torch.any(radius <= 0) or torch.any(temperature <= 0):
            raise ValueError("evolution radius and temperature must be positive")

        blackbody_luminosity = (
            4.0 * math.pi * _SIGMA_SB * radius.square() * temperature.pow(4)
        )
        dilution = torch.where(
            luminosity > 0,
            luminosity / blackbody_luminosity.clamp_min(1e-30),
            torch.zeros_like(luminosity),
        )
        wavelengths_observed = torch.tensor(
            self.wavelengths_angstrom,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        wavelengths_rest = wavelengths_observed / (1.0 + self.redshift)
        x, y = self._coordinates(device=resolved_device, dtype=resolved_dtype)
        profile_function = self.spatial_profile
        if profile_function is None:
            profile = self._built_in_profile(x, y, radius, rest_times, wavelengths_rest)
        else:
            profile = torch.as_tensor(
                profile_function(x, y, radius, rest_times, wavelengths_rest),
                device=resolved_device,
                dtype=resolved_dtype,
            )
        expected = (
            len(rest_times),
            self.source_grid_shape,
            self.source_grid_shape,
            len(self.wavelengths_angstrom),
        )
        try:
            profile = torch.broadcast_to(profile, expected)
        except RuntimeError as error:
            raise ValueError(
                f"spatial profile must broadcast to {expected}, got {profile.shape}"
            ) from error
        if not torch.isfinite(profile).all() or torch.any(profile < 0):
            raise ValueError("spatial profile must be finite and non-negative")
        profile = self._normalized_profile(profile)

        wavelength_m = wavelengths_rest * _ANGSTROM_M
        observed_wavelength_m = wavelengths_observed * _ANGSTROM_M
        exponent = (
            _H
            * _C
            / (_K_B * wavelength_m[None] * temperature[:, None].clamp_min(1e-30))
        ).clamp_max(85.0)
        intensity_lambda = (
            2.0
            * _H
            * _C**2
            / wavelength_m[None].pow(5)
            / torch.expm1(exponent).clamp_min(1e-30)
        )
        modifier_function = self.spectral_modifier
        if modifier_function is None:
            modifier = self._built_in_spectral_modifier(rest_times, wavelengths_rest)
        else:
            modifier = torch.as_tensor(
                modifier_function(rest_times, wavelengths_rest),
                device=resolved_device,
                dtype=resolved_dtype,
            )
        try:
            modifier = torch.broadcast_to(
                modifier,
                (len(rest_times), len(wavelengths_rest)),
            )
        except RuntimeError as error:
            raise ValueError(
                "spectral modifier must broadcast to [time, band]"
            ) from error
        if not torch.isfinite(modifier).all() or torch.any(modifier < 0):
            raise ValueError("spectral modifier must be finite and non-negative")

        distance_to_jy = 1.0e26 / self.luminosity_distance_m**2
        spectral_surface_flux = (
            intensity_lambda
            * observed_wavelength_m[None].square()
            / _C
            * modifier
            * dilution[:, None]
            / (1.0 + self.redshift)
            * distance_to_jy
        )
        brightness = profile * spectral_surface_flux[:, None, None]
        active = (rest_times > 0) & (luminosity > 0)
        return torch.where(
            active[:, None, None, None],
            brightness,
            torch.zeros_like(brightness),
        )

    def metadata(self) -> Mapping[str, object]:
        """Return temporal, geometric, physical, and customization provenance."""

        return {
            "type": "expanding_photosphere",
            "name": self.name,
            "brightness_units": "Jy m^-2 projected source plane",
            "redshift": self.source_redshift,
            "explosion_time_observer_days": self.explosion_time_days,
            "maximum_observer_time_days": self.maximum_observer_time_days,
            "maximum_photosphere_radius_m": self.maximum_photosphere_radius_m,
            "maximum_radius_source": self.maximum_radius_source,
            "source_margin": self.source_margin,
            "source_grid_shape": self.source_grid_shape,
            "luminosity_distance_m": self.luminosity_distance_m,
            "distance_source": self.distance_source,
            "custom_spatial_profile": self.spatial_profile is not None,
            "custom_spectral_modifier": self.spectral_modifier is not None,
            "evolution": dict(self.evolution.metadata()),
            "appearance": dict(self.appearance.metadata()),
        }


def paper_type_ia_supernova_source(
    *,
    redshift: float,
    wavelengths_angstrom: Sequence[float],
    maximum_observer_time_days: float,
    band_names: Sequence[str] | None = None,
    source_grid_shape: int = 256,
    explosion_time_days: float = 0.0,
    luminosity_distance_m: float | None = None,
) -> ExpandingPhotosphereSource:
    """Construct the explicit Type Ia-like prototype used in the paper.

    This named preset is provided for reproducibility. It is a transparent
    photospheric-phase demonstration rather than a precision Type Ia spectral
    or distance model. Applications should normally supply their own temporal,
    spatial, and spectral functions.
    """

    evolution = PowerLawExponentialPhotosphere(
        peak_time_rest_days=18.0,
        peak_luminosity_watts=10.0**36.2,
        rise_power=2.4,
        decline_time_rest_days=28.0,
        photosphere_velocity_km_s=12_000.0,
        initial_radius_m=1.0e9,
        temperature_floor_k=3_000.0,
        temperature_ceiling_k=18_000.0,
    )
    appearance = PhotosphereAppearance(
        axis_ratio=0.9,
        position_angle_deg=25.0,
        limb_darkening=0.5,
        achromatic_phase_rest_days=21.0,
        chromatic_transition_rest_days=7.0,
        chromatic_limb_slope=0.15,
        uv_blanketing_wavelength_angstrom=4_000.0,
        uv_blanketing_tau_early=0.35,
        uv_blanketing_tau_late=1.0,
    )
    return ExpandingPhotosphereSource(
        source_redshift=redshift,
        H0=70.0,
        Om0=0.3,
        wavelengths_angstrom=wavelengths_angstrom,
        maximum_observer_time_days=maximum_observer_time_days,
        evolution=evolution,
        band_names=band_names,
        source_grid_shape=source_grid_shape,
        explosion_time_days=explosion_time_days,
        source_margin=1.05,
        appearance=appearance,
        luminosity_distance_m=luminosity_distance_m,
        name="paper_type_ia_prototype",
    )
