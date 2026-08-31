"""Physical source models that select and materialize their own grids."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from ..geometry import PlaneGrid
from ..lens import LensingDistances
from ..relativity import (
    ObserverScreen,
    add_observer_coordinates,
    axis_lamppost_profile,
    trace_primary_equatorial,
)
from ..runtime import ResolvedRuntime, RuntimeConfig, resolve_runtime
from .analytic import GaussianSource
from .base import PixelatedSource, SourceGeometry
from .reprocessing import ThermalReprocessingSource
from .thin_disk import ThinDiskSource, thin_disk_flux_radius_rg
from .transferred_disk import TransferredThinDiskSource

if TYPE_CHECKING:
    from .variability import DrivingSignal

_G = 6.67430e-11
_C = 299_792_458.0
_M_SUN = 1.988409870698051e30


def _resolve_bands(
    wavelengths_angstrom,
    band_names,
    bands_angstrom: Mapping[str, float] | None,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Normalize either a band mapping or the compatible parallel tuples."""

    wavelengths = tuple(float(value) for value in wavelengths_angstrom)
    names = tuple(str(value) for value in band_names)
    if bands_angstrom is not None:
        mapped_names = tuple(str(name) for name in bands_angstrom)
        mapped_wavelengths = tuple(
            float(value) for value in bands_angstrom.values()
        )
        if (wavelengths or names) and (
            names != mapped_names or wavelengths != mapped_wavelengths
        ):
            raise ValueError(
                "bands_angstrom conflicts with wavelengths_angstrom/band_names"
            )
        names = mapped_names
        wavelengths = mapped_wavelengths
    if not wavelengths or len(wavelengths) != len(names):
        raise ValueError("bands must contain matching names and wavelengths")
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("band names must be non-empty and unique")
    if any(not math.isfinite(value) or value <= 0.0 for value in wavelengths):
        raise ValueError("band wavelengths must be finite and positive")
    return wavelengths, names


def _resolve_sampling(
    grid: SourceGridConfig,
    *,
    source_grid_shape: int | tuple[int, int] | None,
    enclosed_flux_fraction: float | None,
    source_margin: float | None,
) -> SourceGridConfig:
    """Apply convenient scalar sampling overrides to a grid policy."""

    if source_grid_shape is None and enclosed_flux_fraction is None and source_margin is None:
        return grid
    return replace(
        grid,
        **({"shape": source_grid_shape} if source_grid_shape is not None else {}),
        **(
            {"enclosed_flux_fraction": enclosed_flux_fraction}
            if enclosed_flux_fraction is not None
            else {}
        ),
        **({"margin": source_margin} if source_margin is not None else {}),
    )


@dataclass(frozen=True)
class SourceGridConfig:
    """Numerical sampling policy for an automatically sized physical source.

    ``shape`` selects the pixel resolution. ``enclosed_flux_fraction`` and
    ``margin`` define a support criterion rather than an angular field size.
    The quasar model derives that size from its physical disk parameters.
    """

    shape: int | tuple[int, int] = 1024
    enclosed_flux_fraction: float = 0.999
    margin: float = 1.05
    radial_samples: int = 20_000

    def __post_init__(self) -> None:
        shape = (
            (int(self.shape), int(self.shape))
            if isinstance(self.shape, int)
            else tuple(int(value) for value in self.shape)
        )
        if len(shape) != 2 or any(value < 1 for value in shape):
            raise ValueError("shape must contain two positive integers")
        if not 0.0 < self.enclosed_flux_fraction < 1.0:
            raise ValueError("enclosed_flux_fraction must lie in (0, 1)")
        if self.margin < 1.0:
            raise ValueError("margin must be at least one")
        if self.radial_samples < 128:
            raise ValueError("radial_samples must be at least 128")
        object.__setattr__(self, "shape", shape)


@runtime_checkable
class PhysicalSourceModel(Protocol):
    """A source whose physical parameters are independent of pixelization."""

    def recommended_grid(
        self,
        distances: LensingDistances,
        policy: SourceGridConfig | None = None,
    ) -> PlaneGrid:
        """Return a conservative angular source-plane grid."""

        ...

    def pixelate(
        self,
        distances: LensingDistances,
        *,
        grid: PlaneGrid | None = None,
        policy: SourceGridConfig | None = None,
        runtime: RuntimeConfig | ResolvedRuntime | None = None,
    ) -> PixelatedSource:
        """Materialize this physical model on one angular grid."""

        ...


@dataclass(frozen=True)
class ThinDiskModel:
    """A physical thin accretion disk without a predetermined pixel grid.

    The reddest requested band sets the common source support because it has
    the largest thermal-emission radius. ``SourceGridConfig`` controls the
    enclosed-flux fraction, numerical margin, and output resolution.
    """

    black_hole_mass_solar: float
    eddington_ratio: float
    wavelengths_angstrom: tuple[float, ...] = ()
    band_names: tuple[str, ...] = ()
    bands_angstrom: Mapping[str, float] | None = None
    source_redshift: float | None = None
    spin: float = 0.0
    inclination_deg: float = 30.0
    position_angle_deg: float = 0.0
    color_correction: float = 1.0
    temperature_slope_beta: float = 0.75
    support_lamp_fraction: float = 0.0
    support_corona_height_above_isco_rg: float = 20.0
    relativity: str = "none"
    grid: SourceGridConfig = SourceGridConfig()
    source_grid_shape: int | tuple[int, int] | None = None
    enclosed_flux_fraction: float | None = None
    source_margin: float | None = None
    name: str = "thin_disk"

    def __post_init__(self) -> None:
        wavelengths, names = _resolve_bands(
            self.wavelengths_angstrom,
            self.band_names,
            self.bands_angstrom,
        )
        object.__setattr__(self, "wavelengths_angstrom", wavelengths)
        object.__setattr__(self, "band_names", names)
        object.__setattr__(
            self,
            "grid",
            _resolve_sampling(
                self.grid,
                source_grid_shape=self.source_grid_shape,
                enclosed_flux_fraction=self.enclosed_flux_fraction,
                source_margin=self.source_margin,
            ),
        )
        if self.black_hole_mass_solar <= 0.0:
            raise ValueError("black_hole_mass_solar must be positive")
        if self.eddington_ratio <= 0.0:
            raise ValueError("eddington_ratio must be positive")
        if not self.wavelengths_angstrom or any(
            value <= 0.0 for value in self.wavelengths_angstrom
        ):
            raise ValueError("wavelengths_angstrom must be non-empty and positive")
        if len(self.wavelengths_angstrom) != len(self.band_names):
            raise ValueError("wavelength and band-name counts must match")
        if len(set(self.band_names)) != len(self.band_names):
            raise ValueError("band names must be unique")
        if self.source_redshift is not None and self.source_redshift < 0.0:
            raise ValueError("source_redshift must be non-negative")
        if self.support_lamp_fraction < 0.0:
            raise ValueError("support_lamp_fraction must be non-negative")
        if self.support_corona_height_above_isco_rg < 0.0:
            raise ValueError("support_corona_height_above_isco_rg must be non-negative")
        if self.relativity not in {"none", "approximate"}:
            raise ValueError("relativity must be 'none' or 'approximate'")

    def _redshift(self, distances: LensingDistances) -> float:
        value = (
            distances.source_redshift
            if self.source_redshift is None
            else self.source_redshift
        )
        if value is None:
            raise ValueError(
                "ThinDiskModel requires a source redshift. Construct distances "
                "with LensingDistances.from_redshifts or supply source_redshift"
            )
        return float(value)

    def support_radius_m(
        self,
        distances: LensingDistances,
        policy: SourceGridConfig | None = None,
    ) -> float:
        """Return the conservative deprojected outer support in meters."""

        resolved = self.grid if policy is None else policy
        radius_rg = thin_disk_flux_radius_rg(
            black_hole_mass_solar=self.black_hole_mass_solar,
            eddington_ratio=self.eddington_ratio,
            spin=self.spin,
            observed_wavelength_angstrom=max(self.wavelengths_angstrom),
            source_redshift=self._redshift(distances),
            temperature_slope_beta=self.temperature_slope_beta,
            color_correction=self.color_correction,
            lamp_fraction=self.support_lamp_fraction,
            corona_height_above_isco_rg=(self.support_corona_height_above_isco_rg),
            flux_fraction=resolved.enclosed_flux_fraction,
            safety_factor=resolved.margin,
            radial_samples=resolved.radial_samples,
        )
        gravitational_radius_m = _G * _M_SUN / _C**2 * float(self.black_hole_mass_solar)
        return radius_rg * gravitational_radius_m

    def recommended_grid(
        self,
        distances: LensingDistances,
        policy: SourceGridConfig | None = None,
    ) -> PlaneGrid:
        """Return a square field enclosing the projected disk at every angle."""

        resolved = self.grid if policy is None else policy
        radius_m = self.support_radius_m(distances, resolved)
        diameter_uas = float(
            distances.source_length_to_uas(2.0 * radius_m, dtype=torch.float64)
        )
        return PlaneGrid(resolved.shape, (diameter_uas, diameter_uas))

    def pixelate(
        self,
        distances: LensingDistances,
        *,
        grid: PlaneGrid | None = None,
        policy: SourceGridConfig | None = None,
        runtime: RuntimeConfig | ResolvedRuntime | None = None,
    ) -> ThinDiskSource:
        """Materialize the disk using the existing validated source class."""

        del runtime

        resolved_grid = (
            self.recommended_grid(distances, policy) if grid is None else grid
        )
        dy_uas, dx_uas = resolved_grid.pixel_scale_uas
        pixel_scale_m = distances.uas_to_source_length(
            (dy_uas, dx_uas),
            dtype=torch.float64,
        )
        geometry = SourceGeometry(
            shape=resolved_grid.shape,
            pixel_scale_m=(float(pixel_scale_m[0]), float(pixel_scale_m[1])),
            wavelengths_angstrom=tuple(float(v) for v in self.wavelengths_angstrom),
            band_names=tuple(self.band_names),
        )
        return ThinDiskSource.from_lensing_distances(
            geometry,
            self.black_hole_mass_solar,
            self.eddington_ratio,
            distances,
            source_redshift=self._redshift(distances),
            spin=self.spin,
            inclination_deg=self.inclination_deg,
            position_angle_deg=self.position_angle_deg,
            color_correction=self.color_correction,
            temperature_slope_beta=self.temperature_slope_beta,
            relativity=self.relativity,
            name=self.name,
        )


@dataclass(frozen=True)
class KerrDiskModel:
    """A full-Kerr thin disk with optional axial-lamppost reverberation.

    This physical model owns the observer screen, primary-image Kerr trace,
    observer redshift and delay maps, and optional axial-lamppost response.
    Users specify physical and numerical controls rather than reconstructing
    those coupled objects manually. When ``driving_signal`` is omitted,
    :meth:`pixelate` returns a static :class:`TransferredThinDiskSource`.
    Supplying a signal returns a :class:`ThermalReprocessingSource` with the
    same observer transfer and source grid.

    ``grid`` controls source resolution and enclosed-flux support.
    ``lamppost_nalpha`` and ``lamppost_radial_bins`` affect only a variable
    lamppost source. ``compile_solver`` controls the one-time Torch compilation
    of the Kerr integrations and does not alter the physical model.
    """

    black_hole_mass_solar: float
    eddington_ratio: float
    wavelengths_angstrom: tuple[float, ...] = ()
    band_names: tuple[str, ...] = ()
    bands_angstrom: Mapping[str, float] | None = None
    spin: float = 0.0
    inclination_deg: float = 30.0
    position_angle_deg: float = 0.0
    source_redshift: float | None = None
    color_correction: float = 1.0
    temperature_slope_beta: float = 0.75
    lamp_fraction: float = 0.1
    corona_height_above_isco_rg: float = 20.0
    driving_signal: DrivingSignal | None = None
    grid: SourceGridConfig = SourceGridConfig()
    source_grid_shape: int | tuple[int, int] | None = None
    enclosed_flux_fraction: float | None = None
    source_margin: float | None = None
    compile_solver: bool = True
    primary_repair_max_passes: int = 8
    lamppost_nalpha: int = 1024
    lamppost_radial_bins: int = 512
    name: str = "kerr_thin_disk"

    def __post_init__(self) -> None:
        wavelengths, names = _resolve_bands(
            self.wavelengths_angstrom,
            self.band_names,
            self.bands_angstrom,
        )
        object.__setattr__(self, "wavelengths_angstrom", wavelengths)
        object.__setattr__(self, "band_names", names)
        object.__setattr__(
            self,
            "grid",
            _resolve_sampling(
                self.grid,
                source_grid_shape=self.source_grid_shape,
                enclosed_flux_fraction=self.enclosed_flux_fraction,
                source_margin=self.source_margin,
            ),
        )
        support = ThinDiskModel(
            black_hole_mass_solar=self.black_hole_mass_solar,
            eddington_ratio=self.eddington_ratio,
            wavelengths_angstrom=self.wavelengths_angstrom,
            band_names=self.band_names,
            source_redshift=self.source_redshift,
            spin=self.spin,
            inclination_deg=self.inclination_deg,
            position_angle_deg=self.position_angle_deg,
            color_correction=self.color_correction,
            temperature_slope_beta=self.temperature_slope_beta,
            support_lamp_fraction=self.lamp_fraction,
            support_corona_height_above_isco_rg=(self.corona_height_above_isco_rg),
            grid=self.grid,
        )
        del support
        if not -0.998 <= float(self.spin) <= 0.998:
            raise ValueError("spin must lie in [-0.998, 0.998]")
        if not 0.0 <= float(self.inclination_deg) < 90.0:
            raise ValueError("inclination_deg must lie in [0, 90)")
        if self.lamp_fraction < 0.0:
            raise ValueError("lamp_fraction must be non-negative")
        if self.corona_height_above_isco_rg < 0.0:
            raise ValueError("corona_height_above_isco_rg must be non-negative")
        if self.primary_repair_max_passes < 0:
            raise ValueError("primary_repair_max_passes must be non-negative")
        if self.lamppost_nalpha < 16:
            raise ValueError("lamppost_nalpha must be at least 16")
        if self.lamppost_radial_bins < 16:
            raise ValueError("lamppost_radial_bins must be at least 16")

    def _redshift(self, distances: LensingDistances) -> float:
        value = (
            distances.source_redshift
            if self.source_redshift is None
            else self.source_redshift
        )
        if value is None:
            raise ValueError(
                "KerrDiskModel requires a source redshift. Construct distances "
                "with LensingDistances.from_redshifts or supply source_redshift"
            )
        return float(value)

    def _support_model(self) -> ThinDiskModel:
        return ThinDiskModel(
            black_hole_mass_solar=self.black_hole_mass_solar,
            eddington_ratio=self.eddington_ratio,
            wavelengths_angstrom=self.wavelengths_angstrom,
            band_names=self.band_names,
            source_redshift=self.source_redshift,
            spin=self.spin,
            inclination_deg=self.inclination_deg,
            position_angle_deg=self.position_angle_deg,
            color_correction=self.color_correction,
            temperature_slope_beta=self.temperature_slope_beta,
            support_lamp_fraction=self.lamp_fraction,
            support_corona_height_above_isco_rg=(self.corona_height_above_isco_rg),
            grid=self.grid,
        )

    def support_radius_m(
        self,
        distances: LensingDistances,
        policy: SourceGridConfig | None = None,
    ) -> float:
        """Return the common outer support of every requested band."""

        return self._support_model().support_radius_m(distances, policy)

    def with_driving_signal(self, signal: DrivingSignal | None) -> KerrDiskModel:
        """Return the same physical disk with a different variability driver."""

        return replace(self, driving_signal=signal)

    def recommended_grid(
        self,
        distances: LensingDistances,
        policy: SourceGridConfig | None = None,
    ) -> PlaneGrid:
        """Return the square angular field used by the observer screen."""

        return self._support_model().recommended_grid(distances, policy)

    def pixelate(
        self,
        distances: LensingDistances,
        *,
        grid: PlaneGrid | None = None,
        policy: SourceGridConfig | None = None,
        runtime: RuntimeConfig | ResolvedRuntime | None = None,
    ) -> TransferredThinDiskSource | ThermalReprocessingSource:
        """Trace and materialize the full-Kerr source on one angular grid."""

        resolved_grid = (
            self.recommended_grid(distances, policy) if grid is None else grid
        )
        dy_uas, dx_uas = resolved_grid.pixel_scale_uas
        if not math.isclose(dy_uas, dx_uas, rel_tol=1.0e-10, abs_tol=0.0):
            raise ValueError("KerrDiskModel requires square angular pixels")
        resolved_runtime = (
            runtime
            if isinstance(runtime, ResolvedRuntime)
            else resolve_runtime(runtime)
        )
        redshift = self._redshift(distances)
        gravitational_radius_m = _G * _M_SUN / _C**2 * float(self.black_hole_mass_solar)
        fov_m = distances.uas_to_source_length(
            resolved_grid.field_of_view_uas,
            dtype=torch.float64,
        )
        half_width_rg = 0.5 * float(max(fov_m)) / gravitational_radius_m
        screen = ObserverScreen.uniform(
            resolved_grid.shape,
            half_width_rg,
            gravitational_radius_m=gravitational_radius_m,
            observer_distance_m=distances.source_m,
            device=resolved_runtime.device,
            dtype=resolved_runtime.dtype,
        ).rotated(self.position_angle_deg)
        primary = trace_primary_equatorial(
            screen,
            spin=self.spin,
            inclination_deg=self.inclination_deg,
            disk_outer_rg=half_width_rg,
            compile_solver=(
                self.compile_solver and resolved_runtime.device.type == "cuda"
            ),
            repair_max_passes=self.primary_repair_max_passes,
        )
        pixel_scale_m = distances.uas_to_source_length(
            resolved_grid.pixel_scale_uas,
            dtype=torch.float64,
        )
        geometry = SourceGeometry(
            resolved_grid.shape,
            (float(pixel_scale_m[0]), float(pixel_scale_m[1])),
            tuple(float(value) for value in self.wavelengths_angstrom),
            tuple(self.band_names),
        )
        if self.driving_signal is None:
            return TransferredThinDiskSource(
                geometry,
                primary.transfer,
                black_hole_mass_solar=self.black_hole_mass_solar,
                eddington_ratio=self.eddington_ratio,
                spin=self.spin,
                source_redshift=redshift,
                color_correction=self.color_correction,
                temperature_slope_beta=self.temperature_slope_beta,
                name=self.name,
            )
        coordinates = add_observer_coordinates(
            primary,
            screen,
            black_hole_mass_solar=self.black_hole_mass_solar,
            spin=self.spin,
            inclination_deg=self.inclination_deg,
            source_redshift=redshift,
            coordinate_dtype=resolved_runtime.dtype,
        )
        profile = axis_lamppost_profile(
            spin=self.spin,
            height_above_isco_rg=self.corona_height_above_isco_rg,
            disk_outer_rg=half_width_rg,
            nalpha=self.lamppost_nalpha,
            radial_bins=self.lamppost_radial_bins,
            device=resolved_runtime.device,
            dtype=resolved_runtime.dtype,
            compile_solver=(
                self.compile_solver and resolved_runtime.device.type == "cuda"
            ),
        )
        return ThermalReprocessingSource.from_axis_lamppost(
            geometry,
            coordinates.transfer,
            self.driving_signal,
            profile,
            black_hole_mass_solar=self.black_hole_mass_solar,
            eddington_ratio=self.eddington_ratio,
            spin=self.spin,
            source_redshift=redshift,
            lamp_fraction=self.lamp_fraction,
            color_correction=self.color_correction,
            temperature_slope_beta=self.temperature_slope_beta,
            name=self.name,
        )


@dataclass(frozen=True)
class GaussianModel:
    """A physical elliptical Gaussian without a predetermined pixel grid.

    ``sigma_m`` may contain one major-axis width per band. The widest band
    defines the common field. The optional central hole changes the brightness
    profile but not the conservative outer support calculation. ``total_flux``
    is the observed integrated spectral flux density in Jy, either shared by
    all bands or specified once per band.
    """

    sigma_m: float | tuple[float, ...]
    wavelengths_angstrom: tuple[float, ...] = ()
    band_names: tuple[str, ...] = ()
    bands_angstrom: Mapping[str, float] | None = None
    total_flux: float | tuple[float, ...] = 1.0
    axis_ratio: float = 1.0
    position_angle_rad: float = 0.0
    center_m: tuple[float, float] = (0.0, 0.0)
    hole_radius_m: float = 0.0
    hole_power: float = 4.0
    grid: SourceGridConfig = SourceGridConfig()
    source_grid_shape: int | tuple[int, int] | None = None
    enclosed_flux_fraction: float | None = None
    source_margin: float | None = None
    name: str = "gaussian"

    @classmethod
    def from_angular(
        cls,
        distances: LensingDistances,
        *,
        sigma_uas: float | tuple[float, ...],
        bands_angstrom: Mapping[str, float] | None = None,
        wavelengths_angstrom: tuple[float, ...] = (),
        band_names: tuple[str, ...] = (),
        center_uas: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> GaussianModel:
        """Construct a Gaussian from angular widths in microarcseconds.

        This convenience avoids manual angular-to-physical conversions in
        observational workflows. The stored source remains physical and can
        be inspected through the ordinary ``sigma_m`` and ``center_m`` fields.
        """

        widths = torch.as_tensor(sigma_uas, dtype=torch.float64)
        centers = torch.as_tensor(center_uas, dtype=torch.float64)
        width_m = distances.uas_to_source_length(widths, dtype=torch.float64)
        center_m = distances.uas_to_source_length(centers, dtype=torch.float64)
        return cls(
            sigma_m=(
                float(width_m)
                if width_m.ndim == 0
                else tuple(float(value) for value in width_m.reshape(-1))
            ),
            bands_angstrom=bands_angstrom,
            wavelengths_angstrom=wavelengths_angstrom,
            band_names=band_names,
            center_m=tuple(float(value) for value in center_m.reshape(-1)),
            **kwargs,
        )

    def __post_init__(self) -> None:
        wavelengths, names = _resolve_bands(
            self.wavelengths_angstrom,
            self.band_names,
            self.bands_angstrom,
        )
        object.__setattr__(self, "wavelengths_angstrom", wavelengths)
        object.__setattr__(self, "band_names", names)
        object.__setattr__(
            self,
            "grid",
            _resolve_sampling(
                self.grid,
                source_grid_shape=self.source_grid_shape,
                enclosed_flux_fraction=self.enclosed_flux_fraction,
                source_margin=self.source_margin,
            ),
        )
        widths = (
            (float(self.sigma_m),)
            if isinstance(self.sigma_m, (int, float))
            else tuple(float(value) for value in self.sigma_m)
        )
        if not widths or any(value <= 0.0 for value in widths):
            raise ValueError("sigma_m must be positive")
        if len(widths) not in (1, len(self.band_names)):
            raise ValueError("sigma_m must be scalar or contain one value per band")
        if not 0.0 < self.axis_ratio <= 1.0:
            raise ValueError("axis_ratio must lie in (0, 1]")
        if len(self.wavelengths_angstrom) != len(self.band_names):
            raise ValueError("wavelength and band-name counts must match")
        if not self.band_names or len(set(self.band_names)) != len(self.band_names):
            raise ValueError("band names must be non-empty and unique")
        if any(value <= 0.0 for value in self.wavelengths_angstrom):
            raise ValueError("wavelengths must be positive")
        if len(self.center_m) != 2:
            raise ValueError("center_m must contain x and y")
        if self.hole_radius_m < 0.0:
            raise ValueError("hole_radius_m must be non-negative")
        if self.hole_power <= 0.0:
            raise ValueError("hole_power must be positive")

    def support_radius_m(
        self,
        distances: LensingDistances,
        policy: SourceGridConfig | None = None,
    ) -> float:
        """Return the major-axis radius enclosing the requested Gaussian flux."""

        del distances
        resolved = self.grid if policy is None else policy
        widths = torch.as_tensor(self.sigma_m, dtype=torch.float64).reshape(-1)
        quantile = math.sqrt(-2.0 * math.log1p(-resolved.enclosed_flux_fraction))
        return float(widths.max()) * quantile * resolved.margin

    def recommended_grid(
        self,
        distances: LensingDistances,
        policy: SourceGridConfig | None = None,
    ) -> PlaneGrid:
        """Return an angular grid enclosing the rotated Gaussian support."""

        resolved = self.grid if policy is None else policy
        radius = self.support_radius_m(distances, resolved)
        cosine = math.cos(self.position_angle_rad)
        sine = math.sin(self.position_angle_rad)
        half_x = math.hypot(radius * cosine, radius * self.axis_ratio * sine)
        half_y = math.hypot(radius * sine, radius * self.axis_ratio * cosine)
        half_x += abs(float(self.center_m[0]))
        half_y += abs(float(self.center_m[1]))
        fov_uas = distances.source_length_to_uas(
            (2.0 * half_y, 2.0 * half_x),
            dtype=torch.float64,
        )
        return PlaneGrid(
            resolved.shape,
            (float(fov_uas[0]), float(fov_uas[1])),
        )

    def pixelate(
        self,
        distances: LensingDistances,
        *,
        grid: PlaneGrid | None = None,
        policy: SourceGridConfig | None = None,
        runtime: RuntimeConfig | ResolvedRuntime | None = None,
    ) -> GaussianSource:
        """Materialize the analytic profile on an angular source grid."""

        del runtime

        resolved_grid = (
            self.recommended_grid(distances, policy) if grid is None else grid
        )
        dy_uas, dx_uas = resolved_grid.pixel_scale_uas
        pixel_scale_m = distances.uas_to_source_length(
            (dy_uas, dx_uas),
            dtype=torch.float64,
        )
        geometry = SourceGeometry(
            shape=resolved_grid.shape,
            pixel_scale_m=(float(pixel_scale_m[0]), float(pixel_scale_m[1])),
            wavelengths_angstrom=tuple(float(v) for v in self.wavelengths_angstrom),
            band_names=tuple(self.band_names),
        )
        return GaussianSource(
            geometry=geometry,
            sigma_m=self.sigma_m,
            total_flux=self.total_flux,
            axis_ratio=self.axis_ratio,
            position_angle_rad=self.position_angle_rad,
            center_m=self.center_m,
            hole_radius_m=self.hole_radius_m,
            hole_power=self.hole_power,
            name=self.name,
        )
