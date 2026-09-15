"""Physical source models that select and materialize their own grids."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import torch

from ..geometry import PlaneGrid
from ..lens import LensingDistances
from ..lens.models import _source_distances_from_redshift
from ..relativity import (
    ObserverScreen,
    add_observer_coordinates,
    axis_lamppost_profile,
    trace_primary_equatorial,
)
from ..relativity.coordinates import add_observer_coordinates_batch
from ..runtime import ResolvedRuntime, RuntimeConfig, resolve_runtime
from .analytic import GaussianSource
from .base import PixelatedSource, SourceGeometry
from .reprocessing import ThermalReprocessingSource
from .thin_disk import (
    RadiativeEfficiency,
    ThinDiskSource,
    ViscousFluxProfile,
    _validate_viscous_prescriptions,
    thin_disk_flux_radius_rg,
)
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
        mapped_wavelengths = tuple(float(value) for value in bands_angstrom.values())
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

    if (
        source_grid_shape is None
        and enclosed_flux_fraction is None
        and source_margin is None
    ):
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


def _pixelate_source(source, distances, *, grid=None, policy=None, runtime=None):
    """Resolve a model once while accepting portable third-party pixelizers."""
    if not isinstance(source, PhysicalSourceModel):
        return source
    parameters = inspect.signature(source.pixelate).parameters.values()
    accepts_runtime = any(
        item.name == "runtime" or item.kind is inspect.Parameter.VAR_KEYWORD
        for item in parameters
    )
    return source.pixelate(
        distances,
        grid=grid,
        **({"policy": policy} if policy is not None else {}),
        **({"runtime": runtime} if accepts_runtime else {}),
    )


def _resolve_source_distances(
    distances,
    *,
    source_redshift=None,
    model_redshift=None,
    H0=None,
    Om0=None,
    runtime=None,
):
    """Bind cosmology once, accepting either system geometry or source-only inputs."""
    if distances is not None:
        if source_redshift is not None or H0 is not None or Om0 is not None:
            raise ValueError("supply distances or source_redshift/H0/Om0, not both")
        if (
            model_redshift is not None
            and distances.source_redshift is not None
            and not math.isclose(
                float(model_redshift), float(distances.source_redshift), rel_tol=1e-7
            )
        ):
            raise ValueError(
                "source redshift differs from the system's source redshift"
            )
        return distances
    if source_redshift is None:
        source_redshift = model_redshift
    elif model_redshift is not None and not math.isclose(
        float(source_redshift), float(model_redshift), rel_tol=1e-7
    ):
        raise ValueError("source_redshift conflicts with the source model")
    if source_redshift is None:
        raise ValueError(
            "supply source_redshift for a standalone source, or pass the model to a MicrolensingSystem"
        )
    resolved = (
        runtime if isinstance(runtime, ResolvedRuntime) else resolve_runtime(runtime)
    )
    return _source_distances_from_redshift(
        source_redshift,
        H0=67.66 if H0 is None else H0,
        Om0=0.30966 if Om0 is None else Om0,
        device=resolved.device,
        dtype=resolved.dtype,
    )


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
    viscous_flux_profile: ViscousFluxProfile = "novikov-thorne"
    radiative_efficiency: RadiativeEfficiency = None
    support_lamp_fraction: float = 0.0
    support_corona_height_above_isco_rg: float = 20.0
    relativity: str = "none"
    grid: SourceGridConfig = SourceGridConfig()
    source_grid_shape: int | tuple[int, int] | None = None
    enclosed_flux_fraction: float | None = None
    source_margin: float | None = None
    name: str = "thin_disk"

    def __post_init__(self) -> None:
        _validate_viscous_prescriptions(
            self.viscous_flux_profile,
            self.radiative_efficiency,
        )
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

    def with_bands(self, bands_angstrom: Mapping[str, float]) -> ThinDiskModel:
        """Return the same physical disk with different observed bands."""

        return replace(
            self,
            wavelengths_angstrom=(),
            band_names=(),
            bands_angstrom=bands_angstrom,
        )

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
            viscous_flux_profile=self.viscous_flux_profile,
            radiative_efficiency=self.radiative_efficiency,
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
        distances: LensingDistances | None = None,
        *,
        source_redshift: float | None = None,
        H0: float | None = None,
        Om0: float | None = None,
        grid: PlaneGrid | None = None,
        policy: SourceGridConfig | None = None,
        runtime: RuntimeConfig | ResolvedRuntime | None = None,
    ) -> ThinDiskSource:
        """Materialize the disk using the validated source calculation.

        Standalone callers supply ``source_redshift``, ``H0`` and ``Om0``.
        A system instead passes its existing ``distances``. These alternatives
        are mutually exclusive. ``grid`` overrides only the pixelization.
        """

        distances = _resolve_source_distances(
            distances,
            source_redshift=source_redshift,
            model_redshift=self.source_redshift,
            H0=H0,
            Om0=Om0,
            runtime=runtime,
        )

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
            viscous_flux_profile=self.viscous_flux_profile,
            radiative_efficiency=self.radiative_efficiency,
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
    lamppost source. ``observer_coordinate_chunk_size`` controls the fixed,
    padded observer-delay launch shape; the default suits typical modern GPUs
    and is reduced automatically after an out-of-memory error. The emission
    azimuth is omitted by default because an axisymmetric disk needs only the
    delay. ``compile_solver`` controls one-time Torch compilation and does not
    alter the physical model.
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
    viscous_flux_profile: ViscousFluxProfile = "novikov-thorne"
    radiative_efficiency: RadiativeEfficiency = None
    lamp_fraction: float = 0.1
    corona_height_above_isco_rg: float = 20.0
    driving_signal: DrivingSignal | None = None
    grid: SourceGridConfig = SourceGridConfig()
    source_grid_shape: int | tuple[int, int] | None = None
    enclosed_flux_fraction: float | None = None
    source_margin: float | None = None
    compile_solver: bool = True
    warn_on_compile: bool = True
    primary_repair_max_passes: int = 8
    observer_coordinate_chunk_size: int = 524_288
    observer_coordinate_repair_device: str = "cpu"
    compute_emission_azimuth: bool = False
    lamppost_nalpha: int = 1024
    lamppost_radial_bins: int = 512
    name: str = "kerr_thin_disk"

    def __post_init__(self) -> None:
        from .variability import _validate_source_driver

        _validate_source_driver(self)
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
            viscous_flux_profile=self.viscous_flux_profile,
            radiative_efficiency=self.radiative_efficiency,
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
        if self.observer_coordinate_chunk_size < 1:
            raise ValueError("observer_coordinate_chunk_size must be positive")
        if self.observer_coordinate_repair_device not in {"cpu", "same"}:
            raise ValueError(
                "observer_coordinate_repair_device must be 'cpu' or 'same'"
            )
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
            viscous_flux_profile=self.viscous_flux_profile,
            radiative_efficiency=self.radiative_efficiency,
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

    def with_bands(self, bands_angstrom: Mapping[str, float]) -> KerrDiskModel:
        """Return the same physical disk with different observed bands."""

        return replace(
            self,
            wavelengths_angstrom=(),
            band_names=(),
            bands_angstrom=bands_angstrom,
        )

    def recommended_grid(
        self,
        distances: LensingDistances,
        policy: SourceGridConfig | None = None,
    ) -> PlaneGrid:
        """Return the square angular field used by the observer screen."""

        return self._support_model().recommended_grid(distances, policy)

    def pixelate(
        self,
        distances: LensingDistances | None = None,
        *,
        source_redshift: float | None = None,
        H0: float | None = None,
        Om0: float | None = None,
        grid: PlaneGrid | None = None,
        policy: SourceGridConfig | None = None,
        runtime: RuntimeConfig | ResolvedRuntime | None = None,
    ) -> TransferredThinDiskSource | ThermalReprocessingSource:
        """Trace and materialize the full-Kerr source on one angular grid.

        Use ``source_redshift`` with optional ``H0`` and ``Om0`` without a
        lens. Systems pass their existing ``distances`` instead. ``runtime``
        controls device, dtype and compilation as in microlensing calls.
        Resolved angular extent is retained in the transfer metadata for plots.
        """

        plan = _prepare_kerr_pixelation(
            self,
            distances,
            source_redshift=source_redshift,
            H0=H0,
            Om0=Om0,
            grid=grid,
            policy=policy,
            runtime=runtime,
        )
        return _finish_kerr_pixelation(plan)


@dataclass(frozen=True)
class _KerrPixelationPlan:
    model: KerrDiskModel
    distances: LensingDistances
    runtime: ResolvedRuntime
    grid: PlaneGrid
    redshift: float
    half_width_rg: float
    screen: ObserverScreen
    primary: object
    geometry: SourceGeometry


def _prepare_kerr_pixelation(
    model: KerrDiskModel,
    distances: LensingDistances | None,
    *,
    source_redshift: float | None = None,
    H0: float | None = None,
    Om0: float | None = None,
    grid: PlaneGrid | None = None,
    policy: SourceGridConfig | None = None,
    runtime: RuntimeConfig | ResolvedRuntime | None = None,
) -> _KerrPixelationPlan:
    """Resolve geometry and primary transfer before optional pooled delays."""

    distances = _resolve_source_distances(
        distances,
        source_redshift=source_redshift,
        model_redshift=model.source_redshift,
        H0=H0,
        Om0=Om0,
        runtime=runtime,
    )
    resolved_grid = model.recommended_grid(distances, policy) if grid is None else grid
    dy_uas, dx_uas = resolved_grid.pixel_scale_uas
    if not math.isclose(dy_uas, dx_uas, rel_tol=1.0e-10, abs_tol=0.0):
        raise ValueError("KerrDiskModel requires square angular pixels")
    resolved_runtime = (
        runtime if isinstance(runtime, ResolvedRuntime) else resolve_runtime(runtime)
    )
    redshift = model._redshift(distances)
    gravitational_radius_m = (
        _G * _M_SUN / _C**2 * float(model.black_hole_mass_solar)
    )
    fov_m = distances.uas_to_source_length(
        resolved_grid.field_of_view_uas, dtype=torch.float64
    )
    half_width_rg = 0.5 * float(max(fov_m)) / gravitational_radius_m
    screen = ObserverScreen.uniform(
        resolved_grid.shape,
        half_width_rg,
        gravitational_radius_m=gravitational_radius_m,
        observer_distance_m=distances.source_m,
        device=resolved_runtime.device,
        dtype=resolved_runtime.dtype,
    ).rotated(model.position_angle_deg)
    primary = trace_primary_equatorial(
        screen,
        spin=model.spin,
        inclination_deg=model.inclination_deg,
        disk_outer_rg=half_width_rg,
        compile_solver=(model.compile_solver and resolved_runtime.device.type == "cuda"),
        repair_max_passes=model.primary_repair_max_passes,
        compile_mode=(resolved_runtime.torch_compile_mode or "reduce-overhead"),
        warn_on_compile=(model.warn_on_compile and resolved_runtime.warn_on_compile),
    )
    primary = replace(
        primary,
        transfer=replace(
            primary.transfer,
            metadata={
                **primary.transfer.metadata,
                "source_field_of_view_uas": tuple(resolved_grid.field_of_view_uas),
            },
        ),
    )
    pixel_scale_m = distances.uas_to_source_length(
        resolved_grid.pixel_scale_uas, dtype=torch.float64
    )
    geometry = SourceGeometry(
        resolved_grid.shape,
        (float(pixel_scale_m[0]), float(pixel_scale_m[1])),
        tuple(float(value) for value in model.wavelengths_angstrom),
        tuple(model.band_names),
    )
    return _KerrPixelationPlan(
        model,
        distances,
        resolved_runtime,
        resolved_grid,
        redshift,
        half_width_rg,
        screen,
        primary,
        geometry,
    )


def _finish_kerr_pixelation(
    plan: _KerrPixelationPlan,
    coordinates=None,
) -> TransferredThinDiskSource | ThermalReprocessingSource:
    """Construct one source from its primary and optional prepared coordinates."""

    model = plan.model
    if model.driving_signal is None:
        return TransferredThinDiskSource(
            plan.geometry,
            plan.primary.transfer,
            black_hole_mass_solar=model.black_hole_mass_solar,
            eddington_ratio=model.eddington_ratio,
            spin=model.spin,
            source_redshift=plan.redshift,
            color_correction=model.color_correction,
            temperature_slope_beta=model.temperature_slope_beta,
            viscous_flux_profile=model.viscous_flux_profile,
            radiative_efficiency=model.radiative_efficiency,
            name=model.name,
        )
    if coordinates is None:
        coordinates = add_observer_coordinates(
            plan.primary,
            plan.screen,
            black_hole_mass_solar=model.black_hole_mass_solar,
            spin=model.spin,
            inclination_deg=model.inclination_deg,
            source_redshift=plan.redshift,
            coordinate_dtype=plan.runtime.dtype,
            compute_emission_azimuth=model.compute_emission_azimuth,
            chunk_size=model.observer_coordinate_chunk_size,
            compile_solver=(model.compile_solver and plan.runtime.device.type == "cuda"),
            compile_mode=(plan.runtime.torch_compile_mode or "reduce-overhead"),
            fallback_to_eager=not plan.runtime.strict_backend,
            warn_on_compile=(model.warn_on_compile and plan.runtime.warn_on_compile),
            repair_device=model.observer_coordinate_repair_device,
        )
    profile = axis_lamppost_profile(
        spin=model.spin,
        height_above_isco_rg=model.corona_height_above_isco_rg,
        disk_outer_rg=plan.half_width_rg,
        nalpha=model.lamppost_nalpha,
        radial_bins=model.lamppost_radial_bins,
        device=plan.runtime.device,
        dtype=plan.runtime.dtype,
        compile_solver=(model.compile_solver and plan.runtime.device.type == "cuda"),
        compile_mode=(plan.runtime.torch_compile_mode or "reduce-overhead"),
        warn_on_compile=(model.warn_on_compile and plan.runtime.warn_on_compile),
    )
    return ThermalReprocessingSource.from_axis_lamppost(
        plan.geometry,
        coordinates.transfer,
        model.driving_signal,
        profile,
        black_hole_mass_solar=model.black_hole_mass_solar,
        eddington_ratio=model.eddington_ratio,
        spin=model.spin,
        source_redshift=plan.redshift,
        lamp_fraction=model.lamp_fraction,
        color_correction=model.color_correction,
        temperature_slope_beta=model.temperature_slope_beta,
        viscous_flux_profile=model.viscous_flux_profile,
        radiative_efficiency=model.radiative_efficiency,
        name=model.name,
    )


def batched_pixelate_sources(
    models: Sequence[PhysicalSourceModel | PixelatedSource],
    distances: LensingDistances | Sequence[LensingDistances],
    *,
    batch_size: int = 3,
    grids: Sequence[PlaneGrid | None] | None = None,
    runtime: RuntimeConfig | ResolvedRuntime | None = None,
    oom_backoff: bool = True,
) -> tuple[PixelatedSource, ...]:
    """Materialize physical sources while pooling compatible Kerr delay rays.

    Non-Kerr, static, azimuth-dependent, and numerically incompatible models
    retain their ordinary serial implementations. The result order always
    matches ``models``. CUDA OOM reduces only the source-setup batch size.
    """

    models = tuple(models)
    if not models:
        return ()
    requested = int(batch_size)
    if requested < 1:
        raise ValueError("batch_size must be positive")
    distance_values = (
        (distances,) * len(models)
        if isinstance(distances, LensingDistances)
        else tuple(distances)
    )
    if len(distance_values) != len(models):
        raise ValueError("distances must be shared or match models")
    grid_values = (None,) * len(models) if grids is None else tuple(grids)
    if len(grid_values) != len(models):
        raise ValueError("grids must match models")
    outputs: list[PixelatedSource | None] = [None] * len(models)
    current = requested
    start = 0
    while start < len(models):
        stop = min(start + current, len(models))
        selected = models[start:stop]
        try:
            plans = tuple(
                _prepare_kerr_pixelation(
                    model,
                    distance,
                    grid=grid,
                    runtime=runtime,
                )
                if isinstance(model, KerrDiskModel)
                else None
                for model, distance, grid in zip(
                    selected,
                    distance_values[start:stop],
                    grid_values[start:stop],
                    strict=True,
                )
            )
            buckets: dict[tuple[object, ...], list[int]] = {}
            for local, plan in enumerate(plans):
                if plan is None:
                    continue
                model = plan.model
                signature = (
                    model.driving_signal is not None,
                    not model.compute_emission_azimuth,
                    plan.grid.shape,
                    plan.runtime.device,
                    plan.runtime.dtype,
                    model.observer_coordinate_chunk_size,
                    model.observer_coordinate_repair_device,
                    plan.runtime.torch_compile_mode,
                    model.compile_solver,
                )
                buckets.setdefault(signature, []).append(local)
            for members in buckets.values():
                member_plans = tuple(plans[index] for index in members)
                assert all(plan is not None for plan in member_plans)
                driven = member_plans[0].model.driving_signal is not None
                can_pool = (
                    driven
                    and not member_plans[0].model.compute_emission_azimuth
                    and len(member_plans) > 1
                )
                coordinates = (None,) * len(member_plans)
                if can_pool:
                    first = member_plans[0]
                    coordinates = add_observer_coordinates_batch(
                        tuple(plan.primary for plan in member_plans),
                        tuple(plan.screen for plan in member_plans),
                        black_hole_masses_solar=tuple(
                            plan.model.black_hole_mass_solar for plan in member_plans
                        ),
                        spins=tuple(plan.model.spin for plan in member_plans),
                        inclinations_deg=tuple(
                            plan.model.inclination_deg for plan in member_plans
                        ),
                        source_redshifts=tuple(plan.redshift for plan in member_plans),
                        chunk_size=first.model.observer_coordinate_chunk_size,
                        compile_solver=(
                            first.model.compile_solver
                            and first.runtime.device.type == "cuda"
                        ),
                        compile_mode=(
                            first.runtime.torch_compile_mode or "reduce-overhead"
                        ),
                        fallback_to_eager=not first.runtime.strict_backend,
                        warn_on_compile=(
                            first.model.warn_on_compile
                            and first.runtime.warn_on_compile
                        ),
                        repair_device=first.model.observer_coordinate_repair_device,
                    )
                for member, plan, coordinate in zip(
                    members, member_plans, coordinates, strict=True
                ):
                    outputs[start + member] = _finish_kerr_pixelation(plan, coordinate)
            for local, (model, distance, grid, plan) in enumerate(
                zip(
                    selected,
                    distance_values[start:stop],
                    grid_values[start:stop],
                    plans,
                    strict=True,
                )
            ):
                if plan is None:
                    outputs[start + local] = _pixelate_source(
                        model, distance, grid=grid, runtime=runtime
                    )
            start = stop
        except torch.OutOfMemoryError:
            if not oom_backoff or current == 1:
                raise
            current = max(1, current // 2)
            torch.cuda.empty_cache()
    if any(output is None for output in outputs):
        raise RuntimeError("batched source setup did not produce every requested source")
    return cast(tuple[PixelatedSource, ...], tuple(outputs))


@dataclass(frozen=True)
class GaussianModel:
    """A physical elliptical Gaussian without a predetermined pixel grid.

    Supply ``sigma_uas`` or ``sigma_m``, with one major-axis width per band
    or a shared scalar. Angular parameters are resolved by the system. The widest band
    defines the common field. The optional central hole changes the brightness
    profile but not the conservative outer support calculation. ``total_flux``
    is the observed integrated spectral flux density in Jy, either shared by
    all bands or specified once per band.
    """

    sigma_m: float | tuple[float, ...] | None = None
    wavelengths_angstrom: tuple[float, ...] = ()
    band_names: tuple[str, ...] = ()
    bands_angstrom: Mapping[str, float] | None = None
    total_flux: float | tuple[float, ...] = 1.0
    axis_ratio: float = 1.0
    position_angle_deg: float = 0.0
    center_m: tuple[float, float] | None = None
    hole_radius_m: float | None = None
    hole_power: float = 4.0
    grid: SourceGridConfig = SourceGridConfig()
    source_grid_shape: int | tuple[int, int] | None = None
    enclosed_flux_fraction: float | None = None
    source_margin: float | None = None
    name: str = "gaussian"
    sigma_uas: float | tuple[float, ...] | None = None
    center_uas: tuple[float, float] | None = None
    hole_radius_uas: float | None = None

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
        if (self.sigma_m is None) == (self.sigma_uas is None):
            raise ValueError("supply exactly one of sigma_uas or sigma_m")
        width = self.sigma_m if self.sigma_uas is None else self.sigma_uas
        widths = (
            (float(width),)
            if isinstance(width, int | float)
            else tuple(float(value) for value in width)
        )
        if not widths or any(
            not math.isfinite(value) or value <= 0.0 for value in widths
        ):
            raise ValueError("Gaussian widths must be finite and positive")
        if len(widths) not in (1, len(self.band_names)):
            raise ValueError(
                "Gaussian widths must be scalar or contain one value per band"
            )
        if not 0.0 < self.axis_ratio <= 1.0:
            raise ValueError("axis_ratio must lie in (0, 1]")
        if len(self.wavelengths_angstrom) != len(self.band_names):
            raise ValueError("wavelength and band-name counts must match")
        if not self.band_names or len(set(self.band_names)) != len(self.band_names):
            raise ValueError("band names must be non-empty and unique")
        if any(value <= 0.0 for value in self.wavelengths_angstrom):
            raise ValueError("wavelengths must be positive")
        if not math.isfinite(self.position_angle_deg):
            raise ValueError("position_angle_deg must be finite")
        if self.center_m is not None and self.center_uas is not None:
            raise ValueError("supply center_uas or center_m, not both")
        if self.hole_radius_m is not None and self.hole_radius_uas is not None:
            raise ValueError("supply hole_radius_uas or hole_radius_m, not both")
        for name in ("center_m", "center_uas"):
            center = getattr(self, name)
            if center is not None and (
                len(center) != 2 or any(not math.isfinite(v) for v in center)
            ):
                raise ValueError(f"{name} must contain finite x and y coordinates")
        for name in ("hole_radius_m", "hole_radius_uas"):
            radius = getattr(self, name)
            if radius is not None and (not math.isfinite(radius) or radius < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.hole_power <= 0.0:
            raise ValueError("hole_power must be positive")

    def _physical_parameters(self, distances):
        """Convert angular inputs during setup without changing this model."""
        widths = self.sigma_m
        if widths is None:
            converted = distances.uas_to_source_length(
                self.sigma_uas, dtype=torch.float64
            )
            widths = (
                float(converted)
                if converted.ndim == 0
                else tuple(float(v) for v in converted)
            )
        center = (0.0, 0.0) if self.center_m is None else self.center_m
        if self.center_uas is not None:
            center = tuple(
                float(v)
                for v in distances.uas_to_source_length(
                    self.center_uas, dtype=torch.float64
                )
            )
        hole = 0.0 if self.hole_radius_m is None else self.hole_radius_m
        if self.hole_radius_uas is not None:
            hole = float(
                distances.uas_to_source_length(
                    self.hole_radius_uas, dtype=torch.float64
                )
            )
        return widths, center, hole

    def support_radius_m(
        self,
        distances: LensingDistances,
        policy: SourceGridConfig | None = None,
    ) -> float:
        """Return the major-axis radius enclosing the requested Gaussian flux."""

        resolved = self.grid if policy is None else policy
        widths = torch.as_tensor(
            self._physical_parameters(distances)[0], dtype=torch.float64
        ).reshape(-1)
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
        cosine = math.cos(math.radians(self.position_angle_deg))
        sine = math.sin(math.radians(self.position_angle_deg))
        half_x = math.hypot(radius * cosine, radius * self.axis_ratio * sine)
        half_y = math.hypot(radius * sine, radius * self.axis_ratio * cosine)
        center = self._physical_parameters(distances)[1]
        half_x += abs(float(center[0]))
        half_y += abs(float(center[1]))
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
        distances: LensingDistances | None = None,
        *,
        source_redshift: float | None = None,
        H0: float | None = None,
        Om0: float | None = None,
        grid: PlaneGrid | None = None,
        policy: SourceGridConfig | None = None,
        runtime: RuntimeConfig | ResolvedRuntime | None = None,
    ) -> GaussianSource:
        """Materialize the analytic profile on an angular source grid.

        Pass either system ``distances`` or standalone ``source_redshift``
        with optional ``H0`` and ``Om0``. No lens parameters are needed for
        source-only evaluation. Angular inputs are resolved once at setup.
        """

        distances = _resolve_source_distances(
            distances,
            source_redshift=source_redshift,
            model_redshift=None,
            H0=H0,
            Om0=Om0,
            runtime=runtime,
        )

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
        widths, center, hole = self._physical_parameters(distances)
        return GaussianSource(
            geometry=geometry,
            sigma_m=widths,
            total_flux=self.total_flux,
            axis_ratio=self.axis_ratio,
            position_angle_rad=math.radians(self.position_angle_deg),
            center_m=center,
            hole_radius_m=hole,
            hole_power=self.hole_power,
            name=self.name,
        )
