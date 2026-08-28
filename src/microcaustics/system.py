"""High-level physical microlensing systems and their realizations."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

import torch

from .config import (
    CausticConfig,
    DynamicConfig,
    IPMConfig,
    IRSConfig,
    RuntimeConfig,
    production_ipm_config,
)
from .geometry import PlaneGrid, PlaneRegion
from .lens import (
    LensingDistances,
    MacroLens,
    PointMassField,
    rectangular_lens_region,
)
from .lens.stellar import (
    StellarAperture,
    StellarPopulation,
    circular_stellar_aperture,
)
from .runtime import ResolvedRuntime, resolve_runtime
from .simulation import MicrolensingSimulation
from .sources import PhysicalSourceModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .results import LightCurve, MagnificationMap
    from .sources import PixelatedSource
    from .trajectories import SourceTrajectory


class IntegrationDomain(str, Enum):
    """Lens-plane region evaluated by a map construction method."""

    SCOUT = "scout"
    FULL = "full"
    RECTANGLE = "rectangle"


def _retaining_map_observer(
    times_days: Sequence[float],
    keep_maps_at_days: Sequence[float] | None,
    map_observer,
):
    """Compose optional map retention with an existing streaming observer."""

    retained: dict[float, MagnificationMap] = {}
    if keep_maps_at_days is None:
        return map_observer, retained
    times = torch.as_tensor(times_days, dtype=torch.float64).reshape(-1)
    requested = tuple(float(value) for value in keep_maps_at_days)
    indices: dict[int, float] = {}
    for requested_time in requested:
        differences = torch.abs(times - requested_time)
        index = int(torch.argmin(differences))
        tolerance = max(1.0e-6, 1.0e-8 * max(1.0, abs(requested_time)))
        if float(differences[index]) > tolerance:
            raise ValueError(
                f"requested retained-map epoch {requested_time} is not in times_days"
            )
        indices[index] = float(times[index])

    def observer(index, frame):
        if map_observer is not None:
            map_observer(index, frame)
        if index in indices:
            retained[indices[index]] = getattr(frame, "magnification_map", frame)

    return observer, retained


def _source_grid_from_model(
    source: PixelatedSource,
    distances: LensingDistances,
) -> PlaneGrid:
    """Convert a pixelated source's physical geometry to an angular grid."""

    geometry = source.geometry
    fov_m = (
        float(geometry.shape[0]) * float(geometry.pixel_scale_m[0]),
        float(geometry.shape[1]) * float(geometry.pixel_scale_m[1]),
    )
    fov_uas = distances.source_length_to_uas(fov_m, dtype=torch.float64)
    return PlaneGrid(
        shape=geometry.shape,
        field_of_view_uas=(float(fov_uas[0]), float(fov_uas[1])),
    )


def _direct_star_region(
    stars: PointMassField,
    macro: MacroLens,
    source_region: PlaneRegion,
) -> PlaneRegion:
    """Infer a conservative integration box for directly supplied stars."""

    angle = 2.0 * macro.shear_angle_rad
    gamma_1 = macro.shear * math.cos(angle)
    gamma_2 = macro.shear * math.sin(angle)
    a_xx = 1.0 - macro.convergence - gamma_1
    a_xy = -gamma_2
    a_yy = 1.0 - macro.convergence + gamma_1
    determinant = a_xx * a_yy - a_xy * a_xy
    if abs(determinant) <= 1.0e-12:
        raise ValueError("macro-lens matrix is too close to singular")
    inverse_xx = a_yy / determinant
    inverse_xy = -a_xy / determinant
    inverse_yy = a_xx / determinant

    source_fov_y, source_fov_x = source_region.field_of_view_uas
    source_center_y, source_center_x = source_region.center_uas
    half_source_x = 0.5 * source_fov_x
    half_source_y = 0.5 * source_fov_y
    center_x = inverse_xx * source_center_x + inverse_xy * source_center_y
    center_y = inverse_xy * source_center_x + inverse_yy * source_center_y
    half_x = (
        abs(inverse_xx) * half_source_x
        + abs(inverse_xy) * half_source_y
    )
    half_y = (
        abs(inverse_xy) * half_source_x
        + abs(inverse_yy) * half_source_y
    )
    xmin, xmax = center_x - half_x, center_x + half_x
    ymin, ymax = center_y - half_y, center_y + half_y
    if len(stars):
        star_x = stars.x_uas.detach().cpu()
        star_y = stars.y_uas.detach().cpu()
        xmin = min(xmin, float(star_x.min()))
        xmax = max(xmax, float(star_x.max()))
        ymin = min(ymin, float(star_y.min()))
        ymax = max(ymax, float(star_y.max()))
    padding = 0.05 * max(xmax - xmin, ymax - ymin)
    return PlaneRegion(
        (ymax - ymin + 2.0 * padding, xmax - xmin + 2.0 * padding),
        (0.5 * (ymin + ymax), 0.5 * (xmin + xmax)),
    )


@dataclass(frozen=True)
class MicrolensingRealization:
    """One seeded stellar and numerical realization of a physical system."""

    system: MicrolensingSystem
    simulation: MicrolensingSimulation
    source: PixelatedSource | None
    source_grid: PlaneGrid
    lens_region: PlaneRegion
    lens_grid: PlaneGrid
    stellar_aperture: StellarAperture | None
    source_support_radius_uas: float | None = None

    @property
    def source_region(self) -> PlaneRegion:
        """Return the source-plane field without its pixelization."""

        return self.source_grid.region

    @property
    def stars(self) -> PointMassField:
        """Return the realized point-mass field."""

        return self.simulation.point_masses

    def metadata(self) -> Mapping[str, object]:
        """Return serializable physical and derived-geometry provenance."""

        runtime = self.simulation.runtime
        return {
            "integration_domain": self.system.integration_domain.value,
            "seed": self.system.seed,
            "duration_days": float(self.system.duration_days),
            "light_loss": float(self.system.light_loss),
            "rectangle_light_loss": float(
                self.system.light_loss
                if self.system.rectangle_light_loss is None
                else self.system.rectangle_light_loss
            ),
            "safety_scale": float(self.system.safety_scale),
            "stellar_motion_sigma_margin": float(
                self.system.stellar_motion_sigma_margin
            ),
            "source_support_radius_uas": self.source_support_radius_uas,
            "source_grid": {
                "shape": list(self.source_grid.shape),
                "field_of_view_uas": list(self.source_grid.field_of_view_uas),
                "center_uas": list(self.source_grid.center_uas),
            },
            "lens_region": {
                "field_of_view_uas": list(self.lens_region.field_of_view_uas),
                "center_uas": list(self.lens_region.center_uas),
            },
            "stellar_aperture": (
                None
                if self.stellar_aperture is None
                else {
                    "radius_uas": float(self.stellar_aperture.radius_uas),
                    "center_uas": list(self.stellar_aperture.center_uas),
                }
            ),
            "star_count": len(self.stars),
            "stellar_population": (
                None
                if self.system.stellar_population is None
                else self.system.stellar_population.metadata()
            ),
            "runtime": {
                "device": str(runtime.device),
                "dtype": str(runtime.dtype),
                "backend": runtime.backend.value,
                "profiling": runtime.profiling.value,
            },
            "source": (
                None
                if self.source is None
                else dict(self.source.metadata())
            ),
        }

    def magnification_map(
        self,
        *,
        method: IRSConfig | IPMConfig | None = None,
        time_days: float = 0.0,
    ) -> MagnificationMap:
        """Generate one source-independent magnification map."""

        resolved_method = self._method_for_domain(
            production_ipm_config(dynamic=False) if method is None else method
        )
        return self.simulation.magnification_map(
            self.lens_region,
            self.source_grid,
            method=resolved_method,
            time_days=time_days,
        )

    def dynamic_maps(
        self,
        times_days: Sequence[float],
        *,
        method: IRSConfig | IPMConfig | None = None,
        schedule: DynamicConfig | None = None,
    ):
        """Stream source-independent maps for a moving stellar realization."""

        resolved_method = self._method_for_domain(
            production_ipm_config(dynamic=True) if method is None else method
        )
        return self.simulation.dynamic_maps(
            self.lens_region,
            self.source_grid,
            times_days,
            method=resolved_method,
            schedule=schedule,
        )

    def dynamic_labeled_maps(
        self,
        times_days: Sequence[float],
        *,
        method: IRSConfig | IPMConfig | None = None,
        schedule: DynamicConfig | None = None,
        caustics: CausticConfig | None = None,
        diagnostic_grid: PlaneGrid | None = None,
        include_distance_map: bool = False,
    ):
        """Stream maps with same-epoch caustics and label products."""

        resolved_method = self._method_for_domain(
            production_ipm_config(dynamic=True) if method is None else method
        )
        return self.simulation.dynamic_labeled_maps(
            self.lens_region,
            self.source_grid,
            self.lens_grid,
            times_days,
            method=resolved_method,
            map_schedule=schedule,
            caustic_config=CausticConfig() if caustics is None else caustics,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
        )

    def light_curve(
        self,
        times_days: Sequence[float],
        *,
        source: PixelatedSource | None = None,
        method: IRSConfig | IPMConfig | None = None,
        trajectory: SourceTrajectory | None = None,
        schedule: DynamicConfig | None = None,
        strict_coverage: bool = True,
        map_observer=None,
        keep_maps_at_days: Sequence[float] | None = None,
    ) -> LightCurve:
        """Generate a finite-source light curve for this realization."""

        resolved_source = self.source if source is None else source
        if resolved_source is None:
            raise ValueError("light_curve requires a source model")
        resolved_method = self._method_for_domain(
            production_ipm_config(dynamic=True) if method is None else method
        )
        observer, retained = _retaining_map_observer(
            times_days, keep_maps_at_days, map_observer
        )
        result = self.simulation.light_curve(
            self.lens_region,
            self.source_grid,
            times_days,
            resolved_source,
            self.system.distances,
            method=resolved_method,
            trajectory=trajectory,
            schedule=schedule,
            strict_coverage=strict_coverage,
            map_observer=observer,
        )
        return replace(result, maps=retained)

    def light_curves(
        self,
        times_days: Sequence[float],
        requests,
        *,
        method: IRSConfig | IPMConfig | None = None,
        schedule: DynamicConfig | None = None,
        map_observer=None,
    ) -> tuple[LightCurve, ...]:
        """Batch multiple sources or trajectories through one map sequence."""

        resolved_method = self._method_for_domain(
            production_ipm_config(dynamic=True) if method is None else method
        )
        return self.simulation.light_curves(
            self.lens_region,
            self.source_grid,
            times_days,
            requests,
            method=resolved_method,
            schedule=schedule,
            map_observer=map_observer,
        )

    def multirate_light_curve(
        self,
        map_times_days: Sequence[float],
        flux_times_days: Sequence[float],
        *,
        source: PixelatedSource | None = None,
        method: IRSConfig | IPMConfig | None = None,
        trajectory: SourceTrajectory | None = None,
        schedule: DynamicConfig | None = None,
        strict_coverage: bool = True,
        map_observer=None,
        keep_maps_at_days: Sequence[float] | None = None,
    ) -> LightCurve:
        """Use sparse dynamic maps with independently sampled source evolution."""

        resolved_source = self.source if source is None else source
        if resolved_source is None:
            raise ValueError("multirate_light_curve requires a source model")
        resolved_method = self._method_for_domain(
            production_ipm_config(dynamic=True) if method is None else method
        )
        observer, retained = _retaining_map_observer(
            map_times_days, keep_maps_at_days, map_observer
        )
        result = self.simulation.multirate_light_curve(
            self.lens_region,
            self.source_grid,
            map_times_days,
            flux_times_days,
            resolved_source,
            self.system.distances,
            method=resolved_method,
            trajectory=trajectory,
            schedule=schedule,
            strict_coverage=strict_coverage,
            map_observer=observer,
        )
        return replace(result, maps=retained)

    def multirate_light_curve_with_labels(
        self,
        map_times_days: Sequence[float],
        flux_times_days: Sequence[float],
        *,
        source: PixelatedSource | None = None,
        method: IRSConfig | IPMConfig | None = None,
        trajectory: SourceTrajectory | None = None,
        schedule: DynamicConfig | None = None,
        caustics: CausticConfig | None = None,
        strict_coverage: bool = True,
        diagnostic_grid: PlaneGrid | None = None,
        include_distance_map: bool = False,
        map_observer=None,
        keep_maps_at_days: Sequence[float] | None = None,
    ):
        """Return fine-cadence flux and labels at the sparse map epochs."""

        resolved_source = self.source if source is None else source
        if resolved_source is None:
            raise ValueError(
                "multirate_light_curve_with_labels requires a source model"
            )
        resolved_method = self._method_for_domain(
            production_ipm_config(dynamic=True) if method is None else method
        )
        observer, retained = _retaining_map_observer(
            map_times_days, keep_maps_at_days, map_observer
        )
        result = self.simulation.multirate_light_curve_with_labels(
            self.lens_region,
            self.source_grid,
            self.lens_grid,
            map_times_days,
            flux_times_days,
            resolved_source,
            self.system.distances,
            method=resolved_method,
            trajectory=trajectory,
            map_schedule=schedule,
            caustic_config=CausticConfig() if caustics is None else caustics,
            strict_coverage=strict_coverage,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
            map_observer=observer,
        )
        return replace(result, light_curve=replace(result.light_curve, maps=retained))

    def light_curve_with_labels(
        self,
        times_days: Sequence[float],
        *,
        source: PixelatedSource | None = None,
        method: IRSConfig | IPMConfig | None = None,
        trajectory: SourceTrajectory | None = None,
        schedule: DynamicConfig | None = None,
        caustics: CausticConfig | None = None,
        strict_coverage: bool = True,
        diagnostic_grid: PlaneGrid | None = None,
        include_distance_map: bool = False,
        map_observer=None,
        keep_maps_at_days: Sequence[float] | None = None,
    ):
        """Generate a light curve with aligned source-center caustic labels."""

        resolved_source = self.source if source is None else source
        if resolved_source is None:
            raise ValueError("light_curve_with_labels requires a source model")
        resolved_method = self._method_for_domain(
            production_ipm_config(dynamic=True) if method is None else method
        )
        observer, retained = _retaining_map_observer(
            times_days, keep_maps_at_days, map_observer
        )
        result = self.simulation.light_curve_with_labels(
            self.lens_region,
            self.source_grid,
            self.lens_grid,
            times_days,
            resolved_source,
            self.system.distances,
            method=resolved_method,
            trajectory=trajectory,
            map_schedule=schedule,
            caustic_config=CausticConfig() if caustics is None else caustics,
            strict_coverage=strict_coverage,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
            map_observer=observer,
        )
        return replace(result, light_curve=replace(result.light_curve, maps=retained))

    def caustics(
        self,
        *,
        time_days: float = 0.0,
        config: CausticConfig | None = None,
    ):
        """Return critical curves and source-plane caustics at one epoch."""

        resolved = CausticConfig() if config is None else config
        return self.simulation.caustics(
            self.lens_grid,
            time_days=time_days,
            far_field_approx=resolved.far_field_approx,
            ray_chunk_size=resolved.jacobian_chunk_size,
        )

    def labeled_caustics(
        self,
        *,
        time_days: float = 0.0,
        config: CausticConfig | None = None,
        diagnostic_grid: PlaneGrid | None = None,
        include_distance_map: bool = False,
    ):
        """Return one caustic field and its source-region label products."""

        return self.simulation.labeled_caustics(
            self.lens_grid,
            self.source_region,
            time_days=time_days,
            config=CausticConfig() if config is None else config,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
        )

    def steady_transfer_function(
        self,
        delay_edges_days,
        *,
        source=None,
        driver_amplitude: float = 1.0,
        normalize: bool = True,
    ):
        """Return an unlensed response function without constructing a map."""

        from .transfer_functions import steady_transfer_function

        resolved_source = self.source if source is None else source
        if resolved_source is None:
            raise ValueError("steady_transfer_function requires a source")
        return steady_transfer_function(
            resolved_source,
            delay_edges_days,
            driver_amplitude=driver_amplitude,
            normalize=normalize,
        )

    def transfer_functions(
        self,
        times_days: Sequence[float],
        delay_edges_days,
        *,
        source=None,
        method: IRSConfig | IPMConfig | None = None,
        trajectory: SourceTrajectory | None = None,
        schedule: DynamicConfig | None = None,
        strict_coverage: bool = True,
        driver_amplitude: float = 1.0,
        normalize: bool = True,
        map_observer=None,
    ):
        """Stream dynamic maps into microlensed response functions."""

        from .transfer_functions import streaming_microlensed_transfer_functions

        resolved_source = self.source if source is None else source
        if resolved_source is None:
            raise ValueError("transfer_functions requires a source")
        resolved_method = self._method_for_domain(
            production_ipm_config(dynamic=True) if method is None else method
        )
        return streaming_microlensed_transfer_functions(
            self.simulation,
            self.lens_region,
            self.source_grid,
            times_days,
            resolved_source,
            self.system.distances,
            delay_edges_days,
            method=resolved_method,
            trajectory=trajectory,
            schedule=schedule,
            strict_coverage=strict_coverage,
            driver_amplitude=driver_amplitude,
            normalize=normalize,
            map_observer=map_observer,
        )

    def _method_for_domain(self, method: IRSConfig | IPMConfig):
        if not isinstance(method, IPMConfig):
            return method
        tiled = self.system.integration_domain is IntegrationDomain.SCOUT
        if method.tiled == tiled:
            return method
        return replace(
            method,
            tiled=tiled,
            dual_scout_scalar_correction=(
                method.dual_scout_scalar_correction if tiled else False
            ),
        )


@dataclass(frozen=True)
class MicrolensingSystem:
    """A physical microlensing system with automatic numerical geometry.

    Supply either a :class:`StellarPopulation` or a directly constructed
    :class:`PointMassField`. A source model automatically defines the angular
    source grid. Source-independent map calculations can instead provide only
    ``source_grid``. The seeded realization is cached and reused across maps,
    light curves, labels, and future transfer-function calculations.
    """

    macro: MacroLens
    distances: LensingDistances
    source: PixelatedSource | PhysicalSourceModel | None = None
    source_grid: PlaneGrid | None = None
    stellar_population: StellarPopulation | None = None
    stars: PointMassField | None = None
    integration_domain: IntegrationDomain | str = IntegrationDomain.SCOUT
    duration_days: float = 0.0
    light_loss: float = 0.01
    safety_scale: float = 1.5
    stellar_motion_sigma_margin: float = 5.0
    rectangle_light_loss: float | None = None
    source_support_radius_uas: float | None = None
    seed: int | None = None
    runtime: RuntimeConfig | ResolvedRuntime | None = None
    lens_region: PlaneRegion | None = None
    caustic_grid_shape: int | tuple[int, int] = 8192

    @classmethod
    def from_redshifts(
        cls,
        *,
        lens_redshift: float,
        source_redshift: float,
        H0: float = 67.66,
        Om0: float = 0.30966,
        cosmology=None,
        distance_dtype: torch.dtype = torch.float32,
        distance_device: torch.device | str | None = "auto",
        **kwargs,
    ) -> MicrolensingSystem:
        """Construct a system while deriving cosmological distances.

        This is the recommended observational interface. The default flat
        cosmology is evaluated by PyTorch with ``distance_dtype=float32``.
        Supply ``cosmology`` to use an external Astropy-compatible model, or
        construct :class:`LensingDistances` directly for full control.
        """

        if "distances" in kwargs:
            raise TypeError("from_redshifts derives distances; do not also supply distances")
        distances = LensingDistances.from_redshifts(
            lens_redshift,
            source_redshift,
            cosmology=cosmology,
            H0=H0,
            Om0=Om0,
            dtype=distance_dtype,
            device=distance_device,
        )
        return cls(distances=distances, **kwargs)

    def __post_init__(self) -> None:
        if (self.stellar_population is None) == (self.stars is None):
            raise ValueError("supply exactly one of stellar_population or stars")
        if self.source is None and self.source_grid is None:
            raise ValueError("supply a source model or source_grid")
        if (
            self.source is not None
            and self.source_grid is not None
            and not isinstance(self.source, PhysicalSourceModel)
        ):
            inferred_grid = _source_grid_from_model(self.source, self.distances)
            if any(
                actual < expected * (1.0 - 1.0e-10)
                for actual, expected in zip(
                    self.source_grid.field_of_view_uas,
                    inferred_grid.field_of_view_uas,
                    strict=True,
                )
            ):
                raise ValueError(
                    "source_grid field of view must enclose the pixelated source "
                    "geometry. Increase it or omit source_grid to derive the "
                    "minimum field automatically"
                )
        if self.duration_days < 0.0:
            raise ValueError("duration_days must be non-negative")
        if not 0.0 < self.light_loss < 1.0:
            raise ValueError("light_loss must lie strictly between zero and one")
        if self.safety_scale < 1.0:
            raise ValueError("safety_scale must be at least one")
        if self.stellar_motion_sigma_margin < 0.0:
            raise ValueError("stellar_motion_sigma_margin must be non-negative")
        if self.rectangle_light_loss is not None and not (
            0.0 < self.rectangle_light_loss < 1.0
        ):
            raise ValueError("rectangle_light_loss must lie strictly between zero and one")
        if self.source_support_radius_uas is not None and self.source_support_radius_uas <= 0.0:
            raise ValueError("source_support_radius_uas must be positive")
        caustic_shape = (
            (int(self.caustic_grid_shape), int(self.caustic_grid_shape))
            if isinstance(self.caustic_grid_shape, int)
            else tuple(int(value) for value in self.caustic_grid_shape)
        )
        if len(caustic_shape) != 2 or any(value < 2 for value in caustic_shape):
            raise ValueError("caustic_grid_shape must contain values of at least two")
        object.__setattr__(self, "caustic_grid_shape", caustic_shape)
        object.__setattr__(
            self,
            "integration_domain",
            IntegrationDomain(self.integration_domain),
        )

    @cached_property
    def _resolved(self) -> MicrolensingRealization:
        resolved_runtime = (
            self.runtime
            if isinstance(self.runtime, ResolvedRuntime)
            else resolve_runtime(self.runtime)
        )
        source_support_radius_uas = self.source_support_radius_uas
        if isinstance(self.source, PhysicalSourceModel):
            source_grid = (
                self.source.recommended_grid(self.distances)
                if self.source_grid is None
                else self.source_grid
            )
            pixelate_parameters = inspect.signature(
                self.source.pixelate
            ).parameters.values()
            accepts_runtime = any(
                parameter.name == "runtime"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in pixelate_parameters
            )
            source = self.source.pixelate(
                self.distances,
                grid=source_grid,
                **({"runtime": resolved_runtime} if accepts_runtime else {}),
            )
            support_radius_method = getattr(self.source, "support_radius_m", None)
            if source_support_radius_uas is None and support_radius_method is not None:
                source_support_radius_uas = float(
                    self.distances.source_length_to_uas(
                        support_radius_method(self.distances),
                        dtype=torch.float64,
                    )
                )
        else:
            source = self.source
            source_grid = (
                _source_grid_from_model(source, self.distances)
                if self.source_grid is None
                else self.source_grid
            )
            support_radius_method = (
                None if source is None else getattr(source, "support_radius_m", None)
            )
            if source_support_radius_uas is None and support_radius_method is not None:
                source_support_radius_uas = float(
                    self.distances.source_length_to_uas(
                        support_radius_method(self.distances),
                        dtype=torch.float64,
                    )
                )
        assert source_grid is not None
        if source_support_radius_uas is not None:
            available_radius = 0.5 * min(source_grid.field_of_view_uas)
            if source_support_radius_uas > available_radius * (1.0 + 1.0e-10):
                raise ValueError(
                    "source_grid does not enclose source_support_radius_uas. "
                    "Increase its field of view or omit source_grid to use the "
                    "physical model's recommended grid"
                )

        aperture = None
        if self.stellar_population is not None:
            aperture = circular_stellar_aperture(
                self.macro,
                source_grid.region,
                self.distances,
                self.stellar_population,
                light_loss=self.light_loss,
                safety_scale=self.safety_scale,
                duration_days=self.duration_days,
                motion_sigma_margin=self.stellar_motion_sigma_margin,
                source_support_radius_uas=source_support_radius_uas,
            )
            stars = self.stellar_population.realize(
                aperture,
                self.macro,
                self.distances,
                seed=self.seed,
                device=resolved_runtime.device,
                dtype=resolved_runtime.dtype,
            )
            if self.lens_region is not None:
                lens_region = self.lens_region
            elif self.integration_domain is IntegrationDomain.RECTANGLE:
                lens_region = rectangular_lens_region(
                    self.macro,
                    source_grid.region,
                    self.distances,
                    self.stellar_population.mass_function,
                    light_loss=(
                        self.light_loss
                        if self.rectangle_light_loss is None
                        else self.rectangle_light_loss
                    ),
                )
            else:
                lens_region = aperture.bounding_region
        else:
            assert self.stars is not None
            stars = self.stars
            lens_region = (
                _direct_star_region(stars, self.macro, source_grid.region)
                if self.lens_region is None
                else self.lens_region
            )

        simulation = MicrolensingSimulation.create(
            self.macro,
            stars,
            runtime=resolved_runtime,
        )
        lens_grid = PlaneGrid(
            self.caustic_grid_shape,
            lens_region.field_of_view_uas,
            lens_region.center_uas,
        )
        return MicrolensingRealization(
            system=self,
            simulation=simulation,
            source=source,
            source_grid=source_grid,
            lens_region=lens_region,
            lens_grid=lens_grid,
            stellar_aperture=aperture,
            source_support_radius_uas=source_support_radius_uas,
        )

    def realize(self) -> MicrolensingRealization:
        """Return the cached seeded stellar and numerical realization."""

        return self._resolved

    def warmup(
        self,
        times_days: Sequence[float] | None = None,
        *,
        labels: bool = False,
        **kwargs,
    ):
        """Warm the kernels needed by a representative production call.

        With no time axis, one static map is generated. Supplying
        ``times_days`` warms the dynamic light-curve path, including its
        temporal batch shape. Set ``labels=True`` to warm the source-center
        caustic pipeline as well. The computed result is returned so warmup
        work can still be inspected or reused.
        """

        if times_days is None:
            if labels:
                return self.labeled_caustics(**kwargs)
            return self.magnification_map(**kwargs)
        if labels:
            return self.light_curve_with_labels(times_days, **kwargs)
        return self.light_curve(times_days, **kwargs)

    def magnification_map(self, **kwargs) -> MagnificationMap:
        """Generate one map through the cached realization."""

        return self.realize().magnification_map(**kwargs)

    def dynamic_maps(self, times_days: Sequence[float], **kwargs):
        """Stream maps through the cached realization."""

        return self.realize().dynamic_maps(times_days, **kwargs)

    def dynamic_labeled_maps(self, times_days: Sequence[float], **kwargs):
        """Stream maps and aligned label products through the realization."""

        return self.realize().dynamic_labeled_maps(times_days, **kwargs)

    def light_curve(self, times_days: Sequence[float], **kwargs) -> LightCurve:
        """Generate one light curve through the cached realization."""

        return self.realize().light_curve(times_days, **kwargs)

    def light_curves(self, times_days: Sequence[float], requests, **kwargs):
        """Batch multiple light curves through the cached realization."""

        return self.realize().light_curves(times_days, requests, **kwargs)

    def multirate_light_curve(
        self,
        map_times_days: Sequence[float],
        flux_times_days: Sequence[float],
        **kwargs,
    ) -> LightCurve:
        """Evaluate source evolution more finely than the dynamic maps."""

        return self.realize().multirate_light_curve(
            map_times_days,
            flux_times_days,
            **kwargs,
        )

    def multirate_light_curve_with_labels(
        self,
        map_times_days: Sequence[float],
        flux_times_days: Sequence[float],
        **kwargs,
    ):
        """Evaluate fine-cadence flux with labels at map epochs."""

        return self.realize().multirate_light_curve_with_labels(
            map_times_days,
            flux_times_days,
            **kwargs,
        )

    def light_curve_with_labels(self, times_days: Sequence[float], **kwargs):
        """Generate one labeled light curve through the cached realization."""

        return self.realize().light_curve_with_labels(times_days, **kwargs)

    def caustics(self, **kwargs):
        """Return critical curves and caustics through the cached realization."""

        return self.realize().caustics(**kwargs)

    def labeled_caustics(self, **kwargs):
        """Return one labeled caustic frame through the cached realization."""

        return self.realize().labeled_caustics(**kwargs)

    def steady_transfer_function(self, delay_edges_days, **kwargs):
        """Return the source response without microlensing."""

        return self.realize().steady_transfer_function(delay_edges_days, **kwargs)

    def transfer_functions(self, times_days, delay_edges_days, **kwargs):
        """Generate dynamic microlensed transfer functions."""

        return self.realize().transfer_functions(
            times_days,
            delay_edges_days,
            **kwargs,
        )
