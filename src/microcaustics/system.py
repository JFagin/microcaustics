"""High-level physical microlensing systems and their realizations."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
    _production_static_ipm_config,
    production_dynamic_config,
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
from .random import derive_seed
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


def _rotate_cartesian_components(
    x: torch.Tensor,
    y: torch.Tensor,
    angle_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express Cartesian components in axes rotated by ``angle_deg``.

    This is a coordinate-basis change, not an interpolation or a rotation of
    a materialized image.  A positive angle maps sky-frame components to the
    local frame through an active rotation by the negative of that angle.
    """

    angle = torch.as_tensor(
        math.radians(float(angle_deg)),
        device=x.device,
        dtype=x.dtype,
    )
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    return cosine * x + sine * y, -sine * x + cosine * y


def _rotate_point_mass_field(
    stars: PointMassField,
    angle_deg: float,
) -> PointMassField:
    """Return the same physical point lenses in a rotated coordinate basis."""

    x, y = _rotate_cartesian_components(stars.x_uas, stars.y_uas, angle_deg)
    velocity_x = velocity_y = None
    if stars.has_motion:
        assert stars.velocity_x_uas_per_day is not None
        assert stars.velocity_y_uas_per_day is not None
        velocity_x, velocity_y = _rotate_cartesian_components(
            stars.velocity_x_uas_per_day,
            stars.velocity_y_uas_per_day,
            angle_deg,
        )
    return PointMassField(
        x,
        y,
        stars.einstein_radius_uas,
        velocity_x,
        velocity_y,
        stars.mass_solar,
    )


@dataclass(frozen=True)
class _RotatedTrajectory:
    """Internal coordinate-basis view of an arbitrary source trajectory."""

    trajectory: SourceTrajectory
    angle_deg: float

    def position_uas(
        self,
        times_days,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        positions = self.trajectory.position_uas(
            times_days,
            device=device,
            dtype=dtype,
        )
        x, y = _rotate_cartesian_components(
            positions[..., 0], positions[..., 1], self.angle_deg
        )
        return torch.stack((x, y), dim=-1)


def _source_in_rotated_frame(source, angle_deg: float):
    """Return a physical source model expressed in a rotated basis.

    Built-in physical models expose their sky position angle explicitly, so
    changing coordinates only requires subtracting the basis angle before
    pixelization.  Gaussian centers are transformed in the same operation.
    Materialized pixel arrays are deliberately not accepted because rotating
    those would introduce an interpolation into the scientific calculation.
    """

    updates: dict[str, object] = {}
    if hasattr(source, "position_angle_deg"):
        updates["position_angle_deg"] = (
            float(source.position_angle_deg) - float(angle_deg)
        )
    elif hasattr(source, "position_angle_rad"):
        updates["position_angle_rad"] = (
            float(source.position_angle_rad) - math.radians(float(angle_deg))
        )
    else:
        raise TypeError(
            "automatic shear-frame alignment requires a physical source model "
            "with position_angle_deg or position_angle_rad"
        )
    if hasattr(source, "center_m"):
        center = torch.as_tensor(source.center_m, dtype=torch.float64)
        x, y = _rotate_cartesian_components(center[0], center[1], angle_deg)
        updates["center_m"] = (float(x), float(y))
    # Models constructed through the convenient ``bands`` mapping store both
    # that input and normalized parallel tuples after initialization.  A
    # dataclass replacement must retain only the normalized representation.
    if getattr(source, "bands", None) is not None:
        updates["bands"] = None
    return replace(source, **updates)


def _rotate_region_center(region: PlaneRegion, angle_deg: float) -> PlaneRegion:
    """Rotate a region center while preserving a circular aperture's size."""

    center = torch.as_tensor(
        (region.center_uas[1], region.center_uas[0]), dtype=torch.float64
    )
    x, y = _rotate_cartesian_components(center[0], center[1], angle_deg)
    return PlaneRegion(region.field_of_view_uas, (float(y), float(x)))


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


def _cadence_times(
    times_days,
    *,
    duration_days: float | None,
    cadence_days: float | None,
    start_day: float = 0.0,
) -> torch.Tensor:
    """Resolve an explicit time axis or an inclusive regular cadence."""

    if times_days is not None:
        if duration_days is not None or cadence_days is not None:
            raise ValueError(
                "supply times_days or duration_days/cadence_days, not both"
            )
        times = torch.as_tensor(times_days, dtype=torch.float64).reshape(-1)
    else:
        if duration_days is None or cadence_days is None:
            raise ValueError("supply times_days or both duration_days and cadence_days")
        duration = float(duration_days)
        cadence = float(cadence_days)
        if duration < 0.0 or cadence <= 0.0:
            raise ValueError(
                "duration_days must be non-negative and cadence_days positive"
            )
        count = int(math.floor(duration / cadence + 1.0e-10)) + 1
        times = float(start_day) + torch.arange(count, dtype=torch.float64) * cadence
        final = float(start_day) + duration
        if float(times[-1]) < final - 1.0e-10:
            times = torch.cat((times, torch.tensor((final,), dtype=torch.float64)))
    if times.numel() < 1 or not bool(torch.all(torch.isfinite(times))):
        raise ValueError("time axis must contain finite values")
    if times.numel() > 1 and not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("time axis must be strictly increasing")
    return times


def _with_method_options(kwargs: dict, *, dynamic: bool) -> dict:
    """Resolve common plain solver keywords into an advanced configuration."""

    resolved = dict(kwargs)
    option_names = {
        "rays",
        "scout_ratio",
        "refinement",
        "virtual_refinement",
        "scout_dilation_cells",
        "far_field",
    }
    options = {
        name: resolved.pop(name)
        for name in tuple(resolved)
        if name in option_names
    }
    method = resolved.get("method")
    if isinstance(method, str):
        name = method.lower().replace("-", "_")
        rays = int(options.pop("rays", 10_000_000))
        if name == "ipm":
            method = (
                production_ipm_config(rays=rays)
                if dynamic
                else _production_static_ipm_config(rays=rays)
            )
        elif name == "irs":
            if options:
                raise ValueError(
                    "scout/refinement options apply only to method='ipm'"
                )
            method = IRSConfig(rays=rays)
        else:
            raise ValueError("method must be 'ipm', 'irs', or a config object")
        resolved["method"] = method
    elif isinstance(method, IRSConfig):
        rays = options.pop("rays", None)
        if options:
            raise ValueError("scout/refinement options apply only to IPM")
        if rays is not None:
            resolved["method"] = replace(method, rays=int(rays))
        return resolved
    if not options:
        return resolved
    method = resolved.get("method")
    if method is None:
        method = (
            production_ipm_config()
            if dynamic
            else _production_static_ipm_config()
        )
    if not isinstance(method, IPMConfig):
        raise ValueError("IPM numerical options require an IPM method")
    far_field = options.pop("far_field", None)
    if far_field is not None:
        options["far_field_approx"] = replace(
            method.far_field_approx,
            enabled=bool(far_field),
        )
    resolved["method"] = replace(method, **options)
    return resolved


def _production_dynamic_settings(
    method: IPMConfig | IRSConfig,
    schedule: DynamicConfig | None,
    caustics: CausticConfig | None = None,
) -> tuple[DynamicConfig, CausticConfig | None]:
    """Resolve coherent high-level dynamic and optional caustic settings."""

    resolved_schedule = production_dynamic_config() if schedule is None else schedule
    if caustics is None:
        inherited_far_field = (
            method.far_field_approx if isinstance(method, IPMConfig) else None
        )
        resolved_caustics = CausticConfig(
            **(
                {"far_field_approx": inherited_far_field}
                if inherited_far_field is not None
                else {}
            ),
            temporal_batch_size=resolved_schedule.temporal_batch_size,
        )
    else:
        resolved_caustics = caustics
    return resolved_schedule, resolved_caustics


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
    half_x = abs(inverse_xx) * half_source_x + abs(inverse_xy) * half_source_y
    half_y = abs(inverse_xy) * half_source_x + abs(inverse_yy) * half_source_y
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
    trajectory: SourceTrajectory | None = None
    sky_to_local_rotation_deg: float = 0.0

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
            "coordinate_frame": (
                "input"
                if self.sky_to_local_rotation_deg == 0.0
                else "shear_aligned"
            ),
            "sky_to_local_rotation_deg": float(self.sky_to_local_rotation_deg),
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
            "source": (None if self.source is None else dict(self.source.metadata())),
        }

    def magnification_map(
        self,
        *,
        method: IRSConfig | IPMConfig | None = None,
        time_days: float = 0.0,
    ) -> MagnificationMap:
        """Generate one source-independent magnification map."""

        resolved_method = self._method_for_domain(
            _production_static_ipm_config() if method is None else method
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
            production_ipm_config() if method is None else method
        )
        resolved_schedule, _ = _production_dynamic_settings(resolved_method, schedule)
        return self.simulation.dynamic_maps(
            self.lens_region,
            self.source_grid,
            times_days,
            method=resolved_method,
            schedule=resolved_schedule,
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
            production_ipm_config() if method is None else method
        )
        resolved_schedule, resolved_caustics = _production_dynamic_settings(
            resolved_method,
            schedule,
            caustics,
        )
        return self.simulation.dynamic_labeled_maps(
            self.lens_region,
            self.source_grid,
            self.lens_grid,
            times_days,
            method=resolved_method,
            map_schedule=resolved_schedule,
            caustic_config=resolved_caustics,
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
            production_ipm_config() if method is None else method
        )
        resolved_schedule, _ = _production_dynamic_settings(resolved_method, schedule)
        observer, retained = _retaining_map_observer(
            times_days, keep_maps_at_days, map_observer
        )
        resolved_trajectory = self._trajectory_in_local_frame(trajectory)
        result = self.simulation.light_curve(
            self.lens_region,
            self.source_grid,
            times_days,
            resolved_source,
            self.system.distances,
            method=resolved_method,
            trajectory=resolved_trajectory,
            schedule=resolved_schedule,
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
            production_ipm_config() if method is None else method
        )
        resolved_schedule, _ = _production_dynamic_settings(resolved_method, schedule)
        return self.simulation.light_curves(
            self.lens_region,
            self.source_grid,
            times_days,
            requests,
            method=resolved_method,
            schedule=resolved_schedule,
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
            production_ipm_config() if method is None else method
        )
        resolved_schedule, _ = _production_dynamic_settings(resolved_method, schedule)
        observer, retained = _retaining_map_observer(
            map_times_days, keep_maps_at_days, map_observer
        )
        resolved_trajectory = self._trajectory_in_local_frame(trajectory)
        result = self.simulation.multirate_light_curve(
            self.lens_region,
            self.source_grid,
            map_times_days,
            flux_times_days,
            resolved_source,
            self.system.distances,
            method=resolved_method,
            trajectory=resolved_trajectory,
            schedule=resolved_schedule,
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
            production_ipm_config() if method is None else method
        )
        resolved_schedule, resolved_caustics = _production_dynamic_settings(
            resolved_method,
            schedule,
            caustics,
        )
        observer, retained = _retaining_map_observer(
            map_times_days, keep_maps_at_days, map_observer
        )
        resolved_trajectory = self._trajectory_in_local_frame(trajectory)
        result = self.simulation.multirate_light_curve_with_labels(
            self.lens_region,
            self.source_grid,
            self.lens_grid,
            map_times_days,
            flux_times_days,
            resolved_source,
            self.system.distances,
            method=resolved_method,
            trajectory=resolved_trajectory,
            map_schedule=resolved_schedule,
            caustic_config=resolved_caustics,
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
            production_ipm_config() if method is None else method
        )
        resolved_schedule, resolved_caustics = _production_dynamic_settings(
            resolved_method,
            schedule,
            caustics,
        )
        observer, retained = _retaining_map_observer(
            times_days, keep_maps_at_days, map_observer
        )
        resolved_trajectory = self._trajectory_in_local_frame(trajectory)
        result = self.simulation.light_curve_with_labels(
            self.lens_region,
            self.source_grid,
            self.lens_grid,
            times_days,
            resolved_source,
            self.system.distances,
            method=resolved_method,
            trajectory=resolved_trajectory,
            map_schedule=resolved_schedule,
            caustic_config=resolved_caustics,
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
            production_ipm_config() if method is None else method
        )
        resolved_schedule, _ = _production_dynamic_settings(resolved_method, schedule)
        return streaming_microlensed_transfer_functions(
            self.simulation,
            self.lens_region,
            self.source_grid,
            times_days,
            resolved_source,
            self.system.distances,
            delay_edges_days,
            method=resolved_method,
            trajectory=self._trajectory_in_local_frame(trajectory),
            schedule=resolved_schedule,
            strict_coverage=strict_coverage,
            driver_amplitude=driver_amplitude,
            normalize=normalize,
            map_observer=map_observer,
        )

    def _trajectory_in_local_frame(
        self,
        trajectory: SourceTrajectory | None,
    ) -> SourceTrajectory | None:
        """Resolve a trajectory in the realization's numerical frame."""

        if trajectory is None:
            return self.trajectory
        if self.sky_to_local_rotation_deg == 0.0:
            return trajectory
        return _RotatedTrajectory(trajectory, self.sky_to_local_rotation_deg)

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
    source grid. If a trajectory and duration are supplied, the map field is
    enlarged automatically to contain that source throughout the sequence.
    Source-independent map calculations can instead provide only
    ``source_grid``. The seeded realization is cached and reused across maps,
    light curves, labels, and future transfer-function calculations.

    The rectangular integration strategy uses the shear eigenframe
    internally when the source is a built-in physical model. Point-lens
    positions, velocities, source position angle, and trajectory are changed
    to that basis together. This produces the conventional narrow rectangle
    without rotating or resampling a materialized source image. Already
    pixelated custom sources remain in the input frame and therefore use the
    conservative axis-aligned bounding rectangle.
    """

    macro: MacroLens
    distances: LensingDistances
    source: PixelatedSource | PhysicalSourceModel | None = None
    source_grid: PlaneGrid | None = None
    stellar_population: StellarPopulation | None = None
    stars: PointMassField | None = None
    integration_domain: IntegrationDomain | str = IntegrationDomain.SCOUT
    duration_days: float = 0.0
    trajectory: SourceTrajectory | None = None
    trajectory_grid_margin: float = 1.05
    light_loss: float = 0.01
    safety_scale: float = 1.5
    stellar_motion_sigma_margin: float = 5.0
    rectangle_light_loss: float | None = None
    source_support_radius_uas: float | None = None
    seed: int | Mapping[str, int] | None = None
    runtime: RuntimeConfig | ResolvedRuntime | None = None
    lens_region: PlaneRegion | None = None
    caustic_grid_shape: int | tuple[int, int] = 8192
    _duration_realizations: dict[float, MicrolensingRealization] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

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
            raise TypeError(
                "from_redshifts derives distances; do not also supply distances"
            )
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
        if not math.isfinite(float(self.trajectory_grid_margin)) or (
            self.trajectory_grid_margin < 1.0
        ):
            raise ValueError("trajectory_grid_margin must be finite and at least one")
        if not 0.0 < self.light_loss < 1.0:
            raise ValueError("light_loss must lie strictly between zero and one")
        if self.safety_scale < 1.0:
            raise ValueError("safety_scale must be at least one")
        if self.stellar_motion_sigma_margin < 0.0:
            raise ValueError("stellar_motion_sigma_margin must be non-negative")
        if self.rectangle_light_loss is not None and not (
            0.0 < self.rectangle_light_loss < 1.0
        ):
            raise ValueError(
                "rectangle_light_loss must lie strictly between zero and one"
            )
        if (
            self.source_support_radius_uas is not None
            and self.source_support_radius_uas <= 0.0
        ):
            raise ValueError("source_support_radius_uas must be positive")
        if isinstance(self.seed, Mapping):
            supplied = {str(name): int(value) for name, value in self.seed.items()}
            unknown = set(supplied) - {
                "base",
                "stars",
                "variability",
                "observations",
            }
            if unknown:
                raise ValueError(f"unknown seed components: {sorted(unknown)}")
            object.__setattr__(self, "seed", supplied)
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
        physical_source = isinstance(self.source, PhysicalSourceModel)
        source_has_orientation = physical_source and (
            hasattr(self.source, "position_angle_deg")
            or hasattr(self.source, "position_angle_rad")
        )
        explicit_grid_is_rotation_safe = self.source_grid is None or (
            math.isclose(
                self.source_grid.field_of_view_uas[0],
                self.source_grid.field_of_view_uas[1],
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
            and math.isclose(self.source_grid.center_uas[0], 0.0, abs_tol=1.0e-12)
            and math.isclose(self.source_grid.center_uas[1], 0.0, abs_tol=1.0e-12)
        )
        align_rectangle = (
            self.integration_domain is IntegrationDomain.RECTANGLE
            and self.stellar_population is not None
            and self.lens_region is None
            and explicit_grid_is_rotation_safe
            and (self.source is None or source_has_orientation)
            and not math.isclose(
                math.remainder(float(self.macro.shear_angle_deg), 180.0),
                0.0,
                abs_tol=1.0e-12,
            )
        )
        frame_rotation_deg = (
            float(self.macro.shear_angle_deg) if align_rectangle else 0.0
        )
        numerical_macro = (
            replace(self.macro, shear_angle_deg=0.0)
            if align_rectangle
            else self.macro
        )
        numerical_source_model = (
            _source_in_rotated_frame(self.source, frame_rotation_deg)
            if align_rectangle and physical_source
            else self.source
        )
        numerical_trajectory = (
            _RotatedTrajectory(self.trajectory, frame_rotation_deg)
            if align_rectangle and self.trajectory is not None
            else self.trajectory
        )
        source_support_radius_uas = self.source_support_radius_uas
        if physical_source:
            assert isinstance(numerical_source_model, PhysicalSourceModel)
            native_source_grid = (
                numerical_source_model.recommended_grid(self.distances)
                if self.source_grid is None
                else self.source_grid
            )
            pixelate_parameters = inspect.signature(
                numerical_source_model.pixelate
            ).parameters.values()
            accepts_runtime = any(
                parameter.name == "runtime"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in pixelate_parameters
            )
            source = numerical_source_model.pixelate(
                self.distances,
                grid=native_source_grid,
                **({"runtime": resolved_runtime} if accepts_runtime else {}),
            )
            support_radius_method = getattr(
                numerical_source_model, "support_radius_m", None
            )
            if source_support_radius_uas is None and support_radius_method is not None:
                source_support_radius_uas = float(
                    self.distances.source_length_to_uas(
                        support_radius_method(self.distances),
                        dtype=torch.float64,
                    )
                )
        else:
            source = self.source
            native_source_grid = (
                _source_grid_from_model(source, self.distances)
                if self.source_grid is None
                else self.source_grid
            )
            support_radius_method = (
                None if source is None else getattr(source, "support_radius_m", None)
            )
            if source_support_radius_uas is None and support_radius_method is not None:
                physical_support_radius = support_radius_method(self.distances)
                if physical_support_radius is not None:
                    source_support_radius_uas = float(
                        self.distances.source_length_to_uas(
                            physical_support_radius,
                            dtype=torch.float64,
                        )
                    )
        assert native_source_grid is not None
        source_grid = native_source_grid
        if (
            self.source_grid is None
            and numerical_trajectory is not None
            and self.duration_days > 0.0
        ):
            source_grid = native_source_grid.covering_trajectory(
                numerical_trajectory,
                (0.0, self.duration_days),
                margin=self.trajectory_grid_margin,
            )
        if source_support_radius_uas is not None and self.source_grid is not None:
            available_radius = 0.5 * min(native_source_grid.field_of_view_uas)
            if source_support_radius_uas > available_radius * (1.0 + 1.0e-10):
                raise ValueError(
                    "source_grid does not enclose source_support_radius_uas. "
                    "Increase its field of view or omit source_grid to use the "
                    "physical model's recommended grid"
                )

        aperture = None
        if self.stellar_population is not None:
            aperture = circular_stellar_aperture(
                numerical_macro,
                source_grid.region,
                self.distances,
                self.stellar_population,
                light_loss=self.light_loss,
                safety_scale=self.safety_scale,
                duration_days=self.duration_days,
                motion_sigma_margin=self.stellar_motion_sigma_margin,
                source_support_radius_uas=source_support_radius_uas,
            )
            sampling_aperture = aperture
            if align_rectangle:
                sampling_region = _rotate_region_center(
                    aperture.bounding_region, -frame_rotation_deg
                )
                sampling_aperture = StellarAperture(
                    aperture.radius_uas,
                    sampling_region.center_uas,
                )
            stars = self.stellar_population.realize(
                sampling_aperture,
                self.macro,
                self.distances,
                seed=self.seed_for("stars"),
                device=resolved_runtime.device,
                dtype=resolved_runtime.dtype,
            )
            if align_rectangle:
                stars = _rotate_point_mass_field(stars, frame_rotation_deg)
            if self.lens_region is not None:
                lens_region = self.lens_region
            elif self.integration_domain is IntegrationDomain.RECTANGLE:
                lens_region = rectangular_lens_region(
                    numerical_macro,
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
                _direct_star_region(stars, numerical_macro, source_grid.region)
                if self.lens_region is None
                else self.lens_region
            )

        simulation = MicrolensingSimulation.create(
            numerical_macro,
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
            trajectory=numerical_trajectory,
            sky_to_local_rotation_deg=frame_rotation_deg,
        )

    def realize(self) -> MicrolensingRealization:
        """Return the cached seeded stellar and numerical realization."""

        return self._resolved

    def with_integration_domain(
        self,
        integration_domain: IntegrationDomain | str,
    ) -> MicrolensingSystem:
        """Return this physical system with another integration strategy.

        The macro lens, source, stellar seed, kinematics, runtime, and derived
        physical fields are preserved.  Only the numerical lens-plane domain
        changes.  This is useful when comparing source scouting, the complete
        stellar field, and the conventional rectangular preimage without
        rebuilding the physical specification by hand.
        """

        return replace(
            self,
            integration_domain=IntegrationDomain(integration_domain),
        )

    def with_seed(
        self,
        seed: int | Mapping[str, int] | None,
    ) -> MicrolensingSystem:
        """Return the same physical system with a new random seed.

        A new seed creates an independent stellar realization while preserving
        the macro lens, source, numerical settings, and geometry rules.
        Compatible compiled kernels are reused because their cache depends on
        numerical shapes rather than random values.

        Parameters
        ----------
        seed
            Base seed, component-specific seed mapping, or ``None`` for a
            nondeterministic realization.
        """

        return replace(self, seed=seed)

    def seed_for(self, component: str) -> int | None:
        """Return one reproducible component seed from the system seed."""

        if isinstance(self.seed, Mapping):
            if component in self.seed:
                return int(self.seed[component])
            return derive_seed(self.seed.get("base"), component)
        if component == "stars":
            return None if self.seed is None else int(self.seed)
        return derive_seed(self.seed, component)

    @property
    def realization(self) -> MicrolensingRealization:
        """Return the cached realized system without another factory call."""

        return self._resolved

    @property
    def realized_stars(self) -> PointMassField:
        """Return the realized point-mass field."""

        return self._resolved.stars

    @property
    def resolved_source_grid(self) -> PlaneGrid:
        """Return the source grid, including automatically derived geometry."""

        return self._resolved.source_grid

    @property
    def resolved_lens_region(self) -> PlaneRegion:
        """Return the numerical lens region selected for this system."""

        return self._resolved.lens_region

    @property
    def stellar_aperture(self) -> StellarAperture | None:
        """Return the derived circular stellar aperture when applicable."""

        return self._resolved.stellar_aperture

    def metadata(self) -> Mapping[str, object]:
        """Return serializable physical and realized numerical provenance."""

        return self._resolved.metadata()

    def summary(
        self,
        *,
        times_days: Sequence[float] | None = None,
        duration_days: float | None = None,
        display: bool = True,
    ) -> dict[str, object]:
        """Describe derived geometry and resource scale without tracing maps.

        The summary does not sample stars, run the GR source calculation, or
        compile solver kernels. It is therefore safe to call before a large
        production calculation.
        """

        if times_days is not None:
            if duration_days is not None:
                raise ValueError("supply times_days or duration_days, not both")
            times = torch.as_tensor(times_days, dtype=torch.float64).reshape(-1)
            if times.numel() < 1:
                raise ValueError("times_days must not be empty")
            duration = max(abs(float(times.min())), abs(float(times.max())))
        else:
            duration = float(
                self.duration_days if duration_days is None else duration_days
            )
        if duration < 0.0:
            raise ValueError("duration_days must be non-negative")

        if isinstance(self.source, PhysicalSourceModel):
            native_grid = (
                self.source.recommended_grid(self.distances)
                if self.source_grid is None
                else self.source_grid
            )
            support_method = getattr(self.source, "support_radius_m", None)
        elif self.source is not None:
            native_grid = (
                _source_grid_from_model(self.source, self.distances)
                if self.source_grid is None
                else self.source_grid
            )
            support_method = getattr(self.source, "support_radius_m", None)
        else:
            assert self.source_grid is not None
            native_grid = self.source_grid
            support_method = None
        map_grid = native_grid
        if self.source_grid is None and self.trajectory is not None and duration > 0.0:
            map_grid = native_grid.covering_trajectory(
                self.trajectory,
                (0.0, duration),
                margin=self.trajectory_grid_margin,
            )
        support_radius = self.source_support_radius_uas
        if support_radius is None and support_method is not None:
            support_radius = float(
                self.distances.source_length_to_uas(
                    support_method(self.distances),
                    dtype=torch.float64,
                )
            )

        aperture = None
        expected_stars = None
        if self.stellar_population is not None:
            aperture = circular_stellar_aperture(
                self.macro,
                map_grid.region,
                self.distances,
                self.stellar_population,
                light_loss=self.light_loss,
                safety_scale=self.safety_scale,
                duration_days=duration,
                motion_sigma_margin=self.stellar_motion_sigma_margin,
                source_support_radius_uas=support_radius,
            )
            if self.stellar_population.count is not None:
                expected_stars = int(self.stellar_population.count)
            elif self.macro.compact_convergence > 0.0:
                mean_mass = self.stellar_population.mass_function.mean_mass()
                mean_radius = float(
                    self.distances.einstein_radius_uas(
                        mean_mass,
                        dtype=torch.float64,
                    )
                )
                expected_stars = max(
                    1,
                    int(
                        round(
                            self.macro.compact_convergence
                            * aperture.radius_uas**2
                            / mean_radius**2
                        )
                    ),
                )
            lens_region = (
                rectangular_lens_region(
                    self.macro,
                    map_grid.region,
                    self.distances,
                    self.stellar_population.mass_function,
                    light_loss=(
                        self.light_loss
                        if self.rectangle_light_loss is None
                        else self.rectangle_light_loss
                    ),
                )
                if self.integration_domain is IntegrationDomain.RECTANGLE
                else aperture.bounding_region
            )
        else:
            assert self.stars is not None
            expected_stars = len(self.stars)
            lens_region = (
                _direct_star_region(self.stars, self.macro, map_grid.region)
                if self.lens_region is None
                else self.lens_region
            )
        if self.lens_region is not None:
            lens_region = self.lens_region
        runtime = (
            self.runtime
            if isinstance(self.runtime, ResolvedRuntime)
            else resolve_runtime(self.runtime)
        )
        item_size = torch.empty((), dtype=runtime.dtype).element_size()
        result = {
            "duration_days": duration,
            "source_shape": native_grid.shape,
            "source_field_of_view_uas": native_grid.field_of_view_uas,
            "map_field_of_view_uas": map_grid.field_of_view_uas,
            "lens_field_of_view_uas": lens_region.field_of_view_uas,
            "stellar_aperture_radius_uas": (
                None if aperture is None else aperture.radius_uas
            ),
            "estimated_star_count": expected_stars,
            "caustic_grid_shape": self.caustic_grid_shape,
            "device": str(runtime.device),
            "backend": runtime.backend.value,
            "dtype": str(runtime.dtype),
            "minimum_map_storage_bytes": math.prod(map_grid.shape) * item_size,
            "compilation_performed": False,
        }
        if display:
            for name, value in result.items():
                print(f"{name}: {value}")
        return result

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

    def _realize_for_times(self, times_days) -> MicrolensingRealization:
        """Return a realization whose stellar aperture covers ``times_days``."""

        times = torch.as_tensor(times_days, dtype=torch.float64).reshape(-1)
        required = max(abs(float(times.min())), abs(float(times.max())))
        if required <= float(self.duration_days) + 1.0e-10:
            return self.realize()
        key = float(required)
        cached = self._duration_realizations.get(key)
        if cached is None:
            cached = replace(self, duration_days=key).realize()
            self._duration_realizations[key] = cached
        return cached

    def magnification_map(self, **kwargs) -> MagnificationMap:
        """Generate one map through the cached realization."""

        return self.realize().magnification_map(
            **_with_method_options(kwargs, dynamic=False)
        )

    def dynamic_maps(self, times_days: Sequence[float], **kwargs):
        """Stream maps through the cached realization."""

        return self._realize_for_times(times_days).dynamic_maps(
            times_days,
            **_with_method_options(kwargs, dynamic=True),
        )

    def dynamic_labeled_maps(self, times_days: Sequence[float], **kwargs):
        """Stream maps and aligned label products through the realization."""

        return self._realize_for_times(times_days).dynamic_labeled_maps(
            times_days,
            **_with_method_options(kwargs, dynamic=True),
        )

    def light_curve(
        self,
        times_days: Sequence[float] | None = None,
        *,
        duration_days: float | None = None,
        map_cadence_days: float | None = None,
        source_cadence_days: float | None = None,
        start_day: float = 0.0,
        **kwargs,
    ) -> LightCurve:
        """Generate a light curve from explicit times or plain cadence values."""

        map_times = _cadence_times(
            times_days,
            duration_days=duration_days,
            cadence_days=map_cadence_days,
            start_day=start_day,
        )
        resolved = self._realize_for_times(map_times)
        call_kwargs = _with_method_options(kwargs, dynamic=True)
        if source_cadence_days is None or (
            map_cadence_days is not None
            and math.isclose(float(source_cadence_days), float(map_cadence_days))
        ):
            return resolved.light_curve(map_times, **call_kwargs)
        flux_times = _cadence_times(
            None,
            duration_days=(float(map_times[-1]) - float(map_times[0])),
            cadence_days=source_cadence_days,
            start_day=float(map_times[0]),
        )
        return resolved.multirate_light_curve(map_times, flux_times, **call_kwargs)

    def light_curves(self, times_days: Sequence[float], requests, **kwargs):
        """Batch multiple light curves through the cached realization."""

        return self._realize_for_times(times_days).light_curves(
            times_days,
            requests,
            **_with_method_options(kwargs, dynamic=True),
        )

    def multirate_light_curve(
        self,
        map_times_days: Sequence[float],
        flux_times_days: Sequence[float],
        **kwargs,
    ) -> LightCurve:
        """Evaluate source evolution more finely than the dynamic maps."""

        return self._realize_for_times(map_times_days).multirate_light_curve(
            map_times_days,
            flux_times_days,
            **_with_method_options(kwargs, dynamic=True),
        )

    def multirate_light_curve_with_labels(
        self,
        map_times_days: Sequence[float],
        flux_times_days: Sequence[float],
        **kwargs,
    ):
        """Evaluate fine-cadence flux with labels at map epochs."""

        return self._realize_for_times(
            map_times_days
        ).multirate_light_curve_with_labels(
            map_times_days,
            flux_times_days,
            **_with_method_options(kwargs, dynamic=True),
        )

    def light_curve_with_labels(
        self,
        times_days: Sequence[float] | None = None,
        *,
        duration_days: float | None = None,
        map_cadence_days: float | None = None,
        source_cadence_days: float | None = None,
        start_day: float = 0.0,
        **kwargs,
    ):
        """Generate labeled photometry from explicit times or cadence values."""

        map_times = _cadence_times(
            times_days,
            duration_days=duration_days,
            cadence_days=map_cadence_days,
            start_day=start_day,
        )
        resolved = self._realize_for_times(map_times)
        call_kwargs = _with_method_options(kwargs, dynamic=True)
        if source_cadence_days is None or (
            map_cadence_days is not None
            and math.isclose(float(source_cadence_days), float(map_cadence_days))
        ):
            return resolved.light_curve_with_labels(map_times, **call_kwargs)
        flux_times = _cadence_times(
            None,
            duration_days=float(map_times[-1]) - float(map_times[0]),
            cadence_days=source_cadence_days,
            start_day=float(map_times[0]),
        )
        return resolved.multirate_light_curve_with_labels(
            map_times,
            flux_times,
            **call_kwargs,
        )

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
