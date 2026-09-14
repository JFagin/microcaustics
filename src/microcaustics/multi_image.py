"""High-level simulations of resolved multiply imaged sources."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace

import torch

from .config import CausticConfig, DynamicConfig, IPMConfig, IRSConfig
from .geometry import PlaneGrid, PlaneRegion
from .lens import LensingDistances
from .results import (
    MacroImageLightCurve,
    MacroImageTransferFunctions,
    MultiImageLightCurves,
    MultiImageMapFrame,
    MultiImageTransferFunctions,
    TimingBreakdown,
)
from .simulation import MicrolensingSimulation
from .sources import PixelatedSource, ThermalReprocessingSource, TimeShiftedSource
from .trajectories import SourceTrajectory


@dataclass(frozen=True)
class MacroImageConfig:
    """Independent lens and numerical settings for one resolved macroimage.

    ``arrival_time_delay_days`` follows the usual arrival-delay convention:
    flux observed at time ``t`` uses intrinsic source emission at
    ``t - arrival_time_delay_days``. Point-lens motion and the source-plane
    trajectory remain functions of observer time.
    """

    name: str
    simulation: MicrolensingSimulation
    lens_region: PlaneRegion
    source_grid: PlaneGrid
    method: IPMConfig | IRSConfig
    arrival_time_delay_days: float = 0.0
    trajectory: SourceTrajectory | None = None
    schedule: DynamicConfig = field(default_factory=DynamicConfig)
    strict_coverage: bool = True
    lens_grid: PlaneGrid | None = None
    caustic_config: CausticConfig | None = None
    diagnostic_grid: PlaneGrid | None = None
    include_distance_map: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("macroimage name must be a non-empty string")
        if not math.isfinite(float(self.arrival_time_delay_days)):
            raise ValueError("arrival_time_delay_days must be finite")
        if not isinstance(self.method, IPMConfig | IRSConfig):
            raise TypeError("method must be an IPMConfig or IRSConfig")
        if self.caustic_config is not None and self.lens_grid is None:
            raise ValueError("caustic_config requires lens_grid")
        if self.diagnostic_grid is not None and self.lens_grid is None:
            raise ValueError("diagnostic_grid requires lens_grid")
        if self.include_distance_map and self.diagnostic_grid is None:
            raise ValueError("include_distance_map requires diagnostic_grid")


def _times_for_image(
    times_days: Sequence[float] | Mapping[str, Sequence[float]],
    image_name: str,
) -> tuple[float, ...]:
    values = times_days[image_name] if isinstance(times_days, Mapping) else times_days
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"macroimage {image_name!r} has an empty time axis")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"macroimage {image_name!r} times must be finite")
    return result


@dataclass(frozen=True)
class MultiImageSimulation:
    """A resolved lens system sharing one physical source across macroimages.

    Macroimages may have different macro lens parameters, point-mass fields,
    star motions, source trajectories, map geometries, numerical solvers,
    devices, and arrival-time delays. Calculations run image-by-image to keep
    peak memory bounded and to support heterogeneous devices.
    """

    images: tuple[MacroImageConfig, ...]

    def __post_init__(self) -> None:
        images = tuple(self.images)
        if not images:
            raise ValueError("at least one macroimage configuration is required")
        if not all(isinstance(item, MacroImageConfig) for item in images):
            raise TypeError("images must contain MacroImageConfig instances")
        names = tuple(item.name for item in images)
        if len(set(names)) != len(names):
            raise ValueError("macroimage names must be unique")
        object.__setattr__(self, "images", images)

    @classmethod
    def create(cls, images: Sequence[MacroImageConfig]) -> MultiImageSimulation:
        """Construct a resolved system from an ordered image sequence."""

        return cls(tuple(images))

    @property
    def image_names(self) -> tuple[str, ...]:
        """Return configured macroimage names in execution order."""

        return tuple(item.name for item in self.images)

    def image(self, name: str) -> MacroImageConfig:
        """Return one macroimage configuration by name."""

        for item in self.images:
            if item.name == name:
                return item
        raise KeyError(name)

    def with_arrival_time_delays(
        self,
        delays_days: Mapping[str, float],
        *,
        require_all: bool = False,
    ) -> MultiImageSimulation:
        """Return a copy with known or model-derived arrival delays applied.

        A partial mapping is useful when only some delays should override the
        values stored in :class:`MacroImageConfig`. Set ``require_all=True`` to
        enforce a complete solver result. The original simulation is unchanged.
        """

        supplied = {str(name): float(value) for name, value in delays_days.items()}
        unknown = set(supplied) - set(self.image_names)
        missing = set(self.image_names) - set(supplied)
        if unknown:
            raise ValueError(
                f"arrival delays contain unknown images: {sorted(unknown)}"
            )
        if require_all and missing:
            raise ValueError(f"arrival delays omit images: {sorted(missing)}")
        return MultiImageSimulation(
            tuple(
                replace(
                    image,
                    arrival_time_delay_days=supplied.get(
                        image.name,
                        image.arrival_time_delay_days,
                    ),
                )
                for image in self.images
            )
        )

    def _validate_time_mapping(
        self,
        times_days: Sequence[float] | Mapping[str, Sequence[float]],
    ) -> None:
        if not isinstance(times_days, Mapping):
            return
        supplied = set(times_days)
        expected = set(self.image_names)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(
                f"per-image time mapping mismatch. Missing={missing}, extra={extra}"
            )

    def dynamic_maps(
        self,
        times_days: Sequence[float] | Mapping[str, Sequence[float]],
    ) -> Iterator[MultiImageMapFrame]:
        """Stream source-independent maps in image-major, then time order."""

        self._validate_time_mapping(times_days)
        for image in self.images:
            image_times = _times_for_image(times_days, image.name)
            for magnification_map in image.simulation.dynamic_maps(
                image.lens_region,
                image.source_grid,
                image_times,
                method=image.method,
                schedule=image.schedule,
            ):
                magnification_map = replace(
                    magnification_map,
                    metadata={
                        **magnification_map.metadata,
                        "macro_image_name": image.name,
                        "arrival_time_delay_days": float(
                            image.arrival_time_delay_days
                        ),
                        "map_time_convention": "observer_time",
                        "multi_image_simulation": True,
                    },
                )
                yield MultiImageMapFrame(image.name, magnification_map)

    def light_curves(
        self,
        times_days: Sequence[float] | Mapping[str, Sequence[float]],
        source: PixelatedSource,
        distances: LensingDistances,
        *,
        include_labels: bool = False,
        map_observers: Mapping[str, Callable] | None = None,
    ) -> MultiImageLightCurves:
        """Generate resolved multiband light curves for every macroimage.

        A common time sequence produces directly stackable resolved curves;
        a name-to-time mapping permits distinct observing cadences. Optional
        caustic labels refer to the fixed center of each source grid. To avoid
        silently labeling a different path, labels currently require the
        macroimage trajectory to be omitted (the stationary default).
        """

        self._validate_time_mapping(times_days)
        observers = {} if map_observers is None else dict(map_observers)
        unknown_observers = set(observers) - set(self.image_names)
        if unknown_observers:
            names = sorted(unknown_observers)
            raise ValueError(
                f"map observers contain unknown macroimages: {names}"
            )
        outputs = []
        component_seconds: dict[str, float] = {}
        for image in self.images:
            image_times = _times_for_image(times_days, image.name)
            delayed_source = TimeShiftedSource(
                source,
                image.arrival_time_delay_days,
            )
            observer = observers.get(image.name)
            if include_labels:
                if image.lens_grid is None:
                    raise ValueError(
                        f"macroimage {image.name!r} requires lens_grid for labels"
                    )
                if image.trajectory is not None:
                    raise ValueError(
                        "center labels currently require trajectory=None. Encode "
                        "bulk relative motion in the image point-mass velocities"
                    )
                labeled = image.simulation.light_curve_with_labels(
                    image.lens_region,
                    image.source_grid,
                    image.lens_grid,
                    image_times,
                    delayed_source,
                    distances,
                    method=image.method,
                    trajectory=None,
                    map_schedule=image.schedule,
                    caustic_config=image.caustic_config,
                    strict_coverage=image.strict_coverage,
                    diagnostic_grid=image.diagnostic_grid,
                    include_distance_map=image.include_distance_map,
                    map_observer=observer,
                )
                curve = labeled.light_curve
                caustics = labeled.caustics
            else:
                curve = image.simulation.light_curve(
                    image.lens_region,
                    image.source_grid,
                    image_times,
                    delayed_source,
                    distances,
                    method=image.method,
                    trajectory=image.trajectory,
                    schedule=image.schedule,
                    strict_coverage=image.strict_coverage,
                    map_observer=observer,
                )
                caustics = None
            curve = replace(
                curve,
                metadata={
                    **curve.metadata,
                    "macro_image_name": image.name,
                    "arrival_time_delay_days": float(
                        image.arrival_time_delay_days
                    ),
                    "source_time_convention": (
                        "observer_time_minus_arrival_delay"
                    ),
                    "multi_image_simulation": True,
                },
            )
            outputs.append(
                MacroImageLightCurve(
                    image.name,
                    image.arrival_time_delay_days,
                    curve,
                    caustics,
                )
            )
            component_seconds[image.name] = curve.timing.delivered_seconds
        total_seconds = sum(component_seconds.values())
        return MultiImageLightCurves(
            tuple(outputs),
            metadata={
                "method": "resolved_multi_image_light_curves",
                "image_names": self.image_names,
                "image_count": len(self.images),
                "shared_source": dict(source.metadata()),
                "execution_order": "image_major",
                "maps_retained": False,
                "labels_included": bool(include_labels),
            },
            timing=TimingBreakdown(
                collected=all(item.light_curve.timing.collected for item in outputs),
                steady_seconds=total_seconds,
                component_seconds=component_seconds,
            ),
        )

    def multirate_light_curves(
        self,
        map_times_days: Sequence[float] | Mapping[str, Sequence[float]],
        flux_times_days: Sequence[float] | Mapping[str, Sequence[float]],
        source: PixelatedSource,
        distances: LensingDistances,
        *,
        include_labels: bool = False,
        map_observers: Mapping[str, Callable] | None = None,
    ) -> MultiImageLightCurves:
        """Generate fine-cadence resolved curves from sparse dynamic maps.

    This is the efficient multi-rate workflow. Intrinsic source and
        reverberation variability may be evaluated daily while each independent
        microlensing field is generated on a slower cadence. Explicit arrival
        delays on :class:`MacroImageConfig` are applied to source emission only.
        When ``include_labels`` is true, caustic products remain tied to the
        sparse map epochs while source photometry retains its finer cadence.
        """

        self._validate_time_mapping(map_times_days)
        self._validate_time_mapping(flux_times_days)
        observers = {} if map_observers is None else dict(map_observers)
        unknown_observers = set(observers) - set(self.image_names)
        if unknown_observers:
            raise ValueError(
                "map observers contain unknown macroimages: "
                f"{sorted(unknown_observers)}"
            )
        outputs = []
        component_seconds: dict[str, float] = {}
        for image in self.images:
            map_times = _times_for_image(map_times_days, image.name)
            flux_times = _times_for_image(flux_times_days, image.name)
            delayed_source = TimeShiftedSource(
                source,
                image.arrival_time_delay_days,
            )
            if include_labels:
                if image.lens_grid is None:
                    raise ValueError(
                        f"macroimage {image.name!r} requires lens_grid for labels"
                    )
                if image.trajectory is not None:
                    raise ValueError(
                        "center labels currently require trajectory=None. Encode "
                        "bulk relative motion in the image point-mass velocities"
                    )
                labeled = image.simulation.multirate_light_curve_with_labels(
                    image.lens_region,
                    image.source_grid,
                    image.lens_grid,
                    map_times,
                    flux_times,
                    delayed_source,
                    distances,
                    method=image.method,
                    trajectory=None,
                    map_schedule=image.schedule,
                    caustic_config=image.caustic_config,
                    strict_coverage=image.strict_coverage,
                    diagnostic_grid=image.diagnostic_grid,
                    include_distance_map=image.include_distance_map,
                    map_observer=observers.get(image.name),
                )
                curve = labeled.light_curve
                caustics = labeled.caustics
            else:
                curve = image.simulation.multirate_light_curve(
                    image.lens_region,
                    image.source_grid,
                    map_times,
                    flux_times,
                    delayed_source,
                    distances,
                    method=image.method,
                    trajectory=image.trajectory,
                    schedule=image.schedule,
                    strict_coverage=image.strict_coverage,
                    map_observer=observers.get(image.name),
                )
                caustics = None
            curve = replace(
                curve,
                metadata={
                    **curve.metadata,
                    "macro_image_name": image.name,
                    "arrival_time_delay_days": float(
                        image.arrival_time_delay_days
                    ),
                    "source_time_convention": (
                        "observer_time_minus_arrival_delay"
                    ),
                    "multi_image_simulation": True,
                },
            )
            outputs.append(
                MacroImageLightCurve(
                    image.name,
                    image.arrival_time_delay_days,
                    curve,
                    caustics,
                )
            )
            component_seconds[image.name] = curve.timing.delivered_seconds
        total_seconds = sum(component_seconds.values())
        return MultiImageLightCurves(
            tuple(outputs),
            metadata={
                "method": "resolved_multi_image_multirate_light_curves",
                "image_names": self.image_names,
                "image_count": len(self.images),
                "shared_source": dict(source.metadata()),
                "execution_order": "image_major",
                "maps_retained": False,
                "labels_included": bool(include_labels),
                "arrival_delays": "explicit_per_image",
            },
            timing=TimingBreakdown(
                collected=all(item.light_curve.timing.collected for item in outputs),
                steady_seconds=total_seconds,
                component_seconds=component_seconds,
            ),
        )

    def transfer_functions(
        self,
        map_times_days: Sequence[float] | Mapping[str, Sequence[float]],
        source: ThermalReprocessingSource,
        distances: LensingDistances,
        delay_edges_days: torch.Tensor | Sequence[float],
        *,
        normalize: bool = True,
        map_observers: Mapping[str, Callable] | None = None,
        response_batch_size: int | None = None,
        response_spatial_chunk_size: int = 262_144,
    ) -> MultiImageTransferFunctions:
        """Generate microlensing-weighted response functions per image.

        The response is evaluated at the sparse map cadence. Macro arrival
        delays are reported separately from the internal reverberation delay,
        so known or model-derived inter-image delays remain interchangeable.
        """

        from .transfer_functions import streaming_microlensed_transfer_functions

        self._validate_time_mapping(map_times_days)
        observers = {} if map_observers is None else dict(map_observers)
        unknown_observers = set(observers) - set(self.image_names)
        if unknown_observers:
            raise ValueError(
                "map observers contain unknown macroimages: "
                f"{sorted(unknown_observers)}"
            )
        edges = torch.as_tensor(delay_edges_days)
        if edges.ndim != 1 or edges.numel() < 2 or not bool(
            torch.all(edges[1:] > edges[:-1])
        ):
            raise ValueError("delay_edges_days must be strictly increasing")

        outputs = []
        component_seconds: dict[str, float] = {}
        for image in self.images:
            image_times = _times_for_image(map_times_days, image.name)
            runtime = image.simulation.runtime
            device, dtype = runtime.device, runtime.dtype
            series = streaming_microlensed_transfer_functions(
                image.simulation,
                image.lens_region,
                image.source_grid,
                image_times,
                source,
                distances,
                edges,
                method=image.method,
                trajectory=image.trajectory,
                schedule=image.schedule,
                strict_coverage=image.strict_coverage,
                normalize=normalize,
                map_observer=observers.get(image.name),
                response_batch_size=response_batch_size,
                response_spatial_chunk_size=response_spatial_chunk_size,
            )
            outputs.append(
                MacroImageTransferFunctions(
                    image_name=image.name,
                    arrival_time_delay_days=image.arrival_time_delay_days,
                    map_times_days=series.times_days,
                    delay_edges_days=edges.to(device=device, dtype=dtype),
                    values=series.values,
                    mean_delays_days=series.mean_delays_days,
                    band_names=source.geometry.band_names,
                    metadata={
                        **series.metadata,
                        "trajectory_time_convention": "observer_time",
                        "arrival_delay_applied_to_response_bins": False,
                    },
                    timing=series.timing,
                )
            )
            component_seconds[image.name] = series.timing.steady_seconds
        total_seconds = sum(component_seconds.values())
        return MultiImageTransferFunctions(
            tuple(outputs),
            metadata={
                "method": "resolved_multi_image_transfer_functions",
                "image_names": self.image_names,
                "macro_arrival_delays_separate": True,
            },
            timing=TimingBreakdown(
                collected=all(item.timing.collected for item in outputs),
                steady_seconds=total_seconds,
                component_seconds=component_seconds,
            ),
        )
