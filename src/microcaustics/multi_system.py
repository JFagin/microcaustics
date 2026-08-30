"""High-level multi-image microlensing systems."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from .lens import LensingDistances, MacroLens, PointMassField, StellarPopulation
from .multi_image import MacroImageConfig, MultiImageSimulation, _times_for_image
from .runtime import ResolvedRuntime
from .random import derive_seed
from .sources import PhysicalSourceModel, PixelatedSource, ThermalReprocessingSource
from .system import (
    IntegrationDomain,
    MicrolensingSystem,
    _cadence_times,
    _production_dynamic_settings,
    _with_method_options,
)
from .trajectories import SourceTrajectory

if TYPE_CHECKING:
    from .strong_lensing import MacroImageSolution


def _image_value(value, name: str):
    return value.get(name) if isinstance(value, Mapping) else value


def _image_seed(value, name: str, index: int) -> int | None:
    """Resolve a per-image seed from a mapping or reproducible base seed."""

    if isinstance(value, Mapping):
        selected = value.get(name)
        return None if selected is None else int(selected)
    if value is None:
        return None
    return derive_seed(int(value), f"image:{name}")


def _required_image_value(value, name: str, label: str):
    """Resolve a scalar or require one value for the named image."""

    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"{label} omit macroimage {name!r}")
        return value[name]
    return value


@dataclass(frozen=True)
class MultiImageSystem:
    """Resolved macroimages that share one physical source.

    The concise interface maps each image name directly to a
    :class:`~microcaustics.MacroLens` and supplies the shared distances, source,
    and stellar-population prescription once. The class constructs an
    independent seeded :class:`MicrolensingSystem` for every image. Existing
    fully constructed systems remain accepted for per-image expert control.

    Arrival delays shift only source emission. Map evolution remains on the
    observer-time axis. A scalar ``seed`` is treated as a reproducible base
    seed, with stable independent image seeds derived from each image name. A
    seed mapping provides exact per-image values.
    """

    images: Mapping[str, MicrolensingSystem | MacroLens]
    arrival_time_delays_days: Mapping[str, float] = field(default_factory=dict)
    methods: IPMConfig | IRSConfig | Mapping[str, IPMConfig | IRSConfig] | None = None
    schedules: DynamicConfig | Mapping[str, DynamicConfig] = field(
        default_factory=production_dynamic_config
    )
    trajectories: SourceTrajectory | Mapping[str, SourceTrajectory] | None = None
    caustic_configs: CausticConfig | Mapping[str, CausticConfig] | None = None
    source: PixelatedSource | PhysicalSourceModel | None = None
    distances: LensingDistances | None = None
    source_grid: PlaneGrid | None = None
    stellar_population: StellarPopulation | Mapping[str, StellarPopulation] | None = (
        None
    )
    stars: PointMassField | Mapping[str, PointMassField] | None = None
    integration_domain: (
        IntegrationDomain | str | Mapping[str, IntegrationDomain | str]
    ) = IntegrationDomain.SCOUT
    duration_days: float = 0.0
    light_loss: float = 0.01
    safety_scale: float = 1.5
    stellar_motion_sigma_margin: float = 5.0
    rectangle_light_loss: float | None = None
    source_support_radius_uas: float | None = None
    seed: int | Mapping[str, int | None] | None = None
    runtime: (
        RuntimeConfig
        | ResolvedRuntime
        | Mapping[str, RuntimeConfig | ResolvedRuntime]
        | None
    ) = None
    lens_region: PlaneRegion | Mapping[str, PlaneRegion] | None = None
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
    ) -> MultiImageSystem:
        """Construct all macroimages from one shared redshift geometry."""

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

    @classmethod
    def from_macroimage_solutions(
        cls,
        solutions: Sequence[MacroImageSolution],
        *,
        smooth_matter_fraction: float | Mapping[str, float] = 0.0,
        arrival_time_delays_days: Mapping[str, float] | None = None,
        **kwargs,
    ) -> MultiImageSystem:
        """Build directly from global macro-model image solutions.

        Local convergence, shear, shear direction, and model arrival delays are
        read from each solution. Measured delays may replace the modeled values
        through ``arrival_time_delays_days`` without changing the macro lenses.
        All remaining keywords are the shared or per-image physical fields of
        :class:`MultiImageSystem`.
        """

        resolved = tuple(solutions)
        if not resolved:
            raise ValueError("at least one macroimage solution is required")
        names = tuple(item.name for item in resolved)
        if len(set(names)) != len(names):
            raise ValueError("macroimage solution names must be unique")
        images = {
            item.name: item.local_macro_lens(
                smooth_matter_fraction=float(
                    _image_value(smooth_matter_fraction, item.name)
                )
            )
            for item in resolved
        }
        delays = (
            {item.name: float(item.arrival_time_delay_days) for item in resolved}
            if arrival_time_delays_days is None
            else dict(arrival_time_delays_days)
        )
        return cls(
            images=images,
            arrival_time_delays_days=delays,
            **kwargs,
        )

    def __post_init__(self) -> None:
        images = dict(self.images)
        if not images:
            raise ValueError("at least one macroimage is required")
        if any(not name for name in images):
            raise ValueError("macroimage names must be non-empty")
        if not all(
            isinstance(system, (MicrolensingSystem, MacroLens))
            for system in images.values()
        ):
            raise TypeError(
                "images must map names to MacroLens or MicrolensingSystem objects"
            )
        delays = {
            name: float(value) for name, value in self.arrival_time_delays_days.items()
        }
        unknown = set(delays) - set(images)
        if unknown:
            raise ValueError(
                f"arrival delays contain unknown images: {sorted(unknown)}"
            )
        if any(not math.isfinite(value) for value in delays.values()):
            raise ValueError("arrival delays must be finite")
        for value, label in (
            (self.methods, "methods"),
            (self.schedules, "schedules"),
            (self.trajectories, "trajectories"),
            (self.caustic_configs, "caustic_configs"),
            (self.stellar_population, "stellar_population"),
            (self.stars, "stars"),
            (self.integration_domain, "integration_domain"),
            (self.seed, "seed"),
            (self.runtime, "runtime"),
            (self.lens_region, "lens_region"),
        ):
            if isinstance(value, Mapping):
                extra = set(value) - set(images)
                if extra:
                    raise ValueError(f"{label} contain unknown images: {sorted(extra)}")
        resolved_images: dict[str, MicrolensingSystem] = {}
        for index, (name, image) in enumerate(images.items()):
            if isinstance(image, MicrolensingSystem):
                resolved_images[name] = image
                continue
            if self.distances is None:
                raise ValueError(
                    "distances are required when images contain MacroLens objects"
                )
            population = _image_value(self.stellar_population, name)
            stars = _image_value(self.stars, name)
            if (population is None) == (stars is None):
                raise ValueError(
                    "supply exactly one shared or per-image stellar_population "
                    "or stars value when images contain MacroLens objects"
                )
            resolved_images[name] = MicrolensingSystem(
                macro=image,
                distances=self.distances,
                source=self.source,
                source_grid=self.source_grid,
                stellar_population=population,
                stars=stars,
                integration_domain=_image_value(self.integration_domain, name),
                duration_days=self.duration_days,
                trajectory=_image_value(self.trajectories, name),
                light_loss=self.light_loss,
                safety_scale=self.safety_scale,
                stellar_motion_sigma_margin=self.stellar_motion_sigma_margin,
                rectangle_light_loss=self.rectangle_light_loss,
                source_support_radius_uas=self.source_support_radius_uas,
                seed=_image_seed(self.seed, name, index),
                runtime=_image_value(self.runtime, name),
                lens_region=_image_value(self.lens_region, name),
                caustic_grid_shape=self.caustic_grid_shape,
            )
        object.__setattr__(self, "images", resolved_images)
        object.__setattr__(self, "arrival_time_delays_days", delays)

    @property
    def image_names(self) -> tuple[str, ...]:
        """Return macroimage names in execution order."""

        return tuple(self.images)

    def image(self, name: str) -> MicrolensingSystem:
        """Return one resolved physical image system by name."""

        try:
            return self.images[name]
        except KeyError as error:
            raise KeyError(f"unknown macroimage {name!r}") from error

    def summary(
        self,
        *,
        times_days: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
        duration_days: float | None = None,
        display: bool = True,
    ) -> dict[str, dict[str, object]]:
        """Describe all derived image geometries without running a solver."""

        summaries = {
            name: system.summary(
                times_days=(
                    None
                    if times_days is None
                    else _times_for_image(times_days, name)
                ),
                duration_days=duration_days,
                display=False,
            )
            for name, system in self.images.items()
        }
        if display:
            for name, values in summaries.items():
                print(f"[{name}]")
                for key, value in values.items():
                    print(f"{key}: {value}")
        return summaries

    def with_arrival_time_delays(
        self,
        delays_days: Mapping[str, float],
        *,
        require_all: bool = False,
    ) -> MultiImageSystem:
        """Return a copy with measured or model-derived relative delays."""

        supplied = {name: float(value) for name, value in delays_days.items()}
        unknown = set(supplied) - set(self.images)
        if unknown:
            raise ValueError(
                f"arrival delays contain unknown images: {sorted(unknown)}"
            )
        if require_all and set(supplied) != set(self.images):
            omitted = sorted(set(self.images) - set(supplied))
            raise ValueError(f"arrival delays omit macroimages: {omitted}")
        return replace(
            self,
            arrival_time_delays_days={
                **self.arrival_time_delays_days,
                **supplied,
            },
        )

    def _build_simulation(
        self,
        times_days: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
        solver_options: Mapping[str, object] | None = None,
    ) -> tuple[MultiImageSimulation, tuple]:
        """Build the numerical images with geometry covering requested times."""

        configs = []
        realizations = []
        for name, system in self.images.items():
            realization = (
                system.realize()
                if times_days is None
                else system._realize_for_times(_times_for_image(times_days, name))
            )
            realizations.append(realization)
            method = _image_value(self.methods, name)
            if method is None:
                method = production_ipm_config()
            if solver_options:
                method = _with_method_options(
                    {"method": method, **dict(solver_options)}, dynamic=True
                )["method"]
            method = realization._method_for_domain(method)
            schedule = _image_value(self.schedules, name)
            caustic_config = _image_value(self.caustic_configs, name)
            schedule, caustic_config = _production_dynamic_settings(
                method,
                schedule,
                caustic_config,
            )
            configs.append(
                MacroImageConfig(
                    name=name,
                    simulation=realization.simulation,
                    lens_region=realization.lens_region,
                    source_grid=realization.source_grid,
                    method=method,
                    arrival_time_delay_days=self.arrival_time_delays_days.get(
                        name, 0.0
                    ),
                    trajectory=_image_value(self.trajectories, name),
                    schedule=schedule,
                    lens_grid=realization.lens_grid,
                    caustic_config=caustic_config,
                )
            )
        return MultiImageSimulation(tuple(configs)), tuple(realizations)

    @cached_property
    def simulation(self) -> MultiImageSimulation:
        """Return the cached low-level multi-image simulation."""

        return self._build_simulation()[0]

    def _shared_source(
        self,
        source: PixelatedSource | None,
        realizations=None,
    ) -> PixelatedSource:
        if source is not None:
            return source
        if realizations is None:
            realizations = tuple(system.realize() for system in self.images.values())
        resolved = tuple(realization.source for realization in realizations)
        if any(item is None for item in resolved):
            raise ValueError("multi-image light curves require a shared source")
        first = resolved[0]
        assert first is not None
        if any(item.geometry != first.geometry for item in resolved[1:]):
            raise ValueError("all macroimages must resolve the same source geometry")
        return first

    def _shared_distances(self) -> LensingDistances:
        """Return common distances after verifying all image systems."""

        first = next(iter(self.images.values())).distances
        if any(system.distances != first for system in tuple(self.images.values())[1:]):
            raise ValueError("all macroimages must share lensing distances")
        return first

    def dynamic_maps(
        self,
        times_days: Sequence[float] | Mapping[str, Sequence[float]],
        **solver_options,
    ):
        """Stream dynamic maps for every macroimage."""

        return self._build_simulation(times_days, solver_options)[0].dynamic_maps(
            times_days
        )

    def magnification_maps(
        self,
        *,
        time_days: float | Mapping[str, float] = 0.0,
        methods: (
            IPMConfig | IRSConfig | Mapping[str, IPMConfig | IRSConfig] | None
        ) = None,
        **solver_options,
    ):
        """Return one independent magnification map for every macroimage.

        Static maps default to the complete ``k=1`` production scout. Pass one
        shared method or a mapping to compare image-specific numerical choices.
        """

        outputs = {}
        for name, system in self.images.items():
            method = _image_value(methods, name)
            outputs[name] = system.magnification_map(
                method=method or _production_static_ipm_config(),
                time_days=float(_required_image_value(time_days, name, "time_days")),
                **solver_options,
            )
        return outputs

    def caustics(
        self,
        *,
        time_days: float | Mapping[str, float] = 0.0,
        configs: CausticConfig | Mapping[str, CausticConfig] | None = None,
    ):
        """Return critical curves and caustics for every macroimage."""

        return {
            name: system.caustics(
                time_days=float(_required_image_value(time_days, name, "time_days")),
                config=_image_value(configs, name),
            )
            for name, system in self.images.items()
        }

    def labeled_caustics(
        self,
        *,
        time_days: float | Mapping[str, float] = 0.0,
        configs: CausticConfig | Mapping[str, CausticConfig] | None = None,
    ):
        """Return source-center caustic labels for every macroimage."""

        return {
            name: system.labeled_caustics(
                time_days=float(_required_image_value(time_days, name, "time_days")),
                config=_image_value(configs, name),
            )
            for name, system in self.images.items()
        }

    def light_curves(
        self,
        times_days: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
        *,
        duration_days: float | None = None,
        map_cadence_days: float | None = None,
        source_cadence_days: float | None = None,
        start_day: float = 0.0,
        source: PixelatedSource | None = None,
        include_labels: bool = False,
        map_observers=None,
        **solver_options,
    ):
        """Generate delayed curves from explicit times or plain cadences."""

        if times_days is None:
            map_times = _cadence_times(
                None,
                duration_days=duration_days,
                cadence_days=map_cadence_days,
                start_day=start_day,
            )
        else:
            if duration_days is not None or map_cadence_days is not None:
                raise ValueError(
                    "supply times_days or duration_days/map_cadence_days, not both"
                )
            map_times = times_days
        simulation, realizations = self._build_simulation(
            map_times, solver_options
        )
        shared_source = self._shared_source(source, realizations)
        use_multirate = source_cadence_days is not None and not (
            map_cadence_days is not None
            and math.isclose(float(source_cadence_days), float(map_cadence_days))
        )
        if use_multirate:
            if isinstance(map_times, Mapping):
                raise ValueError(
                    "source_cadence_days requires one shared regular map cadence"
                )
            flux_times = _cadence_times(
                None,
                duration_days=float(map_times[-1]) - float(map_times[0]),
                cadence_days=source_cadence_days,
                start_day=float(map_times[0]),
            )
            return simulation.multirate_light_curves(
                map_times,
                flux_times,
                shared_source,
                self._shared_distances(),
                include_labels=include_labels,
                map_observers=map_observers,
            )
        return simulation.light_curves(
            map_times,
            shared_source,
            self._shared_distances(),
            include_labels=include_labels,
            map_observers=map_observers,
        )

    def multirate_light_curves(
        self,
        map_times_days: Sequence[float] | Mapping[str, Sequence[float]],
        flux_times_days: Sequence[float] | Mapping[str, Sequence[float]],
        *,
        source: PixelatedSource | None = None,
        include_labels: bool = False,
        map_observers=None,
        **solver_options,
    ):
        """Generate fine-cadence variability from sparse dynamic maps."""

        simulation, realizations = self._build_simulation(
            map_times_days, solver_options
        )
        return simulation.multirate_light_curves(
            map_times_days,
            flux_times_days,
            self._shared_source(source, realizations),
            self._shared_distances(),
            include_labels=include_labels,
            map_observers=map_observers,
        )

    def transfer_functions(
        self,
        map_times_days: Sequence[float] | Mapping[str, Sequence[float]],
        delay_edges_days,
        *,
        source: ThermalReprocessingSource | None = None,
        normalize: bool = True,
        map_observers=None,
        **solver_options,
    ):
        """Generate microlensed transfer functions for every macroimage."""

        simulation, realizations = self._build_simulation(
            map_times_days, solver_options
        )
        resolved_source = self._shared_source(source, realizations)
        if not isinstance(resolved_source, ThermalReprocessingSource):
            raise TypeError("transfer_functions requires a ThermalReprocessingSource")
        return simulation.transfer_functions(
            map_times_days,
            resolved_source,
            self._shared_distances(),
            delay_edges_days,
            normalize=normalize,
            map_observers=map_observers,
        )


# Import compatibility for code written before the shorter public name.
MultiImageMicrolensingSystem = MultiImageSystem
