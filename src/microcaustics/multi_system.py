"""High-level multi-image microlensing systems."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

import torch

from ._system.scheduling import (
    _light_curve_options,
    _light_curve_times,
    _production_dynamic_settings,
    _retaining_map_observer,
)
from .config import (
    CausticConfig,
    DynamicConfig,
    IPMConfig,
    IRSConfig,
    RuntimeConfig,
    _production_static_ipm_config,
    production_ipm_config,
)
from .geometry import PlaneGrid, PlaneRegion
from .lens import (
    LensingDistances,
    MacroLens,
    PointMassField,
    SkyProjectedKinematics,
    StellarPopulation,
)
from .multi_image import MacroImageConfig, MultiImageSimulation, _times_for_image
from .random import derive_seed
from .runtime import ResolvedRuntime
from .sources import (
    PhysicalSourceModel,
    PixelatedSource,
    ThermalReprocessingSource,
)
from .sources.variability import (
    _FixedHorizonDrivingSignal,
    _source_driving_signal,
    _validate_source_driver,
)
from .system import (
    IntegrationDomain,
    MicrolensingSystem,
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


def _shared_kinematic_seed(seed) -> int | None:
    """Return one physical-motion seed shared by every macroimage."""

    if isinstance(seed, Mapping) and any(value is not None for value in seed.values()):
        label = "kinematics:" + repr(sorted(seed.items()))
        return derive_seed(0, label)
    if isinstance(seed, Mapping):
        return None
    return derive_seed(seed, "kinematics")


def _bind_shared_sky_kinematics(
    population: StellarPopulation | None,
    seed: int | None,
) -> StellarPopulation | None:
    """Bind an unseeded sky prescription before deriving per-image seeds."""

    if population is None or seed is None:
        return population
    kinematics = population.kinematics
    if not isinstance(kinematics, SkyProjectedKinematics) or kinematics.seed is not None:
        return population
    return replace(population, kinematics=replace(kinematics, seed=seed))


@dataclass(frozen=True)
class MultiImageSystem:
    """Resolved macroimages that share one physical source.

    The concise interface maps each image name directly to a
    :class:`~microcaustics.MacroLens` and supplies the shared distances, source,
    and stellar-population prescription once. The class constructs an
    independent seeded :class:`MicrolensingSystem` for every image. Existing
    fully constructed systems remain accepted for per-image expert control.

    Standard workflows supply ``lens_redshift`` and ``source_redshift`` once.
    Explicit :class:`LensingDistances` remain available for another cosmology.

    Arrival delays shift only source emission. Map evolution remains on the
    observer-time axis. A scalar ``seed`` is treated as a reproducible base
    seed, with stable independent image seeds derived from each image name. A
    seed mapping provides exact per-image values.
    """

    images: Mapping[str, MicrolensingSystem | MacroLens]
    arrival_time_delays_days: Mapping[str, float] = field(default_factory=dict)
    methods: IPMConfig | IRSConfig | Mapping[str, IPMConfig | IRSConfig] | None = None
    schedules: DynamicConfig | Mapping[str, DynamicConfig] | None = None
    trajectories: SourceTrajectory | Mapping[str, SourceTrajectory] | None = None
    caustic_configs: CausticConfig | Mapping[str, CausticConfig] | None = None
    source: PixelatedSource | PhysicalSourceModel | None = None
    distances: LensingDistances | None = None
    lens_redshift: float | None = None
    source_redshift: float | None = None
    H0: float = 67.66
    Om0: float = 0.30966
    distance_dtype: torch.dtype = torch.float32
    distance_device: torch.device | str | None = "auto"
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
        _validate_source_driver(self.source)
        if self.distances is None and (
            self.lens_redshift is not None or self.source_redshift is not None
        ):
            if self.lens_redshift is None or self.source_redshift is None:
                raise ValueError("supply both lens_redshift and source_redshift")
            object.__setattr__(
                self,
                "distances",
                LensingDistances.from_redshifts(
                    self.lens_redshift,
                    self.source_redshift,
                    H0=self.H0,
                    Om0=self.Om0,
                    dtype=self.distance_dtype,
                    device=self.distance_device,
                ),
            )
            object.__setattr__(self, "lens_redshift", None)
            object.__setattr__(self, "source_redshift", None)
        elif self.distances is not None and (
            self.lens_redshift is not None or self.source_redshift is not None
        ):
            raise ValueError("supply redshifts or distances, not both")
        images = dict(self.images)
        if not images:
            raise ValueError("at least one macroimage is required")
        if any(not name for name in images):
            raise ValueError("macroimage names must be non-empty")
        if not all(
            isinstance(system, MicrolensingSystem | MacroLens)
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
        shared_kinematic_seed = _shared_kinematic_seed(self.seed)
        resolved_images: dict[str, MicrolensingSystem] = {}
        for index, (name, image) in enumerate(images.items()):
            if isinstance(image, MicrolensingSystem):
                if self.source is not None and image.source is not self.source:
                    image = image.with_source(self.source)
                if self._shared_driving_signal is not None:
                    image = image._with_shared_realization_state(
                        self._shared_driving_signal
                    )
                resolved_images[name] = image
                continue
            if self.distances is None:
                raise ValueError(
                    "distances are required when images contain MacroLens objects"
                )
            population = _bind_shared_sky_kinematics(
                _image_value(self.stellar_population, name),
                shared_kinematic_seed,
            )
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
            if self._shared_driving_signal is not None:
                resolved_images[name] = resolved_images[
                    name
                ]._with_shared_realization_state(self._shared_driving_signal)
        object.__setattr__(self, "images", resolved_images)
        object.__setattr__(self, "arrival_time_delays_days", delays)

    @cached_property
    def _shared_driving_signal(self):
        """One intrinsic driver, independent of the order and star RNG of images."""

        signal = _source_driving_signal(self.source)
        if self.source is None:
            images = tuple(self.images.values())
            drivers = tuple(
                _source_driving_signal(image.source)
                if isinstance(image, MicrolensingSystem)
                else None
                for image in images
            )
            signal = drivers[0]
            if any(driver is not signal for driver in drivers[1:]):
                raise ValueError(
                    "macroimages must share one source driving signal. "
                    "Supply a shared source or use independent system batching."
                )
            if self.seed is None and signal is not None:
                return images[0]._bound_driving_signal
        if isinstance(signal, _FixedHorizonDrivingSignal):
            specification = signal
            if isinstance(self.seed, Mapping):
                seed = derive_seed(0, "variability:" + repr(sorted(self.seed.items())))
            else:
                seed = derive_seed(self.seed, "variability")
            signal = signal.with_seed(seed)
            # Delay-only copies keep an already bound unseeded realization too.
            # A new source specification or inherited seed deliberately rebinds.
            for image in self.images.values():
                if (
                    isinstance(image, MicrolensingSystem)
                    and _source_driving_signal(image.source) is specification
                    and image._shared_driver is not None
                    and image._shared_driver.seed == signal.seed
                ):
                    return image._shared_driver
        return signal

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
                    None if times_days is None else _times_for_image(times_days, name)
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

    def with_source(
        self,
        source: PixelatedSource | PhysicalSourceModel,
    ) -> MultiImageSystem:
        """Return all macroimages with one replacement shared source."""

        return replace(
            self,
            images={
                name: image.with_source(source) for name, image in self.images.items()
            },
            source=source,
            source_grid=None,
        )

    def _build_simulation(
        self,
        times_days: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
        solver_options: Mapping[str, object] | None = None,
        *,
        include_labels: bool = False,
    ) -> tuple[MultiImageSimulation, tuple]:
        """Build the numerical images with geometry covering requested times."""

        configs = []
        realizations = []
        controls_by_image = {}
        for key, value in (solver_options or {}).items():
            if isinstance(value, Mapping) and set(value) - set(self.image_names):
                raise ValueError(f"{key} contains unknown macroimages")
        # Validate every image before realizing any source, stars, or GPU state.
        for name in self.image_names:
            method = _image_value(self.methods, name)
            if method is None:
                method = production_ipm_config()
            schedule = _image_value(self.schedules, name)
            caustic_config = _image_value(self.caustic_configs, name)
            controls = _light_curve_options(
                {
                    "method": method,
                    "schedule": schedule,
                    **({"caustics": caustic_config} if include_labels else {}),
                    **{
                        key: _image_value(value, name)
                        for key, value in (solver_options or {}).items()
                        if not isinstance(value, Mapping) or name in value
                    },
                },
                include_labels=include_labels,
            )
            controls_by_image[name] = controls
        for name, system in self.images.items():
            realization = (
                system.realize()
                if times_days is None
                else system._realize_for_times(_times_for_image(times_days, name))
            )
            realizations.append(realization)
            controls = controls_by_image[name]
            method = realization._method_for_domain(controls["method"])
            schedule = controls["schedule"]
            caustic_config = controls.get("caustics")
            schedule, caustic_config = _production_dynamic_settings(
                method,
                schedule,
                caustic_config,
                include_labels=include_labels,
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
        flux_times_days: Sequence[float] | Mapping[str, Sequence[float]] | None = None,
        start_day: float = 0.0,
        source: PixelatedSource | PhysicalSourceModel | None = None,
        include_labels: bool = False,
        apply_driving_signal: bool | None = None,
        keep_maps_at_days: Sequence[float]
        | Mapping[str, Sequence[float]]
        | None = None,
        map_observers=None,
        **solver_options,
    ):
        """Generate resolved curves sharing one source and intrinsic driver.

        Supply ``duration_days`` and ``map_cadence_days``, or explicit times
        shared by all images or mapped by image name. ``source_cadence_days``
        sets a separate photometry cadence. Arrival delays shift source emission,
        not stellar motion. ``apply_driving_signal=None`` uses an attached driver,
        ``False`` retains its mean heating, and ``True`` requires a driver.

        Plain ``rays``, ``temporal_batch_size``, ``scout_refresh_frames``, and
        ``label_batch_size`` override advanced method and schedule configurations.
        Solver controls may also be mapped by image name. Unknown options raise
        an error before numerical setup. Labels are opt-in with ``include_labels``.

        Each result image exposes ``flux`` in Jy and apparent AB ``magnitude``
        shaped [photometry epoch, band]. ``labels.times_days`` gives the separate
        label epochs. ``maps`` contains only requested ``keep_maps_at_days``,
        ordered by ``map_times_days``. Missing retention epochs warn and are omitted.
        """

        for value, label in (
            (times_days, "times_days"),
            (flux_times_days, "flux_times_days"),
            (keep_maps_at_days, "keep_maps_at_days"),
            (map_observers, "map_observers"),
        ):
            if isinstance(value, Mapping) and set(value) - set(self.image_names):
                raise ValueError(f"{label} contains unknown macroimages")
        map_times, flux_times = {}, {}
        use_multirate = False
        for name in self.image_names:
            maps, flux = _light_curve_times(
                _required_image_value(times_days, name, "times_days"),
                duration_days=duration_days,
                map_cadence_days=map_cadence_days,
                source_cadence_days=source_cadence_days,
                flux_times_days=_required_image_value(
                    flux_times_days, name, "flux_times_days"
                ),
                start_day=start_day,
            )
            map_times[name] = maps
            flux_times[name] = maps if flux is None else flux
            use_multirate = use_multirate or flux is not None
        retained = {}
        observers = dict(map_observers or {})
        for name in self.image_names:
            observer, retained[name] = _retaining_map_observer(
                _times_for_image(map_times, name),
                _image_value(keep_maps_at_days, name),
                observers.get(name),
            )
            if observer is not None:
                observers[name] = observer
        system = self if source is None else self.with_source(source)
        for image in system.images.values():
            _validate_source_driver(image.source, apply_driving_signal)
        simulation, realizations = system._build_simulation(
            map_times,
            solver_options,
            include_labels=include_labels,
        )
        shared_source = system._shared_source(None, realizations)
        if apply_driving_signal is False:
            shared_source = realizations[0]._mean_source
        if use_multirate:
            result = simulation.multirate_light_curves(
                map_times,
                flux_times,
                shared_source,
                self._shared_distances(),
                include_labels=include_labels,
                map_observers=observers,
            )
        else:
            result = simulation.light_curves(
                map_times,
                shared_source,
                self._shared_distances(),
                include_labels=include_labels,
                map_observers=observers,
            )
        return replace(
            result,
            images=tuple(
                replace(
                    image,
                    light_curve=replace(
                        image.light_curve, maps=retained[image.image_name]
                    ),
                )
                for image in result.images
            ),
        )

    def transfer_functions(
        self,
        map_times_days: Sequence[float] | Mapping[str, Sequence[float]],
        delay_edges_days,
        *,
        source: ThermalReprocessingSource | None = None,
        normalize: bool = True,
        map_observers=None,
        response_batch_size: int | None = None,
        response_spatial_chunk_size: int = 262_144,
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
            response_batch_size=response_batch_size,
            response_spatial_chunk_size=response_spatial_chunk_size,
        )
