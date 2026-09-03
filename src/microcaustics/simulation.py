"""The user-facing state for maps, caustics, and finite-source photometry."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .config import (
    DynamicConfig,
    FarFieldApproxConfig,
    IPMConfig,
    IRSConfig,
    RuntimeConfig,
)
from .geometry import PlaneGrid, PlaneRegion
from .lens import LensingDistances, MacroLens, PointMassField
from .results import LightCurve, MagnificationMap
from .runtime import ResolvedRuntime, resolve_runtime
from .sources import PixelatedSource
from .trajectories import SourceTrajectory


@dataclass(frozen=True)
class MicrolensingSimulation:
    """A macro lens, point-mass field, and resolved numerical runtime.

    This is the main user-facing object. It owns no global mutable state and
    can therefore be constructed independently for multiple macroimages. The
    map, dynamic-map, caustic, and light-curve methods all operate from this
    same validated state.

    Parameters
    ----------
    macro_lens:
        Local convergence, shear, and smooth-matter fraction.
    point_masses:
        Point-lens positions, Einstein radii, and optional velocities.
    runtime:
        Either user runtime preferences or an already resolved runtime.
    """

    macro_lens: MacroLens
    point_masses: PointMassField
    runtime: ResolvedRuntime

    @classmethod
    def create(
        cls,
        macro_lens: MacroLens,
        point_masses: PointMassField,
        *,
        runtime: RuntimeConfig | ResolvedRuntime | None = None,
    ) -> MicrolensingSimulation:
        """Construct a simulation and move lens arrays to its runtime.

        Device transfer is explicit at construction so later numerical calls
        cannot accidentally mix CPU and accelerator arrays.
        """

        resolved = (
            runtime
            if isinstance(runtime, ResolvedRuntime)
            else resolve_runtime(runtime)
        )
        masses = point_masses.to(device=resolved.device, dtype=resolved.dtype)
        return cls(macro_lens=macro_lens, point_masses=masses, runtime=resolved)

    def lens_state(self, time_days: float = 0.0) -> PointMassField:
        """Return the point-lens positions at one simulation time."""

        return self.point_masses.at_time(time_days)

    def raytrace_direct(
        self,
        x_uas,
        y_uas,
        *,
        time_days: float = 0.0,
        star_chunk_size: int = 4096,
        ray_chunk_size: int | None = None,
        max_pair_bytes: int = 256 * 1024**2,
    ):
        """Map lens-plane coordinates with the exact point-mass equation.

        This reference path evaluates every requested ray against every point
        lens. It is intentionally independent of far-field approximation and is therefore
        suitable for validation, small maps, and calculations where no
        far-field approximation is desired.

        See :func:`microcaustics.solvers.raytrace_direct` for the chunking
        parameters and return values.
        """

        from .solvers import raytrace_direct

        return raytrace_direct(
            self,
            x_uas,
            y_uas,
            time_days=time_days,
            star_chunk_size=star_chunk_size,
            ray_chunk_size=ray_chunk_size,
            max_pair_bytes=max_pair_bytes,
        )

    def jacobian_determinant_direct(
        self,
        x_uas,
        y_uas,
        *,
        time_days: float = 0.0,
        star_chunk_size: int = 4096,
        ray_chunk_size: int | None = None,
        max_pair_bytes: int = 256 * 1024**2,
    ):
        """Evaluate the exact lens-equation Jacobian determinant.

        This reference calculation uses every point lens and is the numerical
        basis for validating accelerated critical-curve and caustic paths.
        """

        from .solvers import jacobian_determinant_direct

        return jacobian_determinant_direct(
            self,
            x_uas,
            y_uas,
            time_days=time_days,
            star_chunk_size=star_chunk_size,
            ray_chunk_size=ray_chunk_size,
            max_pair_bytes=max_pair_bytes,
        )

    def magnification_map(
        self,
        lens_region: PlaneRegion,
        source_grid: PlaneGrid,
        *,
        method: IRSConfig | IPMConfig,
        time_days: float = 0.0,
    ) -> MagnificationMap:
        """Generate a source-independent magnification map.

        ``lens_region`` defines where inverse rays or IPM cells originate;
        ``source_grid`` independently defines the output field and resolution.
        A source model is not required. IRS and IPM both return absolute
        magnification. IPM may use the nested source scout or evaluate the
        complete lens field, as selected by :class:`IPMConfig`.
        """

        if isinstance(method, IRSConfig):
            from .solvers import uniform_grid_irs

            return uniform_grid_irs(
                self,
                lens_region,
                source_grid,
                method,
                time_days=time_days,
            )
        if isinstance(method, IPMConfig):
            from .solvers import full_field_ipm

            return full_field_ipm(
                self,
                lens_region,
                source_grid,
                method,
                time_days=time_days,
            )
        raise TypeError("method must be an IRSConfig or IPMConfig")

    def caustics(
        self,
        lens_grid: PlaneGrid,
        *,
        time_days: float = 0.0,
        star_chunk_size: int = 4096,
        ray_chunk_size: int | None = None,
        far_field_approx: FarFieldApproxConfig | None = None,
    ):
        """Return direct-reference critical curves and mapped caustics.

        The result retains independent line segments, which are sufficient for
        plotting, length, distance, crossing, winding, and label calculations.
        """

        from .caustics import direct_caustic_field, far_field_caustic_field

        if far_field_approx is not None and far_field_approx.enabled:
            return far_field_caustic_field(
                self,
                lens_grid,
                far_field_approx,
                time_days=time_days,
                star_chunk_size=star_chunk_size,
                ray_chunk_size=ray_chunk_size,
            )

        return direct_caustic_field(
            self,
            lens_grid,
            time_days=time_days,
            star_chunk_size=star_chunk_size,
            ray_chunk_size=ray_chunk_size,
        )

    def labeled_caustics(
        self,
        lens_grid: PlaneGrid,
        source_region: PlaneRegion,
        *,
        time_days: float = 0.0,
        config=None,
        diagnostic_grid: PlaneGrid | None = None,
        include_distance_map: bool = False,
    ):
        """Return one production caustic field with anchor/gauge labels."""

        from .caustics import dynamic_labeled_caustics
        from .config import CausticConfig

        result = dynamic_labeled_caustics(
            self,
            lens_grid,
            source_region,
            [float(time_days)],
            CausticConfig() if config is None else config,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
        )
        return result[0]

    def dynamic_labeled_caustics(
        self,
        lens_grid: PlaneGrid,
        source_region: PlaneRegion,
        times_days: Sequence[float],
        *,
        config=None,
        diagnostic_grid: PlaneGrid | None = None,
        include_distance_map: bool = False,
    ):
        """Return temporally aligned production caustics and center labels."""

        from .caustics import dynamic_labeled_caustics
        from .config import CausticConfig

        return dynamic_labeled_caustics(
            self,
            lens_grid,
            source_region,
            times_days,
            CausticConfig() if config is None else config,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
        )

    def dynamic_labeled_maps(
        self,
        lens_region: PlaneRegion,
        source_grid: PlaneGrid,
        lens_grid: PlaneGrid,
        times_days: Sequence[float],
        *,
        method: IRSConfig | IPMConfig,
        map_schedule: DynamicConfig | None = None,
        caustic_config=None,
        diagnostic_grid: PlaneGrid | None = None,
        include_distance_map: bool = False,
    ):
        """Stream same-epoch maps and production anchor/gauge labels.

        Fused tiled IPM reuses its temporal far-field approximation objects for analytic
        detA, critical-curve extraction, and caustic endpoint mapping.
        Diagnostic label and distance maps remain opt-in.
        """

        from .caustics import dynamic_labeled_maps

        return dynamic_labeled_maps(
            self,
            lens_region,
            source_grid,
            lens_grid,
            times_days,
            method=method,
            map_schedule=map_schedule,
            caustic_config=caustic_config,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
        )

    def light_curve_with_labels(
        self,
        lens_region: PlaneRegion,
        source_grid: PlaneGrid,
        lens_grid: PlaneGrid,
        times_days: Sequence[float],
        source: PixelatedSource,
        distances: LensingDistances,
        *,
        method: IRSConfig | IPMConfig,
        trajectory: SourceTrajectory | None = None,
        map_schedule: DynamicConfig | None = None,
        caustic_config=None,
        strict_coverage: bool = True,
        diagnostic_grid: PlaneGrid | None = None,
        include_distance_map: bool = False,
        map_observer=None,
    ):
        """Stream a finite-source LC with aligned center-crossing labels."""

        from .caustics import streaming_labeled_light_curve

        return streaming_labeled_light_curve(
            self,
            lens_region,
            source_grid,
            lens_grid,
            times_days,
            source,
            distances,
            method=method,
            trajectory=trajectory,
            map_schedule=map_schedule,
            caustic_config=caustic_config,
            strict_coverage=strict_coverage,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
            map_observer=map_observer,
        )

    def dynamic_maps(
        self,
        lens_region: PlaneRegion,
        source_grid: PlaneGrid,
        times_days: Sequence[float],
        *,
        method: IRSConfig | IPMConfig,
        schedule: DynamicConfig | None = None,
    ) -> Iterator[MagnificationMap]:
        """Yield scheduled source-independent maps without retaining the sequence.

        Static lens fields reuse one exact map. Moving tiled-IPM calculations
        may reuse endpoint-union scout selections according to ``schedule``;
        every result records whether that approximation was active. The
        streaming contract remains unchanged when temporal kernels and
        cross-light-curve batching are enabled later.
        """

        from .dynamic import dynamic_maps

        yield from dynamic_maps(
            self,
            lens_region,
            source_grid,
            times_days,
            method=method,
            config=schedule,
        )

    def light_curve_from_maps(
        self,
        maps: Sequence[MagnificationMap],
        source: PixelatedSource,
        times_days: Sequence[float],
        distances: LensingDistances,
        *,
        trajectory: SourceTrajectory | None = None,
        strict_coverage: bool = True,
    ) -> LightCurve:
        """Convolve existing maps with an arbitrary finite source.

        This convenience method is solver-independent. Callers may pass maps
        produced by IRS, IPM, a loaded external calculation, or a future
        custom solver.
        """

        from .photometry import light_curve_from_maps

        return light_curve_from_maps(
            maps,
            source,
            times_days,
            distances,
            trajectory=trajectory,
            strict_coverage=strict_coverage,
        )

    def light_curve(
        self,
        lens_region: PlaneRegion,
        source_grid: PlaneGrid,
        times_days: Sequence[float],
        source: PixelatedSource,
        distances: LensingDistances,
        *,
        method: IRSConfig | IPMConfig,
        trajectory: SourceTrajectory | None = None,
        schedule: DynamicConfig | None = None,
        strict_coverage: bool = True,
        map_observer=None,
    ) -> LightCurve:
        """Stream dynamic maps directly into an arbitrary source light curve.

        Only fluxes are retained unless ``map_observer`` explicitly stores a
        frame. This is the ordinary production interface when a complete map
        sequence is not itself a required output.
        """

        from .photometry import streaming_light_curve

        return streaming_light_curve(
            self,
            lens_region,
            source_grid,
            times_days,
            source,
            distances,
            method=method,
            trajectory=trajectory,
            schedule=schedule,
            strict_coverage=strict_coverage,
            map_observer=map_observer,
        )

    def light_curves(
        self,
        lens_region: PlaneRegion,
        source_grid: PlaneGrid,
        times_days: Sequence[float],
        requests,
        *,
        method: IRSConfig | IPMConfig,
        schedule: DynamicConfig | None = None,
        map_observer=None,
        flux_times_days=None,
    ) -> tuple[LightCurve, ...]:
        """Stream one map sequence into multiple finite-source light curves.

        Compatible source shapes are sampled in accelerator batches controlled
        by :attr:`DynamicConfig.light_curve_batch_size`. Each request may have
        its own source physics, trajectory, bands, distances, and coverage
        policy. Maps are shared because all requests use this simulation's lens
        state and the same requested source-plane map grid.
        """

        from .photometry import streaming_light_curves

        return streaming_light_curves(
            self,
            lens_region,
            source_grid,
            times_days,
            requests,
            method=method,
            schedule=schedule,
            map_observer=map_observer,
            flux_times_days=flux_times_days,
        )

    def multirate_light_curve(
        self,
        lens_region: PlaneRegion,
        source_grid: PlaneGrid,
        map_times_days: Sequence[float],
        flux_times_days: Sequence[float],
        source: PixelatedSource,
        distances: LensingDistances,
        *,
        method: IRSConfig | IPMConfig,
        trajectory: SourceTrajectory | None = None,
        schedule: DynamicConfig | None = None,
        strict_coverage: bool = True,
        map_observer=None,
    ) -> LightCurve:
        """Combine sparse dynamic maps with a finer source/light-curve cadence.

        Flux is obtained from two map--source contractions at each fine epoch,
        so interpolated magnification maps are never materialized. Arrival-time
        shifts remain composable through :class:`TimeShiftedSource`.
        This source-independent interface accepts any ``PixelatedSource``;
        fine cadence is not restricted to quasar variability.
        """

        from .photometry import multirate_streaming_light_curve

        return multirate_streaming_light_curve(
            self,
            lens_region,
            source_grid,
            map_times_days,
            flux_times_days,
            source,
            distances,
            method=method,
            trajectory=trajectory,
            schedule=schedule,
            strict_coverage=strict_coverage,
            map_observer=map_observer,
        )

    def multirate_light_curve_with_labels(
        self,
        lens_region: PlaneRegion,
        source_grid: PlaneGrid,
        lens_grid: PlaneGrid,
        map_times_days: Sequence[float],
        flux_times_days: Sequence[float],
        source: PixelatedSource,
        distances: LensingDistances,
        *,
        method: IRSConfig | IPMConfig,
        trajectory: SourceTrajectory | None = None,
        map_schedule: DynamicConfig | None = None,
        caustic_config=None,
        strict_coverage: bool = True,
        diagnostic_grid: PlaneGrid | None = None,
        include_distance_map: bool = False,
        map_observer=None,
    ):
        """Return fine-cadence flux with labels at the dynamic-map epochs.

        For fused tiled IPM, magnification maps, analytic detA, caustics, and
        anchor/gauge labels share temporal far-field approximation batches. The source may be
        evaluated at a much finer cadence than the dynamic lens map.
        """

        from .caustics import multirate_labeled_light_curve

        return multirate_labeled_light_curve(
            self,
            lens_region,
            source_grid,
            lens_grid,
            map_times_days,
            flux_times_days,
            source,
            distances,
            method=method,
            trajectory=trajectory,
            map_schedule=map_schedule,
            caustic_config=caustic_config,
            strict_coverage=strict_coverage,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
            map_observer=map_observer,
        )

    def transfer_functions(
        self,
        lens_region: PlaneRegion,
        source_grid: PlaneGrid,
        times_days: Sequence[float],
        source,
        distances: LensingDistances,
        delay_edges_days,
        *,
        method: IRSConfig | IPMConfig,
        trajectory: SourceTrajectory | None = None,
        schedule: DynamicConfig | None = None,
        strict_coverage: bool = True,
        driver_amplitude: float = 1.0,
        normalize: bool = True,
        map_observer=None,
    ):
        """Stream dynamic maps into microlensing-weighted transfer functions.

        This single-image method is independent of the resolved multi-image
        interface. Only response products are retained by default.
        """

        from .transfer_functions import streaming_microlensed_transfer_functions

        return streaming_microlensed_transfer_functions(
            self,
            lens_region,
            source_grid,
            times_days,
            source,
            distances,
            delay_edges_days,
            method=method,
            trajectory=trajectory,
            schedule=schedule,
            strict_coverage=strict_coverage,
            driver_amplitude=driver_amplitude,
            normalize=normalize,
            map_observer=map_observer,
        )
