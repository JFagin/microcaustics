"""High-level physical microlensing systems and their realizations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

import torch

from ._system.coordinates import (
    _rotate_point_mass_field,
    _rotate_region_center,
    _RotatedTrajectory,
    _source_in_rotated_frame,
    _stellar_motion_metadata,
    _warn_incomplete_explicit_motion,
)
from ._system.geometry import (
    _direct_star_region,
    _lens_plane_region,
    _source_grid_from_model,
)
from ._system.scheduling import (
    _LABELED_CURVE_CALL_OPTIONS,
    _LIGHT_CURVE_CALL_OPTIONS,
    _evaluate_light_curve,
    _light_curve_options,
    _light_curve_times,
    _production_dynamic_settings,
    _retaining_map_observer,
    _with_method_options,
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
    rectangular_lens_region,
)
from .lens.stellar import (
    StellarAperture,
    StellarPopulation,
    _warn_incomplete_dynamic_kinematics,
    circular_stellar_aperture,
)
from .photometry import LightCurveRequest
from .random import derive_seed
from .runtime import ResolvedRuntime, resolve_runtime
from .simulation import MicrolensingSimulation
from .sources import DrivingSignal, ModulatedSource, PhysicalSourceModel
from .sources.physical import _pixelate_source
from .sources.variability import (
    _FixedHorizonDrivingSignal,
    _source_at_driver_mean,
    _source_driving_signal,
    _source_with_signal,
    _validate_source_driver,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .results import LightCurve, MagnificationMap
    from .sources import PixelatedSource
    from .trajectories import SourceTrajectory


def _macro_magnification(macro: MacroLens) -> float:
    """Return the absolute point-source magnification of the local macro model."""

    determinant = (1.0 - macro.convergence) ** 2 - macro.shear**2
    if determinant == 0.0:
        raise ValueError("the macro model is exactly critical")
    return abs(1.0 / determinant)


def _spectral_light_curve(
    realization,
    map_times,
    flux_times,
    *,
    source,
    trajectory,
    strict_coverage,
    method,
    schedule,
    map_observer,
    keep_maps_at_days,
    bandpasses,
    spectral_model,
    wavelength_samples,
    wavelength_batch_size,
    include_microlensing_only,
    return_spectrum,
    spectrum_wavelengths,
    host_lensing,
    source_evolution,
    response_delay_bin_days,
):
    """Evaluate continuum nodes once, then derive photometry and spectra."""

    from .spectral_photometry import (
        finish_spectral_light_curve,
        prepare_spectral_source,
    )

    base_source = realization.source if source is None else source
    if base_source is None:
        raise ValueError("spectral photometry requires a physical source")
    base_source, plan, resolved_bandpasses, redshift = prepare_spectral_source(
        base_source,
        realization.system.distances,
        bandpasses,
        spectral_model=spectral_model,
        wavelength_samples=wavelength_samples,
        spectrum_wavelengths=spectrum_wavelengths,
        return_spectrum=return_spectrum,
        host_lensing=host_lensing,
    )
    request = LightCurveRequest(
        source=base_source,
        distances=realization.system.distances,
        trajectory=trajectory,
        strict_coverage=strict_coverage,
        bands_angstrom=plan.bands_angstrom,
        source_evolution=source_evolution,
        response_delay_bin_days=response_delay_bin_days,
    )
    requests = [request]
    need_mean = spectral_model is not None or include_microlensing_only
    if need_mean:
        requests.append(
            replace(
                request,
                source=_source_at_driver_mean(base_source),
                source_evolution="exact",
            )
        )
    curves = realization.light_curves(
        map_times,
        tuple(requests),
        method=method,
        schedule=schedule,
        map_observer=map_observer,
        keep_maps_at_days=keep_maps_at_days,
        flux_times_days=flux_times,
        band_batch_size=wavelength_batch_size,
    )
    return finish_spectral_light_curve(
        curves[0],
        curves[1] if need_mean else None,
        plan,
        spectral_model=spectral_model,
        source_redshift=float(redshift),
        luminosity_distance_m=(1.0 + float(redshift)) ** 2
        * realization.system.distances.source_m,
        macro_magnification=_macro_magnification(realization.system.macro),
        include_microlensing_only=include_microlensing_only,
        return_spectrum=return_spectrum,
        host_lensing=host_lensing,
        bandpass_version=resolved_bandpasses.version,
    )


class IntegrationDomain(str, Enum):
    """Lens-plane region evaluated by a map construction method."""

    SCOUT = "scout"
    FULL = "full"
    RECTANGLE = "rectangle"


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
    stellar_population: StellarPopulation | None = None
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
                "input" if self.sky_to_local_rotation_deg == 0.0 else "shear_aligned"
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
            "stellar_motion": _stellar_motion_metadata(self.stars),
            "stellar_population": (
                None
                if self.stellar_population is None
                else self.stellar_population.metadata(self.system.distances)
            ),
            "runtime": {
                "device": str(runtime.device),
                "dtype": str(runtime.dtype),
                "backend": runtime.backend.value,
                "profiling": runtime.profiling.value,
            },
            "source": (None if self.source is None else dict(self.source.metadata())),
        }

    @cached_property
    def _mean_source(self):
        """The same resolved source with constant mean driver heating."""

        return _source_at_driver_mean(self.source)

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
        resolved_schedule, _ = _production_dynamic_settings(
            resolved_method, schedule, include_labels=False
        )
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

    def _photometry(
        self,
        map_times,
        flux_times,
        *,
        include_labels,
        source,
        method,
        trajectory,
        schedule,
        strict_coverage,
        map_observer,
        keep_maps_at_days,
        caustics=None,
        diagnostic_grid=None,
        include_distance_map=False,
        include_microlensing_only=False,
        band_batch_size=None,
        source_evolution="exact",
        response_delay_bin_days=0.25,
    ):
        """Resolve geometry and observers once for all four photometry schedulers."""

        source = self.source if source is None else source
        if source is None:
            raise ValueError("light_curve requires a source model")
        source = _pixelate_source(
            source, self.system.distances, runtime=self.simulation.runtime
        )
        microlensing_only_source = None
        if include_microlensing_only:
            _validate_source_driver(source, True)
            microlensing_only_source = _source_at_driver_mean(source)
        method = self._method_for_domain(
            production_ipm_config() if method is None else method
        )
        schedule, caustics = _production_dynamic_settings(
            method, schedule, caustics, include_labels=include_labels
        )
        observer, retained = _retaining_map_observer(
            map_times, keep_maps_at_days, map_observer
        )
        args = [self.lens_region, self.source_grid]
        if include_labels:
            args.append(self.lens_grid)
        args.append(map_times)
        if flux_times is not None:
            args.append(flux_times)
        args.extend((source, self.system.distances))
        kwargs = dict(
            method=method,
            trajectory=self._trajectory_in_local_frame(trajectory),
            strict_coverage=strict_coverage,
            map_observer=observer,
            microlensing_only_source=microlensing_only_source,
            band_batch_size=band_batch_size,
            source_evolution=source_evolution,
            response_delay_bin_days=response_delay_bin_days,
        )
        if include_labels:
            kwargs.update(
                map_schedule=schedule,
                caustic_config=caustics,
                diagnostic_grid=diagnostic_grid,
                include_distance_map=include_distance_map,
            )
            calculate = (
                self.simulation.light_curve_with_labels
                if flux_times is None
                else self.simulation.multirate_light_curve_with_labels
            )
        else:
            kwargs["schedule"] = schedule
            calculate = (
                self.simulation.light_curve
                if flux_times is None
                else self.simulation.multirate_light_curve
            )
        result = calculate(*args, **kwargs)
        if include_labels:
            return replace(
                result, light_curve=replace(result.light_curve, maps=retained)
            )
        return replace(result, maps=retained)

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
        bands_angstrom: Mapping[str, float] | None = None,
        band_batch_size: int | None = None,
        include_microlensing_only: bool = False,
        bandpasses=None,
        spectral_model=None,
        wavelength_samples: int = 32,
        wavelength_batch_size: int | None = None,
        return_spectrum: bool = False,
        spectrum_wavelengths=None,
        host_lensing: str = "omit",
        source_evolution: str = "exact",
        response_delay_bin_days: float = 0.25,
    ) -> LightCurve:
        """Generate a finite-source light curve for this realization."""

        if wavelength_batch_size is not None and band_batch_size is not None:
            raise ValueError(
                "supply wavelength_batch_size or band_batch_size, not both"
            )
        if spectral_model is not None and bandpasses is None:
            raise ValueError("spectral_model requires bandpasses")
        if return_spectrum and bandpasses is None:
            raise ValueError("return_spectrum requires bandpasses")
        if bandpasses is not None:
            return _spectral_light_curve(
                self,
                times_days,
                None,
                source=source,
                trajectory=trajectory,
                strict_coverage=strict_coverage,
                method=production_ipm_config() if method is None else method,
                schedule=schedule,
                map_observer=map_observer,
                keep_maps_at_days=keep_maps_at_days,
                bandpasses=bandpasses,
                spectral_model=spectral_model,
                wavelength_samples=wavelength_samples,
                wavelength_batch_size=(
                    band_batch_size
                    if wavelength_batch_size is None
                    else wavelength_batch_size
                ),
                include_microlensing_only=include_microlensing_only,
                return_spectrum=return_spectrum,
                spectrum_wavelengths=spectrum_wavelengths,
                host_lensing=host_lensing,
                source_evolution=source_evolution,
                response_delay_bin_days=response_delay_bin_days,
            )

        if band_batch_size is not None or bands_angstrom is not None:
            request = LightCurveRequest(
                source=self.source if source is None else source,
                distances=self.system.distances,
                trajectory=trajectory,
                strict_coverage=strict_coverage,
                bands_angstrom=bands_angstrom,
                source_evolution=source_evolution,
                response_delay_bin_days=response_delay_bin_days,
            )
            curves = self.light_curves(
                times_days,
                (
                    request,
                    *(
                        (
                            replace(
                                request,
                                source=_source_at_driver_mean(request.source),
                                source_evolution="exact",
                            ),
                        )
                        if include_microlensing_only
                        else ()
                    ),
                ),
                method=method,
                schedule=schedule,
                map_observer=map_observer,
                keep_maps_at_days=keep_maps_at_days,
                band_batch_size=band_batch_size,
            )
            if not include_microlensing_only:
                return curves[0]
            return curves[0].with_microlensing_only(curves[1])

        return self._photometry(
            times_days,
            None,
            include_labels=False,
            source=source,
            method=method,
            trajectory=trajectory,
            schedule=schedule,
            strict_coverage=strict_coverage,
            map_observer=map_observer,
            keep_maps_at_days=keep_maps_at_days,
            include_microlensing_only=include_microlensing_only,
            band_batch_size=band_batch_size,
            source_evolution=source_evolution,
            response_delay_bin_days=response_delay_bin_days,
        )

    def source_light_curve(
        self,
        times_days,
        *,
        bands_angstrom: Mapping[str, float] | None = None,
        apply_driving_signal: bool | None = None,
        batch_size: int = 16,
        band_batch_size: int | None = None,
        include_microlensing_only: bool = False,
        bandpasses=None,
        spectral_model=None,
        wavelength_samples: int = 32,
        wavelength_batch_size: int | None = None,
        return_spectrum: bool = False,
        spectrum_wavelengths=None,
    ) -> LightCurve:
        """Integrate the source without generating magnification maps.

        With ``bandpasses``, the disk continuum is evaluated on sparse
        wavelength nodes before the empirical spectrum and filter integrals
        are evaluated on a dense one-dimensional grid. ``return_spectrum``
        retains the spectrum produced by that same calculation.
        """

        from .photometry import source_light_curve

        if wavelength_batch_size is not None and band_batch_size is not None:
            raise ValueError(
                "supply wavelength_batch_size or band_batch_size, not both"
            )
        if spectral_model is not None and bandpasses is None:
            raise ValueError("spectral_model requires bandpasses")
        if return_spectrum and bandpasses is None:
            raise ValueError("return_spectrum requires bandpasses")
        source = self.source
        if source is None:
            raise ValueError("source_light_curve requires a system source")
        plan = None
        resolved_bandpasses = None
        redshift = self.system.distances.source_redshift
        if bandpasses is not None:
            from .bandpasses import resolve_bandpasses
            from .spectral_photometry import spectral_sampling_plan

            if redshift is None:
                raise ValueError(
                    "spectral photometry requires distances with a source_redshift"
                )
            resolved_bandpasses = resolve_bandpasses(bandpasses)
            required = (
                ()
                if spectral_model is None
                else spectral_model.required_wavelengths(float(redshift))
            )
            plan = spectral_sampling_plan(
                resolved_bandpasses,
                continuum_samples=wavelength_samples,
                spectrum_wavelengths=spectrum_wavelengths,
                return_spectrum=return_spectrum,
                required_wavelengths=required,
            )
            with_bands = getattr(source, "with_bands", None)
            if with_bands is None:
                raise TypeError("bandpasses require a source that supports with_bands")
            source = with_bands(plan.bands_angstrom)
        if bands_angstrom is not None:
            if bandpasses is not None:
                raise ValueError("supply bands_angstrom or bandpasses, not both")
            with_bands = getattr(source, "with_bands", None)
            if with_bands is None:
                raise TypeError("bands_angstrom requires a source with with_bands")
            source = with_bands(bands_angstrom)
        _validate_source_driver(source, apply_driving_signal)
        if apply_driving_signal is False:
            source = _source_at_driver_mean(source)
        runtime = self.simulation.runtime
        curve = source_light_curve(
            source,
            times_days,
            batch_size=batch_size,
            band_batch_size=(
                band_batch_size
                if wavelength_batch_size is None
                else wavelength_batch_size
            ),
            device=runtime.device,
            dtype=runtime.dtype,
        )
        if plan is None:
            return curve
        mean_curve = None
        if spectral_model is not None or include_microlensing_only:
            mean_single = source_light_curve(
                _source_at_driver_mean(source),
                curve.times_days[:1],
                batch_size=batch_size,
                band_batch_size=(
                    band_batch_size
                    if wavelength_batch_size is None
                    else wavelength_batch_size
                ),
                device=runtime.device,
                dtype=runtime.dtype,
            )
            mean_curve = replace(
                mean_single,
                times_days=curve.times_days,
                flux=mean_single.flux.expand(curve.flux.shape),
                unlensed_flux=mean_single.unlensed_flux.expand(curve.flux.shape),
            )
        from .spectral_photometry import finish_spectral_light_curve

        return finish_spectral_light_curve(
            curve,
            mean_curve,
            plan,
            spectral_model=spectral_model,
            source_redshift=float(redshift),
            luminosity_distance_m=(1.0 + float(redshift)) ** 2
            * self.system.distances.source_m,
            macro_magnification=1.0,
            include_microlensing_only=include_microlensing_only,
            return_spectrum=return_spectrum,
            host_lensing="unlensed",
            bandpass_version=resolved_bandpasses.version,
        )

    def light_curves(
        self,
        times_days: Sequence[float],
        requests,
        *,
        method: IRSConfig | IPMConfig | None = None,
        schedule: DynamicConfig | None = None,
        map_observer=None,
        keep_maps_at_days: Sequence[float] | None = None,
        flux_times_days=None,
        request_flux_times_days=None,
        band_batch_size: int | None = None,
    ) -> tuple[LightCurve, ...]:
        """Batch multiple sources or trajectories through one map sequence.

        Selected evaluated epochs may be retained on every returned curve with
        ``keep_maps_at_days``. Retention composes with ``map_observer`` and does
        not repeat map generation.
        """

        resolved_requests = []
        for request in requests:
            distances = request.distances or self.system.distances
            source = self.source if request.source is None else request.source
            if source is None:
                raise ValueError(
                    "a request without a source requires a system source to inherit"
                )
            source = _pixelate_source(
                source,
                distances,
                runtime=self.simulation.runtime,
            )
            if request.bands_angstrom is not None:
                with_bands = getattr(source, "with_bands", None)
                if with_bands is None:
                    raise TypeError(
                        "bands_angstrom requires a source that supports with_bands"
                    )
                source = with_bands(request.bands_angstrom)
            _validate_source_driver(source, request.apply_driving_signal)
            if request.apply_driving_signal is False:
                source = _source_at_driver_mean(source)
            resolved_requests.append(
                replace(request, source=source, distances=distances)
            )
        requests = tuple(resolved_requests)

        resolved_method = self._method_for_domain(
            production_ipm_config() if method is None else method
        )
        resolved_schedule, _ = _production_dynamic_settings(
            resolved_method, schedule, include_labels=False
        )
        observer, retained = _retaining_map_observer(
            times_days, keep_maps_at_days, map_observer
        )
        curves = self.simulation.light_curves(
            self.lens_region,
            self.source_grid,
            times_days,
            requests,
            method=resolved_method,
            schedule=resolved_schedule,
            map_observer=observer,
            flux_times_days=flux_times_days,
            request_flux_times_days=request_flux_times_days,
            band_batch_size=band_batch_size,
        )
        return tuple(replace(curve, maps=retained) for curve in curves)

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
        band_batch_size: int | None = None,
        include_microlensing_only: bool = False,
        bandpasses=None,
        spectral_model=None,
        wavelength_samples: int = 32,
        wavelength_batch_size: int | None = None,
        return_spectrum: bool = False,
        spectrum_wavelengths=None,
        host_lensing: str = "omit",
        source_evolution: str = "exact",
        response_delay_bin_days: float = 0.25,
    ) -> LightCurve:
        """Use sparse dynamic maps with independently sampled source evolution."""

        if wavelength_batch_size is not None and band_batch_size is not None:
            raise ValueError(
                "supply wavelength_batch_size or band_batch_size, not both"
            )
        if spectral_model is not None and bandpasses is None:
            raise ValueError("spectral_model requires bandpasses")
        if return_spectrum and bandpasses is None:
            raise ValueError("return_spectrum requires bandpasses")
        if bandpasses is not None:
            return _spectral_light_curve(
                self,
                map_times_days,
                flux_times_days,
                source=source,
                trajectory=trajectory,
                strict_coverage=strict_coverage,
                method=production_ipm_config() if method is None else method,
                schedule=schedule,
                map_observer=map_observer,
                keep_maps_at_days=keep_maps_at_days,
                bandpasses=bandpasses,
                spectral_model=spectral_model,
                wavelength_samples=wavelength_samples,
                wavelength_batch_size=(
                    band_batch_size
                    if wavelength_batch_size is None
                    else wavelength_batch_size
                ),
                include_microlensing_only=include_microlensing_only,
                return_spectrum=return_spectrum,
                spectrum_wavelengths=spectrum_wavelengths,
                host_lensing=host_lensing,
                source_evolution=source_evolution,
                response_delay_bin_days=response_delay_bin_days,
            )

        if band_batch_size is not None:
            request = LightCurveRequest(
                source=self.source if source is None else source,
                distances=self.system.distances,
                trajectory=trajectory,
                strict_coverage=strict_coverage,
                source_evolution=source_evolution,
                response_delay_bin_days=response_delay_bin_days,
            )
            curves = self.light_curves(
                map_times_days,
                (
                    request,
                    *(
                        (
                            replace(
                                request,
                                source=_source_at_driver_mean(request.source),
                                source_evolution="exact",
                            ),
                        )
                        if include_microlensing_only
                        else ()
                    ),
                ),
                method=method,
                schedule=schedule,
                map_observer=map_observer,
                keep_maps_at_days=keep_maps_at_days,
                flux_times_days=flux_times_days,
                band_batch_size=band_batch_size,
            )
            if not include_microlensing_only:
                return curves[0]
            return curves[0].with_microlensing_only(curves[1])

        return self._photometry(
            map_times_days,
            flux_times_days,
            include_labels=False,
            source=source,
            method=method,
            trajectory=trajectory,
            schedule=schedule,
            strict_coverage=strict_coverage,
            map_observer=map_observer,
            keep_maps_at_days=keep_maps_at_days,
            include_microlensing_only=include_microlensing_only,
            band_batch_size=band_batch_size,
            source_evolution=source_evolution,
            response_delay_bin_days=response_delay_bin_days,
        )

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
        include_microlensing_only: bool = False,
        band_batch_size: int | None = None,
        source_evolution: str = "exact",
        response_delay_bin_days: float = 0.25,
    ):
        """Return fine-cadence flux and labels at the sparse map epochs."""

        return self._photometry(
            map_times_days,
            flux_times_days,
            include_labels=True,
            source=source,
            method=method,
            trajectory=trajectory,
            schedule=schedule,
            strict_coverage=strict_coverage,
            map_observer=map_observer,
            keep_maps_at_days=keep_maps_at_days,
            caustics=caustics,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
            include_microlensing_only=include_microlensing_only,
            band_batch_size=band_batch_size,
            source_evolution=source_evolution,
            response_delay_bin_days=response_delay_bin_days,
        )

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
        include_microlensing_only: bool = False,
        band_batch_size: int | None = None,
        source_evolution: str = "exact",
        response_delay_bin_days: float = 0.25,
    ):
        """Generate a light curve with aligned source-center caustic labels."""

        return self._photometry(
            times_days,
            None,
            include_labels=True,
            source=source,
            method=method,
            trajectory=trajectory,
            schedule=schedule,
            strict_coverage=strict_coverage,
            map_observer=map_observer,
            keep_maps_at_days=keep_maps_at_days,
            caustics=caustics,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
            include_microlensing_only=include_microlensing_only,
            band_batch_size=band_batch_size,
            source_evolution=source_evolution,
            response_delay_bin_days=response_delay_bin_days,
        )

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
        response_batch_size: int | None = None,
        response_spatial_chunk_size: int = 262_144,
    ):
        """Stream dynamic maps into microlensed response functions."""

        from .transfer_functions import streaming_microlensed_transfer_functions

        resolved_source = self.source if source is None else source
        if resolved_source is None:
            raise ValueError("transfer_functions requires a source")
        resolved_method = self._method_for_domain(
            production_ipm_config() if method is None else method
        )
        resolved_schedule, _ = _production_dynamic_settings(
            resolved_method, schedule, include_labels=False
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
            trajectory=self._trajectory_in_local_frame(trajectory),
            schedule=resolved_schedule,
            strict_coverage=strict_coverage,
            driver_amplitude=driver_amplitude,
            normalize=normalize,
            map_observer=map_observer,
            response_batch_size=response_batch_size,
            response_spatial_chunk_size=response_spatial_chunk_size,
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

    Supply ``lens_redshift`` and ``source_redshift`` for the standard flat
    cosmology, or provide :class:`LensingDistances` directly. Supply either a
    :class:`StellarPopulation` or a directly constructed
    :class:`PointMassField`. A source model automatically defines the angular
    source grid. If a trajectory and duration are supplied, the map field is
    enlarged automatically to contain that source throughout the sequence.
    Source-independent map calculations can provide ``map_width_uas`` and
    ``map_pixels`` directly to a map method. An explicit ``source_grid`` is
    retained for advanced rectangular or off-center fields. Direct point
    lenses use solar masses and are converted with the system distances. The
    lens plane is inferred automatically unless ``lens_plane_uas`` or an
    advanced ``lens_region`` is supplied. The seeded realization is cached
    and reused across maps, light curves, labels, and future transfer-function
    calculations. Advanced batched workflows may supply a complete
    ``stellar_aperture_override`` shared by several systems to enable fused
    map generation. The override may enlarge, but never truncate, the
    automatically required circular aperture.

    The rectangular integration strategy uses the shear eigenframe
    internally when the source is a built-in physical model. Point-lens
    positions, velocities, source position angle, and trajectory are changed
    to that basis together. This produces the conventional narrow rectangle
    without rotating or resampling a materialized source image. Already
    pixelated custom sources remain in the input frame and therefore use the
    conservative axis-aligned bounding rectangle.
    """

    macro: MacroLens
    distances: LensingDistances | None = None
    lens_redshift: float | None = None
    source_redshift: float | None = None
    H0: float = 67.66
    Om0: float = 0.30966
    distance_dtype: torch.dtype = torch.float32
    distance_device: torch.device | str | None = "auto"
    source: PixelatedSource | PhysicalSourceModel | None = None
    source_grid: PlaneGrid | None = None
    map_grid_shape: int | tuple[int, int] | None = None
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
    stellar_aperture_override: StellarAperture | None = None
    seed: int | Mapping[str, int] | None = None
    runtime: RuntimeConfig | ResolvedRuntime | None = None
    lens_plane_uas: str | float | tuple[float, float] = "auto"
    lens_region: PlaneRegion | None = None
    caustic_grid_shape: int | tuple[int, int] = 8192
    _duration_realizations: dict[float, MicrolensingRealization] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _map_grid_systems: dict[tuple[float, int], MicrolensingSystem] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _stellar_realizations: dict[tuple[object, ...], PointMassField] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    # Internal geometry variants share the already bound driver. Ordinary
    # dataclass replacement resets this field so a new seed binds a new signal.
    _shared_driver: DrivingSignal | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_source_driver(self.source)
        if self.distances is None:
            if self.lens_redshift is None or self.source_redshift is None:
                raise ValueError(
                    "supply lens_redshift and source_redshift, or distances"
                )
            distances = LensingDistances.from_redshifts(
                self.lens_redshift,
                self.source_redshift,
                H0=self.H0,
                Om0=self.Om0,
                dtype=self.distance_dtype,
                device=self.distance_device,
            )
            object.__setattr__(self, "distances", distances)
            object.__setattr__(self, "lens_redshift", None)
            object.__setattr__(self, "source_redshift", None)
        elif self.lens_redshift is not None or self.source_redshift is not None:
            raise ValueError("supply redshifts or distances, not both")
        assert self.distances is not None
        if (self.stellar_population is None) == (self.stars is None):
            raise ValueError("supply exactly one of stellar_population or stars")
        explicit_lens_plane = _lens_plane_region(self.lens_plane_uas)
        if explicit_lens_plane is not None and self.lens_region is not None:
            raise ValueError("supply lens_plane_uas or lens_region, not both")
        if self.map_grid_shape is not None:
            if self.source_grid is not None:
                raise ValueError("supply map_grid_shape or source_grid, not both")
            raw_shape = (
                (self.map_grid_shape, self.map_grid_shape)
                if isinstance(self.map_grid_shape, int)
                and not isinstance(self.map_grid_shape, bool)
                else tuple(self.map_grid_shape)
                if not isinstance(self.map_grid_shape, (int, bool))
                else ()
            )
            if (
                len(raw_shape) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in raw_shape)
                or any(value < 1 for value in raw_shape)
            ):
                raise ValueError("map_grid_shape must contain positive integers")
            shape = tuple(raw_shape)
            object.__setattr__(self, "map_grid_shape", shape)
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
                "kinematics",
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
    def _bound_stellar_population(self) -> StellarPopulation | None:
        """Bind the kinematic seed once for geometry, stars, and provenance."""

        population = self.stellar_population
        if population is None:
            return None
        kinematics = population.kinematics
        seed = self.seed_for("kinematics")
        if (
            isinstance(kinematics, SkyProjectedKinematics)
            and kinematics.seed is None
            and seed is not None
        ):
            return replace(population, kinematics=replace(kinematics, seed=seed))
        return population

    @cached_property
    def _bound_driving_signal(self):
        """Bind the independent variability seed without generating the signal."""

        if self._shared_driver is not None:
            return self._shared_driver
        signal = _source_driving_signal(self.source)
        return (
            signal.with_seed(self.seed_for("variability"))
            if isinstance(signal, _FixedHorizonDrivingSignal)
            else signal
        )

    def _with_shared_realization_state(self, signal, **changes):
        """Copy numerical geometry while retaining randomized physical state."""

        system = replace(self, **changes)
        object.__setattr__(system, "_shared_driver", signal)
        object.__setattr__(system, "_stellar_realizations", self._stellar_realizations)
        return system

    @cached_property
    def _resolved(self) -> MicrolensingRealization:
        if self.source is None and self.source_grid is None:
            raise ValueError(
                "this operation requires a source model or map geometry. "
                "For a centered source-independent map, pass map_width_uas "
                "and map_pixels to magnification_map or dynamic_maps"
            )
        resolved_runtime = (
            self.runtime
            if isinstance(self.runtime, ResolvedRuntime)
            else resolve_runtime(self.runtime)
        )
        stellar_population = self._bound_stellar_population
        if self.duration_days > 0.0 and stellar_population is not None:
            _warn_incomplete_dynamic_kinematics(
                stellar_population.kinematics,
                stacklevel=3,
            )
        elif self.duration_days > 0.0 and self.stars is not None:
            _warn_incomplete_explicit_motion(self.stars)
        # Rectangular preimages are smallest in the shear-aligned frame. Rotate
        # the complete numerical problem together so the physical sky geometry
        # and resulting light curve remain unchanged.
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
            and stellar_population is not None
            and self.lens_region is None
            and _lens_plane_region(self.lens_plane_uas) is None
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
            replace(self.macro, shear_angle_deg=0.0) if align_rectangle else self.macro
        )
        numerical_source_model = (
            _source_in_rotated_frame(self.source, frame_rotation_deg)
            if align_rectangle and physical_source
            else self.source
        )
        model_accepts_signal = hasattr(numerical_source_model, "with_driving_signal")
        if self._bound_driving_signal is not None and model_accepts_signal:
            numerical_source_model = _source_with_signal(
                numerical_source_model, self._bound_driving_signal
            )
        numerical_trajectory = (
            _RotatedTrajectory(self.trajectory, frame_rotation_deg)
            if align_rectangle and self.trajectory is not None
            else self.trajectory
        )
        source_support_radius_uas = self.source_support_radius_uas
        # Resolve physical source models only after the runtime and coordinate
        # frame are known; this keeps pixelization on the requested device and
        # lets the model provide its own physically motivated support grid.
        if physical_source:
            assert isinstance(numerical_source_model, PhysicalSourceModel)
            native_source_grid = (
                numerical_source_model.recommended_grid(self.distances)
                if self.source_grid is None
                else self.source_grid
            )
            source = _pixelate_source(
                numerical_source_model,
                self.distances,
                grid=native_source_grid,
                runtime=resolved_runtime,
            )
            support_radius_method = getattr(
                # Match explicit pixelization when a physical model is wrapped.
                source
                if isinstance(numerical_source_model, ModulatedSource)
                else numerical_source_model,
                "support_radius_m",
                None,
            )
            if source_support_radius_uas is None and support_radius_method is not None:
                support_radius = support_radius_method(self.distances)
                if support_radius is not None:
                    source_support_radius_uas = float(
                        self.distances.source_length_to_uas(
                            support_radius,
                            dtype=torch.float64,
                        )
                    )
        else:
            source = numerical_source_model
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
        if (
            source is not None
            and self._bound_driving_signal is not None
            and not model_accepts_signal
        ):
            source = _source_with_signal(source, self._bound_driving_signal)
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
        if self.map_grid_shape is not None:
            source_grid = PlaneGrid(
                self.map_grid_shape,
                source_grid.field_of_view_uas,
                source_grid.center_uas,
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
        if stellar_population is not None:
            # Aperture sizing includes the requested duration and source
            # support. Its physical key intentionally excludes cadence and
            # variability seeds, which must not change the stellar draw.
            required_aperture = circular_stellar_aperture(
                numerical_macro,
                source_grid.region,
                self.distances,
                stellar_population,
                light_loss=self.light_loss,
                safety_scale=self.safety_scale,
                duration_days=self.duration_days,
                motion_sigma_margin=self.stellar_motion_sigma_margin,
                source_support_radius_uas=source_support_radius_uas,
            )
            aperture = (
                required_aperture
                if self.stellar_aperture_override is None
                else self.stellar_aperture_override
            )
            if self.stellar_aperture_override is not None:
                if aperture.center_uas != required_aperture.center_uas:
                    raise ValueError(
                        "stellar_aperture center must match the automatically "
                        "required aperture center"
                    )
                if aperture.radius_uas < required_aperture.radius_uas:
                    raise ValueError(
                        "stellar_aperture is smaller than the complete aperture "
                        "required by the source, lens, duration, and kinematics"
                    )
            sampling_aperture = aperture
            if align_rectangle:
                # Sampling happens in the sky frame so a frame rotation cannot
                # alter a seeded stellar realization; only the returned field
                # is transformed into the numerical shear frame.
                sampling_region = _rotate_region_center(
                    aperture.bounding_region, -frame_rotation_deg
                )
                sampling_aperture = StellarAperture(
                    aperture.radius_uas,
                    sampling_region.center_uas,
                )
            stellar_key = (
                float(sampling_aperture.radius_uas),
                *map(float, sampling_aperture.center_uas),
                str(resolved_runtime.device),
                resolved_runtime.dtype,
            )
            stars = self._stellar_realizations.get(stellar_key)
            if stars is None:
                stars = stellar_population.realize(
                    sampling_aperture,
                    self.macro,
                    self.distances,
                    seed=self.seed_for("stars"),
                    device=resolved_runtime.device,
                    dtype=resolved_runtime.dtype,
                )
                self._stellar_realizations[stellar_key] = stars
            if align_rectangle:
                stars = _rotate_point_mass_field(stars, frame_rotation_deg)
            explicit_lens_region = (
                self.lens_region
                if self.lens_region is not None
                else _lens_plane_region(self.lens_plane_uas)
            )
            if explicit_lens_region is not None:
                lens_region = explicit_lens_region
            elif self.integration_domain is IntegrationDomain.RECTANGLE:
                lens_region = rectangular_lens_region(
                    numerical_macro,
                    source_grid.region,
                    self.distances,
                    stellar_population.mass_function,
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
            stars = self.stars.to(
                device=resolved_runtime.device,
                dtype=resolved_runtime.dtype,
            ).resolve(self.distances)
            explicit_lens_region = (
                self.lens_region
                if self.lens_region is not None
                else _lens_plane_region(self.lens_plane_uas)
            )
            lens_region = (
                _direct_star_region(stars, numerical_macro, source_grid.region)
                if explicit_lens_region is None
                else explicit_lens_region
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
            stellar_population=stellar_population,
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

        return self._with_shared_realization_state(
            self._bound_driving_signal,
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

    def metadata(
        self,
        *,
        map_width_uas: float | None = None,
        map_pixels: int | None = None,
    ) -> Mapping[str, object]:
        """Return serializable physical and realized numerical provenance.

        Source-independent systems accept the same map geometry arguments as
        :meth:`magnification_map`. If exactly one such geometry was used
        already, it is selected automatically.
        """

        system = self._system_for_geometry_inspection(
            map_width_uas=map_width_uas,
            map_pixels=map_pixels,
        )
        return system._resolved.metadata()

    def summary(
        self,
        *,
        times_days: Sequence[float] | None = None,
        duration_days: float | None = None,
        map_width_uas: float | None = None,
        map_pixels: int | None = None,
        display: bool = True,
    ) -> dict[str, object]:
        """Describe derived geometry and resource scale without tracing maps.

        The summary does not sample stars, run the GR source calculation, or
        compile solver kernels. It is therefore safe to call before a large
        production calculation.
        """

        system = self._system_for_geometry_inspection(
            map_width_uas=map_width_uas,
            map_pixels=map_pixels,
        )
        if system is not self:
            return system.summary(
                times_days=times_days,
                duration_days=duration_days,
                display=display,
            )

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
        population = self._bound_stellar_population
        if population is not None:
            aperture = circular_stellar_aperture(
                self.macro,
                map_grid.region,
                self.distances,
                population,
                light_loss=self.light_loss,
                safety_scale=self.safety_scale,
                duration_days=duration,
                motion_sigma_margin=self.stellar_motion_sigma_margin,
                source_support_radius_uas=support_radius,
            )
            if population.count is not None:
                expected_stars = int(population.count)
            elif self.macro.compact_convergence > 0.0:
                mean_mass = population.mass_function.mean_mass()
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
                    population.mass_function,
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
        include_labels: bool = False,
        **kwargs,
    ):
        """Warm the kernels needed by a representative production call.

        With no time axis, one static map is generated. Supplying
        ``times_days`` warms the dynamic light-curve path, including its
        temporal batch shape. Set ``include_labels=True`` to warm the source-center
        caustic pipeline as well. The computed result is returned so warmup
        work can still be inspected or reused.
        """

        if times_days is None:
            if include_labels:
                return self.labeled_caustics(**kwargs)
            return self.magnification_map(**kwargs)
        return self.light_curve(times_days, include_labels=include_labels, **kwargs)

    def warmup_light_curve(
        self,
        *,
        include_labels: bool = False,
        map_cadence_days: float = 25.0,
        method: IRSConfig | IPMConfig | None = None,
        schedule: DynamicConfig | None = None,
        **kwargs,
    ):
        """Warm one representative temporal batch without building a time axis.

        The batch length comes from the selected dynamic schedule. This helper
        is optional. A normal light-curve call performs the same one-time
        compilation when a compatible kernel is not already cached.
        """

        options = _light_curve_options(
            {"schedule": schedule, "method": method, **kwargs},
            include_labels=include_labels,
            allowed_options=_LIGHT_CURVE_CALL_OPTIONS
            | (_LABELED_CURVE_CALL_OPTIONS if include_labels else frozenset())
            | {"apply_driving_signal", "source_cadence_days", "flux_times_days"},
        )
        resolved_schedule = options["schedule"]
        batch = int(resolved_schedule.temporal_batch_size or 1)
        times = torch.arange(batch, dtype=torch.float32) * float(map_cadence_days)
        return self.light_curve(times, include_labels=include_labels, **options)

    def with_source(
        self,
        source: PixelatedSource | PhysicalSourceModel,
    ) -> MicrolensingSystem:
        """Replace the source and its driver without inheriting the old driver."""

        return replace(self, source=source, source_grid=None)

    def _with_square_map_grid(
        self,
        *,
        map_width_uas: float | None,
        map_pixels: int | None,
    ) -> MicrolensingSystem:
        """Return a system with a centered square source-independent grid.

        The public map methods use these two scalar arguments for their common
        source-independent case. ``PlaneGrid`` remains available when a caller
        needs a rectangular field or a nonzero map center.
        """

        supplied = map_width_uas is not None or map_pixels is not None
        if not supplied:
            if self.source is None and self.source_grid is None:
                raise ValueError(
                    "source-independent maps require map_width_uas and map_pixels"
                )
            return self
        if map_width_uas is None or map_pixels is None:
            raise ValueError("supply map_width_uas and map_pixels together")
        if self.source is not None or self.source_grid is not None:
            raise ValueError(
                "map_width_uas and map_pixels are only for systems without a "
                "source or source_grid"
            )
        width = float(map_width_uas)
        pixels = int(map_pixels)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("map_width_uas must be finite and positive")
        if pixels < 1 or pixels != map_pixels:
            raise ValueError("map_pixels must be a positive integer")
        key = (width, pixels)
        cached = self._map_grid_systems.get(key)
        if cached is None:
            cached = self._with_shared_realization_state(
                self._bound_driving_signal,
                source_grid=PlaneGrid(
                    shape=(pixels, pixels),
                    field_of_view_uas=(width, width),
                ),
            )
            self._map_grid_systems[key] = cached
        return cached

    def _system_for_geometry_inspection(
        self,
        *,
        map_width_uas: float | None,
        map_pixels: int | None,
    ) -> MicrolensingSystem:
        """Resolve source-independent geometry for metadata and summaries."""

        if self.source is not None or self.source_grid is not None:
            if map_width_uas is not None or map_pixels is not None:
                raise ValueError(
                    "map_width_uas and map_pixels apply only to source-independent systems"
                )
            return self
        if map_width_uas is not None or map_pixels is not None:
            return self._with_square_map_grid(
                map_width_uas=map_width_uas,
                map_pixels=map_pixels,
            )
        if len(self._map_grid_systems) == 1:
            return next(iter(self._map_grid_systems.values()))
        if not self._map_grid_systems:
            raise ValueError(
                "source-independent inspection requires map_width_uas and map_pixels. "
                "Pass the intended map geometry to summary or metadata"
            )
        raise ValueError(
            "more than one source-independent map geometry has been used. "
            "Pass map_width_uas and map_pixels to select one"
        )

    def _realize_for_times(self, times_days) -> MicrolensingRealization:
        """Return a realization whose stellar aperture covers ``times_days``."""

        times = torch.as_tensor(times_days, dtype=torch.float64).reshape(-1)
        required = max(abs(float(times.min())), abs(float(times.max())))
        if required <= float(self.duration_days) + 1.0e-10:
            return self.realize()
        key = float(required)
        cached = self._duration_realizations.get(key)
        if cached is None:
            duration_system = self._with_shared_realization_state(
                self._bound_driving_signal, duration_days=key
            )
            cached = duration_system.realize()
            self._duration_realizations[key] = cached
        return cached

    def magnification_map(
        self,
        *,
        map_width_uas: float | None = None,
        map_pixels: int | None = None,
        **kwargs,
    ) -> MagnificationMap:
        """Generate one map through the cached realization.

        A physical source determines the map geometry automatically. For a
        centered source-independent map, supply ``map_width_uas`` and
        ``map_pixels``. Advanced rectangular or off-center maps may instead
        use ``source_grid`` when constructing the system.
        """

        system = self._with_square_map_grid(
            map_width_uas=map_width_uas,
            map_pixels=map_pixels,
        )
        return system.realize().magnification_map(
            **_with_method_options(kwargs, dynamic=False)
        )

    def dynamic_maps(
        self,
        times_days: Sequence[float],
        *,
        map_width_uas: float | None = None,
        map_pixels: int | None = None,
        **kwargs,
    ):
        """Stream maps through the cached realization.

        Source-independent sequences accept the same ``map_width_uas`` and
        ``map_pixels`` convenience arguments as :meth:`magnification_map`.
        """

        system = self._with_square_map_grid(
            map_width_uas=map_width_uas,
            map_pixels=map_pixels,
        )
        return system._realize_for_times(times_days).dynamic_maps(
            times_days,
            **_with_method_options(kwargs, dynamic=True),
        )

    def dynamic_labeled_maps(
        self,
        times_days: Sequence[float],
        *,
        map_width_uas: float | None = None,
        map_pixels: int | None = None,
        **kwargs,
    ):
        """Stream maps and aligned label products through the realization."""

        system = self._with_square_map_grid(
            map_width_uas=map_width_uas,
            map_pixels=map_pixels,
        )
        return system._realize_for_times(times_days).dynamic_labeled_maps(
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
        flux_times_days: Sequence[float] | None = None,
        include_labels: bool = False,
        apply_driving_signal: bool | None = None,
        include_microlensing_only: bool = False,
        bands_angstrom: Mapping[str, float] | None = None,
        band_batch_size: int | None = None,
        bandpasses=None,
        spectral_model=None,
        wavelength_samples: int = 32,
        wavelength_batch_size: int | None = None,
        return_spectrum: bool = False,
        spectrum_wavelengths=None,
        host_lensing: str = "omit",
        source_evolution: str = "exact",
        response_delay_bin_days: float = 0.25,
        start_day: float = 0.0,
        **kwargs,
    ) -> LightCurve:
        """Generate photometry with optional source-center labels.

        Supply a duration and map cadence, or explicit ``times_days``.
        ``source_cadence_days`` independently controls intrinsic evolution
        and photometry sampling. Irregular photometry uses ``flux_times_days``.
        Plain ``rays``, ``temporal_batch_size`` and ``scout_refresh_frames``
        control the compute budget without constructing configuration objects.
        ``label_batch_size`` optionally overrides the inherited label batch.

        Returns a LightCurve with apparent AB ``magnitude``, physical ``flux``
        in Jy, and optional ``labels`` and integer-indexed retained ``maps``.
        Labels are calculated only when ``include_labels=True``.

        ``apply_driving_signal=None`` uses the source's driver when
        present. ``False`` keeps its mean heating while disabling fluctuations,
        and ``True`` requires a configured source driver. A ``source`` override
        uses its own geometry and driver, never the previous source's driver.
        ``include_microlensing_only=True`` also returns the mean-driver flux
        through the same map stream; its static source brightness is reused.

        Plain numerical controls override ``method``, ``schedule`` and ``caustics``.
        Unrecognized controls raise before realization. ``keep_maps_at_days``
        retains only evaluated map epochs and warns about unavailable requests.
        Photometry arrays have shape [time, band]. Label epochs are separate in
        ``labels.times_days`` and retained map epochs in ``map_times_days``.
        """

        if include_microlensing_only and apply_driving_signal is False:
            raise ValueError(
                "include_microlensing_only=True requires the driven light curve"
            )
        if source_evolution not in {
            "exact", "linear_response", "quadratic_response"
        }:
            raise ValueError(
                "source_evolution must be 'exact', 'linear_response', "
                "or 'quadratic_response'"
            )
        if not math.isfinite(response_delay_bin_days) or response_delay_bin_days <= 0:
            raise ValueError("response_delay_bin_days must be finite and positive")

        map_times, flux_times = _light_curve_times(
            times_days,
            duration_days=duration_days,
            map_cadence_days=map_cadence_days,
            source_cadence_days=source_cadence_days,
            flux_times_days=flux_times_days,
            start_day=start_day,
        )
        if wavelength_batch_size is not None and band_batch_size is not None:
            raise ValueError(
                "supply wavelength_batch_size or band_batch_size, not both"
            )
        if spectral_model is not None and bandpasses is None:
            raise ValueError("spectral_model requires bandpasses")
        if return_spectrum and bandpasses is None:
            raise ValueError("return_spectrum requires bandpasses")
        if bandpasses is not None:
            call_kwargs = _light_curve_options(
                kwargs,
                include_labels=include_labels,
                allowed_options=_LIGHT_CURVE_CALL_OPTIONS
                | (_LABELED_CURVE_CALL_OPTIONS if include_labels else frozenset()),
            )
            observer, retained = _retaining_map_observer(
                map_times,
                call_kwargs.pop("keep_maps_at_days", None),
                call_kwargs.get("map_observer"),
            )
            call_kwargs["map_observer"] = observer
            source_override = call_kwargs.pop("source", None)
            system = self if source_override is None else self.with_source(source_override)
            _validate_source_driver(system.source, apply_driving_signal)
            from .spectral_photometry import (
                finish_spectral_light_curve,
                prepare_spectral_source,
            )

            prepared_source, plan, resolved_bandpasses, redshift = (
                prepare_spectral_source(
                system.source,
                system.distances,
                bandpasses,
                spectral_model=spectral_model,
                wavelength_samples=wavelength_samples,
                spectrum_wavelengths=spectrum_wavelengths,
                return_spectrum=return_spectrum,
                host_lensing=host_lensing,
                )
            )
            system = system.with_source(prepared_source)
            resolved = system._realize_for_times(map_times)
            selected_source = resolved.source
            if apply_driving_signal is False:
                selected_source = resolved._mean_source
            need_mean = spectral_model is not None or include_microlensing_only
            result = _evaluate_light_curve(
                resolved,
                map_times,
                flux_times,
                include_labels=include_labels,
                include_microlensing_only=need_mean,
                source=selected_source,
                band_batch_size=(
                    band_batch_size
                    if wavelength_batch_size is None
                    else wavelength_batch_size
                ),
                source_evolution=source_evolution,
                response_delay_bin_days=response_delay_bin_days,
                **call_kwargs,
            )
            mean_curve = None
            if result.microlensing_only_flux is not None:
                mean_curve = replace(
                    result,
                    flux=result.microlensing_only_flux,
                    unlensed_flux=result.microlensing_only_unlensed_flux,
                    microlensing_only_flux=None,
                    microlensing_only_unlensed_flux=None,
                )
            result = finish_spectral_light_curve(
                result,
                mean_curve,
                plan,
                spectral_model=spectral_model,
                source_redshift=float(redshift),
                luminosity_distance_m=(1.0 + float(redshift)) ** 2
                * system.distances.source_m,
                macro_magnification=_macro_magnification(system.macro),
                include_microlensing_only=include_microlensing_only,
                return_spectrum=return_spectrum,
                host_lensing=host_lensing,
                bandpass_version=resolved_bandpasses.version,
            )
            return replace(result, maps=retained)
        if band_batch_size is not None or bands_angstrom is not None:
            if include_labels:
                raise ValueError(
                    "wavelength selection and batching are not supported with labels"
                )
            source_override = kwargs.pop("source", None)
            trajectory = kwargs.pop("trajectory", None)
            strict_coverage = kwargs.pop("strict_coverage", True)
            request = LightCurveRequest(
                source=source_override,
                trajectory=trajectory,
                strict_coverage=strict_coverage,
                bands_angstrom=bands_angstrom,
                apply_driving_signal=apply_driving_signal,
                source_evolution=source_evolution,
                response_delay_bin_days=response_delay_bin_days,
            )
            requests = [request]
            if include_microlensing_only:
                requests.append(
                    replace(
                        request,
                        apply_driving_signal=False,
                        source_evolution="exact",
                    )
                )
            curves = self.light_curves(
                map_times,
                requests=tuple(requests),
                flux_times_days=map_times if flux_times is None else flux_times,
                band_batch_size=band_batch_size,
                **kwargs,
            )
            if not include_microlensing_only:
                return curves[0]
            return curves[0].with_microlensing_only(curves[1])
        call_kwargs = _light_curve_options(
            kwargs,
            include_labels=include_labels,
            allowed_options=_LIGHT_CURVE_CALL_OPTIONS
            | (_LABELED_CURVE_CALL_OPTIONS if include_labels else frozenset()),
        )
        observer, retained = _retaining_map_observer(
            map_times,
            call_kwargs.pop("keep_maps_at_days", None),
            call_kwargs.get("map_observer"),
        )
        call_kwargs["map_observer"] = observer
        source_override = call_kwargs.pop("source", None)
        system = self if source_override is None else self.with_source(source_override)
        _validate_source_driver(system.source, apply_driving_signal)
        resolved = system._realize_for_times(map_times)
        if apply_driving_signal is False:
            call_kwargs["source"] = resolved._mean_source
        result = _evaluate_light_curve(
            resolved,
            map_times,
            flux_times,
            include_labels=include_labels,
            include_microlensing_only=include_microlensing_only,
            source_evolution=source_evolution,
            response_delay_bin_days=response_delay_bin_days,
            **call_kwargs,
        )
        return replace(result, maps=retained)

    def source_light_curve(self, times_days, **kwargs) -> LightCurve:
        """Integrate the physical source without generating lensing maps."""

        system = self
        if kwargs.get("bandpasses") is not None:
            from .spectral_photometry import prepare_spectral_source

            source, _, _, _ = prepare_spectral_source(
                self.source,
                self.distances,
                kwargs["bandpasses"],
                spectral_model=kwargs.get("spectral_model"),
                wavelength_samples=kwargs.get("wavelength_samples", 32),
                spectrum_wavelengths=kwargs.get("spectrum_wavelengths"),
                return_spectrum=kwargs.get("return_spectrum", False),
                host_lensing="unlensed",
            )
            system = self.with_source(source)
        return system.realize().source_light_curve(times_days, **kwargs)

    def light_curves(
        self,
        times_days: Sequence[float] | None = None,
        requests=None,
        *,
        duration_days: float | None = None,
        map_cadence_days: float | None = None,
        source_cadence_days: float | None = None,
        flux_times_days=None,
        band_batch_size: int | None = None,
        start_day: float = 0.0,
        **kwargs,
    ):
        """Sample several sources or trajectories from one shared map sequence.

        Requests inherit this system's distances unless explicitly overridden.
        Duration, cadence, rays and batch controls match :meth:`light_curve`.
        ``keep_maps_at_days`` retains selected evaluated map epochs and attaches
        the same ordered maps to every returned light curve. It composes with
        an optional ``map_observer`` without generating the maps twice.
        Use ``batched_system_light_curves`` for independent stellar fields.
        """
        if requests is None:
            raise ValueError(
                "supply requests containing the sources or trajectories to sample"
            )
        requests = tuple(requests)
        if not requests:
            raise ValueError("at least one light-curve request is required")
        if not all(isinstance(request, LightCurveRequest) for request in requests):
            raise TypeError("requests must contain LightCurveRequest instances")
        map_times, flux_times = _light_curve_times(
            times_days,
            duration_days=duration_days,
            map_cadence_days=map_cadence_days,
            source_cadence_days=source_cadence_days,
            flux_times_days=flux_times_days,
            start_day=start_day,
        )
        options = _light_curve_options(
            kwargs,
            include_labels=False,
            allowed_options={"keep_maps_at_days", "map_observer"},
        )
        default_flux_times = map_times if flux_times is None else flux_times
        request_times = []
        for request in requests:
            if request.flux_times_days is not None:
                _, selected = _light_curve_times(
                    map_times,
                    flux_times_days=request.flux_times_days,
                )
            elif request.flux_cadence_days is not None:
                _, selected = _light_curve_times(
                    map_times,
                    source_cadence_days=request.flux_cadence_days,
                )
            else:
                selected = default_flux_times
            request_times.append(map_times if selected is None else selected)
        shared_times = request_times[0]
        request_times_argument = (
            None
            if all(torch.equal(shared_times, item) for item in request_times[1:])
            else tuple(request_times)
        )
        return self._realize_for_times(map_times).light_curves(
            map_times,
            requests,
            flux_times_days=(shared_times if request_times_argument is None else None),
            request_flux_times_days=request_times_argument,
            band_batch_size=band_batch_size,
            **options,
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
