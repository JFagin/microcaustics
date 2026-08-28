"""Finite-source photometry on precomputed magnification maps."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn.functional as functional

from .config import DynamicConfig, IPMConfig, IRSConfig
from .geometry import PlaneGrid, PlaneRegion
from .lens import LensingDistances
from .results import LightCurve, MagnificationMap, TimingBreakdown
from .sources import ModulatedSource, PixelatedSource
from .trajectories import LinearTrajectory, SourceTrajectory


def zero_point_flux_for_magnitude(reference_flux, reference_magnitude) -> torch.Tensor:
    """Return the zero-point flux that assigns a reference magnitude.

    This is useful for normalized or otherwise instrument-independent source
    models. For physical flux densities in Jy, use a zero point of 3631 Jy
    directly instead.
    """

    flux = torch.as_tensor(reference_flux)
    if not flux.is_floating_point():
        flux = flux.to(torch.get_default_dtype())
    magnitude = torch.as_tensor(
        reference_magnitude,
        device=flux.device,
        dtype=flux.dtype,
    )
    if bool(torch.any(~torch.isfinite(flux))) or bool(torch.any(flux <= 0.0)):
        raise ValueError("reference_flux must be finite and positive")
    if bool(torch.any(~torch.isfinite(magnitude))):
        raise ValueError("reference_magnitude must be finite")
    return flux * torch.pow(flux.new_tensor(10.0), 0.4 * magnitude)


def flux_to_magnitude(flux, zero_point_flux) -> torch.Tensor:
    """Convert positive flux to astronomical magnitudes.

    Inputs broadcast according to ordinary Torch rules. Non-positive or
    non-finite fluxes produce ``nan`` rather than a silently clipped value.
    The function remains differentiable for valid fluxes.
    """

    values = torch.as_tensor(flux)
    if not values.is_floating_point():
        values = values.to(torch.get_default_dtype())
    zero_point = torch.as_tensor(
        zero_point_flux,
        device=values.device,
        dtype=values.dtype,
    )
    if bool(torch.any(~torch.isfinite(zero_point))) or bool(
        torch.any(zero_point <= 0.0)
    ):
        raise ValueError("zero_point_flux must be finite and positive")
    valid = torch.isfinite(values) & (values > 0.0)
    safe = torch.where(valid, values, torch.ones_like(values))
    magnitude = -2.5 * torch.log10(safe / zero_point)
    return torch.where(valid, magnitude, torch.full_like(magnitude, float("nan")))


@dataclass(frozen=True)
class LightCurveRequest:
    """One finite-source observation of a shared magnification-map sequence.

    Requests may use different source models, trajectories, physical source
    scales, band names, and coverage policies. Compatible source array shapes
    are sampled together on an accelerator. Incompatible shapes are grouped
    automatically without changing their numerical result.
    """

    source: PixelatedSource
    distances: LensingDistances
    trajectory: SourceTrajectory | None = None
    strict_coverage: bool = True
    name: str | None = None


def _source_offsets_uas(
    source: PixelatedSource,
    distances: LensingDistances,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Cartesian angular coordinates of source-pixel centers."""

    ny, nx = source.geometry.shape
    dy_m, dx_m = source.geometry.pixel_scale_m
    x_m = (torch.arange(nx, device=device, dtype=dtype) + 0.5 - 0.5 * nx) * dx_m
    y_m = (torch.arange(ny, device=device, dtype=dtype) + 0.5 - 0.5 * ny) * dy_m
    y_m, x_m = torch.meshgrid(y_m, x_m, indexing="ij")
    return (
        distances.source_length_to_uas(x_m, device=device, dtype=dtype),
        distances.source_length_to_uas(y_m, device=device, dtype=dtype),
    )


def _sample_map(
    magnification_map: MagnificationMap,
    x_uas: torch.Tensor,
    y_uas: torch.Tensor,
    *,
    strict_coverage: bool,
) -> torch.Tensor:
    """Bilinearly sample one map at Cartesian source-plane coordinates."""

    values = magnification_map.values
    xmin, xmax, ymin, ymax = magnification_map.grid.bounds_uas
    if strict_coverage and bool(
        torch.any((x_uas < xmin) | (x_uas > xmax) | (y_uas < ymin) | (y_uas > ymax))
    ):
        raise ValueError(
            "the magnification map does not cover every source pixel. Enlarge "
            "the source grid or set strict_coverage=False to use zero padding"
        )
    normalized_x = 2.0 * (x_uas - xmin) / (xmax - xmin) - 1.0
    normalized_y = 2.0 * (y_uas - ymin) / (ymax - ymin) - 1.0
    sample_grid = torch.stack((normalized_x, normalized_y), dim=-1).unsqueeze(0)
    sampled = functional.grid_sample(
        values[None, None],
        sample_grid,
        mode="bilinear",
        padding_mode="border" if strict_coverage else "zeros",
        align_corners=False,
    )
    return sampled[0, 0]


def _sample_map_batch(
    magnification_map: MagnificationMap,
    x_uas: torch.Tensor,
    y_uas: torch.Tensor,
    *,
    strict_coverage: bool,
) -> torch.Tensor:
    """Sample one map at a batch of equally shaped coordinate grids."""

    if x_uas.ndim != 3 or y_uas.shape != x_uas.shape:
        raise ValueError("batched source coordinates must have shape [batch, y, x]")
    values = magnification_map.values
    xmin, xmax, ymin, ymax = magnification_map.grid.bounds_uas
    if strict_coverage and bool(
        torch.any((x_uas < xmin) | (x_uas > xmax) | (y_uas < ymin) | (y_uas > ymax))
    ):
        raise ValueError(
            "the magnification map does not cover every source pixel. Enlarge "
            "the source grid or set strict_coverage=False to use zero padding"
        )
    normalized_x = 2.0 * (x_uas - xmin) / (xmax - xmin) - 1.0
    normalized_y = 2.0 * (y_uas - ymin) / (ymax - ymin) - 1.0
    sample_grid = torch.stack((normalized_x, normalized_y), dim=-1)
    sampled = functional.grid_sample(
        values[None, None].expand(x_uas.shape[0], -1, -1, -1),
        sample_grid,
        mode="bilinear",
        padding_mode="border" if strict_coverage else "zeros",
        align_corners=False,
    )
    return sampled[:, 0]


@torch.no_grad()
def light_curve_from_maps(
    maps: Sequence[MagnificationMap],
    source: PixelatedSource,
    times_days: torch.Tensor | Sequence[float],
    distances: LensingDistances,
    *,
    trajectory: SourceTrajectory | None = None,
    strict_coverage: bool = True,
) -> LightCurve:
    """Convolve time-aligned maps with an arbitrary pixelated source.

    Source brightness is evaluated in physical source-plane pixels and each
    magnification map is bilinearly sampled at their angular positions. Maps
    may come from any solver. The routine deliberately refuses incomplete map
    coverage by default so missing flux cannot silently bias a light curve.

    Parameters
    ----------
    maps:
        One source-plane magnification map per requested time.
    source:
        Any object implementing :class:`~microcaustics.PixelatedSource`.
    times_days:
        Observer-frame sampling times, paired with ``maps`` in order.
    distances:
        Lens geometry used to convert source pixels from meters to angle.
    trajectory:
        Source-center motion. The default is a stationary source at the map
        origin. Motion is independent of point-lens velocities.
    strict_coverage:
        Raise when any source pixel lies beyond its corresponding map. If
        false, out-of-map samples contribute zero magnification.
    """

    times = torch.as_tensor(times_days).reshape(-1)
    if not times.is_floating_point():
        times = times.to(torch.get_default_dtype())
    if len(maps) != int(times.numel()):
        raise ValueError("maps and times_days must have the same length")
    if not maps:
        raise ValueError("at least one magnification map is required")
    device = maps[0].values.device
    dtype = maps[0].values.dtype
    if any(item.values.device != device or item.values.dtype != dtype for item in maps):
        raise ValueError("all magnification maps must share a device and dtype")
    trajectory = LinearTrajectory() if trajectory is None else trajectory
    started = perf_counter()
    brightness = source.brightness(times, device=device, dtype=dtype)
    centers = trajectory.position_uas(times, device=device, dtype=dtype)
    if centers.shape != (times.numel(), 2):
        raise ValueError("trajectory positions must have shape [time, 2]")
    offset_x, offset_y = _source_offsets_uas(
        source,
        distances,
        device=device,
        dtype=dtype,
    )
    flux_rows = []
    for index, magnification_map in enumerate(maps):
        sample = _sample_map(
            magnification_map,
            offset_x + centers[index, 0],
            offset_y + centers[index, 1],
            strict_coverage=strict_coverage,
        )
        flux_rows.append((brightness[index] * sample[..., None]).sum(dim=(0, 1)))
    pixel_area_m2 = float(
        source.geometry.pixel_scale_m[0] * source.geometry.pixel_scale_m[1]
    )
    lensed_flux = torch.stack(flux_rows) * pixel_area_m2
    unlensed_flux = brightness.sum(dim=(1, 2)) * pixel_area_m2
    elapsed = perf_counter() - started
    return LightCurve(
        times_days=times.to(device=device, dtype=dtype),
        flux=lensed_flux,
        band_names=source.geometry.band_names,
        unlensed_flux=unlensed_flux,
        metadata={
            "method": "finite_source_map_sampling",
            "map_methods": sorted({item.method for item in maps}),
            "strict_coverage": bool(strict_coverage),
            "source": dict(source.metadata()),
        },
        timing=TimingBreakdown(
            collected=True,
            steady_seconds=elapsed,
            component_seconds={"source_and_map_convolution": elapsed},
        ),
    )


@torch.no_grad()
def streaming_light_curve(
    simulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    times_days: torch.Tensor | Sequence[float],
    source: PixelatedSource,
    distances: LensingDistances,
    *,
    method: IRSConfig | IPMConfig,
    trajectory: SourceTrajectory | None = None,
    schedule: DynamicConfig | None = None,
    strict_coverage: bool = True,
    map_observer: Callable[[int, MagnificationMap], None] | None = None,
) -> LightCurve:
    """Generate maps and finite-source fluxes without retaining a map cube.

    The source is evaluated in temporal batches while magnification maps are
    consumed from the dynamic scheduler one at a time. ``map_observer`` may
    inspect or save selected frames. Omitting it leaves storage ownership with
    the caller and permits each map tensor to be released after convolution.
    """

    request = LightCurveRequest(
        source=source,
        distances=distances,
        trajectory=trajectory,
        strict_coverage=strict_coverage,
    )
    return streaming_light_curves(
        simulation,
        lens_region,
        source_grid,
        times_days,
        (request,),
        method=method,
        schedule=schedule,
        map_observer=map_observer,
    )[0]


def _increasing_times(
    values: torch.Tensor | Sequence[float],
    *,
    name: str,
) -> torch.Tensor:
    """Return a finite, strictly increasing floating time axis."""

    times = torch.as_tensor(values).reshape(-1)
    if not times.is_floating_point():
        times = times.to(torch.get_default_dtype())
    if times.numel() < 1:
        raise ValueError(f"{name} must contain at least one time")
    if not bool(torch.all(torch.isfinite(times))):
        raise ValueError(f"{name} must contain only finite values")
    if times.numel() > 1 and not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError(f"{name} must be strictly increasing")
    return times


@torch.no_grad()
def multirate_streaming_light_curve(
    simulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    map_times_days: torch.Tensor | Sequence[float],
    flux_times_days: torch.Tensor | Sequence[float],
    source: PixelatedSource,
    distances: LensingDistances,
    *,
    method: IRSConfig | IPMConfig,
    trajectory: SourceTrajectory | None = None,
    schedule: DynamicConfig | None = None,
    strict_coverage: bool = True,
    map_observer: Callable[[int, MagnificationMap], None] | None = None,
    _map_iterator=None,
) -> LightCurve:
    """Evaluate a fine-cadence source using a sparse dynamic-map cadence.

    The dynamic magnification field is generated only at ``map_times_days``.
    At every requested ``flux_times_days`` epoch, this function evaluates the
    two contractions with the bracketing maps and linearly interpolates those
    contractions. It therefore never materializes an interpolated map cube and
    retains at most two magnification maps. This is the multi-rate construction
    used for rapidly varying sources whose microlens field evolves more slowly.
    ``source`` is fully generic. The same path supports accretion disks,
    expanding supernovae, built-in analytic profiles, and user-defined
    :class:`~microcaustics.CallableSource` implementations.

    Both time axes use observer time. Source-emission delays can be applied
    independently with :class:`~microcaustics.TimeShiftedSource`. Source-plane
    trajectories and point-lens motion remain functions of observer time.
    One map is treated as a static field. With multiple maps, all flux epochs
    must lie inside the map-time interval.
    """

    map_times = _increasing_times(map_times_days, name="map_times_days")
    flux_times = _increasing_times(flux_times_days, name="flux_times_days")
    if map_times.numel() > 1 and (
        float(flux_times[0]) < float(map_times[0])
        or float(flux_times[-1]) > float(map_times[-1])
    ):
        raise ValueError("flux_times_days must lie within the dynamic map cadence")

    resolved_schedule = DynamicConfig() if schedule is None else schedule
    runtime = simulation.runtime
    device, dtype = runtime.device, runtime.dtype
    map_times_device = map_times.to(device=device, dtype=dtype)
    flux_times_device = flux_times.to(device=device, dtype=dtype)
    trajectory = LinearTrajectory() if trajectory is None else trajectory
    centers = trajectory.position_uas(
        flux_times_device,
        device=device,
        dtype=dtype,
    )
    if centers.shape != (flux_times.numel(), 2):
        raise ValueError("trajectory positions must have shape [time, 2]")
    offset_x, offset_y = _source_offsets_uas(
        source,
        distances,
        device=device,
        dtype=dtype,
    )

    if map_times.numel() == 1:
        interval_indices = torch.zeros(
            flux_times.numel(),
            dtype=torch.long,
            device=device,
        )
        fractions = torch.zeros_like(flux_times_device)
    else:
        interval_indices = torch.searchsorted(
            map_times_device,
            flux_times_device,
            right=True,
        ).sub(1).clamp(0, map_times.numel() - 2)
        left_times = map_times_device[interval_indices]
        right_times = map_times_device[interval_indices + 1]
        fractions = (flux_times_device - left_times) / (right_times - left_times)

    map_iterator = iter(
        simulation.dynamic_maps(
            lens_region,
            source_grid,
            map_times.tolist(),
            method=method,
            schedule=resolved_schedule,
        )
        if _map_iterator is None
        else _map_iterator
    )
    runtime.synchronize(detailed=False)
    started = perf_counter()
    map_seconds = 0.0
    source_seconds = 0.0
    convolution_seconds = 0.0
    map_methods: set[str] = set()
    dynamic_metadata: dict[str, object] = {}
    flux_rows: list[torch.Tensor | None] = [None] * int(flux_times.numel())
    unlensed_rows: list[torch.Tensor | None] = [None] * int(flux_times.numel())

    # A spatially coherent modulation is exactly separable.  Evaluating the
    # wrapped physical disk once avoids rebuilding a [time,y,x,band] source
    # cube while preserving identical lensed and unlensed fluxes.
    factorized_base = None
    factorized_amplitudes = None
    if isinstance(source, ModulatedSource) and bool(
        getattr(source.source, "is_time_static", False)
    ):
        runtime.synchronize()
        source_started = perf_counter()
        factorized_base = source.source.brightness(
            flux_times_device[:1], device=device, dtype=dtype
        )[0]
        factorized_amplitudes = source.signal.amplitudes(
            flux_times_device,
            bands=len(source.geometry.band_names),
            device=device,
            dtype=dtype,
        )
        runtime.synchronize()
        source_seconds += perf_counter() - source_started

    def next_map(index: int) -> MagnificationMap:
        nonlocal map_seconds, dynamic_metadata
        runtime.synchronize()
        map_started = perf_counter()
        frame = next(map_iterator)
        runtime.synchronize()
        map_seconds += perf_counter() - map_started
        map_methods.add(frame.method)
        dynamic_metadata = dict(frame.metadata)
        if map_observer is not None:
            map_observer(index, frame)
        return frame

    left_map = next_map(0)
    interval_count = max(1, int(map_times.numel()) - 1)
    for interval in range(interval_count):
        right_map = (
            left_map if map_times.numel() == 1 else next_map(interval + 1)
        )
        selected = torch.nonzero(interval_indices == interval).reshape(-1)
        if selected.numel() > 0:
            runtime.synchronize()
            source_started = perf_counter()
            selected_times = flux_times_device[selected]
            brightness = (
                source.brightness(selected_times, device=device, dtype=dtype)
                if factorized_base is None
                else factorized_base[None].expand(selected.numel(), -1, -1, -1)
            )
            runtime.synchronize()
            source_seconds += perf_counter() - source_started

            runtime.synchronize()
            convolution_started = perf_counter()
            selected_centers = centers[selected]
            x_uas = offset_x[None] + selected_centers[:, 0, None, None]
            y_uas = offset_y[None] + selected_centers[:, 1, None, None]
            left_values = _sample_map_batch(
                left_map,
                x_uas,
                y_uas,
                strict_coverage=strict_coverage,
            )
            if right_map is left_map:
                sampled = left_values
            else:
                right_values = _sample_map_batch(
                    right_map,
                    x_uas,
                    y_uas,
                    strict_coverage=strict_coverage,
                )
                weight = fractions[selected, None, None]
                sampled = left_values + weight * (right_values - left_values)
            selected_flux = (brightness * sampled[..., None]).sum(dim=(1, 2))
            selected_unlensed = brightness.sum(dim=(1, 2))
            if factorized_amplitudes is not None:
                amplitudes = factorized_amplitudes[selected]
                selected_flux = selected_flux * amplitudes
                selected_unlensed = selected_unlensed * amplitudes
            pixel_area_m2 = float(
                source.geometry.pixel_scale_m[0]
                * source.geometry.pixel_scale_m[1]
            )
            selected_flux = selected_flux * pixel_area_m2
            selected_unlensed = selected_unlensed * pixel_area_m2
            for local_index, output_index in enumerate(selected.tolist()):
                flux_rows[output_index] = selected_flux[local_index]
                unlensed_rows[output_index] = selected_unlensed[local_index]
            runtime.synchronize()
            convolution_seconds += perf_counter() - convolution_started
        left_map = right_map

    try:
        next(map_iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("dynamic map iterator yielded more maps than requested")
    if any(value is None for value in flux_rows):
        raise RuntimeError("not every flux epoch was assigned to a map interval")
    runtime.synchronize(detailed=False)
    elapsed = perf_counter() - started
    flux = torch.stack([value for value in flux_rows if value is not None])
    unlensed = torch.stack([value for value in unlensed_rows if value is not None])
    return LightCurve(
        times_days=flux_times_device,
        flux=flux,
        band_names=source.geometry.band_names,
        unlensed_flux=unlensed,
        metadata={
            "method": "multirate_interpolated_map_photometry",
            "map_methods": sorted(map_methods),
            "map_epochs": int(map_times.numel()),
            "flux_epochs": int(flux_times.numel()),
            "map_interpolation": "linear_two_contractions",
            "interpolated_maps_materialized": False,
            "maximum_maps_retained": 2,
            "coherent_source_factorized": factorized_base is not None,
            "strict_coverage": bool(strict_coverage),
            "source": dict(source.metadata()),
            "dynamic_map_metadata": dynamic_metadata,
        },
        timing=TimingBreakdown(
            collected=runtime.profiling_enabled,
            steady_seconds=elapsed,
            component_seconds={
                "dynamic_maps": map_seconds,
                "source_brightness": source_seconds,
                "map_source_contractions": convolution_seconds,
            } if runtime.profiling.value == "detailed" else {},
        ),
    )


@torch.no_grad()
def streaming_light_curves(
    simulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    times_days: torch.Tensor | Sequence[float],
    requests: Sequence[LightCurveRequest],
    *,
    method: IRSConfig | IPMConfig,
    schedule: DynamicConfig | None = None,
    map_observer: Callable[[int, MagnificationMap], None] | None = None,
    _map_iterator=None,
) -> tuple[LightCurve, ...]:
    """Generate several light curves from one streamed map sequence.

    The expensive magnification map for each time is generated only once.
    Sources with compatible array shapes and band counts are sampled together
    in batches of ``DynamicConfig.light_curve_batch_size``. This supports
    independent source models and trajectories without retaining a map cube.

    This API batches photometry for sources behind the same macroimage and
    point-mass realization. It does not claim fused map generation across
    different :class:`~microcaustics.MicrolensingSimulation` objects.
    """

    requests = tuple(requests)
    if not requests:
        raise ValueError("at least one light-curve request is required")
    if not all(isinstance(item, LightCurveRequest) for item in requests):
        raise TypeError("requests must contain LightCurveRequest instances")
    resolved_schedule = DynamicConfig() if schedule is None else schedule
    times = torch.as_tensor(times_days).reshape(-1)
    if not times.is_floating_point():
        times = times.to(torch.get_default_dtype())
    if times.numel() < 1:
        raise ValueError("at least one light-curve time is required")
    runtime = simulation.runtime
    device, dtype = runtime.device, runtime.dtype
    times_device = times.to(device=device, dtype=dtype)

    centers: list[torch.Tensor] = []
    offsets: list[tuple[torch.Tensor, torch.Tensor]] = []
    groups: dict[tuple[tuple[int, int], int, bool], list[int]] = {}
    for request_index, request in enumerate(requests):
        trajectory = (
            LinearTrajectory() if request.trajectory is None else request.trajectory
        )
        position = trajectory.position_uas(
            times_device,
            device=device,
            dtype=dtype,
        )
        if position.shape != (times.numel(), 2):
            raise ValueError("trajectory positions must have shape [time, 2]")
        centers.append(position)
        offsets.append(
            _source_offsets_uas(
                request.source,
                request.distances,
                device=device,
                dtype=dtype,
            )
        )
        key = (
            request.source.geometry.shape,
            len(request.source.geometry.band_names),
            bool(request.strict_coverage),
        )
        groups.setdefault(key, []).append(request_index)

    temporal_batch = resolved_schedule.temporal_batch_size
    if temporal_batch is None:
        temporal_batch = min(int(times.numel()), 40 if device.type == "cuda" else 1)
    temporal_batch = max(1, int(temporal_batch))
    curve_batch = resolved_schedule.light_curve_batch_size
    if curve_batch is None:
        curve_batch = len(requests)
    curve_batch = max(1, int(curve_batch))

    map_iterator = iter(
        simulation.dynamic_maps(
            lens_region,
            source_grid,
            times.tolist(),
            method=method,
            schedule=resolved_schedule,
        )
        if _map_iterator is None
        else _map_iterator
    )
    runtime.synchronize(detailed=False)
    started = perf_counter()
    source_seconds = 0.0
    convolution_seconds = 0.0
    map_seconds = 0.0
    flux_rows: list[list[torch.Tensor]] = [[] for _ in requests]
    unlensed_rows: list[list[torch.Tensor]] = [[] for _ in requests]
    map_methods: set[str] = set()
    dynamic_metadata: dict[str, object] = {}

    factorized_bases: list[torch.Tensor | None] = [None] * len(requests)
    factorized_amplitudes: list[torch.Tensor | None] = [None] * len(requests)
    runtime.synchronize()
    factorized_started = perf_counter()
    for index, request in enumerate(requests):
        if isinstance(request.source, ModulatedSource) and bool(
            getattr(request.source.source, "is_time_static", False)
        ):
            factorized_bases[index] = request.source.source.brightness(
                times_device[:1], device=device, dtype=dtype
            )[0]
            factorized_amplitudes[index] = request.source.signal.amplitudes(
                times_device,
                bands=len(request.source.geometry.band_names),
                device=device,
                dtype=dtype,
            )
    runtime.synchronize()
    source_seconds += perf_counter() - factorized_started

    for batch_start in range(0, int(times.numel()), temporal_batch):
        batch_stop = min(int(times.numel()), batch_start + temporal_batch)
        runtime.synchronize()
        source_started = perf_counter()
        brightness = [
            (
                request.source.brightness(
                    times_device[batch_start:batch_stop],
                    device=device,
                    dtype=dtype,
                )
                if factorized_bases[index] is None
                else None
            )
            for index, request in enumerate(requests)
        ]
        runtime.synchronize()
        source_seconds += perf_counter() - source_started

        for local, frame_index in enumerate(range(batch_start, batch_stop)):
            magnification_map = next(map_iterator)
            map_seconds += magnification_map.timing.delivered_seconds
            map_methods.add(magnification_map.method)
            dynamic_metadata = {
                key: value
                for key, value in magnification_map.metadata.items()
                if key.startswith("dynamic_") or key.startswith("dual_scout_")
            }
            for group_indices in groups.values():
                for group_start in range(0, len(group_indices), curve_batch):
                    active = group_indices[group_start : group_start + curve_batch]
                    x_batch = torch.stack(
                        [
                            offsets[index][0] + centers[index][frame_index, 0]
                            for index in active
                        ]
                    )
                    y_batch = torch.stack(
                        [
                            offsets[index][1] + centers[index][frame_index, 1]
                            for index in active
                        ]
                    )
                    runtime.synchronize()
                    convolution_started = perf_counter()
                    samples = _sample_map_batch(
                        magnification_map,
                        x_batch,
                        y_batch,
                        strict_coverage=requests[active[0]].strict_coverage,
                    )
                    source_frames = torch.stack([
                        (
                            factorized_bases[index]
                            if factorized_bases[index] is not None
                            else brightness[index][local]
                        )
                        for index in active
                    ])
                    lensed = (source_frames * samples[..., None]).sum(dim=(1, 2))
                    unlensed = source_frames.sum(dim=(1, 2))
                    modulation = torch.stack([
                        (
                            factorized_amplitudes[index][frame_index]
                            if factorized_amplitudes[index] is not None
                            else torch.ones(
                                len(requests[index].source.geometry.band_names),
                                device=device,
                                dtype=dtype,
                            )
                        )
                        for index in active
                    ])
                    lensed = lensed * modulation
                    unlensed = unlensed * modulation
                    runtime.synchronize()
                    convolution_seconds += perf_counter() - convolution_started
                    for row, index in enumerate(active):
                        flux_rows[index].append(lensed[row])
                        unlensed_rows[index].append(unlensed[row])
            if map_observer is not None:
                map_observer(frame_index, magnification_map)

    try:
        next(map_iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("dynamic scheduler returned more maps than requested")
    runtime.synchronize(detailed=False)
    elapsed = perf_counter() - started
    timing = TimingBreakdown(
        collected=runtime.profiling_enabled,
        steady_seconds=elapsed,
        component_seconds={
            "map_generation": map_seconds,
            "source_evaluation": source_seconds,
            "map_convolution": convolution_seconds,
            "scheduling_and_other": max(
                0.0,
                elapsed - map_seconds - source_seconds - convolution_seconds,
            ),
        } if runtime.profiling.value == "detailed" else {},
    )
    results = []
    for index, request in enumerate(requests):
        pixel_area_m2 = float(
            request.source.geometry.pixel_scale_m[0]
            * request.source.geometry.pixel_scale_m[1]
        )
        results.append(
            LightCurve(
                times_days=times_device,
                flux=torch.stack(flux_rows[index]) * pixel_area_m2,
                band_names=request.source.geometry.band_names,
                unlensed_flux=torch.stack(unlensed_rows[index]) * pixel_area_m2,
                metadata={
                    "method": "streaming_finite_source_map_sampling",
                    "request_name": request.name,
                    "map_methods": sorted(map_methods),
                    "strict_coverage": bool(request.strict_coverage),
                    "source": dict(request.source.metadata()),
                    "maps_retained": False,
                    "source_batch_size": temporal_batch,
                    "light_curve_batch_size": curve_batch,
                    "shared_map_request_count": len(requests),
                    "coherent_source_factorized": (
                        factorized_bases[index] is not None
                    ),
                    **dynamic_metadata,
                },
                timing=timing,
            )
        )
    return tuple(results)
