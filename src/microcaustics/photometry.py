"""Finite-source photometry on precomputed magnification maps."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional

from .compile import run_tensor_kernel
from .config import DynamicConfig, IPMConfig, IRSConfig
from .geometry import PlaneGrid, PlaneRegion
from .lens import LensingDistances
from .results import LightCurve, MagnificationMap, TimingBreakdown
from .sources import ModulatedSource, PixelatedSource
from .trajectories import LinearTrajectory, SourceTrajectory

if TYPE_CHECKING:
    from .sources import PhysicalSourceModel

AB_ZERO_POINT_JY = 3631.0


def mag_to_flux(magnitude) -> torch.Tensor:
    """Convert apparent AB magnitudes to physical flux densities in Jy.

    Positive infinite magnitude maps to zero flux. NaN remains NaN. For
    simulated curves use ``result.flux`` to retain the original values
    without a magnitude round trip.
    """

    values = torch.as_tensor(magnitude)
    if not values.is_floating_point():
        values = values.to(torch.get_default_dtype())
    return values.new_tensor(AB_ZERO_POINT_JY) * torch.pow(10.0, -0.4 * values)


def flux_to_magnitude(flux_jy) -> torch.Tensor:
    """Convert physical flux density in Jy to AB magnitude.

    Inputs broadcast according to ordinary Torch rules. Non-positive or
    non-finite fluxes produce ``nan`` rather than a silently clipped value.
    The function remains differentiable for valid fluxes.
    """

    values = torch.as_tensor(flux_jy)
    if not values.is_floating_point():
        values = values.to(torch.get_default_dtype())
    valid = torch.isfinite(values) & (values > 0.0)
    safe = torch.where(valid, values, torch.ones_like(values))
    magnitude = -2.5 * torch.log10(safe / values.new_tensor(AB_ZERO_POINT_JY))
    return torch.where(valid, magnitude, torch.full_like(magnitude, float("nan")))


@dataclass(frozen=True)
class LightCurveRequest:
    """One finite-source observation of a shared magnification-map sequence.

    Requests may use different source models, trajectories, physical source
    scales, band names, and coverage policies. Compatible source array shapes
    are sampled together on an accelerator. Incompatible shapes are grouped
    automatically without changing their numerical result.
    ``distances=None`` inherits geometry from a high-level system. Low-level
    simulation calls must supply distances because they contain no cosmology.
    """

    source: PixelatedSource | PhysicalSourceModel | None = None
    distances: LensingDistances | None = None
    trajectory: SourceTrajectory | None = None
    strict_coverage: bool = True
    name: str | None = None
    bands_angstrom: Mapping[str, float] | None = None
    apply_driving_signal: bool | None = None
    flux_cadence_days: float | None = None
    flux_times_days: Sequence[float] | torch.Tensor | None = None
    source_evolution: str = "exact"
    response_delay_bin_days: float = 0.25

    def __post_init__(self) -> None:
        if self.flux_cadence_days is not None and self.flux_times_days is not None:
            raise ValueError(
                "supply flux_cadence_days or flux_times_days per request, not both"
            )
        if self.flux_cadence_days is not None and (
            not math.isfinite(float(self.flux_cadence_days))
            or self.flux_cadence_days <= 0.0
        ):
            raise ValueError("flux_cadence_days must be finite and positive")
        if self.apply_driving_signal is not None and not isinstance(
            self.apply_driving_signal, bool
        ):
            raise TypeError("apply_driving_signal must be True, False, or None")
        if self.source_evolution not in {
            "exact", "linear_response", "quadratic_response"
        }:
            raise ValueError(
                "source_evolution must be 'exact', 'linear_response', "
                "or 'quadratic_response'"
            )
        if (
            not math.isfinite(self.response_delay_bin_days)
            or self.response_delay_bin_days <= 0
        ):
            raise ValueError("response_delay_bin_days must be finite and positive")


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


def _source_is_map_aligned(
    source: PixelatedSource,
    distances: LensingDistances,
    trajectory: SourceTrajectory | None,
    source_grid: PlaneGrid,
) -> bool:
    """Whether source pixels coincide exactly with the map pixel centers.

    A centered, stationary source with the map's shape and angular field needs
    no interpolation.  This check uses scalar geometry only, so enabling the
    zero-copy path does not introduce an accelerator synchronization.
    """

    resolved_trajectory = LinearTrajectory() if trajectory is None else trajectory
    if not isinstance(resolved_trajectory, LinearTrajectory):
        return False
    if tuple(float(value) for value in resolved_trajectory.initial_position_uas) != (
        0.0,
        0.0,
    ) or tuple(float(value) for value in resolved_trajectory.velocity_uas_per_day) != (
        0.0,
        0.0,
    ):
        return False
    if source.geometry.shape != source_grid.shape:
        return False
    source_fov_m = tuple(
        scale * pixels
        for scale, pixels in zip(
            source.geometry.pixel_scale_m,
            source.geometry.shape,
            strict=True,
        )
    )
    radians_to_uas = 180.0 / torch.pi * 3600.0 * 1.0e6
    source_fov_uas = tuple(
        length / distances.source_m * float(radians_to_uas) for length in source_fov_m
    )
    scale = max(*source_fov_uas, *source_grid.field_of_view_uas, 1.0)
    tolerance = 16.0 * torch.finfo(torch.float32).eps * scale
    return (
        max(
            abs(actual - expected)
            for actual, expected in zip(
                source_fov_uas,
                source_grid.field_of_view_uas,
                strict=True,
            )
        )
        <= tolerance
        and max(abs(value) for value in source_grid.center_uas) <= tolerance
    )


@torch.no_grad()
def light_curve_from_maps(
    maps: Sequence[MagnificationMap],
    source: PixelatedSource,
    times_days: torch.Tensor | Sequence[float],
    distances: LensingDistances,
    *,
    trajectory: SourceTrajectory | None = None,
    strict_coverage: bool = True,
    batch_size: int | None = None,
    band_batch_size: int | None = None,
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
    batch_size:
        Number of source epochs to evaluate together. The default evaluates
        every epoch at once; use a smaller value to bound memory for sources
        with many wavelength channels.
    band_batch_size:
        Number of wavelength channels evaluated together. The final batch is
        padded internally to this size and trimmed from the result.
    """

    times = torch.as_tensor(times_days).reshape(-1)
    if not times.is_floating_point():
        times = times.to(torch.get_default_dtype())
    if len(maps) != int(times.numel()):
        raise ValueError("maps and times_days must have the same length")
    if not maps:
        raise ValueError("at least one magnification map is required")
    if batch_size is None:
        batch_size = len(maps)
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer or None")
    if band_batch_size is not None and (
        not isinstance(band_batch_size, int)
        or isinstance(band_batch_size, bool)
        or band_batch_size < 1
    ):
        raise ValueError("band_batch_size must be a positive integer or None")
    device = maps[0].values.device
    dtype = maps[0].values.dtype
    if any(item.values.device != device or item.values.dtype != dtype for item in maps):
        raise ValueError("all magnification maps must share a device and dtype")
    trajectory = LinearTrajectory() if trajectory is None else trajectory
    started = perf_counter()
    centers = trajectory.position_uas(times, device=device, dtype=dtype)
    if centers.shape != (times.numel(), 2):
        raise ValueError("trajectory positions must have shape [time, 2]")
    aligned = all(item.grid == maps[0].grid for item in maps) and (
        _source_is_map_aligned(source, distances, trajectory, maps[0].grid)
    )
    offsets = (
        None
        if aligned
        else _source_offsets_uas(
            source,
            distances,
            device=device,
            dtype=dtype,
        )
    )
    pixel_area_m2 = float(
        source.geometry.pixel_scale_m[0] * source.geometry.pixel_scale_m[1]
    )
    lensed_chunks = []
    unlensed_chunks = []
    source_chunks = _source_band_chunks(source, band_batch_size)
    prepare = getattr(source, "_brightness_state", None)
    for start in range(0, len(maps), batch_size):
        stop = min(start + batch_size, len(maps))
        if aligned:
            samples = [item.values for item in maps[start:stop]]
        else:
            assert offsets is not None
            offset_x, offset_y = offsets
            samples = [
                _sample_map(
                    magnification_map,
                    offset_x + centers[index, 0],
                    offset_y + centers[index, 1],
                    strict_coverage=strict_coverage,
                )
                for index, magnification_map in enumerate(maps[start:stop], start=start)
            ]
        lensed_parts = []
        unlensed_parts = []
        state = (
            None
            if prepare is None
            else prepare(times[start:stop], device=device, dtype=dtype)
        )
        sample_batch = None if state is None else torch.stack(samples)
        for chunk_source, valid_count in source_chunks:
            if state is not None:
                lensed, unlensed = chunk_source._flux_from_brightness_state(
                    state, sample_batch, sample_batch, times.new_zeros(stop - start)
                )
                lensed_parts.append(lensed[:, :valid_count])
                unlensed_parts.append(
                    unlensed[:, :valid_count].expand(stop - start, -1)
                )
                continue
            brightness = chunk_source.brightness(
                times[start:stop], device=device, dtype=dtype
            )
            rows = [
                (brightness[index] * sample[..., None]).sum(dim=(0, 1))
                for index, sample in enumerate(samples)
            ]
            lensed_parts.append(torch.stack(rows)[:, :valid_count])
            unlensed_parts.append(brightness.sum(dim=(1, 2))[:, :valid_count])
        lensed_chunks.append(torch.cat(lensed_parts, dim=1) * pixel_area_m2)
        unlensed_chunks.append(torch.cat(unlensed_parts, dim=1) * pixel_area_m2)
    lensed_flux = torch.cat(lensed_chunks, dim=0)
    unlensed_flux = torch.cat(unlensed_chunks, dim=0)
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
            "map_aligned_source_fast_path": aligned,
            "source_batch_size": batch_size,
            "band_batch_size": band_batch_size,
            "source": dict(source.metadata()),
        },
        timing=TimingBreakdown(
            collected=True,
            steady_seconds=elapsed,
            component_seconds={"source_and_map_convolution": elapsed},
        ),
    )


@torch.no_grad()
def source_light_curve(
    source: PixelatedSource,
    times_days: torch.Tensor | Sequence[float],
    *,
    batch_size: int = 16,
    band_batch_size: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> LightCurve:
    """Integrate a time-dependent pixelated source without microlensing.

    This is the direct continuum-reverberation or intrinsic-source light
    curve. The returned flux density is in Jy when the source brightness uses
    the package's standard physical surface-brightness convention.

    ``batch_size`` and ``band_batch_size`` only control memory. They do not
    alter the result. The last wavelength batch is padded and trimmed.
    ``device`` and ``dtype`` optionally override the time tensor's placement.
    """

    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if band_batch_size is not None and (
        not isinstance(band_batch_size, int)
        or isinstance(band_batch_size, bool)
        or band_batch_size < 1
    ):
        raise ValueError("band_batch_size must be a positive integer or None")
    times = torch.as_tensor(times_days)
    if times.ndim != 1 or times.numel() < 1:
        raise ValueError("times_days must be a non-empty one-dimensional sequence")
    resolved_device = times.device if device is None else torch.device(device)
    resolved_dtype = (
        times.dtype if dtype is None and times.is_floating_point() else dtype
    )
    if resolved_dtype is None:
        resolved_dtype = torch.get_default_dtype()
    times = times.to(device=resolved_device, dtype=resolved_dtype)
    pixel_area_m2 = float(
        source.geometry.pixel_scale_m[0] * source.geometry.pixel_scale_m[1]
    )
    rows = []
    source_chunks = _source_band_chunks(source, band_batch_size)
    prepare = getattr(source, "_brightness_state", None)
    static_state = (
        prepare(times[:1], device=resolved_device, dtype=resolved_dtype)
        if prepare is not None and getattr(source, "is_time_static", False)
        else None
    )
    # A scalar unity map broadcasts inside the compiled intrinsic contraction.
    # No image-sized dummy magnification map or time-dependent map is allocated.
    unity = times.new_ones((1, 1, 1))
    static_flux = None
    for chunk in times.split(int(batch_size)):
        if static_flux is not None:
            rows.append(static_flux.expand(chunk.numel(), -1))
            continue
        parts = []
        padded = (
            _pad_source_times(chunk, int(batch_size)) if prepare is not None else chunk
        )
        state = static_state
        if state is None and prepare is not None:
            state = prepare(padded, device=resolved_device, dtype=resolved_dtype)
        for chunk_source, valid_count in source_chunks:
            if state is not None:
                _, intrinsic = chunk_source._flux_from_brightness_state(
                    state, unity, unity, times.new_zeros(1)
                )
                parts.append(intrinsic[: chunk.numel(), :valid_count])
                continue
            brightness = chunk_source.brightness(
                chunk, device=resolved_device, dtype=resolved_dtype
            )
            parts.append(brightness.sum(dim=(1, 2))[:, :valid_count])
        flux_chunk = torch.cat(parts, dim=1) * pixel_area_m2
        if getattr(source, "is_time_static", False):
            static_flux = flux_chunk[:1]
            flux_chunk = static_flux.expand(chunk.numel(), -1)
        rows.append(flux_chunk)
    flux = torch.cat(rows, dim=0)
    return LightCurve(
        times_days=times,
        flux=flux,
        band_names=source.geometry.band_names,
        unlensed_flux=flux,
        metadata={
            "method": "integrated_source_brightness",
            "band_batch_size": band_batch_size,
            "source": dict(source.metadata()),
        },
        timing=TimingBreakdown(),
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

    if getattr(source, "_brightness_state", None) is not None:
        return streaming_light_curves(
            simulation,
            lens_region,
            source_grid,
            map_times_days,
            (LightCurveRequest(source, distances, trajectory, strict_coverage),),
            method=method,
            schedule=schedule,
            map_observer=map_observer,
            flux_times_days=flux_times_days,
            _map_iterator=_map_iterator,
        )[0]

    # Plan interpolation brackets on the CPU once. This avoids synchronizing
    # CUDA for every map interval merely to recover Python output indices.
    map_times = _increasing_times(map_times_days, name="map_times_days").cpu()
    flux_times = _increasing_times(flux_times_days, name="flux_times_days").cpu()
    if map_times.numel() > 1 and (
        float(flux_times[0]) < float(map_times[0])
        or float(flux_times[-1]) > float(map_times[-1])
    ):
        raise ValueError("flux_times_days must lie within the dynamic map cadence")

    resolved_schedule = DynamicConfig() if schedule is None else schedule
    runtime = simulation.runtime
    device, dtype = runtime.device, runtime.dtype
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
        )
        fractions = torch.zeros_like(flux_times)
    else:
        interval_indices = (
            torch.searchsorted(
                map_times,
                flux_times,
                right=True,
            )
            .sub(1)
            .clamp(0, map_times.numel() - 2)
        )
        left_times = map_times[interval_indices]
        right_times = map_times[interval_indices + 1]
        fractions = (flux_times - left_times) / (right_times - left_times)
    fractions_device = fractions.to(device=device, dtype=dtype)

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
    elif bool(getattr(source, "is_time_static", False)):
        runtime.synchronize()
        source_started = perf_counter()
        factorized_base = source.brightness(
            flux_times_device[:1], device=device, dtype=dtype
        )[0]
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
        right_map = left_map if map_times.numel() == 1 else next_map(interval + 1)
        selected_host = torch.nonzero(interval_indices == interval).reshape(-1)
        if selected_host.numel() > 0:
            selected = selected_host.to(device=device)
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
                weight = fractions_device[selected, None, None]
                sampled = left_values + weight * (right_values - left_values)
            selected_flux = (brightness * sampled[..., None]).sum(dim=(1, 2))
            selected_unlensed = brightness.sum(dim=(1, 2))
            if factorized_amplitudes is not None:
                amplitudes = factorized_amplitudes[selected]
                selected_flux = selected_flux * amplitudes
                selected_unlensed = selected_unlensed * amplitudes
            pixel_area_m2 = float(
                source.geometry.pixel_scale_m[0] * source.geometry.pixel_scale_m[1]
            )
            selected_flux = selected_flux * pixel_area_m2
            selected_unlensed = selected_unlensed * pixel_area_m2
            for local_index, output_index in enumerate(selected_host.tolist()):
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
            }
            if runtime.profiling.value == "detailed"
            else {},
        ),
    )


def _source_band_chunks(source, band_batch_size):
    """Split a wavelength-aware source without duplicating its spatial state."""

    band_count = len(source.geometry.band_names)
    if band_batch_size is None or band_batch_size >= band_count:
        return ((source, band_count),)
    with_bands = getattr(source, "with_bands", None)
    if with_bands is None:
        raise TypeError("band_batch_size requires a source that supports with_bands")
    names = source.geometry.band_names
    wavelengths = source.geometry.wavelengths_angstrom
    chunks = []
    for start in range(0, band_count, band_batch_size):
        stop = min(start + band_batch_size, band_count)
        chunk_names = list(names[start:stop])
        chunk_wavelengths = list(wavelengths[start:stop])
        valid_count = len(chunk_names)
        if valid_count < band_batch_size:
            chunk_names.extend(
                f"__padding_band_{index:04d}"
                for index in range(valid_count, band_batch_size)
            )
            chunk_wavelengths.extend(
                [chunk_wavelengths[-1]] * (band_batch_size - valid_count)
            )
        chunks.append(
            (
                with_bands(dict(zip(chunk_names, chunk_wavelengths, strict=True))),
                valid_count,
            )
        )
    return tuple(chunks)


def _flux_reduction(brightness, left, right, fraction):
    """Fuse map interpolation and the lensed/unlensed spatial contractions."""

    sampled = left + fraction[:, None, None] * (right - left)
    return (
        (brightness * sampled[..., None]).sum(dim=(1, 2)),
        brightness.sum(dim=(1, 2)),
    )


def _pad_source_times(values, size):
    """Keep temporal tail shapes fixed without querying outside driver bounds."""

    if values.shape[0] == size:
        return values
    return torch.cat(
        (values, values[-1:].expand(size - values.shape[0], *values.shape[1:]))
    )


@torch.no_grad()
def _flexible_streaming_light_curves(
    simulation,
    lens_region,
    source_grid,
    map_times,
    requests,
    request_times,
    *,
    method,
    schedule,
    map_observer,
    band_batch_size,
    map_iterator,
):
    """Share dynamic maps across request-specific time and wavelength batches."""

    from .sources.linear_response import LinearResponsePlan

    runtime = simulation.runtime
    device, dtype = runtime.device, runtime.dtype
    map_times = _increasing_times(map_times, name="times_days").cpu()
    request_times = tuple(
        _increasing_times(values, name="request flux_times_days").cpu()
        for values in request_times
    )
    if len(request_times) != len(requests):
        raise ValueError("request_flux_times_days must match the request count")
    if map_times.numel() > 1 and any(
        times[0] < map_times[0] or times[-1] > map_times[-1] for times in request_times
    ):
        raise ValueError("request flux times must lie within the dynamic map cadence")
    if band_batch_size is not None and (
        not isinstance(band_batch_size, int)
        or isinstance(band_batch_size, bool)
        or band_batch_size < 1
    ):
        raise ValueError("band_batch_size must be a positive integer or None")

    resolved_schedule = DynamicConfig() if schedule is None else schedule
    temporal_batch = resolved_schedule.temporal_batch_size
    if temporal_batch is None:
        temporal_batch = 49 if device.type == "cuda" else 1
    temporal_batch = max(1, int(temporal_batch))
    times_device = tuple(
        times.to(device=device, dtype=dtype) for times in request_times
    )
    centers = []
    offsets = []
    chunks = []
    aligned = []
    stationary = []
    static_states = []
    static_matrices = []
    source_temporal_batches = []
    response_plans = []
    # Bound the combined static cache across requests. This is at most 1/32
    # of the configured device budget (and never more than 512 MiB). Larger
    # spectra keep streaming wavelength chunks through the fused thermal path.
    static_cache_remaining = min(
        512 * 2**20, (runtime.available_memory_bytes or 512 * 2**20) // 32
    )
    right_indices = []
    fractions = []
    for request, times, local_times in zip(
        requests, request_times, times_device, strict=True
    ):
        trajectory = (
            LinearTrajectory() if request.trajectory is None else request.trajectory
        )
        position = trajectory.position_uas(local_times, device=device, dtype=dtype)
        if position.shape != (times.numel(), 2):
            raise ValueError("trajectory positions must have shape [time, 2]")
        centers.append(position)
        is_aligned = _source_is_map_aligned(
            request.source, request.distances, request.trajectory, source_grid
        )
        aligned.append(is_aligned)
        stationary.append(
            isinstance(trajectory, LinearTrajectory)
            and tuple(trajectory.velocity_uas_per_day) == (0.0, 0.0)
        )
        offsets.append(
            None
            if is_aligned
            else _source_offsets_uas(
                request.source,
                request.distances,
                device=device,
                dtype=dtype,
            )
        )
        chunks.append(_source_band_chunks(request.source, band_batch_size))
        if request.source_evolution != "exact":
            if not (
                isinstance(trajectory, LinearTrajectory)
                and tuple(trajectory.velocity_uas_per_day) == (0.0, 0.0)
            ):
                raise ValueError(
                    "linear_response currently requires a stationary source trajectory"
                )
            response_plans.append(
                LinearResponsePlan(
                    request.source,
                    chunks[-1],
                    local_times,
                    delay_bin_days=request.response_delay_bin_days,
                    response_order=(
                        2 if request.source_evolution == "quadratic_response" else 1
                    ),
                    runtime=runtime,
                )
            )
        else:
            response_plans.append(None)
        is_static = bool(getattr(request.source, "is_time_static", False))
        prepare = getattr(request.source, "_brightness_state", None)
        static_states.append(
            prepare(local_times[:1], device=device, dtype=dtype)
            if is_static and prepare is not None
            else None
        )
        band_count = len(request.source.geometry.band_names)
        pixel_count = math.prod(request.source.geometry.shape)
        cache_bytes = pixel_count * band_count * local_times.element_size()
        matrix = None
        if is_static and cache_bytes <= static_cache_remaining:
            matrix = torch.empty((pixel_count, band_count), device=device, dtype=dtype)
            column = 0
            for chunk, valid in chunks[-1]:
                frame = (
                    chunk._brightness_from_state(static_states[-1])
                    if static_states[-1] is not None
                    else chunk.brightness(local_times[:1], device=device, dtype=dtype)
                )
                matrix[:, column : column + valid] = frame[0, ..., :valid].reshape(
                    pixel_count, valid
                )
                column += valid
            static_cache_remaining -= cache_bytes
        static_matrices.append(
            None if matrix is None else (matrix, matrix.sum(dim=0)[None])
        )
        if map_times.numel() == 1:
            right_indices.append(torch.zeros(times.numel(), dtype=torch.long))
            fractions.append(torch.zeros(times.numel(), dtype=dtype, device=device))
        else:
            right = torch.searchsorted(map_times, times).clamp(1, map_times.numel() - 1)
            right_indices.append(right)
            fraction = (times - map_times[right - 1]) / (
                map_times[right] - map_times[right - 1]
            )
            fractions.append(fraction.to(device=device, dtype=dtype))
        # A sparse request can have far fewer source epochs per map interval
        # than the map/label batch size. Keep one compiled source shape for
        # this request without padding every interval to the map batch size.
        source_temporal_batches.append(
            min(temporal_batch, int(torch.bincount(right_indices[-1]).max()))
        )

    iterator = iter(
        simulation.dynamic_maps(
            lens_region,
            source_grid,
            map_times.tolist(),
            method=method,
            schedule=resolved_schedule,
        )
        if map_iterator is None
        else map_iterator
    )
    runtime.synchronize(detailed=False)
    started = perf_counter()
    outputs = [
        torch.empty(
            (times.numel(), len(request.source.geometry.band_names)),
            device=device,
            dtype=dtype,
        )
        for request, times in zip(requests, request_times, strict=True)
    ]
    unlensed_outputs = [torch.empty_like(value) for value in outputs]
    written = [0] * len(requests)
    static_map_flux = [None] * len(requests)
    response_map_flux = [None] * len(requests)
    map_methods = set()
    dynamic_metadata = {}

    def take_map(index):
        nonlocal dynamic_metadata
        frame = next(iterator)
        map_methods.add(frame.method)
        dynamic_metadata = {
            key: value
            for key, value in frame.metadata.items()
            if key == "rasterizer"
            or key.startswith(("dynamic_", "dual_scout_", "cross_system_"))
        }
        if map_observer is not None:
            map_observer(index, frame)
        return frame

    def sample(frame, request_index, positions):
        if aligned[request_index]:
            return frame.values[None]
        x_offset, y_offset = offsets[request_index]
        return _sample_map_batch(
            frame,
            x_offset[None] + positions[:, 0, None, None],
            y_offset[None] + positions[:, 1, None, None],
            strict_coverage=requests[request_index].strict_coverage,
        )

    def contract(request_index, times, left, right, fraction):
        cached_matrix = static_matrices[request_index]
        if cached_matrix is not None:
            matrix, intrinsic = cached_matrix
            # These are BLAS matrix products against a disk evaluated once.
            # Interpolate the small flux vectors, never daily static images.
            left_flux = left.flatten(1) @ matrix
            right_flux = left_flux if right is left else right.flatten(1) @ matrix
            return left_flux + fraction[:, None] * (right_flux - left_flux), intrinsic
        source = requests[request_index].source
        state = static_states[request_index]
        prepare = getattr(source, "_brightness_state", None)
        if state is None and prepare is not None:
            state = prepare(times, device=device, dtype=dtype)
        lensed, unlensed = [], []
        for chunk, valid in chunks[request_index]:
            if state is not None:
                flux, intrinsic = chunk._flux_from_brightness_state(
                    state, left, right, fraction
                )
            else:
                brightness = chunk.brightness(times, device=device, dtype=dtype)
                (flux, intrinsic), _ = run_tensor_kernel(
                    runtime,
                    "source photometry reduction",
                    _flux_reduction,
                    brightness,
                    left,
                    right,
                    fraction,
                )
            lensed.append(flux[:, :valid])
            unlensed.append(intrinsic[:, :valid])
        return torch.cat(lensed, dim=1), torch.cat(unlensed, dim=1)

    def static_contract(request_index, frame):
        cached = static_map_flux[request_index]
        if cached is not None and cached[0] is frame:
            return cached[1]
        values = sample(frame, request_index, centers[request_index][:1])
        result = contract(
            request_index,
            times_device[request_index][:1],
            values,
            values,
            torch.zeros(1, device=device, dtype=dtype),
        )
        static_map_flux[request_index] = (frame, result)
        return result

    def response_contract(request_index, frame):
        cached = response_map_flux[request_index]
        if cached is not None and cached[0] is frame:
            return cached[1]
        values = sample(frame, request_index, centers[request_index][:1])
        result = response_plans[request_index].project_map(values[0])
        response_map_flux[request_index] = (frame, result)
        return result

    left_map = take_map(0)
    interval_values = (0,) if map_times.numel() == 1 else range(1, map_times.numel())
    for right_index in interval_values:
        right_map = left_map if right_index == 0 else take_map(right_index)
        for request_index, request in enumerate(requests):
            selected = torch.nonzero(
                right_indices[request_index] == right_index
            ).reshape(-1)
            if selected.numel() == 0:
                continue
            if response_plans[request_index] is not None:
                plan = response_plans[request_index]
                left_projection = response_contract(request_index, left_map)
                right_projection = response_contract(request_index, right_map)
                lo, hi = int(selected[0]), int(selected[-1]) + 1
                left_flux = plan.flux(left_projection, lo, hi)
                right_flux = plan.flux(right_projection, lo, hi)
                weight = fractions[request_index][lo:hi, None]
                outputs[request_index][lo:hi] = left_flux + weight * (
                    right_flux - left_flux
                )
                unlensed_outputs[request_index][lo:hi] = plan.unlensed_flux(lo, hi)
                written[request_index] += hi - lo
                continue
            is_static = bool(getattr(request.source, "is_time_static", False))
            if is_static and stationary[request_index]:
                # For a fixed disk, flux is linear in the map. Contract each
                # bracketing map once and interpolate tiny band vectors, not
                # daily maps/brightness cubes. This is exact, not a variability
                # or reverberation approximation.
                left_flux, intrinsic = static_contract(request_index, left_map)
                right_flux, _ = static_contract(request_index, right_map)
                lo, hi = int(selected[0]), int(selected[-1]) + 1
                weight = fractions[request_index][lo:hi, None]
                outputs[request_index][lo:hi] = left_flux + weight * (
                    right_flux - left_flux
                )
                unlensed_outputs[request_index][lo:hi] = intrinsic
                written[request_index] += hi - lo
                continue
            source_batch = source_temporal_batches[request_index]
            for start in range(0, selected.numel(), source_batch):
                local = selected[start : start + source_batch]
                lo, hi = int(local[0]), int(local[-1]) + 1
                times = _pad_source_times(
                    times_device[request_index][lo:hi], source_batch
                )
                weight = _pad_source_times(
                    fractions[request_index][lo:hi], source_batch
                )
                positions = (
                    centers[request_index][:1]
                    if stationary[request_index]
                    else _pad_source_times(centers[request_index][lo:hi], source_batch)
                )
                left_values = sample(left_map, request_index, positions)
                right_values = (
                    left_values
                    if right_map is left_map
                    else sample(right_map, request_index, positions)
                )
                lensed, unlensed = contract(
                    request_index, times, left_values, right_values, weight
                )
                outputs[request_index][lo:hi] = lensed[: hi - lo]
                unlensed_outputs[request_index][lo:hi] = unlensed[: hi - lo]
                written[request_index] += hi - lo
        left_map = right_map

    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("dynamic scheduler returned more maps than requested")
    runtime.synchronize(detailed=False)
    elapsed = perf_counter() - started
    results = []
    for index, request in enumerate(requests):
        if written[index] != request_times[index].numel():
            raise RuntimeError("not every request flux epoch was evaluated")
        pixel_area = float(
            request.source.geometry.pixel_scale_m[0]
            * request.source.geometry.pixel_scale_m[1]
        )
        results.append(
            LightCurve(
                times_days=times_device[index],
                flux=outputs[index] * pixel_area,
                band_names=request.source.geometry.band_names,
                unlensed_flux=unlensed_outputs[index] * pixel_area,
                metadata={
                    "method": "streaming_finite_source_map_sampling",
                    "request_name": request.name,
                    "map_methods": sorted(map_methods),
                    "strict_coverage": bool(request.strict_coverage),
                    "source": dict(request.source.metadata()),
                    "maps_retained": False,
                    "source_batch_size": source_temporal_batches[index],
                    "source_evolution": request.source_evolution,
                    "response_delay_bin_days": (
                        request.response_delay_bin_days
                        if response_plans[index] is not None
                        else None
                    ),
                    "band_batch_size": band_batch_size,
                    "map_aligned_source_fast_path": aligned[index],
                    "coherent_source_factorized": bool(
                        getattr(request.source, "is_time_static", False)
                    ),
                    "shared_map_request_count": len(requests),
                    "map_epochs": int(map_times.numel()),
                    "flux_epochs": int(request_times[index].numel()),
                    "map_interpolation": (
                        "linear"
                        if not torch.equal(request_times[index], map_times)
                        else "none"
                    ),
                    "dynamic_map_metadata": dynamic_metadata,
                    **dynamic_metadata,
                },
                timing=TimingBreakdown(
                    collected=runtime.profiling_enabled,
                    steady_seconds=elapsed,
                ),
            )
        )
    return tuple(results)


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
    flux_times_days: torch.Tensor | Sequence[float] | None = None,
    request_flux_times_days=None,
    band_batch_size: int | None = None,
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
    Optional ``flux_times_days`` samples every source on a finer or irregular
    grid using the two bracketing maps. No interpolated map cube is built.
    """

    requests = tuple(requests)
    if not requests:
        raise ValueError("at least one light-curve request is required")
    if not all(isinstance(item, LightCurveRequest) for item in requests):
        raise TypeError("requests must contain LightCurveRequest instances")
    if any(item.distances is None for item in requests):
        raise ValueError(
            "low-level requests require distances; use system.light_curves to inherit them"
        )
    if any(item.source is None for item in requests):
        raise ValueError("low-level requests require resolved sources")
    if request_flux_times_days is not None:
        if flux_times_days is not None:
            raise ValueError(
                "supply shared flux_times_days or request_flux_times_days, not both"
            )
        return _flexible_streaming_light_curves(
            simulation,
            lens_region,
            source_grid,
            times_days,
            requests,
            request_flux_times_days,
            method=method,
            schedule=schedule,
            map_observer=map_observer,
            band_batch_size=band_batch_size,
            map_iterator=_map_iterator,
        )
    if band_batch_size is not None or any(
        getattr(request.source, "_brightness_state", None) is not None
        for request in requests
    ):
        shared_times = times_days if flux_times_days is None else flux_times_days
        return _flexible_streaming_light_curves(
            simulation,
            lens_region,
            source_grid,
            times_days,
            requests,
            (shared_times,) * len(requests),
            method=method,
            schedule=schedule,
            map_observer=map_observer,
            band_batch_size=band_batch_size,
            map_iterator=_map_iterator,
        )
    resolved_schedule = DynamicConfig() if schedule is None else schedule
    map_times = _increasing_times(times_days, name="times_days").cpu()
    times = (
        map_times
        if flux_times_days is None
        else _increasing_times(flux_times_days, name="flux_times_days").cpu()
    )
    if map_times.numel() > 1 and (times[0] < map_times[0] or times[-1] > map_times[-1]):
        raise ValueError("flux_times_days must lie within the dynamic map cadence")
    # Plan brackets on the CPU once. CUDA work below never synchronizes just
    # to decide which dynamic frame a photometry epoch needs.
    if map_times.numel() == 1:
        right_indices = [0] * times.numel()
        fractions = [0.0] * times.numel()
    else:
        right = torch.searchsorted(map_times, times).clamp(1, map_times.numel() - 1)
        right_indices = right.tolist()
        fractions = (
            (times - map_times[right - 1]) / (map_times[right] - map_times[right - 1])
        ).tolist()
    runtime = simulation.runtime
    device, dtype = runtime.device, runtime.dtype
    times_device = times.to(device=device, dtype=dtype)

    centers: list[torch.Tensor] = []
    offsets: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    map_aligned: list[bool] = []
    groups: dict[tuple[tuple[int, int], int, bool, bool], list[int]] = {}
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
        aligned = _source_is_map_aligned(
            request.source,
            request.distances,
            request.trajectory,
            source_grid,
        )
        map_aligned.append(aligned)
        offsets.append(
            None
            if aligned
            else _source_offsets_uas(
                request.source, request.distances, device=device, dtype=dtype
            )
        )
        key = (
            request.source.geometry.shape,
            len(request.source.geometry.band_names),
            bool(request.strict_coverage),
            aligned,
        )
        # Only shape-compatible sources can share a tensor contraction. The
        # physical models and trajectories within a group may still differ.
        groups.setdefault(key, []).append(request_index)

    temporal_batch = resolved_schedule.temporal_batch_size
    if temporal_batch is None:
        temporal_batch = min(int(times.numel()), 49 if device.type == "cuda" else 1)
    temporal_batch = max(1, int(temporal_batch))
    curve_batch = resolved_schedule.light_curve_batch_size
    if curve_batch is None:
        curve_batch = len(requests)
    curve_batch = max(1, int(curve_batch))

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
        elif bool(getattr(request.source, "is_time_static", False)):
            factorized_bases[index] = request.source.brightness(
                times_device[:1], device=device, dtype=dtype
            )[0]
    factorized_unlensed = [
        None if base is None else base.sum(dim=(0, 1)) for base in factorized_bases
    ]
    runtime.synchronize()
    source_seconds += perf_counter() - factorized_started

    left_map = right_map = None
    map_index = -1

    def advance_map():
        # Keep only the bracketing pair needed by the next photometry epoch.
        nonlocal left_map, right_map, map_index, map_seconds, dynamic_metadata
        left_map, right_map = right_map, next(map_iterator)
        map_index += 1
        map_seconds += right_map.timing.delivered_seconds
        map_methods.add(right_map.method)
        dynamic_metadata = {
            key: value
            for key, value in right_map.metadata.items()
            if key == "rasterizer"
            or key.startswith(("dynamic_", "dual_scout_", "cross_system_"))
        }
        if map_observer is not None:
            map_observer(map_index, right_map)

    def sample_frame(frame, active, frame_index):
        if all(map_aligned[index] for index in active):
            return frame.values.unsqueeze(0).expand(len(active), -1, -1)
        if any(offsets[index] is None for index in active):
            raise RuntimeError("mixed aligned and sampled sources cannot share a group")
        x_batch = torch.stack(
            [offsets[index][0] + centers[index][frame_index, 0] for index in active]
        )
        y_batch = torch.stack(
            [offsets[index][1] + centers[index][frame_index, 1] for index in active]
        )
        return _sample_map_batch(
            frame, x_batch, y_batch, strict_coverage=requests[active[0]].strict_coverage
        )

    for batch_start in range(0, int(times.numel()), temporal_batch):
        batch_stop = min(int(times.numel()), batch_start + temporal_batch)
        runtime.synchronize()
        source_started = perf_counter()
        brightness = [
            request.source.brightness(
                times_device[batch_start:batch_stop], device=device, dtype=dtype
            )
            if factorized_bases[index] is None
            else None
            for index, request in enumerate(requests)
        ]
        runtime.synchronize()
        source_seconds += perf_counter() - source_started
        for local, frame_index in enumerate(range(batch_start, batch_stop)):
            while map_index < right_indices[frame_index]:
                advance_map()
            weight = fractions[frame_index]
            for group_indices in groups.values():
                for group_start in range(0, len(group_indices), curve_batch):
                    active = group_indices[group_start : group_start + curve_batch]
                    runtime.synchronize()
                    convolution_started = perf_counter()
                    if left_map is None or weight == 1.0:
                        samples = sample_frame(right_map, active, frame_index)
                    else:
                        # Interpolate the two map-source contractions, not the
                        # full maps. This is algebraically equivalent for a
                        # fixed source frame and avoids a map-sized temporary.
                        samples = sample_frame(left_map, active, frame_index)
                        if weight != 0.0:
                            other = sample_frame(right_map, active, frame_index)
                            samples = samples + weight * (other - samples)
                    source_items = [
                        (
                            factorized_bases[index]
                            if factorized_bases[index] is not None
                            else brightness[index][local]
                        )
                        for index in active
                    ]
                    source_frames = (
                        source_items[0].unsqueeze(0)
                        if len(source_items) == 1
                        else torch.stack(source_items)
                    )
                    lensed = (source_frames * samples[..., None]).sum(dim=(1, 2))
                    unlensed_items = [
                        (
                            factorized_unlensed[index]
                            if factorized_unlensed[index] is not None
                            else source_items[row].sum(dim=(0, 1))
                        )
                        for row, index in enumerate(active)
                    ]
                    unlensed = (
                        unlensed_items[0].unsqueeze(0)
                        if len(unlensed_items) == 1
                        else torch.stack(unlensed_items)
                    )
                    if any(
                        factorized_amplitudes[index] is not None for index in active
                    ):
                        modulation = torch.stack(
                            [
                                (
                                    factorized_amplitudes[index][frame_index]
                                    if factorized_amplitudes[index] is not None
                                    else torch.ones_like(unlensed_items[row])
                                )
                                for row, index in enumerate(active)
                            ]
                        )
                        lensed = lensed * modulation
                        unlensed = unlensed * modulation
                    runtime.synchronize()
                    convolution_seconds += perf_counter() - convolution_started
                    for row, index in enumerate(active):
                        flux_rows[index].append(lensed[row])
                        unlensed_rows[index].append(unlensed[row])
    while map_index < map_times.numel() - 1:
        advance_map()
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
        }
        if runtime.profiling.value == "detailed"
        else {},
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
                    "map_epochs": int(map_times.numel()),
                    "flux_epochs": int(times.numel()),
                    "map_interpolation": "linear"
                    if flux_times_days is not None
                    else "none",
                    "coherent_source_factorized": (factorized_bases[index] is not None),
                    "map_aligned_source_fast_path": map_aligned[index],
                    "dynamic_map_metadata": dynamic_metadata,
                    **dynamic_metadata,
                },
                timing=timing,
            )
        )
    return tuple(results)
