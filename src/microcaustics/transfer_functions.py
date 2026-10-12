"""Standalone steady and microlensed reverberation-response products."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from time import perf_counter
from typing import Protocol, runtime_checkable

import torch

from .config import DynamicConfig, IPMConfig, IRSConfig
from .geometry import PlaneGrid, PlaneRegion
from .lens import LensingDistances
from .photometry import _sample_map, _source_is_map_aligned, _source_offsets_uas
from .response_rebin import CDFResponseRebinner
from .results import (
    MagnificationMap,
    TimingBreakdown,
    TransferFunction,
    TransferFunctionSeries,
)
from .sources import SourceGeometry, ThermalReprocessingSource
from .trajectories import LinearTrajectory, SourceTrajectory


def rebin_transfer_function(product, delay_edges_days, *, runtime=None):
    """Reconstruct a response product on new edges using conservative CDF/PCHIP.

    Retains the original mean delays, rather than recomputing them from the
    reconstructed curve. A cropped output retains only its included response
    mass, even when the input was normalized. No extrapolation is performed.
    For repeated tensor batches, reuse :class:`CDFResponseRebinner` directly.
    """
    if not isinstance(product, (TransferFunction, TransferFunctionSeries)):
        raise TypeError("product must be a TransferFunction or TransferFunctionSeries")
    plan = CDFResponseRebinner(
        product.delay_edges_days, delay_edges_days, runtime=runtime
    )
    return replace(
        product,
        delay_edges_days=plan.output_edges_days,
        values=plan(product.values),
        metadata={
            **product.metadata,
            "response_reconstruction": "cdf_pchip",
            "native_delay_bins": product.values.shape[-2],
            "mean_delays_preserved": True,
            "output_mass_policy": "retain_included_mass_without_renormalization",
        },
    )


@runtime_checkable
class TransferFunctionSource(Protocol):
    """Source contract for producing reverberation transfer functions."""

    geometry: SourceGeometry

    def transfer_function(
        self,
        delay_edges_days,
        *,
        magnification: torch.Tensor | None = None,
        driver_amplitude: float = 1.0,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Return response mass with shape ``[delay_bin, band]``."""

        ...

    def metadata(self) -> Mapping[str, object]:
        """Return source-model provenance."""

        ...


@runtime_checkable
class MeanResponseDelaySource(Protocol):
    """Source contract for direct response-weighted mean delays."""

    geometry: SourceGeometry

    def mean_response_delays(
        self,
        *,
        magnification: torch.Tensor | None = None,
        driver_amplitude: float = 1.0,
    ) -> torch.Tensor:
        """Return one response-weighted mean delay per band."""

        ...


def steady_mean_response_delays(
    source: MeanResponseDelaySource,
    *,
    driver_amplitude: float = 1.0,
) -> torch.Tensor:
    """Return exact unlensed response-weighted mean delays without binning."""

    if not isinstance(source, MeanResponseDelaySource):
        raise TypeError("source must implement the MeanResponseDelaySource protocol")
    return source.mean_response_delays(driver_amplitude=driver_amplitude)


def microlensed_mean_response_delays(
    source: MeanResponseDelaySource,
    magnification_map: MagnificationMap,
    distances: LensingDistances,
    *,
    source_center_uas: tuple[float, float] = (0.0, 0.0),
    strict_coverage: bool = True,
    driver_amplitude: float = 1.0,
) -> torch.Tensor:
    """Return exact microlensed mean delays without delay-bin construction."""

    if not isinstance(source, MeanResponseDelaySource):
        raise TypeError("source must implement the MeanResponseDelaySource protocol")
    if len(source_center_uas) != 2:
        raise ValueError("source_center_uas must contain Cartesian x and y")
    values_map = magnification_map.values
    device, dtype = values_map.device, values_map.dtype
    offset_x, offset_y = _source_offsets_uas(
        source,
        distances,
        device=device,
        dtype=dtype,
    )
    magnification = _sample_map(
        magnification_map,
        offset_x + float(source_center_uas[0]),
        offset_y + float(source_center_uas[1]),
        strict_coverage=strict_coverage,
    )
    return source.mean_response_delays(
        magnification=magnification,
        driver_amplitude=driver_amplitude,
    )


def microlensed_mean_response_delays_batch(
    source: MeanResponseDelaySource,
    magnification_maps: Sequence[MagnificationMap],
    distances: LensingDistances,
    *,
    source_centers_uas: Sequence[tuple[float, float]] | None = None,
    strict_coverage: bool = True,
    driver_amplitude: float = 1.0,
    spatial_chunk_size: int = 262_144,
) -> torch.Tensor:
    """Return exact mean delays for a bounded batch of magnification maps."""

    if not isinstance(source, MeanResponseDelaySource):
        raise TypeError("source must implement the MeanResponseDelaySource protocol")
    maps = tuple(magnification_maps)
    if not maps:
        raise ValueError("at least one magnification map is required")
    centers = (
        ((0.0, 0.0),) * len(maps)
        if source_centers_uas is None
        else tuple(source_centers_uas)
    )
    if len(centers) != len(maps) or any(len(center) != 2 for center in centers):
        raise ValueError("source centers must contain one Cartesian pair per map")
    device, dtype = maps[0].values.device, maps[0].values.dtype
    if any(item.values.device != device or item.values.dtype != dtype for item in maps):
        raise ValueError("all magnification maps must share device and dtype")
    aligned = all(center == (0.0, 0.0) for center in centers) and all(
        item.grid == maps[0].grid for item in maps
    )
    aligned = aligned and _source_is_map_aligned(
        source,
        distances,
        None,
        maps[0].grid,
    )
    if aligned:
        sampled = torch.stack([item.values for item in maps])
    else:
        offset_x, offset_y = _source_offsets_uas(
            source,
            distances,
            device=device,
            dtype=dtype,
        )
        sampled = torch.stack(
            [
                _sample_map(
                    magnification_map,
                    offset_x + float(center[0]),
                    offset_y + float(center[1]),
                    strict_coverage=strict_coverage,
                )
                for magnification_map, center in zip(maps, centers, strict=True)
            ]
        )
    batched = getattr(source, "batched_mean_response_delays", None)
    if callable(batched):
        return batched(
            magnification=sampled,
            driver_amplitude=driver_amplitude,
            spatial_chunk_size=spatial_chunk_size,
        )
    return torch.stack(
        [
            source.mean_response_delays(
                magnification=magnification,
                driver_amplitude=driver_amplitude,
            )
            for magnification in sampled
        ]
    )


def _mean_delays(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    centers = 0.5 * (edges[:-1] + edges[1:])
    normalizer = values.sum(dim=0).clamp_min(1.0e-30)
    return (values * centers[:, None]).sum(dim=0) / normalizer


def _batched_mean_delays(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    centers = 0.5 * (edges[:-1] + edges[1:])
    normalizer = values.sum(dim=1).clamp_min(1.0e-30)
    return (values * centers[None, :, None]).sum(dim=1) / normalizer


def _cdf_source(source):
    return (
        isinstance(source, ThermalReprocessingSource)
        and getattr(source._runtime, "response_reconstruction", "cdf") == "cdf"
    )


def _native_options(source, native_edges):
    # Explicit native grids must accumulate once, not fit CDF twice.
    return (
        {"reconstruction": "ordinary"}
        if (native_edges is not None and isinstance(source, ThermalReprocessingSource))
        else {}
    )


def steady_transfer_function(
    source: TransferFunctionSource,
    delay_edges_days: torch.Tensor | Sequence[float],
    *,
    driver_amplitude: float = 1.0,
    normalize: bool = True,
    native_delay_edges_days=None,
    runtime=None,
) -> TransferFunction:
    """Return a source's transfer function without microlensing.

    This calculation requires no magnification map, point-mass field, or
    multi-image configuration and can be used for ordinary continuum
    reverberation studies.

    Supply ``native_delay_edges_days`` to accumulate on that grid and use
    CDF/PCHIP reconstruction on ``delay_edges_days``. Thermal disk sources use
    an automatically selected CDF grid by default. Direct pixel mean delays
    are retained for thermal sources; generic sources keep their own method.
    """

    if not isinstance(source, TransferFunctionSource):
        raise TypeError("source must implement the TransferFunctionSource protocol")
    started = perf_counter()
    native_edges = (
        delay_edges_days if native_delay_edges_days is None else native_delay_edges_days
    )
    values = source.transfer_function(
        native_edges,
        magnification=None,
        driver_amplitude=driver_amplitude,
        normalize=normalize,
        **_native_options(source, native_delay_edges_days),
    )
    edges = torch.as_tensor(
        native_edges,
        device=values.device,
        dtype=values.dtype,
    )
    means = (
        source.mean_response_delays(driver_amplitude=driver_amplitude)
        if _cdf_source(source)
        else _mean_delays(values, edges)
    )
    if native_delay_edges_days is not None:
        if isinstance(source, MeanResponseDelaySource):
            means = source.mean_response_delays(driver_amplitude=driver_amplitude)
        plan = CDFResponseRebinner(edges, delay_edges_days, runtime=runtime)
        values = plan(values, validate=False)
        edges = plan.output_edges_days
    elapsed = perf_counter() - started
    return TransferFunction(
        delay_edges_days=edges,
        values=values,
        mean_delays_days=means,
        band_names=source.geometry.band_names,
        metadata={
            "method": "steady_source_transfer_function",
            "response_reconstruction": "cdf_pchip"
            if native_delay_edges_days is not None or _cdf_source(source)
            else "histogram",
            "mean_delay_method": "direct_pixels"
            if (native_delay_edges_days is not None or _cdf_source(source))
            and isinstance(source, MeanResponseDelaySource)
            else "native_bin_centers",
            "microlensed": False,
            "normalized": bool(normalize),
            "driver_amplitude": float(driver_amplitude),
            "source": dict(source.metadata()),
        },
        timing=TimingBreakdown(steady_seconds=elapsed),
    )


def microlensed_transfer_function(
    source: TransferFunctionSource,
    magnification_map: MagnificationMap,
    distances: LensingDistances,
    delay_edges_days: torch.Tensor | Sequence[float],
    *,
    source_center_uas: tuple[float, float] = (0.0, 0.0),
    strict_coverage: bool = True,
    driver_amplitude: float = 1.0,
    normalize: bool = True,
    native_delay_edges_days=None,
    runtime=None,
) -> TransferFunction:
    """Weight one source response by one magnification map.

    The map is sampled onto the physical source pixels using exactly the same
    angular conversion and bilinear sampling as finite-source photometry.
    Optional ``native_delay_edges_days`` selects CDF/PCHIP reconstruction, as
    in :func:`steady_transfer_function`.
    """

    if not isinstance(source, TransferFunctionSource):
        raise TypeError("source must implement the TransferFunctionSource protocol")
    if len(source_center_uas) != 2:
        raise ValueError("source_center_uas must contain Cartesian x and y")
    values_map = magnification_map.values
    device, dtype = values_map.device, values_map.dtype
    started = perf_counter()
    offset_x, offset_y = _source_offsets_uas(
        source,
        distances,
        device=device,
        dtype=dtype,
    )
    magnification = _sample_map(
        magnification_map,
        offset_x + float(source_center_uas[0]),
        offset_y + float(source_center_uas[1]),
        strict_coverage=strict_coverage,
    )
    native_edges = (
        delay_edges_days if native_delay_edges_days is None else native_delay_edges_days
    )
    response = source.transfer_function(
        native_edges,
        magnification=magnification,
        driver_amplitude=driver_amplitude,
        normalize=normalize,
        **_native_options(source, native_delay_edges_days),
    )
    edges = torch.as_tensor(
        native_edges,
        device=response.device,
        dtype=response.dtype,
    )
    means = (
        source.mean_response_delays(
            magnification=magnification, driver_amplitude=driver_amplitude
        )
        if _cdf_source(source)
        else _mean_delays(response, edges)
    )
    if native_delay_edges_days is not None:
        if isinstance(source, MeanResponseDelaySource):
            means = source.mean_response_delays(
                magnification=magnification, driver_amplitude=driver_amplitude
            )
        plan = CDFResponseRebinner(edges, delay_edges_days, runtime=runtime)
        response = plan(response, validate=False)
        edges = plan.output_edges_days
    elapsed = perf_counter() - started
    return TransferFunction(
        delay_edges_days=edges,
        values=response,
        mean_delays_days=means,
        band_names=source.geometry.band_names,
        metadata={
            "method": "microlensed_transfer_function",
            "response_reconstruction": "cdf_pchip"
            if native_delay_edges_days is not None or _cdf_source(source)
            else "histogram",
            "mean_delay_method": "direct_pixels"
            if (native_delay_edges_days is not None or _cdf_source(source))
            and isinstance(source, MeanResponseDelaySource)
            else "native_bin_centers",
            "microlensed": True,
            "map_method": magnification_map.method,
            "map_time_days": float(magnification_map.time_days),
            "source_center_uas": tuple(float(value) for value in source_center_uas),
            "strict_coverage": bool(strict_coverage),
            "normalized": bool(normalize),
            "driver_amplitude": float(driver_amplitude),
            "source": dict(source.metadata()),
        },
        timing=TimingBreakdown(steady_seconds=elapsed),
    )


@torch.no_grad()
def streaming_microlensed_transfer_functions(
    simulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    times_days: torch.Tensor | Sequence[float],
    source: TransferFunctionSource,
    distances: LensingDistances,
    delay_edges_days: torch.Tensor | Sequence[float],
    *,
    method: IRSConfig | IPMConfig,
    trajectory: SourceTrajectory | None = None,
    schedule: DynamicConfig | None = None,
    strict_coverage: bool = True,
    driver_amplitude: float = 1.0,
    normalize: bool = True,
    map_observer: Callable[[int, MagnificationMap], None] | None = None,
    response_batch_size: int | None = None,
    response_spatial_chunk_size: int = 262_144,
    native_delay_edges_days=None,
) -> TransferFunctionSeries:
    """Stream dynamic maps into time-dependent microlensed responses.

    Only response functions and mean lags are retained. The maps are consumed
    one at a time unless ``map_observer`` explicitly stores selected frames.
    With ``native_delay_edges_days``, one CDF/PCHIP grid plan is reused across
    response batches. Physical mean lags are not taken from reconstructed bins.
    """

    if not isinstance(source, TransferFunctionSource):
        raise TypeError("source must implement the TransferFunctionSource protocol")
    times = torch.as_tensor(times_days).reshape(-1)
    if not times.is_floating_point():
        times = times.to(torch.get_default_dtype())
    if times.numel() < 1:
        raise ValueError("at least one transfer-function epoch is required")
    runtime = simulation.runtime
    device, dtype = runtime.device, runtime.dtype
    native_edges = (
        delay_edges_days if native_delay_edges_days is None else native_delay_edges_days
    )
    rebin = (
        None
        if native_delay_edges_days is None
        else CDFResponseRebinner(
            torch.as_tensor(native_edges, device=device, dtype=dtype),
            delay_edges_days,
            runtime=runtime,
        )
    )
    times_device = times.to(device=device, dtype=dtype)
    trajectory = LinearTrajectory() if trajectory is None else trajectory
    centers = trajectory.position_uas(times_device, device=device, dtype=dtype)
    if centers.shape != (times.numel(), 2):
        raise ValueError("trajectory positions must have shape [time, 2]")
    resolved_schedule = DynamicConfig() if schedule is None else schedule
    runtime.synchronize(detailed=False)
    started = perf_counter()
    if response_batch_size is None:
        response_batch_size = resolved_schedule.temporal_batch_size
    if response_batch_size is None:
        response_batch_size = 1
    response_batch_size = int(response_batch_size)
    if response_batch_size < 1:
        raise ValueError("response_batch_size must be positive")
    if response_spatial_chunk_size < 1:
        raise ValueError("response_spatial_chunk_size must be positive")
    batched_response = getattr(source, "batched_transfer_function", None)
    use_batched_response = callable(batched_response) and response_batch_size > 1
    offset_x, offset_y = _source_offsets_uas(
        source,
        distances,
        device=device,
        dtype=dtype,
    )
    responses: list[torch.Tensor] = []
    means: list[torch.Tensor] = []
    sampled_batch: list[torch.Tensor] = []
    map_methods: set[str] = set()

    def flush_response_batch() -> None:
        if not sampled_batch:
            return
        assert callable(batched_response)
        sampled = torch.stack(sampled_batch)
        batch = batched_response(
            native_edges,
            magnification=sampled,
            driver_amplitude=driver_amplitude,
            normalize=normalize,
            **_native_options(source, native_delay_edges_days),
            spatial_chunk_size=response_spatial_chunk_size,
        )
        batch_edges = torch.as_tensor(
            native_edges,
            device=batch.device,
            dtype=batch.dtype,
        )
        batch_means = _batched_mean_delays(batch, batch_edges)
        if rebin is not None or _cdf_source(source):
            direct = getattr(source, "batched_mean_response_delays", None)
            if callable(direct):
                batch_means = direct(
                    magnification=sampled,
                    driver_amplitude=driver_amplitude,
                    spatial_chunk_size=response_spatial_chunk_size,
                )
            elif isinstance(source, MeanResponseDelaySource):
                batch_means = torch.stack(
                    [
                        source.mean_response_delays(
                            magnification=frame, driver_amplitude=driver_amplitude
                        )
                        for frame in sampled_batch
                    ]
                )
        if rebin is not None:
            batch = rebin(batch, validate=False)
        responses.extend(batch.unbind(0))
        means.extend(batch_means.unbind(0))
        sampled_batch.clear()

    for index, magnification_map in enumerate(
        simulation.dynamic_maps(
            lens_region,
            source_grid,
            times.tolist(),
            method=method,
            schedule=resolved_schedule,
        )
    ):
        map_methods.add(magnification_map.method)
        if map_observer is not None:
            map_observer(index, magnification_map)
        if use_batched_response:
            sampled_batch.append(
                _sample_map(
                    magnification_map,
                    offset_x + centers[index, 0],
                    offset_y + centers[index, 1],
                    strict_coverage=strict_coverage,
                )
            )
            if len(sampled_batch) == response_batch_size:
                flush_response_batch()
        else:
            # Reuse the prepared coordinates and sampled map for both the
            # response and its direct moments. Do not resample it a second time.
            sampled = _sample_map(
                magnification_map,
                offset_x + centers[index, 0],
                offset_y + centers[index, 1],
                strict_coverage=strict_coverage,
            )
            values = source.transfer_function(
                native_edges,
                magnification=sampled,
                driver_amplitude=driver_amplitude,
                normalize=normalize,
                **_native_options(source, native_delay_edges_days),
            )
            mean = _mean_delays(
                values,
                torch.as_tensor(native_edges, device=values.device, dtype=values.dtype),
            )
            if rebin is not None:
                values = rebin(values, validate=False)
            if (rebin is not None or _cdf_source(source)) and isinstance(
                source, MeanResponseDelaySource
            ):
                mean = source.mean_response_delays(
                    magnification=sampled,
                    driver_amplitude=driver_amplitude,
                )
            responses.append(values)
            means.append(mean)
    if use_batched_response:
        flush_response_batch()
    runtime.synchronize(detailed=False)
    elapsed = perf_counter() - started
    edges = torch.as_tensor(
        delay_edges_days,
        device=responses[0].device,
        dtype=responses[0].dtype,
    )
    return TransferFunctionSeries(
        times_days=times_device,
        delay_edges_days=edges,
        values=torch.stack(responses),
        mean_delays_days=torch.stack(means),
        band_names=source.geometry.band_names,
        metadata={
            "method": "streaming_microlensed_transfer_functions",
            "response_reconstruction": "cdf_pchip"
            if rebin is not None or _cdf_source(source)
            else "histogram",
            "mean_delay_method": "direct_pixels"
            if (rebin is not None or _cdf_source(source))
            and isinstance(source, MeanResponseDelaySource)
            else "native_bin_centers",
            "microlensed": True,
            "map_methods": sorted(map_methods),
            "maps_retained": False,
            "strict_coverage": bool(strict_coverage),
            "normalized": bool(normalize),
            "driver_amplitude": float(driver_amplitude),
            "response_batch_size": (response_batch_size if use_batched_response else 1),
            "response_spatial_chunk_size": (
                response_spatial_chunk_size if use_batched_response else None
            ),
            "source": dict(source.metadata()),
        },
        timing=TimingBreakdown(
            collected=runtime.profiling_enabled,
            steady_seconds=elapsed,
        ),
    )
