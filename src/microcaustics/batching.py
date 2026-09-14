"""Batch scientifically independent map and light-curve calculations."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import local
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from ._batch_storage import StoredSystemLightCurves
from ._system.scheduling import (
    _evaluate_light_curve,
    _light_curve_options,
    _light_curve_times,
    _retaining_map_observer,
    _with_method_options,
)
from .config import IPMConfig
from .geometry import PlaneGrid, PlaneRegion

if TYPE_CHECKING:
    from .config import CausticConfig, DynamicConfig, IRSConfig
    from .multi_system import MultiImageSystem
    from .results import MagnificationMap
    from .simulation import MicrolensingSimulation
    from .system import MicrolensingRealization, MicrolensingSystem


_INDEPENDENT_STREAM_STATE = local()


@dataclass(frozen=True)
class IndependentLightCurveBatch:
    """Results and execution metadata for independently simulated systems.

    ``curves_per_batch`` counts concurrent macroimage light curves. A returned
    item is either one :class:`LightCurve` or one reconstructed
    :class:`MultiImageLightCurves`. Disk-backed calls keep only compact records;
    :meth:`load_system` restores either result type on demand.
    """

    light_curves: tuple[object, ...]
    requested_curves_per_batch: int
    executed_batch_sizes: tuple[int, ...]
    oom_reductions: int
    wall_seconds: float | None
    stored_systems: tuple[StoredSystemLightCurves, ...] = ()
    output_path: Path | None = None
    storage_mode: str = "memory"
    compute_seconds: float | None = None
    write_seconds: float | None = None
    writer_wait_seconds: float | None = None

    @property
    def completed_systems(self) -> int:
        """Number of complete input systems delivered or stored."""

        return (
            len(self.stored_systems) if self.stored_systems else len(self.light_curves)
        )

    @property
    def completed_light_curves(self) -> int:
        """Number of individual macroimage curves represented by the batch."""

        if self.stored_systems:
            return sum(item.image_count for item in self.stored_systems)
        from .results import MultiImageLightCurves

        return sum(
            len(item.images) if isinstance(item, MultiImageLightCurves) else 1
            for item in self.light_curves
        )

    def load_system(self, index: int, *, device="cpu"):
        """Load one result by input index, or return its in-memory value."""

        if self.stored_systems:
            return self.stored_systems[index].load(device=device)
        return self.light_curves[index]

    @property
    def seconds_per_curve(self) -> float | None:
        """Return the amortized wall time per independent light curve."""

        return (
            None
            if self.wall_seconds is None
            else self.wall_seconds / max(self.completed_light_curves, 1)
        )


@dataclass(frozen=True)
class IndependentBatchTuningTrial:
    """One measured independent-curve concurrency candidate."""

    curves_per_batch: int
    wall_seconds: float | None
    seconds_per_curve: float | None
    peak_memory_bytes: int | None
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True)
class IndependentBatchTuningResult:
    """Selected concurrency and all trials from representative systems."""

    curves_per_batch: int
    trials: tuple[IndependentBatchTuningTrial, ...]


@dataclass(frozen=True)
class StaticMapRequest:
    """One source-independent static magnification-map request.

    Requests in a fused batch may have different point-mass fields and static
    epochs, but must share the macro lens, lens/source geometry, runtime, and
    numerical method. This is the common use case for batching independent
    stellar realizations or mass functions on one macroimage.
    """

    simulation: MicrolensingSimulation
    lens_region: PlaneRegion
    source_grid: PlaneGrid
    time_days: float = 0.0
    name: str | None = None


def _validate_compatible_requests(
    requests: Sequence[StaticMapRequest],
    method: IPMConfig,
) -> None:
    if not requests:
        raise ValueError("at least one static map request is required")
    if method.dual_scout_scalar_correction:
        raise ValueError(
            "independent static maps should use k=1 rather than the dynamic "
            "k=1-to-k=2 scalar correction"
        )
    first = requests[0]
    first_runtime = first.simulation.runtime
    for request in requests[1:]:
        runtime = request.simulation.runtime
        if request.lens_region != first.lens_region:
            raise ValueError("batched static maps must share the lens region")
        if request.source_grid != first.source_grid:
            raise ValueError("batched static maps must share the source grid")
        if request.simulation.macro_lens != first.simulation.macro_lens:
            raise ValueError("batched static maps must share the macro lens")
        if (
            runtime.device != first_runtime.device
            or runtime.dtype != first_runtime.dtype
            or runtime.backend != first_runtime.backend
        ):
            raise ValueError("batched static maps must share one resolved runtime")
    if method.tiled and method.scout_ratio != 1:
        raise ValueError(
            "independent static tiled maps use scout_ratio=1. Use k=2 plus its "
            "scalar correction is reserved for dynamic sequences"
        )
    if not method.far_field_approx.enabled and len(requests) > 1:
        raise ValueError(
            "fused independent-map batching currently requires the Taylor "
            "far-field approximation"
        )


def _calculate_compatible_batch(
    requests: Sequence[StaticMapRequest],
    method: IPMConfig,
) -> tuple[MagnificationMap, ...]:
    from .solvers.far_field import TaylorFarFieldApproximation
    from .solvers.ipm import _source_scout_cells, temporal_batch_ipm

    first = requests[0]
    far_fields = tuple(
        TaylorFarFieldApproximation(
            request.simulation,
            request.lens_region,
            method.far_field_approx,
            time_days=float(request.time_days),
        )
        for request in requests
    )
    selected = None
    selected_shape = None
    selection_metadata: dict[str, object] = {
        "independent_map_batch": True,
        "independent_map_batch_size": len(requests),
    }
    per_request_counts: list[int] = []
    if method.tiled:
        selections = []
        for request, far_field in zip(requests, far_fields, strict=True):
            indices, fine_ny, fine_nx, _ = _source_scout_cells(
                request.simulation,
                far_field,
                request.lens_region,
                request.source_grid,
                method,
                time_days=float(request.time_days),
            )
            selections.append(indices)
            per_request_counts.append(int(indices.numel()))
            shape = (int(fine_ny), int(fine_nx))
            if selected_shape is None:
                selected_shape = shape
            elif selected_shape != shape:
                raise ValueError("independent scouts produced incompatible grids")
        selected = torch.unique(torch.cat(selections), sorted=True)
        selection_metadata.update(
            {
                "independent_selected_cells_per_request": per_request_counts,
                "independent_union_selected_cells": int(selected.numel()),
                "selected_fine_cells": int(selected.numel()),
                "selected_fine_fraction": float(
                    selected.numel() / max(selected_shape[0] * selected_shape[1], 1)
                ),
            }
        )

    outputs = temporal_batch_ipm(
        first.simulation,
        first.lens_region,
        first.source_grid,
        method,
        [float(request.time_days) for request in requests],
        far_fields=far_fields,
        selected_cell_indices=selected,
        selected_cell_shape=selected_shape,
        selection_metadata=selection_metadata,
    )
    return tuple(
        replace(
            output,
            metadata={
                **output.metadata,
                "independent_map_batch": True,
                "independent_map_batch_index": index,
                "independent_map_request_name": request.name,
                "temporal_reuse": False,
            },
        )
        for index, (request, output) in enumerate(zip(requests, outputs, strict=True))
    )


def batched_magnification_maps(
    requests: Sequence[StaticMapRequest],
    *,
    method: IPMConfig,
    batch_size: int | None = None,
) -> tuple[MagnificationMap, ...]:
    """Generate compatible independent maps in fused solver batches.

    The function is not a temporal approximation. Every request builds its own
    stellar far-field coefficients and complete ``k=1`` source scout. Their
    selected cells are conservatively unioned only to permit one fused raster
    launch. CUDA out-of-memory failures halve the current batch without
    changing any numerical setting.
    """

    requests = tuple(requests)
    _validate_compatible_requests(requests, method)
    requested_batch = len(requests) if batch_size is None else int(batch_size)
    if requested_batch < 1:
        raise ValueError("batch_size must be positive")
    outputs: list[MagnificationMap] = []
    start = 0
    current_batch = min(requested_batch, len(requests))
    while start < len(requests):
        count = min(current_batch, len(requests) - start)
        chunk = requests[start : start + count]
        try:
            outputs.extend(_calculate_compatible_batch(chunk, method))
            start += count
        except torch.cuda.OutOfMemoryError:
            if count == 1:
                raise
            current_batch = max(1, count // 2)
            torch.cuda.empty_cache()
    return tuple(outputs)


def batched_system_maps(
    systems: Sequence[MicrolensingSystem | MicrolensingRealization],
    *,
    map_width_uas: float | None = None,
    map_pixels: int | None = None,
    method: IPMConfig | None = None,
    batch_size: int | None = None,
    time_days: float = 0.0,
    **solver_options,
) -> tuple[MagnificationMap, ...]:
    """Generate compatible independent maps from high-level systems.

    Every system retains an independent stellar realization. Centered square
    source-independent systems may share ``map_width_uas`` and ``map_pixels``
    instead of carrying explicit ``PlaneGrid`` objects. This helper only groups
    compatible numerical work and applies no temporal or shared-field
    approximation. Automatic OOM recovery is inherited from
    :func:`batched_magnification_maps`.
    Plain ``rays``, ``refinement``, ``virtual_refinement``, ``scout_ratio`` and
    ``far_field`` overrides match ``system.magnification_map``. An advanced
    IPM configuration remains optional.
    """

    from .config import _production_static_ipm_config
    from .system import MicrolensingRealization

    options = _with_method_options(dict(method=method, **solver_options), dynamic=False)
    unknown = set(options) - {"method"}
    if unknown:
        raise TypeError(f"unsupported static-map options {sorted(unknown)}")
    method = options["method"]
    if method is not None and not isinstance(method, IPMConfig):
        raise ValueError("independent fused static-map batching requires an IPM method")

    resolved = []
    for item in systems:
        if isinstance(item, MicrolensingRealization):
            if map_width_uas is not None or map_pixels is not None:
                raise ValueError(
                    "map_width_uas and map_pixels cannot override a realized system"
                )
            resolved.append(item)
        else:
            resolved.append(
                item._with_square_map_grid(
                    map_width_uas=map_width_uas,
                    map_pixels=map_pixels,
                ).realize()
            )
    resolved = tuple(resolved)
    if not resolved:
        raise ValueError("at least one microlensing system is required")
    requested_method = _production_static_ipm_config() if method is None else method
    methods = tuple(item._method_for_domain(requested_method) for item in resolved)
    if any(candidate != methods[0] for candidate in methods[1:]):
        raise ValueError("batched systems must use the same integration domain")
    requests = tuple(
        StaticMapRequest(
            item.simulation,
            item.lens_region,
            item.source_grid,
            time_days=float(time_days),
            name=f"system-{index}",
        )
        for index, item in enumerate(resolved)
    )
    return batched_magnification_maps(
        requests,
        method=methods[0],
        batch_size=batch_size,
    )


def _is_cuda_oom(error: BaseException) -> bool:
    """Return whether an exception represents CUDA memory exhaustion."""

    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "out of memory" in message and ("cuda" in message or "triton" in message)


def _curve_flux(result: object) -> torch.Tensor:
    curve = getattr(result, "light_curve", result)
    return torch.as_tensor(curve.flux)


def _curve_labels(result: object) -> torch.Tensor | None:
    labels = getattr(getattr(result, "labels", result), "crossing_labels", None)
    return None if labels is None else torch.as_tensor(labels)


def _result_curves(result: object) -> tuple[object, ...]:
    from .results import MultiImageLightCurves

    if isinstance(result, MultiImageLightCurves):
        return tuple(item.light_curve for item in result.images)
    return (getattr(result, "light_curve", result),)


def _system_device(system) -> torch.device:
    from .multi_system import MultiImageSystem

    if isinstance(system, MultiImageSystem):
        system = next(iter(system.images.values()))
    realization = system if hasattr(system, "simulation") else system.realize()
    return realization.simulation.runtime.device


@dataclass(frozen=True)
class _CurveJob:
    """One independently executable macroimage plus its parent-system identity."""

    parent_index: int
    image_index: int
    image_name: str
    arrival_time_delay_days: float
    realization: object
    map_times_days: Sequence[float]
    flux_times_days: Sequence[float] | None
    include_labels: bool
    method: object
    schedule: object
    caustics: object
    map_observer: object | None
    trajectory: object | None = None
    strict_coverage: bool | None = None
    diagnostic_grid: object | None = None
    include_distance_map: bool = False
    multi_image: bool = False


@dataclass(frozen=True)
class _ParentInfo:
    """Information needed to undo the flat execution layout."""

    image_names: tuple[str, ...]
    arrival_time_delays_days: tuple[float, ...]
    metadata: dict[str, object]
    multi_image: bool


def _run_independent_curve(job: _CurveJob, stream: torch.cuda.Stream | None) -> object:
    """Execute one independent curve on an optional private CUDA stream."""

    kwargs = {
        "method": job.method,
        "schedule": job.schedule,
        "map_observer": job.map_observer,
    }
    if job.include_labels:
        kwargs["caustics"] = job.caustics
        if job.multi_image:
            kwargs["diagnostic_grid"] = job.diagnostic_grid
            kwargs["include_distance_map"] = job.include_distance_map
    if job.multi_image:
        kwargs["trajectory"] = job.trajectory
        kwargs["strict_coverage"] = job.strict_coverage

    def calculate() -> object:
        curve = _evaluate_light_curve(
            job.realization,
            job.map_times_days,
            job.flux_times_days,
            include_labels=job.include_labels,
            **kwargs,
        )
        if job.multi_image:
            curve = replace(
                curve,
                metadata={
                    **curve.metadata,
                    "macro_image_name": job.image_name,
                    "arrival_time_delay_days": job.arrival_time_delay_days,
                    "source_time_convention": "observer_time_minus_arrival_delay",
                    "multi_image_simulation": True,
                },
            )
        return curve

    if stream is None:
        return calculate()
    with (
        torch.cuda.device(job.realization.simulation.runtime.device),
        torch.cuda.stream(stream),
    ):
        result = calculate()
        stream.synchronize()
    return result


def _resolved_fused_ipm_settings(job: _CurveJob):
    """Return production map settings when a job can enter the fused solver."""

    from ._system.scheduling import _production_dynamic_settings
    from .config import production_ipm_config

    realization = job.realization
    if not hasattr(realization, "_method_for_domain"):
        return None
    method = realization._method_for_domain(
        production_ipm_config() if job.method is None else job.method
    )
    if not isinstance(method, IPMConfig) or not method.tiled:
        return None
    schedule, caustics = _production_dynamic_settings(
        method, job.schedule, job.caustics, include_labels=job.include_labels
    )
    runtime = realization.simulation.runtime
    if not (
        schedule.fused_temporal_ipm
        and method.far_field_approx.enabled
        and not method.scout_trace_centers
        and runtime.device.type == "cuda"
        and runtime.backend.value == "triton"
        and runtime.dtype == torch.float32
        and realization.simulation.point_masses.has_motion
    ):
        return None
    if job.include_labels and (
        caustics is None or caustics.far_field_approx != method.far_field_approx
    ):
        return None
    return method, schedule, caustics


def _cross_system_signature(job: _CurveJob):
    """Return the exact numerical contract that may share fused launches."""

    if not isinstance(job, _CurveJob):
        return None
    setting = _resolved_fused_ipm_settings(job)
    if setting is None:
        return None
    realization = job.realization
    return (
        setting,
        tuple(float(value) for value in job.map_times_days),
        realization.lens_region,
        realization.source_grid,
        realization.simulation.macro_lens,
        realization.simulation.runtime.device,
        realization.simulation.runtime.dtype,
        realization.simulation.runtime.backend,
        bool(job.include_labels),
    )


def _run_cross_system_ipm_group(jobs: Sequence[_CurveJob]) -> tuple[object, ...] | None:
    """Fuse compatible independent systems without sharing their scout cells."""

    if len(jobs) < 2 or any(not isinstance(job, _CurveJob) for job in jobs):
        return None
    signatures = tuple(_cross_system_signature(job) for job in jobs)
    if any(item is None for item in signatures) or any(
        item != signatures[0] for item in signatures[1:]
    ):
        return None
    settings = tuple(signature[0] for signature in signatures)
    first = jobs[0]
    first_method, first_schedule, _ = settings[0]

    from .dynamic import _cross_system_tiled_ipm_maps
    from .photometry import LightCurveRequest, streaming_light_curves
    from .results import _unified_light_curve

    map_sequences, label_sequences = _cross_system_tiled_ipm_maps(
        tuple(
            (
                job.realization.simulation,
                job.realization.lens_region,
                job.realization.source_grid,
                job.map_times_days,
                setting[0],
                setting[1],
            )
            for job, setting in zip(jobs, settings, strict=True)
        ),
        caustic_requests=(
            tuple(
                (
                    job.realization.lens_grid,
                    setting[2],
                    job.diagnostic_grid,
                    job.include_distance_map,
                    job.map_observer,
                )
                for job, setting in zip(jobs, settings, strict=True)
            )
            if first.include_labels
            else None
        ),
    )
    results = []
    for job_index, (job, maps, setting) in enumerate(
        zip(jobs, map_sequences, settings, strict=True)
    ):
        realization = job.realization
        if realization.source is None:
            raise ValueError("light_curve requires a source model")
        labels = None if label_sequences is None else label_sequences[job_index]

        def observed_maps(
            _maps=maps,
            _labels=labels,
            _observer=job.map_observer,
        ):
            if _labels is None:
                for index, frame in enumerate(_maps):
                    if _observer is not None:
                        _observer(index, frame)
                    yield frame
                return
            from .results import LabeledMapFrame

            for index, (frame, labeled) in enumerate(
                zip(_maps, _labels, strict=True)
            ):
                if _observer is not None:
                    _observer(index, LabeledMapFrame(frame, labeled))
                yield frame

        curve = streaming_light_curves(
            realization.simulation,
            realization.lens_region,
            realization.source_grid,
            job.map_times_days,
            (
                LightCurveRequest(
                    realization.source,
                    realization.system.distances,
                    realization._trajectory_in_local_frame(job.trajectory),
                    True if job.strict_coverage is None else job.strict_coverage,
                ),
            ),
            method=setting[0],
            schedule=setting[1],
            flux_times_days=job.flux_times_days,
            _map_iterator=observed_maps(),
        )[0]
        if labels is not None:
            from .results import LabeledLightCurve, MultirateLabeledLightCurve

            wrapper = (
                LabeledLightCurve(curve, labels)
                if job.flux_times_days is None
                else MultirateLabeledLightCurve(curve, labels)
            )
            curve = _unified_light_curve(wrapper)
        else:
            curve = _unified_light_curve(curve)
        if job.multi_image:
            curve = replace(
                curve,
                metadata={
                    **curve.metadata,
                    "macro_image_name": job.image_name,
                    "arrival_time_delay_days": job.arrival_time_delay_days,
                    "source_time_convention": "observer_time_minus_arrival_delay",
                    "multi_image_simulation": True,
                },
            )
        results.append(curve)
    return tuple(results)


def _run_independent_group(
    jobs: Sequence[_CurveJob],
) -> tuple[object, ...]:
    """Run one concurrency group while preserving input order."""

    device = jobs[0].realization.simulation.runtime.device
    if any(job.realization.simulation.runtime.device != device for job in jobs[1:]):
        raise ValueError("one independent batch must use a single device")
    fused = _run_cross_system_ipm_group(jobs)
    if fused is not None:
        return fused
    # A flattened batch may interleave A/B/C/D macroimages from several lens
    # systems. Fuse matching image contracts across systems while preserving
    # the caller's original result order; incompatible leftovers retain the
    # established private-stream execution path.
    buckets: list[list[tuple[int, _CurveJob]]] = []
    bucket_signatures = []
    for index, job in enumerate(jobs):
        signature = _cross_system_signature(job)
        match = next(
            (
                bucket_index
                for bucket_index, existing in enumerate(bucket_signatures)
                if signature is not None and signature == existing
            ),
            None,
        )
        if match is None:
            bucket_signatures.append(signature)
            buckets.append([(index, job)])
        else:
            buckets[match].append((index, job))
    if any(
        len(bucket) > 1 and signature is not None
        for bucket, signature in zip(buckets, bucket_signatures, strict=True)
    ):
        ordered: list[object | None] = [None] * len(jobs)
        leftovers = []
        for bucket, signature in zip(buckets, bucket_signatures, strict=True):
            if signature is not None and len(bucket) > 1:
                calculated = _run_cross_system_ipm_group(
                    tuple(job for _, job in bucket)
                )
                assert calculated is not None
                for (index, _), result in zip(bucket, calculated, strict=True):
                    ordered[index] = result
            else:
                leftovers.extend(bucket)
        if leftovers:
            remaining = _run_independent_stream_group(
                tuple(job for _, job in leftovers)
            )
            for (index, _), result in zip(leftovers, remaining, strict=True):
                ordered[index] = result
        return tuple(item for item in ordered if item is not None)
    return _run_independent_stream_group(jobs)


def _run_independent_stream_group(
    jobs: Sequence[_CurveJob],
) -> tuple[object, ...]:
    """Execute independent jobs on private streams without solver fusion."""

    device = jobs[0].realization.simulation.runtime.device
    if device.type != "cuda" or len(jobs) == 1:
        return tuple(_run_independent_curve(job, None) for job in jobs)
    # Every curve gets a private stream. Waiting on the caller's current stream
    # makes prior setup visible without introducing a device-wide barrier.
    pools = getattr(_INDEPENDENT_STREAM_STATE, "pools", None)
    if pools is None:
        pools = {}
        _INDEPENDENT_STREAM_STATE.pools = pools
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device.type, device_index)
    streams = pools.setdefault(key, [])
    while len(streams) < len(jobs):
        streams.append(torch.cuda.Stream(device=device_index))
    streams = tuple(streams[: len(jobs)])
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = tuple(
            executor.submit(_run_independent_curve, job, stream)
            for job, stream in zip(jobs, streams, strict=True)
        )
        return tuple(future.result() for future in futures)


def _observer_for_image(observer, image_name: str, *, multi_image: bool):
    if not multi_image:
        return observer
    if observer is None:
        return None
    if not isinstance(observer, Mapping):
        raise TypeError("a multi-image map observer must map image names to callbacks")
    return observer.get(image_name)


def _jobs_for_system(
    item,
    parent_index: int,
    map_times_days,
    flux_times_days,
    *,
    include_labels: bool,
    apply_driving_signal: bool | None,
    single_options: dict[str, object],
    raw_options: dict[str, object],
    observer,
) -> tuple[_ParentInfo, tuple[_CurveJob, ...]]:
    from .multi_system import MultiImageSystem
    from .sources import TimeShiftedSource
    from .sources.variability import _validate_source_driver

    if isinstance(item, MultiImageSystem):
        if observer is not None:
            if not isinstance(observer, Mapping):
                raise TypeError(
                    "each multi-image map_observers entry must be a name mapping"
                )
            unknown = set(observer) - set(item.image_names)
            if unknown:
                raise ValueError(
                    f"map observers contain unknown macroimages: {sorted(unknown)}"
                )
        simulation, realizations = item._build_simulation(
            map_times_days, raw_options, include_labels=include_labels
        )
        shared_source = item._shared_source(None, realizations)
        _validate_source_driver(shared_source, apply_driving_signal)
        if apply_driving_signal is False:
            shared_source = realizations[0]._mean_source
        jobs = []
        for image_index, (config, realization) in enumerate(
            zip(simulation.images, realizations, strict=True)
        ):
            if include_labels and config.trajectory is not None:
                raise ValueError(
                    "center labels currently require trajectory=None. Encode bulk "
                    "relative motion in the image point-mass velocities"
                )
            # The source clock is shared, but each macroimage observes it at
            # t-delay. Lens motion and trajectories remain on observer time.
            delayed = TimeShiftedSource(shared_source, config.arrival_time_delay_days)
            realization = replace(realization, source=delayed)
            jobs.append(
                _CurveJob(
                    parent_index,
                    image_index,
                    config.name,
                    float(config.arrival_time_delay_days),
                    realization,
                    map_times_days,
                    flux_times_days,
                    include_labels,
                    config.method,
                    config.schedule,
                    config.caustic_config,
                    _observer_for_image(observer, config.name, multi_image=True),
                    config.trajectory,
                    config.strict_coverage,
                    config.diagnostic_grid,
                    config.include_distance_map,
                    True,
                )
            )
        metadata = {
            "method": "resolved_multi_image_multirate_light_curves"
            if flux_times_days is not None
            else "resolved_multi_image_light_curves",
            "image_names": simulation.image_names,
            "image_count": len(simulation.images),
            "shared_source": dict(shared_source.metadata()),
            "execution_order": "independent_curve_batches",
            "maps_retained": False,
            "labels_included": bool(include_labels),
            "arrival_delays": "explicit_per_image",
        }
        return (
            _ParentInfo(
                simulation.image_names,
                tuple(
                    float(config.arrival_time_delay_days)
                    for config in simulation.images
                ),
                metadata,
                True,
            ),
            tuple(jobs),
        )

    _validate_source_driver(item.source, apply_driving_signal)
    realization = (
        item if hasattr(item, "simulation") else item._realize_for_times(map_times_days)
    )
    if apply_driving_signal is False:
        realization = replace(realization, source=realization._mean_source)
    job = _CurveJob(
        parent_index,
        0,
        "image",
        0.0,
        realization,
        map_times_days,
        flux_times_days,
        include_labels,
        single_options.get("method"),
        single_options["schedule"],
        single_options.get("caustics"),
        observer,
    )
    return _ParentInfo(("image",), (0.0,), {}, False), (job,)


def _assemble_parent(info: _ParentInfo, curves: Sequence[object]):
    if not info.multi_image:
        return curves[0]
    from .results import MacroImageLightCurve, MultiImageLightCurves, TimingBreakdown

    images = tuple(
        MacroImageLightCurve(name, delay, curve)
        for name, delay, curve in zip(
            info.image_names, info.arrival_time_delays_days, curves, strict=True
        )
    )
    components = {
        image.image_name: image.light_curve.timing.delivered_seconds for image in images
    }
    return MultiImageLightCurves(
        images,
        metadata=info.metadata,
        timing=TimingBreakdown(
            collected=all(image.light_curve.timing.collected for image in images),
            steady_seconds=sum(components.values()),
            component_seconds=components,
        ),
    )


def batched_system_light_curves(
    systems: Sequence[MicrolensingSystem | MicrolensingRealization | MultiImageSystem],
    map_times_days: Sequence[float] | None = None,
    flux_times_days: Sequence[float] | None = None,
    *,
    curves_per_batch: int = 1,
    duration_days: float | None = None,
    map_cadence_days: float | None = None,
    source_cadence_days: float | None = None,
    start_day: float = 0.0,
    include_labels: bool = False,
    apply_driving_signal: bool | None = None,
    method: IPMConfig | IRSConfig | None = None,
    schedule: DynamicConfig | None = None,
    caustics: CausticConfig | None = None,
    map_observers: Sequence[object | None] | None = None,
    oom_backoff: bool = True,
    keep_maps_at_days: Sequence[float] | None = None,
    output_path: str | Path | None = None,
    compression: bool = True,
    writer_queue_size: int = 2,
    overwrite: bool = False,
    resume: bool = False,
    profile: bool = False,
    **solver_options,
) -> IndependentLightCurveBatch:
    """Generate arbitrary single- or multi-image systems in curve batches.

    ``curves_per_batch`` counts individual macroimage curves, so doubles, quads,
    and single-image systems can be mixed without padding. Input systems are
    realized lazily, their images are flattened for execution, and results are
    reconstructed in the original system and image order. A CUDA out-of-memory
    error halves concurrency without changing numerical settings.

    With ``output_path=None`` results remain in memory. A directory writes flat
    per-image NPZ files plus ``manifest.json``; a path ending in ``.npz`` writes
    one combined archive. Disk writes use a bounded background queue and the
    returned index loads complete systems with :meth:`load_system`.

    Tune ``curves_per_batch`` with :func:`tune_system_light_curve_batch` on
    representative systems. It is intentionally explicit because the optimum
    depends on stellar count, map geometry, labels and accelerator memory.
    """

    from ._batch_storage import BatchOutputWriter

    map_times_days, flux_times_days = _light_curve_times(
        map_times_days,
        duration_days=duration_days,
        map_cadence_days=map_cadence_days,
        source_cadence_days=source_cadence_days,
        flux_times_days=flux_times_days,
        start_day=start_day,
    )
    raw_options = dict(solver_options)
    if method is not None:
        raw_options["method"] = method
    if schedule is not None:
        raw_options["schedule"] = schedule
    if caustics is not None:
        raw_options["caustics"] = caustics
    options = _light_curve_options(
        raw_options,
        include_labels=include_labels,
    )
    systems = tuple(systems)
    if not systems:
        raise ValueError("at least one microlensing system is required")
    requested = int(curves_per_batch)
    if requested < 1:
        raise ValueError("curves_per_batch must be positive")
    if int(writer_queue_size) < 1:
        raise ValueError("writer_queue_size must be positive")
    if output_path is not None and keep_maps_at_days is not None:
        raise ValueError(
            "disk-backed batches do not serialize retained maps; use map_observers "
            "to save map products separately"
        )
    observers = (
        (None,) * len(systems) if map_observers is None else tuple(map_observers)
    )
    if len(observers) != len(systems):
        raise ValueError("map_observers must match the number of systems")
    # Validate global controls before realizing any stellar field. This keeps
    # configuration errors cheap while realization itself remains lazy.
    from .multi_system import MultiImageSystem
    from .sources.variability import _validate_source_driver

    for item in systems:
        if isinstance(item, MultiImageSystem):
            sources = (
                (item.source,)
                if item.source is not None
                else tuple(image.source for image in item.images.values())
            )
            for source in sources:
                _validate_source_driver(source, apply_driving_signal)
        else:
            _validate_source_driver(item.source, apply_driving_signal)
    if keep_maps_at_days is not None:
        for observer in observers:
            callbacks = (
                observer.values() if isinstance(observer, Mapping) else (observer,)
            )
            for callback in callbacks:
                _retaining_map_observer(map_times_days, keep_maps_at_days, callback)
    writer = (
        None
        if output_path is None
        else BatchOutputWriter(
            output_path,
            compression=compression,
            overwrite=overwrite,
            resume=resume,
        )
    )
    start_time = perf_counter() if profile else None
    compute_seconds = 0.0
    writer_wait_seconds = 0.0
    # Parent-indexed slots preserve caller order even when images from adjacent
    # systems share one execution group or finish storage out of order.
    outputs: list[object | None] = [None] * len(systems)
    records: list[object | None] = [None] * len(systems)
    parent_info: dict[int, _ParentInfo] = {}
    parent_curves: dict[int, list[object | None]] = {}
    executed: list[int] = []
    oom_reductions = 0
    current = requested
    pending: list[_CurveJob] = []
    write_futures = []
    # One writer owns the combined ZIP archive. The bounded future list limits
    # CPU copies and serialized results waiting behind that writer.
    write_executor = ThreadPoolExecutor(max_workers=1) if writer is not None else None
    batch_device = None

    def collect_oldest_write() -> None:
        nonlocal writer_wait_seconds
        started = perf_counter()
        parent_index, future = write_futures.pop(0)
        records[parent_index] = future.result()
        writer_wait_seconds += perf_counter() - started

    def deliver(job: _CurveJob, curve: object) -> None:
        slot = parent_curves[job.parent_index]
        slot[job.image_index] = curve
        if any(item is None for item in slot):
            return
        # A parent becomes externally visible only after all of its images are
        # complete, so users never observe a partially assembled lens system.
        result = _assemble_parent(parent_info[job.parent_index], slot)
        if writer is None:
            outputs[job.parent_index] = result
        else:
            assert write_executor is not None
            future = write_executor.submit(
                writer.write_system, job.parent_index, result
            )
            write_futures.append((job.parent_index, future))
            if len(write_futures) >= int(writer_queue_size):
                collect_oldest_write()
        del parent_curves[job.parent_index]

    def execute_pending(*, drain: bool = False) -> None:
        nonlocal current, oom_reductions, compute_seconds
        # Leave a short tail queued so images from the next system can fill the
        # same CUDA concurrency group. ``drain`` handles the final partial group.
        while len(pending) >= current or (drain and pending):
            count = min(current, len(pending))
            group = tuple(pending[:count])
            device = group[0].realization.simulation.runtime.device
            try:
                if profile and device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = perf_counter()
                calculated = _run_independent_group(group)
                if profile and device.type == "cuda":
                    torch.cuda.synchronize(device)
                compute_seconds += perf_counter() - started
                executed.append(count)
                del pending[:count]
                for job, curve in zip(group, calculated, strict=True):
                    deliver(job, curve)
            except BaseException as error:
                if not oom_backoff or not _is_cuda_oom(error) or count == 1:
                    raise
                oom_reductions += 1
                for job in group:
                    reset = getattr(job.map_observer, "reset", None)
                    if callable(reset):
                        reset()
                current = max(1, count // 2)
                warnings.warn(
                    f"CUDA memory was insufficient for {count} concurrent light "
                    f"curves. Retrying with curves_per_batch={current}; numerical "
                    "settings are unchanged",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()

    try:
        for parent_index, (item, observer) in enumerate(
            zip(systems, observers, strict=True)
        ):
            info, jobs = _jobs_for_system(
                item,
                parent_index,
                map_times_days,
                flux_times_days,
                include_labels=include_labels,
                apply_driving_signal=apply_driving_signal,
                single_options=options,
                raw_options=raw_options,
                observer=observer,
            )
            devices = {job.realization.simulation.runtime.device for job in jobs}
            if len(devices) != 1:
                raise ValueError("one independent batch must use a single device")
            system_device = next(iter(devices))
            if batch_device is None:
                batch_device = system_device
            elif system_device != batch_device:
                raise ValueError("one independent batch must use a single device")
            if keep_maps_at_days is not None:
                retained_jobs = []
                for job in jobs:
                    combined, maps = _retaining_map_observer(
                        map_times_days, keep_maps_at_days, job.map_observer
                    )
                    retained_jobs.append(replace(job, map_observer=combined))
                    # Attach after calculation without changing the scheduler result.
                    parent_info.setdefault(parent_index, info)
                    info.metadata.setdefault("_retained_maps", {})[job.image_index] = (
                        maps
                    )
                jobs = tuple(retained_jobs)
            parent_info[parent_index] = info
            parent_curves[parent_index] = [None] * len(jobs)
            if writer is not None:
                # Resume is checked at system granularity. If any expected image
                # is absent, the writer later replaces every image in that system.
                resumed = writer.resume_record(
                    parent_index,
                    info.image_names,
                    info.arrival_time_delays_days,
                    info.multi_image,
                    info.metadata,
                )
                if resumed is not None:
                    records[parent_index] = resumed
                    del parent_curves[parent_index]
                    continue
            pending.extend(jobs)
            execute_pending()
        execute_pending(drain=True)
        while write_futures:
            collect_oldest_write()
        if writer is not None:
            writer.finalize()
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        if write_executor is not None:
            write_executor.shutdown(wait=True, cancel_futures=True)

    # Restore explicitly retained maps only for in-memory results.
    if keep_maps_at_days is not None:
        for index, result in enumerate(outputs):
            maps_by_image = parent_info[index].metadata.pop("_retained_maps")
            if parent_info[index].multi_image:
                images = tuple(
                    replace(
                        image,
                        light_curve=replace(image.light_curve, maps=maps_by_image[i]),
                    )
                    for i, image in enumerate(result.images)
                )
                outputs[index] = replace(result, images=images)
            else:
                outputs[index] = replace(result, maps=maps_by_image[0])

    wall_seconds = None if start_time is None else perf_counter() - start_time
    stored = () if writer is None else tuple(records)
    return IndependentLightCurveBatch(
        light_curves=tuple(outputs) if writer is None else (),
        requested_curves_per_batch=requested,
        executed_batch_sizes=tuple(executed),
        oom_reductions=oom_reductions,
        wall_seconds=wall_seconds,
        stored_systems=stored,
        output_path=None if writer is None else writer.output_path,
        storage_mode="memory" if writer is None else writer.mode,
        compute_seconds=compute_seconds if profile else None,
        write_seconds=None if writer is None or not profile else writer.write_seconds,
        writer_wait_seconds=None
        if writer is None or not profile
        else writer_wait_seconds,
    )


def tune_system_light_curve_batch(
    systems: Sequence[MicrolensingSystem | MicrolensingRealization | MultiImageSystem],
    map_times_days: Sequence[float] | None = None,
    flux_times_days: Sequence[float] | None = None,
    *,
    candidates: Sequence[int] = (1, 2, 3, 4),
    include_labels: bool = False,
    method: IPMConfig | IRSConfig | None = None,
    schedule: DynamicConfig | None = None,
    caustics: CausticConfig | None = None,
    verify_numerics: bool = True,
    rtol: float = 5.0e-5,
    atol: float = 5.0e-6,
    duration_days: float | None = None,
    map_cadence_days: float | None = None,
    source_cadence_days: float | None = None,
    start_day: float = 0.0,
    apply_driving_signal: bool | None = None,
    **solver_options,
) -> IndependentBatchTuningResult:
    """Benchmark macroimage-curve concurrency on representative systems.

    Inputs may mix single- and multi-image systems and should resemble the
    intended workload. The function reports rejected OOM candidates and
    verifies every macroimage's fluxes and labels against sequential execution
    by default. Tuning is never run implicitly by production calls. Duration,
    cadence and plain solver controls match
    :func:`batched_system_light_curves`.
    """

    map_times_days, flux_times_days = _light_curve_times(
        map_times_days,
        duration_days=duration_days,
        map_cadence_days=map_cadence_days,
        source_cadence_days=source_cadence_days,
        flux_times_days=flux_times_days,
        start_day=start_day,
    )
    systems = tuple(systems)
    values = tuple(dict.fromkeys(int(value) for value in candidates))
    if not values or any(value < 1 for value in values):
        raise ValueError("candidates must contain positive integers")
    common = dict(
        include_labels=include_labels,
        method=method,
        schedule=schedule,
        caustics=caustics,
        oom_backoff=False,
        profile=True,
        apply_driving_signal=apply_driving_signal,
        **solver_options,
    )
    # Pay first-call compilation before collecting the sequential reference.
    batched_system_light_curves(
        systems,
        map_times_days,
        flux_times_days,
        curves_per_batch=1,
        **common,
    )
    reference = batched_system_light_curves(
        systems,
        map_times_days,
        flux_times_days,
        curves_per_batch=1,
        **common,
    )
    trials: list[IndependentBatchTuningTrial] = []
    for candidate in values:
        device = _system_device(systems[0])
        try:
            if candidate != 1:
                batched_system_light_curves(
                    systems,
                    map_times_days,
                    flux_times_days,
                    curves_per_batch=candidate,
                    **common,
                )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            measured = (
                reference
                if candidate == 1
                else batched_system_light_curves(
                    systems,
                    map_times_days,
                    flux_times_days,
                    curves_per_batch=candidate,
                    **common,
                )
            )
            accepted = True
            reason = None
            if verify_numerics:
                for expected_system, actual_system in zip(
                    reference.light_curves, measured.light_curves, strict=True
                ):
                    expected_curves = _result_curves(expected_system)
                    actual_curves = _result_curves(actual_system)
                    if len(expected_curves) != len(actual_curves):
                        accepted = False
                        reason = "macroimage-count verification failed"
                        break
                    for expected, actual in zip(
                        expected_curves, actual_curves, strict=True
                    ):
                        if not torch.allclose(
                            _curve_flux(expected),
                            _curve_flux(actual),
                            rtol=rtol,
                            atol=atol,
                        ):
                            accepted = False
                            reason = "flux verification failed"
                            break
                        expected_labels = _curve_labels(expected)
                        actual_labels = _curve_labels(actual)
                        if expected_labels is not None and not torch.equal(
                            expected_labels, actual_labels
                        ):
                            accepted = False
                            reason = "label verification failed"
                            break
                    if not accepted:
                        break
            peak = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            )
            trials.append(
                IndependentBatchTuningTrial(
                    candidate,
                    measured.wall_seconds,
                    measured.seconds_per_curve,
                    peak,
                    accepted,
                    reason,
                )
            )
        except BaseException as error:
            if not _is_cuda_oom(error):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            trials.append(
                IndependentBatchTuningTrial(
                    candidate, None, None, None, False, "CUDA out of memory"
                )
            )
    accepted_trials = [trial for trial in trials if trial.accepted]
    if not accepted_trials:
        raise RuntimeError("no independent-curve batch candidate was accepted")
    selected = min(accepted_trials, key=lambda trial: trial.seconds_per_curve)
    return IndependentBatchTuningResult(selected.curves_per_batch, tuple(trials))
