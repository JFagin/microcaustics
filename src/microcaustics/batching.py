"""Batch scientifically independent map and light-curve calculations."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from .config import IPMConfig
from .geometry import PlaneGrid, PlaneRegion

if TYPE_CHECKING:
    from .config import CausticConfig, DynamicConfig, IRSConfig
    from .results import MagnificationMap
    from .simulation import MicrolensingSimulation
    from .system import MicrolensingRealization, MicrolensingSystem


@dataclass(frozen=True)
class IndependentLightCurveBatch:
    """Results and execution metadata for independent light curves.

    ``curves_per_batch`` is CUDA concurrency, not the number of sources
    contracted through one shared magnification-map sequence. Every returned
    curve was generated from its own :class:`MicrolensingSystem` realization.
    """

    light_curves: tuple[object, ...]
    requested_curves_per_batch: int
    executed_batch_sizes: tuple[int, ...]
    oom_reductions: int
    wall_seconds: float | None

    @property
    def seconds_per_curve(self) -> float | None:
        """Return the amortized wall time per independent light curve."""

        return (
            None
            if self.wall_seconds is None
            else self.wall_seconds / max(len(self.light_curves), 1)
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
    from .system import MicrolensingRealization, _with_method_options

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


def _run_independent_curve(
    realization: MicrolensingRealization,
    map_times_days: Sequence[float],
    flux_times_days: Sequence[float] | None,
    *,
    include_labels: bool,
    method: IPMConfig | IRSConfig | None,
    schedule: DynamicConfig | None,
    caustics: CausticConfig | None,
    map_observer,
    stream: torch.cuda.Stream | None,
) -> object:
    """Execute one independent curve on an optional private CUDA stream."""

    kwargs = {
        "method": method,
        "schedule": schedule,
        "map_observer": map_observer,
    }
    if include_labels:
        kwargs["caustics"] = caustics

    def calculate() -> object:
        from .system import _evaluate_light_curve

        return _evaluate_light_curve(
            realization,
            map_times_days,
            flux_times_days,
            include_labels=include_labels,
            **kwargs,
        )

    if stream is None:
        return calculate()
    with torch.cuda.device(realization.simulation.runtime.device):
        with torch.cuda.stream(stream):
            result = calculate()
        stream.synchronize()
    return result


def _run_independent_group(
    realizations: Sequence[MicrolensingRealization],
    map_times_days: Sequence[float],
    flux_times_days: Sequence[float] | None,
    *,
    include_labels: bool,
    method: IPMConfig | IRSConfig | None,
    schedule: DynamicConfig | None,
    caustics: CausticConfig | None,
    map_observers: Sequence[object | None],
) -> tuple[object, ...]:
    """Run one concurrency group while preserving input order."""

    device = realizations[0].simulation.runtime.device
    if device.type != "cuda" or len(realizations) == 1:
        return tuple(
            _run_independent_curve(
                realization,
                map_times_days,
                flux_times_days,
                include_labels=include_labels,
                method=method,
                schedule=schedule,
                caustics=caustics,
                map_observer=observer,
                stream=None,
            )
            for realization, observer in zip(realizations, map_observers, strict=True)
        )
    streams = tuple(torch.cuda.Stream(device=device) for _ in realizations)
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))
    with ThreadPoolExecutor(max_workers=len(realizations)) as executor:
        futures = tuple(
            executor.submit(
                _run_independent_curve,
                realization,
                map_times_days,
                flux_times_days,
                include_labels=include_labels,
                method=method,
                schedule=schedule,
                caustics=caustics,
                map_observer=observer,
                stream=stream,
            )
            for realization, observer, stream in zip(
                realizations, map_observers, streams, strict=True
            )
        )
        return tuple(future.result() for future in futures)


def batched_system_light_curves(
    systems: Sequence[MicrolensingSystem | MicrolensingRealization],
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
    profile: bool = False,
    **solver_options,
) -> IndependentLightCurveBatch:
    """Generate independent systems concurrently on one CUDA device.

    Every input owns its stellar realization, source and trajectory. Compatible
    compiled kernels are reused, but no map or physical state is shared. On a
    CUDA out-of-memory error the current concurrency is halved and retried.
    CPU and Apple MPS calls retain the same API and execute sequentially.

    Tune ``curves_per_batch`` with :func:`tune_system_light_curve_batch` on
    representative systems. It is intentionally explicit because the optimum
    depends on stellar count, map geometry, labels and accelerator memory.
    """

    from .sources.variability import _validate_source_driver
    from .system import (
        _light_curve_options,
        _light_curve_times,
        _retaining_map_observer,
    )

    map_times_days, flux_times_days = _light_curve_times(
        map_times_days,
        duration_days=duration_days,
        map_cadence_days=map_cadence_days,
        source_cadence_days=source_cadence_days,
        flux_times_days=flux_times_days,
        start_day=start_day,
    )
    options = _light_curve_options(
        dict(method=method, schedule=schedule, caustics=caustics, **solver_options),
        include_labels=include_labels,
    )
    method, schedule, caustics = (
        options.get("method"),
        options["schedule"],
        options.get("caustics"),
    )
    systems = tuple(systems)
    if not systems:
        raise ValueError("at least one microlensing system is required")
    requested = int(curves_per_batch)
    if requested < 1:
        raise ValueError("curves_per_batch must be positive")
    observers = (
        (None,) * len(systems) if map_observers is None else tuple(map_observers)
    )
    if len(observers) != len(systems):
        raise ValueError("map_observers must match the number of systems")
    retained = []
    if keep_maps_at_days is not None:
        composed = []
        for observer in observers:
            combined, maps = _retaining_map_observer(
                map_times_days, keep_maps_at_days, observer
            )
            composed.append(combined)
            retained.append(maps)
        observers = tuple(composed)
    for item in systems:
        _validate_source_driver(item.source, apply_driving_signal)
    realizations = tuple(
        item if hasattr(item, "simulation") else item._realize_for_times(map_times_days)
        for item in systems
    )
    if apply_driving_signal is False:
        realizations = tuple(
            replace(item, source=item._mean_source) for item in realizations
        )
    device = realizations[0].simulation.runtime.device
    for realization in realizations[1:]:
        if realization.simulation.runtime.device != device:
            raise ValueError("one independent batch must use a single device")

    if profile and device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = perf_counter() if profile else None
    outputs: list[object] = []
    executed: list[int] = []
    oom_reductions = 0
    start = 0
    current = min(requested, len(realizations))
    while start < len(realizations):
        count = min(current, len(realizations) - start)
        group = realizations[start : start + count]
        group_observers = observers[start : start + count]
        try:
            outputs.extend(
                _run_independent_group(
                    group,
                    map_times_days,
                    flux_times_days,
                    include_labels=include_labels,
                    method=method,
                    schedule=schedule,
                    caustics=caustics,
                    map_observers=group_observers,
                )
            )
            executed.append(count)
            start += count
        except BaseException as error:
            if not oom_backoff or not _is_cuda_oom(error) or count == 1:
                raise
            oom_reductions += 1
            for observer in group_observers:
                reset = getattr(observer, "reset", None)
                if callable(reset):
                    reset()
            current = max(1, count // 2)
            warnings.warn(
                f"CUDA memory was insufficient for {count} concurrent light curves. "
                f"Retrying with curves_per_batch={current}; numerical settings are unchanged",
                RuntimeWarning,
                stacklevel=2,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    if profile and device.type == "cuda":
        torch.cuda.synchronize(device)
    if retained:
        outputs = [
            replace(item, maps=maps)
            for item, maps in zip(outputs, retained, strict=True)
        ]
    return IndependentLightCurveBatch(
        light_curves=tuple(outputs),
        requested_curves_per_batch=requested,
        executed_batch_sizes=tuple(executed),
        oom_reductions=oom_reductions,
        wall_seconds=None if start_time is None else perf_counter() - start_time,
    )


def tune_system_light_curve_batch(
    systems: Sequence[MicrolensingSystem | MicrolensingRealization],
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
    """Benchmark independent-curve concurrency on representative systems.

    The systems should resemble the intended workload. The function reports
    rejected OOM candidates and verifies fluxes and labels against sequential
    execution by default. Tuning is never run implicitly by production calls.
    Duration, cadence and plain solver controls match
    :func:`batched_system_light_curves`.
    """

    from .system import _light_curve_times

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
        device = (
            systems[0].simulation.runtime.device
            if hasattr(systems[0], "simulation")
            else systems[0].realize().simulation.runtime.device
        )
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
                for expected, actual in zip(
                    reference.light_curves, measured.light_curves, strict=True
                ):
                    if not torch.allclose(
                        _curve_flux(expected), _curve_flux(actual), rtol=rtol, atol=atol
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
