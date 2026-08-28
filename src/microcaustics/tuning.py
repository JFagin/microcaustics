"""Optional steady-state tuning for production temporal and spatial work."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, replace
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from .config import (
    AutoTuningConfig,
    CausticConfig,
    DynamicConfig,
    IPMConfig,
    IRSConfig,
)
from .geometry import PlaneGrid, PlaneRegion

if TYPE_CHECKING:
    from .simulation import MicrolensingSimulation


@dataclass(frozen=True)
class TuningTrial:
    """Outcome of one warmed temporal/spatial candidate."""

    temporal_batch_size: int
    spatial_chunk_size: int
    seconds_per_frame: float | None
    peak_device_memory_bytes: int | None
    accepted: bool
    rejection_reason: str | None = None


@dataclass(frozen=True)
class TuningResult:
    """Selected working sizes and the complete candidate audit."""

    operation: str
    temporal_batch_size: int
    spatial_chunk_size: int
    memory_budget_bytes: int | None
    trials: tuple[TuningTrial, ...]
    tuning_seconds: float
    cache_hit: bool = False

    def metadata(self) -> dict[str, object]:
        """Return compact provenance suitable for numerical result metadata."""

        return {
            "autotune_operation": self.operation,
            "autotune_temporal_batch_size": self.temporal_batch_size,
            "autotune_spatial_chunk_size": self.spatial_chunk_size,
            "autotune_memory_budget_bytes": self.memory_budget_bytes,
            "autotune_trial_count": len(self.trials),
            "autotune_accepted_trials": sum(trial.accepted for trial in self.trials),
            "autotune_seconds_excluded": self.tuning_seconds,
            "autotune_cache_hit": self.cache_hit,
        }


_TUNING_CACHE: dict[Hashable, TuningResult] = {}


def clear_tuning_cache() -> None:
    """Clear in-process tuning decisions without touching compiler caches."""

    _TUNING_CACHE.clear()


def _is_oom(error: BaseException) -> bool:
    out_of_memory = getattr(torch, "OutOfMemoryError", ())
    if out_of_memory and isinstance(error, out_of_memory):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _memory_budget(runtime, config: AutoTuningConfig) -> int | None:
    if runtime.device.type != "cuda":
        return None
    properties = torch.cuda.get_device_properties(runtime.device)
    fraction = (
        runtime.memory_fraction
        if config.memory_fraction is None
        else config.memory_fraction
    )
    return int(
        properties.total_memory
        * float(fraction)
        * float(config.memory_headroom_fraction)
    )


def _candidate_trial(
    evaluate: Callable[[int, int], torch.Tensor | None],
    *,
    temporal: int,
    spatial: int,
    work_units: int,
    runtime,
    config: AutoTuningConfig,
    budget: int | None,
    verification_reference: list[torch.Tensor | None],
) -> TuningTrial:
    peak: int | None = None
    try:
        for _ in range(config.warmup_runs):
            evaluate(temporal, spatial)
            runtime.synchronize(force=True)
        durations = []
        signature = None
        for _ in range(config.benchmark_runs):
            if runtime.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(runtime.device)
            runtime.synchronize(force=True)
            started = perf_counter()
            signature = evaluate(temporal, spatial)
            runtime.synchronize(force=True)
            durations.append(perf_counter() - started)
            if runtime.device.type == "cuda":
                measured = int(torch.cuda.max_memory_allocated(runtime.device))
                peak = measured if peak is None else max(peak, measured)
        if budget is not None and peak is not None and peak > budget:
            return TuningTrial(
                temporal,
                spatial,
                None,
                peak,
                False,
                "memory budget exceeded",
            )
        if config.verify_numerics and signature is not None:
            signature = signature.detach().to(device="cpu")
            if verification_reference[0] is None:
                verification_reference[0] = signature
            elif not torch.allclose(
                signature,
                verification_reference[0],
                rtol=config.verification_rtol,
                atol=config.verification_atol,
                equal_nan=True,
            ):
                return TuningTrial(
                    temporal,
                    spatial,
                    None,
                    peak,
                    False,
                    "numerical verification failed",
                )
        return TuningTrial(
            temporal,
            spatial,
            median(durations) / max(1, int(work_units)),
            peak,
            True,
        )
    except Exception as error:
        if not _is_oom(error):
            raise
        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()
        return TuningTrial(
            temporal,
            spatial,
            None,
            peak,
            False,
            "CUDA out of memory",
        )


def _coordinate_tune(
    evaluate: Callable[[int, int], torch.Tensor | None],
    *,
    operation: str,
    base_temporal: int,
    base_spatial: int,
    work_units: int,
    runtime,
    config: AutoTuningConfig,
    cache_key: Hashable,
) -> TuningResult:
    full_key = (operation, cache_key, config)
    if config.cache and full_key in _TUNING_CACHE:
        return replace(
            _TUNING_CACHE[full_key],
            tuning_seconds=0.0,
            cache_hit=True,
        )
    started = perf_counter()
    budget = _memory_budget(runtime, config)
    trials_by_pair: dict[tuple[int, int], TuningTrial] = {}
    reference: list[torch.Tensor | None] = [None]

    def trial(temporal: int, spatial: int) -> TuningTrial:
        key = (int(temporal), int(spatial))
        if key not in trials_by_pair:
            trials_by_pair[key] = _candidate_trial(
                evaluate,
                temporal=key[0],
                spatial=key[1],
                work_units=work_units,
                runtime=runtime,
                config=config,
                budget=budget,
                verification_reference=reference,
            )
        return trials_by_pair[key]

    spatial_values = (
        config.spatial_candidates
        if config.tune_spatial_chunk
        else (int(base_spatial),)
    )
    for spatial in spatial_values:
        trial(base_temporal, spatial)
    accepted_spatial = [
        item
        for item in trials_by_pair.values()
        if item.accepted and item.temporal_batch_size == base_temporal
    ]
    if not accepted_spatial:
        raise RuntimeError(f"no {operation} spatial tuning candidate succeeded")
    selected_spatial = min(
        accepted_spatial,
        key=lambda item: float(item.seconds_per_frame),
    ).spatial_chunk_size

    temporal_values = (
        config.temporal_candidates
        if config.tune_temporal_batch
        else (int(base_temporal),)
    )
    for temporal in temporal_values:
        trial(temporal, selected_spatial)
    accepted_temporal = [
        item
        for item in trials_by_pair.values()
        if item.accepted and item.spatial_chunk_size == selected_spatial
    ]
    if not accepted_temporal:
        raise RuntimeError(f"no {operation} temporal tuning candidate succeeded")
    selected = min(
        accepted_temporal,
        key=lambda item: float(item.seconds_per_frame),
    )
    result = TuningResult(
        operation=operation,
        temporal_batch_size=selected.temporal_batch_size,
        spatial_chunk_size=selected.spatial_chunk_size,
        memory_budget_bytes=budget,
        trials=tuple(trials_by_pair.values()),
        tuning_seconds=perf_counter() - started,
    )
    if config.cache:
        _TUNING_CACHE[full_key] = result
    return result


def _sample_signature(values: torch.Tensor, maximum_axis: int = 64) -> torch.Tensor:
    """Return a small deterministic numerical signature without retaining a map."""

    if values.ndim < 2:
        return values.reshape(-1).detach().cpu()
    step_y = max(1, values.shape[-2] // maximum_axis)
    step_x = max(1, values.shape[-1] // maximum_axis)
    return values[..., ::step_y, ::step_x].reshape(-1).detach().cpu()


def autotune_dynamic_ipm(
    simulation: MicrolensingSimulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    times_days,
    method: IPMConfig,
    schedule: DynamicConfig,
) -> TuningResult:
    """Tune an IPM map sequence using warmed representative production calls."""

    from .dynamic import dynamic_maps

    tuning = schedule.tuning
    if not tuning.enabled:
        raise ValueError("schedule.tuning.enabled must be true")
    times = tuple(float(value) for value in times_days)
    if not times:
        raise ValueError("at least one time is required for tuning")
    trial_count = min(len(times), tuning.maximum_trial_frames)
    trial_times = times[:trial_count]
    base_temporal = min(trial_count, (
        int(schedule.temporal_batch_size)
        if schedule.temporal_batch_size is not None
        else min(20, trial_count)
    ))
    base_spatial = int(method.cell_chunk_size)
    disabled = AutoTuningConfig(enabled=False)

    def evaluate(temporal: int, spatial: int) -> torch.Tensor:
        candidate_method = replace(method, cell_chunk_size=int(spatial))
        candidate_schedule = replace(
            schedule,
            temporal_batch_size=int(temporal),
            tuning=disabled,
        )
        signatures = []
        for result in (
            dynamic_maps(
                simulation,
                lens_region,
                source_grid,
                trial_times,
                method=candidate_method,
                config=candidate_schedule,
            )
        ):
            signatures.append(_sample_signature(result.values, maximum_axis=16))
        if not signatures:
            raise RuntimeError("dynamic map tuning produced no frames")
        return torch.cat(signatures)

    field = simulation.point_masses
    key = (
        simulation.runtime.device.type,
        simulation.runtime.device.index,
        str(simulation.runtime.dtype),
        simulation.runtime.backend.value,
        id(field),
        len(field),
        lens_region,
        source_grid,
        replace(method, cell_chunk_size=1),
        replace(schedule, temporal_batch_size=1, tuning=disabled),
        trial_times,
    )
    tune_temporal = tuning.tune_temporal_batch and field.has_motion
    temporal_candidates = tuple(
        value for value in tuning.temporal_candidates if value <= trial_count
    )
    if trial_count not in temporal_candidates:
        temporal_candidates += (trial_count,)
    effective = replace(
        tuning,
        tune_temporal_batch=tune_temporal,
        temporal_candidates=temporal_candidates,
    )
    return _coordinate_tune(
        evaluate,
        operation="dynamic_ipm",
        base_temporal=base_temporal,
        base_spatial=base_spatial,
        work_units=trial_count,
        runtime=simulation.runtime,
        config=effective,
        cache_key=key,
    )


def autotune_dynamic_irs(
    simulation: MicrolensingSimulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    times_days,
    method: IRSConfig,
    schedule: DynamicConfig,
) -> TuningResult:
    """Tune the lossless IRS ray chunk under the active memory budget.

    The current portable IRS implementation streams frames independently, so
    temporal batching has no numerical kernel to tune. The returned temporal
    value therefore preserves the requested scheduler batch while the spatial
    sweep selects ``ray_chunk_size``.
    """

    from .dynamic import dynamic_maps

    tuning = schedule.tuning
    if not tuning.enabled:
        raise ValueError("schedule.tuning.enabled must be true")
    times = tuple(float(value) for value in times_days)
    if not times:
        raise ValueError("at least one time is required for tuning")
    trial_count = min(len(times), tuning.maximum_trial_frames)
    trial_times = times[:trial_count]
    base_temporal = min(
        trial_count,
        int(schedule.temporal_batch_size)
        if schedule.temporal_batch_size is not None
        else (40 if simulation.runtime.device.type == "cuda" else 1),
    )
    disabled = AutoTuningConfig(enabled=False)

    def evaluate(temporal: int, spatial: int) -> torch.Tensor:
        candidate_method = replace(method, ray_chunk_size=int(spatial))
        candidate_schedule = replace(
            schedule,
            temporal_batch_size=int(temporal),
            tuning=disabled,
        )
        signatures = [
            _sample_signature(result.values, maximum_axis=16)
            for result in dynamic_maps(
                simulation,
                lens_region,
                source_grid,
                trial_times,
                method=candidate_method,
                config=candidate_schedule,
            )
        ]
        if not signatures:
            raise RuntimeError("IRS tuning produced no frames")
        return torch.cat(signatures)

    field = simulation.point_masses
    key = (
        simulation.runtime.device.type,
        simulation.runtime.device.index,
        str(simulation.runtime.dtype),
        simulation.runtime.backend.value,
        id(field),
        len(field),
        lens_region,
        source_grid,
        replace(method, ray_chunk_size=1),
        replace(schedule, temporal_batch_size=1, tuning=disabled),
        trial_times,
    )
    effective = replace(tuning, tune_temporal_batch=False)
    return _coordinate_tune(
        evaluate,
        operation="dynamic_irs",
        base_temporal=base_temporal,
        base_spatial=method.ray_chunk_size,
        work_units=trial_count,
        runtime=simulation.runtime,
        config=effective,
        cache_key=key,
    )


def autotune_dynamic_maps(
    simulation: MicrolensingSimulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    times_days,
    method: IPMConfig | IRSConfig,
    schedule: DynamicConfig,
) -> TuningResult:
    """Dispatch optional dynamic-map tuning to the selected solver."""

    if isinstance(method, IPMConfig):
        return autotune_dynamic_ipm(
            simulation,
            lens_region,
            source_grid,
            times_days,
            method,
            schedule,
        )
    if isinstance(method, IRSConfig):
        return autotune_dynamic_irs(
            simulation,
            lens_region,
            source_grid,
            times_days,
            method,
            schedule,
        )
    raise TypeError("automatic dynamic tuning requires IPMConfig or IRSConfig")


def autotune_caustics(
    simulation: MicrolensingSimulation,
    lens_grid: PlaneGrid,
    source_region: PlaneRegion,
    times_days,
    config: CausticConfig,
) -> TuningResult:
    """Tune caustic detA chunks and temporal batches with full label checks."""

    from .caustics.production import dynamic_labeled_caustics

    tuning = config.tuning
    if not tuning.enabled:
        raise ValueError("config.tuning.enabled must be true")
    times = tuple(float(value) for value in times_days)
    if not times:
        raise ValueError("at least one time is required for tuning")
    trial_count = min(len(times), tuning.maximum_trial_frames)
    trial_times = times[:trial_count]
    base_temporal = min(trial_count, (
        int(config.temporal_batch_size)
        if config.temporal_batch_size is not None
        else min(20, trial_count)
    ))
    disabled = AutoTuningConfig(enabled=False)

    def evaluate(temporal: int, spatial: int) -> torch.Tensor:
        candidate = replace(
            config,
            temporal_batch_size=int(temporal),
            jacobian_chunk_size=int(spatial),
            tuning=disabled,
        )
        frames = dynamic_labeled_caustics(
            simulation,
            lens_grid,
            source_region,
            trial_times,
            candidate,
        )
        labels = torch.tensor(
            [frame.labels.center_label for frame in frames],
            dtype=torch.float64,
        )
        coordinates = []
        for frame in frames:
            values = frame.caustics.caustic_segments_uas.reshape(-1)
            fixed = torch.zeros(257, dtype=torch.float64)
            count = min(values.numel(), 256)
            fixed[0] = float(values.numel())
            if count:
                fixed[1 : count + 1] = values[:count].detach().cpu().to(torch.float64)
            coordinates.append(fixed)
        return torch.cat((labels, *coordinates))

    field = simulation.point_masses
    key = (
        simulation.runtime.device.type,
        simulation.runtime.device.index,
        str(simulation.runtime.dtype),
        simulation.runtime.backend.value,
        id(field),
        len(field),
        lens_grid,
        source_region,
        replace(config, temporal_batch_size=1, jacobian_chunk_size=1, tuning=disabled),
        trial_times,
    )
    tune_temporal = tuning.tune_temporal_batch and field.has_motion
    temporal_candidates = tuple(
        value for value in tuning.temporal_candidates if value <= trial_count
    )
    if trial_count not in temporal_candidates:
        temporal_candidates += (trial_count,)
    effective = replace(
        tuning,
        tune_temporal_batch=tune_temporal,
        temporal_candidates=temporal_candidates,
    )
    return _coordinate_tune(
        evaluate,
        operation="caustics",
        base_temporal=base_temporal,
        base_spatial=config.jacobian_chunk_size,
        work_units=trial_count,
        runtime=simulation.runtime,
        config=effective,
        cache_key=key,
    )
