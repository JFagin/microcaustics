"""Memory-safe scheduling for static and moving microlensing maps."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from .config import DynamicConfig, IPMConfig, IRSConfig
from .geometry import PlaneGrid, PlaneRegion
from .results import MagnificationMap, TimingBreakdown

if TYPE_CHECKING:
    from .simulation import MicrolensingSimulation


def _scheduled_map(
    result: MagnificationMap,
    *,
    time_days: float,
    metadata: dict[str, object],
) -> MagnificationMap:
    """Attach dynamic scheduling provenance without copying map values."""

    combined = dict(result.metadata)
    combined.update(metadata)
    return replace(result, time_days=float(time_days), metadata=combined)


def _is_cuda_oom(error: BaseException) -> bool:
    """Recognize PyTorch CUDA allocation failures without masking other bugs."""

    out_of_memory = getattr(torch, "OutOfMemoryError", ())
    if out_of_memory and isinstance(error, out_of_memory):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


class DynamicMapScheduler:
    """Stream source-independent maps with explicit reuse provenance.

    The scheduler never changes map normalization or numerical configuration.
    Its exact optimizations are static-map reuse and memory-safe chunk backoff.
    Endpoint-union scout reuse for moving lenses is intentionally marked as an
    approximation in every returned map.
    """

    def __init__(
        self,
        simulation: MicrolensingSimulation,
        lens_region: PlaneRegion,
        source_grid: PlaneGrid,
        times_days: Sequence[float],
        method: IRSConfig | IPMConfig,
        config: DynamicConfig,
        *,
        far_field_batch_observer=None,
    ) -> None:
        """Store an immutable calculation request and validate its time axis."""

        self.simulation = simulation
        self.lens_region = lens_region
        self.source_grid = source_grid
        self.times = tuple(float(value) for value in times_days)
        if any(not torch.isfinite(torch.tensor(value)).item() for value in self.times):
            raise ValueError("dynamic map times must be finite")
        self.tuning_result = None
        if config.tuning.enabled and self.times:
            from .tuning import autotune_dynamic_maps

            self.tuning_result = autotune_dynamic_maps(
                simulation,
                lens_region,
                source_grid,
                self.times,
                method,
                config,
            )
            if isinstance(method, IPMConfig):
                method = replace(
                    method,
                    cell_chunk_size=self.tuning_result.spatial_chunk_size,
                )
            else:
                method = replace(
                    method,
                    ray_chunk_size=self.tuning_result.spatial_chunk_size,
                )
            config = replace(
                config,
                temporal_batch_size=self.tuning_result.temporal_batch_size,
                tuning=replace(config.tuning, enabled=False),
            )
        self.method = method
        self.config = config
        self._far_field_batch_observer = far_field_batch_observer

    def _batch_size(self) -> int:
        """Choose a deterministic temporal scheduling batch size."""

        if not self.times:
            return 1
        if self.config.temporal_batch_size is not None:
            return min(len(self.times), int(self.config.temporal_batch_size))
        # This is a scheduling batch, not a promise that every solver kernel
        # fuses the temporal dimension. Forty is the validated paper default.
        # Smaller sequences are never padded solely for this interface.
        return min(len(self.times), 40 if self.simulation.runtime.device.type == "cuda" else 1)

    def _calculate_with_backoff(
        self,
        method: IRSConfig | IPMConfig,
        *,
        time_days: float,
        **internal,
    ) -> tuple[MagnificationMap, int]:
        """Retry IPM after CUDA OOM by halving only its lossless cell chunk."""

        current = method
        reductions = 0
        while True:
            try:
                if isinstance(current, IPMConfig):
                    from .solvers.ipm import full_field_ipm

                    result = full_field_ipm(
                        self.simulation,
                        self.lens_region,
                        self.source_grid,
                        current,
                        time_days=time_days,
                        **internal,
                    )
                else:
                    result = self.simulation.magnification_map(
                        self.lens_region,
                        self.source_grid,
                        method=current,
                        time_days=time_days,
                    )
                return result, reductions
            except Exception as error:
                if not _is_cuda_oom(error) or not isinstance(current, IPMConfig):
                    raise
                minimum = int(self.config.minimum_cell_chunk_size)
                if current.cell_chunk_size <= minimum:
                    raise
                next_chunk = max(minimum, current.cell_chunk_size // 2)
                if next_chunk >= current.cell_chunk_size:
                    raise
                current = replace(current, cell_chunk_size=next_chunk)
                reductions += 1
                if self.simulation.runtime.device.type == "cuda":
                    torch.cuda.empty_cache()

    def _calculate_ipm_batch_with_backoff(
        self,
        method: IPMConfig,
        times_days,
        *,
        far_fields=None,
        selected_cell_indices=None,
        selected_cell_shape=None,
        selection_metadata=None,
        scalar_correction=None,
        real_frame_count: int | None = None,
    ) -> tuple[tuple[MagnificationMap, ...], int, int]:
        """Run one fused IPM batch with lossless chunk and batch backoff.

        Spatial OOM recovery first halves only the cell working chunk. If the
        minimum chunk still cannot hold the temporal stack, the real frames
        are split recursively. Neither recovery changes selected cells,
        refinement, virtual refinement, or map values.
        """

        from .solvers.ipm import temporal_batch_ipm

        times = tuple(float(value) for value in times_days)
        far_fields = None if far_fields is None else tuple(far_fields)
        real_frames = len(times) if real_frame_count is None else int(real_frame_count)
        current = method
        reductions = 0
        while True:
            try:
                result = temporal_batch_ipm(
                    self.simulation,
                    self.lens_region,
                    self.source_grid,
                    current,
                    times,
                    far_fields=far_fields,
                    selected_cell_indices=selected_cell_indices,
                    selected_cell_shape=selected_cell_shape,
                    selection_metadata=selection_metadata,
                    scalar_correction=scalar_correction,
                    real_frame_count=real_frames,
                )
                return result, reductions, 0
            except Exception as error:
                if not _is_cuda_oom(error):
                    raise
                minimum = int(self.config.minimum_cell_chunk_size)
                if current.cell_chunk_size > minimum:
                    next_chunk = max(minimum, current.cell_chunk_size // 2)
                    current = replace(current, cell_chunk_size=next_chunk)
                    reductions += 1
                    if self.simulation.runtime.device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                if real_frames <= 1:
                    raise
                # Padded entries are an optimization only. Remove them before
                # recursively reducing the real temporal batch.
                real_times = times[:real_frames]
                real_far_fields = None if far_fields is None else far_fields[:real_frames]
                midpoint = max(1, real_frames // 2)
                left, left_reductions, left_splits = (
                    self._calculate_ipm_batch_with_backoff(
                        current,
                        real_times[:midpoint],
                        far_fields=(
                            None if real_far_fields is None else real_far_fields[:midpoint]
                        ),
                        selected_cell_indices=selected_cell_indices,
                        selected_cell_shape=selected_cell_shape,
                        selection_metadata=selection_metadata,
                        scalar_correction=scalar_correction,
                    )
                )
                right, right_reductions, right_splits = (
                    self._calculate_ipm_batch_with_backoff(
                        current,
                        real_times[midpoint:],
                        far_fields=(
                            None if real_far_fields is None else real_far_fields[midpoint:]
                        ),
                        selected_cell_indices=selected_cell_indices,
                        selected_cell_shape=selected_cell_shape,
                        selection_metadata=selection_metadata,
                        scalar_correction=scalar_correction,
                    )
                )
                return (
                    (*left, *right),
                    reductions + left_reductions + right_reductions,
                    1 + left_splits + right_splits,
                )

    def _static(self) -> Iterator[MagnificationMap]:
        """Reuse one exact map when point lenses have no time dependence."""

        first, reductions = self._calculate_with_backoff(
            self.method,
            time_days=self.times[0],
        )
        for index, time_days in enumerate(self.times):
            scheduled = _scheduled_map(
                first,
                time_days=time_days,
                metadata={
                    "dynamic_frame_index": index,
                    "dynamic_frame_count": len(self.times),
                    "dynamic_static_map_reuse": True,
                    "dynamic_scout_reuse_approximate": False,
                    "dynamic_oom_chunk_reductions": reductions,
                    "dynamic_temporal_batch_size": self._batch_size(),
                },
            )
            if index > 0:
                scheduled = replace(scheduled, timing=TimingBreakdown())
            yield scheduled

    def _ordinary(self) -> Iterator[MagnificationMap]:
        """Stream frames independently for IRS or non-reused IPM requests."""

        batch_size = self._batch_size()
        if (
            isinstance(self.method, IPMConfig)
            and not self.method.tiled
            and self.config.fused_temporal_ipm
        ):
            for batch_start in range(0, len(self.times), batch_size):
                batch_stop = min(len(self.times), batch_start + batch_size)
                real_times = self.times[batch_start:batch_stop]
                real_indices = list(range(batch_start, batch_stop))
                real_far_fields = None
                if self.method.far_field_approx.enabled:
                    from .solvers.far_field import temporal_taylor_far_field_window

                    real_far_fields, _ = temporal_taylor_far_field_window(
                        self.simulation,
                        self.lens_region,
                        self.method.far_field_approx,
                        self.times,
                        real_indices,
                    )
                padded_times = tuple(real_times)
                padded_far_fields = None if real_far_fields is None else list(real_far_fields)
                if (
                    self.config.pad_temporal_batches
                    and self.simulation.runtime.device.type == "cuda"
                    and len(real_times) < batch_size
                ):
                    padded_times += (real_times[-1],) * (batch_size - len(real_times))
                    if padded_far_fields is not None:
                        padded_far_fields.extend(
                            [real_far_fields[-1]] * (batch_size - len(real_times))
                        )
                maps, reductions, splits = self._calculate_ipm_batch_with_backoff(
                    self.method,
                    padded_times,
                    far_fields=padded_far_fields,
                    real_frame_count=len(real_times),
                )
                for local, result in enumerate(maps):
                    index = batch_start + local
                    yield _scheduled_map(
                        result,
                        time_days=self.times[index],
                        metadata={
                            "dynamic_frame_index": index,
                            "dynamic_frame_count": len(self.times),
                            "dynamic_static_map_reuse": False,
                            "dynamic_scout_reuse_approximate": False,
                            "dynamic_oom_chunk_reductions": reductions,
                            "dynamic_temporal_oom_splits": splits,
                            "dynamic_temporal_batch_size": batch_size,
                            "dynamic_temporal_batch_index": batch_start // batch_size,
                            "dynamic_temporal_solver_fused": bool(
                                result.metadata.get("temporal_solver_fused", False)
                            ),
                        },
                    )
            return
        scalar_correction: float | None = None
        for batch_start in range(0, len(self.times), batch_size):
            batch_stop = min(len(self.times), batch_start + batch_size)
            for index in range(batch_start, batch_stop):
                internal = {}
                if (
                    isinstance(self.method, IPMConfig)
                    and self.method.dual_scout_scalar_correction
                    and scalar_correction is not None
                ):
                    internal["_scalar_correction"] = scalar_correction
                result, reductions = self._calculate_with_backoff(
                    self.method,
                    time_days=self.times[index],
                    **internal,
                )
                if (
                    isinstance(self.method, IPMConfig)
                    and self.method.dual_scout_scalar_correction
                    and scalar_correction is None
                ):
                    scalar_correction = float(
                        result.metadata["dual_scout_scalar_correction"]
                    )
                yield _scheduled_map(
                    result,
                    time_days=self.times[index],
                    metadata={
                        "dynamic_frame_index": index,
                        "dynamic_frame_count": len(self.times),
                        "dynamic_static_map_reuse": False,
                        "dynamic_scout_reuse_approximate": False,
                        "dynamic_oom_chunk_reductions": reductions,
                        "dynamic_temporal_batch_size": batch_size,
                        "dynamic_temporal_batch_index": batch_start // batch_size,
                    },
                )

    def _tiled_ipm(self) -> Iterator[MagnificationMap]:
        """Reuse endpoint-union tiled selections over moving-lens intervals."""

        if self.config.fused_temporal_ipm:
            yield from self._tiled_ipm_fused()
            return

        from .solvers.far_field import TaylorFarFieldApproximation
        from .solvers.ipm import _source_scout_cells

        assert isinstance(self.method, IPMConfig)
        refresh = int(self.config.scout_refresh_frames)
        batch_size = self._batch_size()
        scalar_correction: torch.Tensor | None = None
        correction_diagnostics: dict[str, object] = {}
        for anchor_start in range(0, len(self.times), refresh):
            anchor_stop = min(len(self.times), anchor_start + refresh)
            first_index = anchor_start
            last_index = anchor_stop - 1
            far_fields: dict[int, TaylorFarFieldApproximation | None] = {}
            temporal_far_field_metadata: dict[str, object] = {}
            if self.method.far_field_approx.enabled:
                from .solvers.far_field import temporal_taylor_far_field_window

                interval_indices = list(range(anchor_start, anchor_stop))
                interval_far_fields, temporal_far_field_metadata = temporal_taylor_far_field_window(
                    self.simulation,
                    self.lens_region,
                    self.method.far_field_approx,
                    self.times,
                    interval_indices,
                )
                far_fields.update(zip(interval_indices, interval_far_fields, strict=True))

            def far_field_for(
                index: int,
                *,
                _far_fields=far_fields,
            ):
                if index not in _far_fields:
                    _far_fields[index] = (
                        TaylorFarFieldApproximation(
                            self.simulation,
                            self.lens_region,
                            self.method.far_field_approx,
                            time_days=self.times[index],
                        )
                        if self.method.far_field_approx.enabled
                        else None
                    )
                return _far_fields[index]

            first_cells, fine_ny, fine_nx, first_meta = _source_scout_cells(
                self.simulation,
                far_field_for(first_index),
                self.lens_region,
                self.source_grid,
                self.method,
                time_days=self.times[first_index],
            )
            if self.method.dual_scout_scalar_correction and scalar_correction is None:
                from .solvers.ipm import dual_scout_scalar_correction

                scalar_correction, correction_diagnostics = (
                    dual_scout_scalar_correction(
                        self.simulation,
                        far_field_for(0),
                        self.lens_region,
                        self.source_grid,
                        self.method,
                        time_days=self.times[0],
                    )
                )
            endpoint_cells = first_cells
            last_meta = first_meta
            if self.config.endpoint_union and last_index != first_index:
                last_cells, last_ny, last_nx, last_meta = _source_scout_cells(
                    self.simulation,
                    far_field_for(last_index),
                    self.lens_region,
                    self.source_grid,
                    self.method,
                    time_days=self.times[last_index],
                )
                if (last_ny, last_nx) != (fine_ny, fine_nx):
                    raise RuntimeError("dynamic scout fine-grid shape changed between frames")
                endpoint_cells = torch.unique(torch.cat((first_cells, last_cells)))
            selection_metadata: dict[str, object] = {
                "scout_ratio": int(self.method.scout_ratio),
                "selected_fine_cells": int(endpoint_cells.numel()),
                "selected_fine_fraction": float(
                    endpoint_cells.numel() / max(fine_ny * fine_nx, 1)
                ),
                "dynamic_scout_anchor_first": first_index,
                "dynamic_scout_anchor_last": last_index,
                "dynamic_scout_first_cells": int(first_cells.numel()),
                "dynamic_scout_last_cells": int(last_meta["selected_fine_cells"]),
                "dynamic_scout_endpoint_union": bool(
                    self.config.endpoint_union and last_index != first_index
                ),
                **temporal_far_field_metadata,
                **correction_diagnostics,
            }
            for batch_start in range(anchor_start, anchor_stop, batch_size):
                batch_stop = min(anchor_stop, batch_start + batch_size)
                for index in range(batch_start, batch_stop):
                    result, reductions = self._calculate_with_backoff(
                        self.method,
                        time_days=self.times[index],
                        _far_field=far_field_for(index),
                        _selected_cell_indices=endpoint_cells,
                        _selected_cell_shape=(fine_ny, fine_nx),
                        _selection_metadata=selection_metadata,
                        _scalar_correction=scalar_correction,
                    )
                    yield _scheduled_map(
                        result,
                        time_days=self.times[index],
                        metadata={
                            "dynamic_frame_index": index,
                            "dynamic_frame_count": len(self.times),
                            "dynamic_static_map_reuse": False,
                            "dynamic_scout_reuse_approximate": anchor_stop - anchor_start > 1,
                            "dynamic_oom_chunk_reductions": reductions,
                            "dynamic_temporal_batch_size": batch_size,
                            "dynamic_temporal_batch_index": batch_start // batch_size,
                        },
                    )

    def _tiled_ipm_fused(self) -> Iterator[MagnificationMap]:
        """Fuse temporal batches while preserving refresh-interval scouts.

        Each temporal batch uses the union of every endpoint pair belonging to
        the refresh intervals it spans. This is exactly the union of the
        scalar interval selections plus harmless extra cells, allowing one
        shared queue and one temporal raster launch without weakening source
        coverage.
        """

        from .solvers.far_field import temporal_taylor_far_field_window
        from .solvers.ipm import _source_scout_cells, dual_scout_scalar_correction

        assert isinstance(self.method, IPMConfig)
        batch_size = self._batch_size()
        refresh = int(self.config.scout_refresh_frames)
        scalar_correction: torch.Tensor | None = None
        correction_diagnostics: dict[str, object] = {}
        for batch_start in range(0, len(self.times), batch_size):
            runtime = self.simulation.runtime
            runtime.synchronize()
            preparation_started = perf_counter()
            batch_stop = min(len(self.times), batch_start + batch_size)
            real_indices = list(range(batch_start, batch_stop))
            interval_pairs = sorted(
                {
                    (
                        (index // refresh) * refresh,
                        min(
                            len(self.times) - 1,
                            ((index // refresh) + 1) * refresh - 1,
                        ),
                    )
                    for index in real_indices
                }
            )
            if self.config.endpoint_union:
                scout_anchor_indices = [
                    endpoint
                    for pair in interval_pairs
                    for endpoint in pair
                ]
            else:
                # Disabling endpoint reuse means every real frame receives an
                # exact scout. The shared queue remains conservative by using
                # the union of those independently selected cells.
                scout_anchor_indices = list(real_indices)
            scout_anchor_indices = sorted(set(scout_anchor_indices))
            requested_far_field_indices = sorted(
                set(real_indices) | set(scout_anchor_indices)
            )
            if self.method.far_field_approx.enabled:
                requested_far_fields, temporal_far_field_metadata = temporal_taylor_far_field_window(
                    self.simulation,
                    self.lens_region,
                    self.method.far_field_approx,
                    self.times,
                    requested_far_field_indices,
                )
                far_field_by_index = dict(
                    zip(
                        requested_far_field_indices,
                        requested_far_fields,
                        strict=True,
                    )
                )
                far_fields = tuple(
                    far_field_by_index[index] for index in real_indices
                )
            else:
                far_fields = (None,) * len(real_indices)
                far_field_by_index = {
                    index: None for index in requested_far_field_indices
                }
                temporal_far_field_metadata = {
                    "far_field_enabled": False,
                    "far_field_frame_count": 0,
                    "far_field_exact_each_frame": True,
                    "far_field_batched_accumulator": False,
                    "far_field_batched_local_pack": False,
                }
            runtime.synchronize()
            far_fields_prepared = perf_counter()
            from .solvers.ipm import _source_scout_union_temporal

            batched_scout = _source_scout_union_temporal(
                self.simulation,
                tuple(far_field_by_index[index] for index in scout_anchor_indices),
                self.lens_region,
                self.source_grid,
                self.method,
            )
            if batched_scout is not None:
                endpoint_cells, fine_ny, fine_nx, scout_metadata_rows = (
                    batched_scout
                )
            else:
                all_cells = []
                fine_shape = None
                scout_metadata_rows = []
                for anchor in scout_anchor_indices:
                    cells, fine_ny, fine_nx, metadata = _source_scout_cells(
                        self.simulation,
                        far_field_by_index[anchor],
                        self.lens_region,
                        self.source_grid,
                        self.method,
                        time_days=self.times[anchor],
                    )
                    if fine_shape is None:
                        fine_shape = (fine_ny, fine_nx)
                    elif fine_shape != (fine_ny, fine_nx):
                        raise RuntimeError("dynamic scout fine-grid shape changed")
                    all_cells.append(cells)
                    scout_metadata_rows.append(metadata)
                assert fine_shape is not None
                fine_ny, fine_nx = fine_shape
                endpoint_cells = torch.unique(torch.cat(all_cells))
            if (
                self.method.dual_scout_scalar_correction
                and scalar_correction is None
                and batch_start == 0
            ):
                scalar_correction, correction_diagnostics = (
                    dual_scout_scalar_correction(
                        self.simulation,
                        far_field_by_index[0],
                        self.lens_region,
                        self.source_grid,
                        self.method,
                        time_days=self.times[0],
                    )
                )
            runtime.synchronize()
            scouts_prepared = perf_counter()
            if self._far_field_batch_observer is not None:
                # The observer is deliberately outside map component timing.
                # Production caustics use this hook to reuse the exact same
                # temporal far-field approximations without charging label work to LC-only
                # timings or rebuilding the far field.
                self._far_field_batch_observer(
                    real_indices,
                    far_fields,
                    endpoint_cells,
                    (fine_ny, fine_nx),
                )
            far_field_preparation_seconds = far_fields_prepared - preparation_started
            scout_preparation_seconds = scouts_prepared - far_fields_prepared
            selection_metadata: dict[str, object] = {
                "scout_ratio": int(self.method.scout_ratio),
                "selected_fine_cells": int(endpoint_cells.numel()),
                "selected_fine_fraction": float(
                    endpoint_cells.numel() / max(fine_ny * fine_nx, 1)
                ),
                "dynamic_scout_anchor_frames": scout_anchor_indices,
                "dynamic_scout_interval_pairs": interval_pairs,
                "dynamic_scout_anchor_first": interval_pairs[0][0],
                "dynamic_scout_anchor_last": interval_pairs[-1][1],
                "dynamic_scout_endpoint_union": bool(
                    self.config.endpoint_union
                    and any(first != last for first, last in interval_pairs)
                ),
                "dynamic_scout_component_selected_cells": [
                    int(row["selected_fine_cells"])
                    for row in scout_metadata_rows
                ],
                **temporal_far_field_metadata,
                **correction_diagnostics,
            }
            padded_indices = list(real_indices)
            padded_far_fields = list(far_fields)
            if (
                self.config.pad_temporal_batches
                and self.simulation.runtime.device.type == "cuda"
                and len(real_indices) < batch_size
            ):
                padding = batch_size - len(real_indices)
                padded_indices.extend([real_indices[-1]] * padding)
                padded_far_fields.extend([far_fields[-1]] * padding)
            maps, reductions, splits = self._calculate_ipm_batch_with_backoff(
                self.method,
                [self.times[index] for index in padded_indices],
                far_fields=padded_far_fields,
                selected_cell_indices=endpoint_cells,
                selected_cell_shape=(fine_ny, fine_nx),
                selection_metadata=selection_metadata,
                scalar_correction=scalar_correction,
                real_frame_count=len(real_indices),
            )
            for local, result in enumerate(maps):
                index = real_indices[local]
                component_seconds = dict(result.timing.component_seconds)
                component_seconds["dynamic_far_field_preparation"] = (
                    far_field_preparation_seconds / len(real_indices)
                )
                component_seconds["dynamic_source_scout_and_correction"] = (
                    scout_preparation_seconds / len(real_indices)
                )
                result = replace(
                    result,
                    timing=replace(
                        result.timing,
                        steady_seconds=(
                            result.timing.steady_seconds
                            + (far_field_preparation_seconds + scout_preparation_seconds)
                            / len(real_indices)
                        ),
                        component_seconds=component_seconds,
                    ),
                )
                yield _scheduled_map(
                    result,
                    time_days=self.times[index],
                    metadata={
                        "dynamic_frame_index": index,
                        "dynamic_frame_count": len(self.times),
                        "dynamic_static_map_reuse": False,
                        "dynamic_scout_reuse_approximate": bool(
                            self.config.endpoint_union
                            and any(first != last for first, last in interval_pairs)
                        ),
                        "dynamic_oom_chunk_reductions": reductions,
                        "dynamic_temporal_oom_splits": splits,
                        "dynamic_temporal_batch_size": batch_size,
                        "dynamic_temporal_batch_index": batch_start // batch_size,
                        "dynamic_batch_preparation_seconds": (
                            far_field_preparation_seconds + scout_preparation_seconds
                        ),
                        "dynamic_temporal_solver_fused": bool(
                            result.metadata.get("temporal_solver_fused", False)
                        ),
                    },
                )

    def maps(self) -> Iterator[MagnificationMap]:
        """Yield maps in requested order without retaining the full sequence."""

        if not self.times:
            return
        if self.config.reuse_static_maps and not self.simulation.point_masses.has_motion:
            iterator = self._static()
        elif isinstance(self.method, IPMConfig) and self.method.tiled:
            iterator = self._tiled_ipm()
        else:
            iterator = self._ordinary()
        for result in iterator:
            if self.tuning_result is not None:
                result = replace(
                    result,
                    metadata={
                        **result.metadata,
                        **self.tuning_result.metadata(),
                    },
                )
            yield result


def dynamic_maps(
    simulation: MicrolensingSimulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    times_days: Sequence[float],
    *,
    method: IRSConfig | IPMConfig,
    config: DynamicConfig | None = None,
    _far_field_batch_observer=None,
) -> Iterator[MagnificationMap]:
    """Return a streaming iterator over scheduled dynamic magnification maps."""

    return DynamicMapScheduler(
        simulation,
        lens_region,
        source_grid,
        times_days,
        method,
        DynamicConfig() if config is None else config,
        far_field_batch_observer=_far_field_batch_observer,
    ).maps()
