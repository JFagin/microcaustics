"""Batch compatible, scientifically independent static map requests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from .config import IPMConfig
from .geometry import PlaneGrid, PlaneRegion

if TYPE_CHECKING:
    from .results import MagnificationMap
    from .simulation import MicrolensingSimulation
    from .system import MicrolensingRealization, MicrolensingSystem


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
        for index, (request, output) in enumerate(
            zip(requests, outputs, strict=True)
        )
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
    method: IPMConfig | None = None,
    batch_size: int | None = None,
    time_days: float = 0.0,
) -> tuple[MagnificationMap, ...]:
    """Generate compatible independent maps from high-level systems.

    Every system retains an independent stellar realization. This helper only
    groups compatible numerical work and applies no temporal or shared-field
    approximation. Automatic OOM recovery is inherited from
    :func:`batched_magnification_maps`.
    """

    from .config import _production_static_ipm_config
    from .system import MicrolensingRealization

    resolved = tuple(
        item if isinstance(item, MicrolensingRealization) else item.realize()
        for item in systems
    )
    if not resolved:
        raise ValueError("at least one microlensing system is required")
    requested_method = (
        _production_static_ipm_config() if method is None else method
    )
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
