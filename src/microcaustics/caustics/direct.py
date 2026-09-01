"""Exact point-mass critical curves and source-plane caustics."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import torch

from ..config import FarFieldApproxConfig
from ..geometry import PlaneGrid
from ..results import CausticField, TimingBreakdown
from ..runtime import warn_backend_fallback
from ..solvers import jacobian_determinant_direct, raytrace_direct
from .marching import marching_squares_zero

if TYPE_CHECKING:
    from ..simulation import MicrolensingSimulation


def _extract_zero_segments(
    determinant: torch.Tensor,
    lens_grid: PlaneGrid,
) -> tuple[torch.Tensor, str]:
    """Use compact CUDA marching squares with an exact portable fallback."""

    if determinant.device.type == "cuda" and determinant.dtype == torch.float32:
        try:
            from .triton_caustics import (
                marching_squares_zero_triton,
                triton_caustics_available,
            )

            if triton_caustics_available():
                return (
                    marching_squares_zero_triton(determinant, lens_grid),
                    "triton_compact",
                )
        except Exception as error:
            warn_backend_fallback("Triton complete-field marching squares", error)
    x_grid, y_grid = lens_grid.mesh(
        device=determinant.device,
        dtype=determinant.dtype,
    )
    return marching_squares_zero(determinant, x_grid, y_grid), "torch_portable"


@torch.no_grad()
def direct_caustic_field(
    simulation: MicrolensingSimulation,
    lens_grid: PlaneGrid,
    *,
    time_days: float = 0.0,
    star_chunk_size: int = 4096,
    ray_chunk_size: int | None = None,
) -> CausticField:
    """Calculate exact-reference critical and caustic line segments.

    The Jacobian determinant is sampled at ``lens_grid`` cell centers. Linear
    zero crossings are extracted with marching squares and each endpoint is
    then mapped through the exact point-mass lens equation. This path is
    intended as a portable reference and for modest grids. Accelerated
    analytic/Triton paths will preserve the same result contract.
    """

    runtime = simulation.runtime
    runtime.synchronize(detailed=False)
    started = perf_counter()
    x_grid, y_grid = lens_grid.mesh(device=runtime.device, dtype=runtime.dtype)
    determinant, determinant_diagnostics = jacobian_determinant_direct(
        simulation,
        x_grid,
        y_grid,
        time_days=time_days,
        star_chunk_size=star_chunk_size,
        ray_chunk_size=ray_chunk_size,
    )
    determinant_finished = perf_counter()
    critical, rasterizer = _extract_zero_segments(determinant, lens_grid)
    marching_finished = perf_counter()
    if critical.numel():
        source_x, source_y, _ = raytrace_direct(
            simulation,
            critical[..., 0],
            critical[..., 1],
            time_days=time_days,
            star_chunk_size=star_chunk_size,
            ray_chunk_size=ray_chunk_size,
        )
        caustic = torch.stack((source_x, source_y), dim=-1)
    else:
        caustic = critical.clone()
    runtime.synchronize(detailed=False)
    finished = perf_counter()
    timing = TimingBreakdown(
        collected=runtime.profiling_enabled,
        steady_seconds=finished - started,
        component_seconds={
            "jacobian_determinant": determinant_finished - started,
            "marching_squares": marching_finished - determinant_finished,
            "map_critical_segments": finished - marching_finished,
        } if runtime.profiling.value == "detailed" else {},
        peak_device_memory_bytes=(
            int(torch.cuda.max_memory_allocated(runtime.device))
            if runtime.profiling.value == "detailed" and runtime.device.type == "cuda"
            else None
        ),
    )
    return CausticField(
        critical,
        caustic,
        lens_grid,
        time_days=float(time_days),
        metadata={
            "method": "direct_point_mass",
            "determinant_grid_shape": list(lens_grid.shape),
            "determinant_ray_chunks": determinant_diagnostics.ray_chunks,
            "marching_squares": rasterizer,
            "segment_representation": "independent_linear_segments",
        },
        timing=timing,
    )


@torch.no_grad()
def far_field_caustic_field(
    simulation: MicrolensingSimulation,
    lens_grid: PlaneGrid,
    config: FarFieldApproxConfig,
    *,
    time_days: float = 0.0,
    star_chunk_size: int = 4096,
    ray_chunk_size: int | None = None,
) -> CausticField:
    """Calculate critical curves and caustics with analytic Taylor far-field approximation."""

    from ..solvers import TaylorFarFieldApproximation

    runtime = simulation.runtime
    runtime.synchronize(detailed=False)
    started = perf_counter()
    approximation = TaylorFarFieldApproximation(
        simulation,
        lens_grid.region,
        config,
        time_days=time_days,
        star_chunk_size=star_chunk_size,
    )
    built = perf_counter()
    point_count = lens_grid.shape[0] * lens_grid.shape[1]
    chunk_size = (
        int(ray_chunk_size)
        if ray_chunk_size is not None
        else min(point_count, 1_048_576)
    )
    if chunk_size < 1:
        raise ValueError("ray_chunk_size must be positive")
    determinant = torch.empty(
        point_count,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    ny, nx = lens_grid.shape
    dy, dx = lens_grid.pixel_scale_uas
    xmin, _, ymin, _ = lens_grid.bounds_uas
    regular_grid_triton = False
    regular_grid_query = None
    if runtime.device.type == "cuda" and runtime.dtype == torch.float32:
        try:
            from ..solvers.triton_taylor import (
                evaluate_far_field_p4_regular_grid_jacobian_triton,
                triton_taylor_available,
            )

            if triton_taylor_available() and config.taylor_order == 4:
                regular_grid_query = (
                    evaluate_far_field_p4_regular_grid_jacobian_triton
                )
                regular_grid_triton = True
        except Exception as error:
            warn_backend_fallback("Triton regular-grid determinant", error)
    if regular_grid_query is not None and ray_chunk_size is None:
        # Grid coordinates are loaded from two short axes inside the kernel,
        # so one full launch requires no point-sized coordinate workspace.
        chunk_size = point_count
    for start in range(0, point_count, chunk_size):
        stop = min(point_count, start + chunk_size)
        if regular_grid_query is not None:
            try:
                regular_grid_query(
                    approximation,
                    lens_grid,
                    determinant,
                    start=start,
                    stop=stop,
                )
                continue
            except Exception as error:
                warn_backend_fallback("Triton regular-grid determinant", error)
                regular_grid_query = None
                regular_grid_triton = False
        linear = torch.arange(start, stop, device=runtime.device, dtype=torch.int64)
        row = torch.div(linear, nx, rounding_mode="floor")
        column = linear - row * nx
        x = xmin + (column.to(runtime.dtype) + 0.5) * dx
        y = ymin + (row.to(runtime.dtype) + 0.5) * dy
        determinant[start:stop] = approximation.jacobian_determinant(x, y)
    determinant = determinant.reshape(ny, nx)
    determinant_finished = perf_counter()
    critical, rasterizer = _extract_zero_segments(determinant, lens_grid)
    marching_finished = perf_counter()
    if critical.numel():
        source_x, source_y = approximation.raytrace(
            critical[..., 0],
            critical[..., 1],
        )
        caustic = torch.stack((source_x, source_y), dim=-1)
    else:
        caustic = critical.clone()
    runtime.synchronize(detailed=False)
    finished = perf_counter()
    return CausticField(
        critical,
        caustic,
        lens_grid,
        time_days=float(time_days),
        metadata={
            "method": "local_exact_complex_taylor_far_field",
            "determinant_grid_shape": list(lens_grid.shape),
            "segment_representation": "independent_linear_segments",
            "marching_squares": rasterizer,
            "determinant_ray_chunks": (point_count + chunk_size - 1) // chunk_size,
            "regular_grid_determinant": (
                "triton_indexed" if regular_grid_triton else "coordinate_tensors"
            ),
            "far_field_cells": list(approximation.diagnostics.cells),
            "far_field_nodes_per_cell": list(
                approximation.diagnostics.nodes_per_cell
            ),
        },
        timing=TimingBreakdown(
            collected=runtime.profiling_enabled,
            steady_seconds=finished - started,
            component_seconds={
                "far_field_build": built - started,
                "jacobian_determinant": determinant_finished - built,
                "marching_squares": marching_finished - determinant_finished,
                "map_critical_segments": finished - marching_finished,
            } if runtime.profiling.value == "detailed" else {},
            peak_device_memory_bytes=(
                int(torch.cuda.max_memory_allocated(runtime.device))
                if runtime.profiling.value == "detailed" and runtime.device.type == "cuda"
                else None
            ),
        ),
    )
