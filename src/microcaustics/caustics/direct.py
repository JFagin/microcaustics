"""Exact point-mass critical curves and source-plane caustics."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import torch

from ..config import FarFieldApproxConfig
from ..geometry import PlaneGrid
from ..results import CausticField, TimingBreakdown
from ..solvers import jacobian_determinant_direct, raytrace_direct
from .marching import marching_squares_zero

if TYPE_CHECKING:
    from ..simulation import MicrolensingSimulation


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
    critical = marching_squares_zero(determinant, x_grid, y_grid)
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
    x_grid, y_grid = lens_grid.mesh(device=runtime.device, dtype=runtime.dtype)
    determinant = approximation.jacobian_determinant(x_grid, y_grid)
    determinant_finished = perf_counter()
    critical = marching_squares_zero(determinant, x_grid, y_grid)
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
