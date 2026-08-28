"""Source-independent uniform-grid inverse ray shooting."""

from __future__ import annotations

import math
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from ..compile import run_tensor_kernel
from ..config import IRSConfig
from ..geometry import PlaneGrid, PlaneRegion
from ..results import MagnificationMap, TimingBreakdown
from .direct import raytrace_direct

if TYPE_CHECKING:
    from ..simulation import MicrolensingSimulation


def _regular_ray_shape(rays: int, region: PlaneRegion) -> tuple[int, int]:
    """Choose a nearly exact ray count while preserving field aspect ratio."""

    fov_y, fov_x = region.field_of_view_uas
    nx = max(1, int(round(math.sqrt(int(rays) * fov_x / fov_y))))
    ny = max(1, int(round(int(rays) / nx)))
    return ny, nx


def _deposit_regular_rays(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    geometry: torch.Tensor,
    source_ny: int,
    source_nx: int,
) -> torch.Tensor:
    """Deposit one ray chunk without Python-side boolean compaction."""

    source_xmin, source_xmax, source_ymin, source_ymax, source_dx, source_dy = (
        geometry.unbind()
    )
    valid = (
        (source_x >= source_xmin)
        & (source_x < source_xmax)
        & (source_y >= source_ymin)
        & (source_y < source_ymax)
    )
    column = torch.floor((source_x - source_xmin) / source_dx).long()
    row = torch.floor((source_y - source_ymin) / source_dy).long()
    column = column.clamp(0, source_nx - 1)
    row = row.clamp(0, source_ny - 1)
    pixels = source_ny * source_nx
    linear = torch.where(valid, row * source_nx + column, pixels)
    weights = valid.to(torch.int64)
    return torch.zeros(
        pixels + 1,
        device=source_x.device,
        dtype=torch.int64,
    ).scatter_add_(0, linear, weights)[:-1]


@torch.no_grad()
def uniform_grid_irs(
    simulation: MicrolensingSimulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    config: IRSConfig,
    *,
    time_days: float = 0.0,
) -> MagnificationMap:
    """Generate an absolute magnification map with regular inverse rays.

    Rays uniformly sample ``lens_region`` and are deposited into the half-open
    pixels of ``source_grid``. Each ray carries its lens-plane area, so the
    returned values are absolute magnifications rather than count maps or
    maps normalized to their own mean. Rays mapped outside the requested
    source field are correctly omitted.

    This first extracted IRS implementation uses the exact point-mass lens
    equation. Far-field acceleration is enabled only after its package version
    passes the frozen paper regression fixtures.
    """

    runtime = simulation.runtime
    runtime.synchronize(detailed=False)
    if runtime.profiling_enabled and runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    started = perf_counter()
    far_field = None
    if config.far_field_approx.enabled:
        from .far_field import TaylorFarFieldApproximation

        far_field = TaylorFarFieldApproximation(
            simulation,
            lens_region,
            config.far_field_approx,
            time_days=time_days,
            star_chunk_size=config.star_chunk_size,
        )
    ray_ny, ray_nx = _regular_ray_shape(config.rays, lens_region)
    actual_rays = ray_ny * ray_nx
    lens_y_size, lens_x_size = lens_region.field_of_view_uas
    lens_xmin, _, lens_ymin, _ = lens_region.bounds_uas
    lens_dx = lens_x_size / ray_nx
    lens_dy = lens_y_size / ray_ny
    source_ny, source_nx = source_grid.shape
    source_dy, source_dx = source_grid.pixel_scale_uas
    source_xmin, source_xmax, source_ymin, source_ymax = source_grid.bounds_uas
    deposition_geometry = torch.tensor(
        [
            source_xmin,
            source_xmax,
            source_ymin,
            source_ymax,
            source_dx,
            source_dy,
        ],
        device=runtime.device,
        dtype=runtime.dtype,
    )
    counts = torch.zeros(
        source_ny * source_nx,
        device=runtime.device,
        dtype=torch.int64,
    )
    ray_chunk = min(int(config.ray_chunk_size), actual_rays)
    raytrace_backends: set[str] = set()
    deposition_compiled = True
    deposition_calls = 0
    for start in range(0, actual_rays, ray_chunk):
        stop = min(actual_rays, start + ray_chunk)
        linear = torch.arange(start, stop, device=runtime.device)
        row = torch.div(linear, ray_nx, rounding_mode="floor")
        column = linear - row * ray_nx
        lens_x = lens_xmin + (column.to(runtime.dtype) + 0.5) * lens_dx
        lens_y = lens_ymin + (row.to(runtime.dtype) + 0.5) * lens_dy
        if far_field is None:
            source_x, source_y, raytrace_diagnostics = raytrace_direct(
                simulation,
                lens_x,
                lens_y,
                time_days=time_days,
                star_chunk_size=config.star_chunk_size,
                ray_chunk_size=stop - start,
            )
            raytrace_backends.add(raytrace_diagnostics.effective_backend)
        else:
            source_x, source_y = far_field.raytrace(lens_x, lens_y)
            raytrace_backends.add(far_field.last_query_backend)
        deposited, used = run_tensor_kernel(
            runtime,
            "IRS deposition",
            _deposit_regular_rays,
            source_x.reshape(-1),
            source_y.reshape(-1),
            deposition_geometry,
            source_ny,
            source_nx,
        )
        counts.add_(deposited)
        deposition_calls += 1
        deposition_compiled = deposition_compiled and used
    lens_area_per_ray = lens_x_size * lens_y_size / actual_rays
    source_pixel_area = source_dx * source_dy
    magnification = counts.to(runtime.dtype).reshape(source_grid.shape)
    magnification.mul_(lens_area_per_ray / source_pixel_area)
    runtime.synchronize(detailed=False)
    elapsed = perf_counter() - started
    far_field_build_seconds = (
        0.0 if far_field is None else far_field.diagnostics.build_seconds
    )
    effective_components = {
        "raytrace": (
            next(iter(raytrace_backends))
            if len(raytrace_backends) == 1
            else "mixed:" + ",".join(sorted(raytrace_backends))
        ),
        "deposition": (
            "torch-compile"
            if deposition_calls and deposition_compiled
            else "torch-eager"
        ),
    }
    if far_field is not None:
        effective_components["far_field_coefficient_build"] = (
            far_field.coefficient_build_backend
        )
    effective_backend = (
        "torch-compile"
        if set(effective_components.values()).issubset(
            {"torch-compile", "precomputed"}
        )
        else (
            "triton"
            if effective_components["raytrace"] == "triton"
            else (
                "torch-compile-partial"
                if "torch-compile" in effective_components.values()
                else "torch-eager"
            )
        )
    )
    timing = TimingBreakdown(
        collected=runtime.profiling_enabled,
        steady_seconds=elapsed,
        component_seconds={
            "far_field_build": far_field_build_seconds,
            "raytrace_and_deposition": max(0.0, elapsed - far_field_build_seconds),
        } if runtime.profiling.value == "detailed" else {},
        peak_device_memory_bytes=(
            int(torch.cuda.max_memory_allocated(runtime.device))
            if runtime.profiling.value == "detailed" and runtime.device.type == "cuda"
            else None
        ),
    )
    return MagnificationMap(
        magnification,
        source_grid,
        time_days=float(time_days),
        method="uniform_grid_irs_direct",
        metadata={
            "requested_rays": int(config.rays),
            "actual_rays": actual_rays,
            "ray_grid_shape": [ray_ny, ray_nx],
            "lens_region": {
                "field_of_view_uas": list(lens_region.field_of_view_uas),
                "center_uas": list(lens_region.center_uas),
            },
            "absolute_magnification": True,
            "requested_backend": runtime.backend.value,
            "effective_backend": effective_backend,
            "backend_components": effective_components,
            "far_field": (
                {"enabled": False}
                if far_field is None
                else {
                    "enabled": True,
                    "cells": list(far_field.diagnostics.cells),
                    "nodes_per_cell": list(far_field.diagnostics.nodes_per_cell),
                    "maximum_local_stars": far_field.diagnostics.maximum_local_stars,
                    "mean_local_stars": far_field.diagnostics.mean_local_stars,
                    "taylor_order": config.far_field_approx.taylor_order,
                    "center_translation_order": config.far_field_approx.center_translation_order,
                }
            ),
            "dtype": str(runtime.dtype),
        },
        timing=timing,
    )
