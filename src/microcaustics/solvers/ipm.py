"""Portable inverse polygon mapping reference implementation."""

from __future__ import annotations

import math
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as torch_functional

from ..config import Backend, IPMConfig
from ..geometry import PlaneGrid, PlaneRegion
from ..results import MagnificationMap, TimingBreakdown
from ..runtime import warn_backend_fallback, warn_compilation

_PREPARED_TRITON_IPM_SPECIALIZATIONS: set[tuple[object, ...]] = set()


def _warn_new_triton_ipm_specialization(
    runtime,
    *,
    temporal: bool,
    frames: int,
    source_shape: tuple[int, int],
    refinement: int,
    virtual_refinement: int,
) -> None:
    # These are the constexpr/layout controls used by the Triton rasterizers;
    # changing physical star values alone does not create a new warning key.
    key = (
        str(runtime.device),
        str(runtime.dtype),
        runtime.warn_on_compile,
        bool(temporal),
        int(frames),
        tuple(source_shape),
        int(refinement),
        int(virtual_refinement),
    )
    if key in _PREPARED_TRITON_IPM_SPECIALIZATIONS:
        return
    warn_compilation(
        "batched IPM rasterization" if temporal else "IPM rasterization",
        backend="Triton",
        device=runtime.device,
        dtype=runtime.dtype,
        enabled=runtime.warn_on_compile,
    )
    _PREPARED_TRITON_IPM_SPECIALIZATIONS.add(key)


if TYPE_CHECKING:
    from ..simulation import MicrolensingSimulation


def _ipm_cell_shape(
    rays: int,
    region: PlaneRegion,
    scout_ratio: int,
) -> tuple[int, int]:
    """Recover the validated production IPM lattice convention.

    The research implementation interpreted ``rays`` as the approximate
    vertex budget, converted this to cell counts, and then rounded each cell
    count upward only as needed for exact ``k=scout_ratio`` nesting.  Keeping
    that convention is important for reproducibility. Independently rounding
    ``N / n_x`` can turn a nominally square field into a slightly rectangular
    lattice and shift every finite IPM cell boundary.
    """

    requested = max(4, int(rays))
    ratio = max(1, int(scout_ratio))
    fov_y, fov_x = region.field_of_view_uas
    vertex_nx = max(2, int(math.sqrt(requested * fov_x / fov_y)))
    vertex_ny = max(2, int(requested // vertex_nx))
    cell_nx = max(ratio, ratio * math.ceil((vertex_nx - 1) / ratio))
    cell_ny = max(ratio, ratio * math.ceil((vertex_ny - 1) / ratio))
    return cell_ny, cell_nx


def _compact_sparse_node_plan(
    cell_indices: torch.Tensor,
    *,
    cell_ny: int,
    cell_nx: int,
    refinement: int,
    lens_region: PlaneRegion,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unique traced-node coordinates and a cell-local inverse map.

    Adjacent selected IPM cells share true lens-equation nodes.  Building the
    integer refined-lattice IDs first lets the production path trace each
    physical node once without allocating dense mapped-coordinate grids over
    the complete lens plane.  The returned inverse has shape
    ``(selected_cells, r + 1, r + 1)`` and reconstructs each cell's node
    lattice directly from the compact traced queue.
    """

    refinement = int(refinement)
    row = torch.div(cell_indices, int(cell_nx), rounding_mode="floor")
    column = cell_indices - row * int(cell_nx)
    local = torch.arange(
        refinement + 1,
        device=cell_indices.device,
        dtype=torch.long,
    )
    local_y, local_x = torch.meshgrid(local, local, indexing="ij")
    node_row = row[:, None] * refinement + local_y.reshape(1, -1)
    node_column = column[:, None] * refinement + local_x.reshape(1, -1)
    refined_nx = int(cell_nx) * refinement + 1
    node_ids = node_row * refined_nx + node_column
    unique_ids, inverse = torch.unique(
        node_ids.reshape(-1),
        sorted=True,
        return_inverse=True,
    )
    unique_row = torch.div(unique_ids, refined_nx, rounding_mode="floor")
    unique_column = unique_ids - unique_row * refined_nx
    fov_y, fov_x = lens_region.field_of_view_uas
    xmin, _, ymin, _ = lens_region.bounds_uas
    node_x = xmin + unique_column.to(dtype) * (
        float(fov_x) / float(int(cell_nx) * refinement)
    )
    node_y = ymin + unique_row.to(dtype) * (
        float(fov_y) / float(int(cell_ny) * refinement)
    )
    return (
        node_x,
        node_y,
        inverse.reshape(-1, refinement + 1, refinement + 1),
    )


def _dilate_mask(mask: torch.Tensor, cells: int) -> torch.Tensor:
    """Dilate a 2D boolean scout mask by a square cell neighborhood."""

    cells = int(cells)
    if cells <= 0:
        return mask
    pooled = torch_functional.max_pool2d(
        mask[None, None].to(torch.float32),
        kernel_size=2 * cells + 1,
        stride=1,
        padding=cells,
    )
    return pooled[0, 0].to(torch.bool)


@torch.no_grad()
def _source_scout_cells(
    simulation: MicrolensingSimulation,
    far_field,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    config: IPMConfig,
    *,
    time_days: float,
    _fine_shape: tuple[int, int] | None = None,
    _pretraced_corners: tuple[torch.Tensor, torch.Tensor] | None = None,
    _return_corners: bool = False,
):
    """Select fine lens cells through a nested source-box scout.

    ``k=scout_ratio`` traces a grid coarsened by ``k`` along each axis. Each
    retained coarse tile expands to its exact ``k×k`` fine cells, so changing
    ``k`` changes scout cost and conservatism but never the fine-cell geometry.
    """

    runtime = simulation.runtime
    ratio = int(config.scout_ratio)
    if _fine_shape is None:
        fine_ny, fine_nx = _ipm_cell_shape(
            config.rays,
            lens_region,
            ratio,
        )
    else:
        fine_ny, fine_nx = (int(value) for value in _fine_shape)
        if fine_ny < 1 or fine_nx < 1:
            raise ValueError("fine scout shape must be positive")
        if fine_ny % ratio or fine_nx % ratio:
            raise ValueError("fine scout shape must be divisible by scout_ratio")
    coarse_ny = fine_ny // ratio
    coarse_nx = fine_nx // ratio
    lens_xmin, lens_xmax, lens_ymin, lens_ymax = lens_region.bounds_uas
    x_edges = torch.linspace(
        lens_xmin,
        lens_xmax,
        coarse_nx + 1,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    y_edges = torch.linspace(
        lens_ymin,
        lens_ymax,
        coarse_ny + 1,
        device=runtime.device,
        dtype=runtime.dtype,
    )

    def _trace_grid(
        x_axis: torch.Tensor,
        y_axis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(x_axis.numel() * y_axis.numel())
        traced_x = torch.empty(count, device=runtime.device, dtype=runtime.dtype)
        traced_y = torch.empty_like(traced_x)
        nx = int(x_axis.numel())
        for start in range(0, count, int(config.cell_chunk_size)):
            stop = min(count, start + int(config.cell_chunk_size))
            linear = torch.arange(start, stop, device=runtime.device)
            row = torch.div(linear, nx, rounding_mode="floor")
            column = linear - row * nx
            lens_x = x_axis[column]
            lens_y = y_axis[row]
            if far_field is None:
                source_x, source_y, _ = simulation.raytrace_direct(
                    lens_x,
                    lens_y,
                    time_days=time_days,
                )
            else:
                source_x, source_y = far_field.raytrace(lens_x, lens_y)
            traced_x[start:stop] = source_x
            traced_y[start:stop] = source_y
        shape = (int(y_axis.numel()), int(x_axis.numel()))
        return traced_x.reshape(shape), traced_y.reshape(shape)

    if _pretraced_corners is None:
        corner_x, corner_y = _trace_grid(x_edges, y_edges)
        corner_trace_count = int(corner_x.numel())
    else:
        corner_x, corner_y = _pretraced_corners
        expected = (coarse_ny + 1, coarse_nx + 1)
        if tuple(corner_x.shape) != expected or tuple(corner_y.shape) != expected:
            raise ValueError("pretraced scout corners have the wrong grid shape")
        corner_trace_count = 0
    source_xmin, source_xmax, source_ymin, source_ymax = source_grid.bounds_uas
    source_dy, source_dx = source_grid.pixel_scale_uas
    halo_x = float(config.scout_halo_pixels) * source_dx
    halo_y = float(config.scout_halo_pixels) * source_dy
    source_xmin -= halo_x
    source_xmax += halo_x
    source_ymin -= halo_y
    source_ymax += halo_y
    center_trace_count = 0
    mapped_center_x = None
    mapped_center_y = None
    if config.scout_trace_centers:
        center_x = 0.5 * (x_edges[:-1] + x_edges[1:])
        center_y = 0.5 * (y_edges[:-1] + y_edges[1:])
        mapped_center_x, mapped_center_y = _trace_grid(center_x, center_y)
        center_trace_count = int(mapped_center_x.numel())
    use_triton_scout = bool(
        runtime.backend is Backend.TRITON
        and runtime.device.type == "cuda"
        and runtime.dtype == torch.float32
    )
    selected = None
    if use_triton_scout:
        try:
            from .triton_scout import select_source_tiles_triton

            selected = select_source_tiles_triton(
                corner_x,
                corner_y,
                bounds=(source_xmin, source_xmax, source_ymin, source_ymax),
                center_x=mapped_center_x,
                center_y=mapped_center_y,
            )
        except Exception as error:
            if runtime.strict_backend:
                raise
            warn_backend_fallback("Triton IPM source-tile scout", error)
            use_triton_scout = False
    if selected is None:
        mapped_corners_x = torch.stack(
            (
                corner_x[:-1, :-1],
                corner_x[:-1, 1:],
                corner_x[1:, :-1],
                corner_x[1:, 1:],
            ),
            dim=-1,
        )
        mapped_corners_y = torch.stack(
            (
                corner_y[:-1, :-1],
                corner_y[:-1, 1:],
                corner_y[1:, :-1],
                corner_y[1:, 1:],
            ),
            dim=-1,
        )
        selected = (
            (mapped_corners_x.amin(dim=-1) <= source_xmax)
            & (mapped_corners_x.amax(dim=-1) >= source_xmin)
            & (mapped_corners_y.amin(dim=-1) <= source_ymax)
            & (mapped_corners_y.amax(dim=-1) >= source_ymin)
        )
        if mapped_center_x is not None and mapped_center_y is not None:
            selected |= (
                (mapped_center_x >= source_xmin)
                & (mapped_center_x <= source_xmax)
                & (mapped_center_y >= source_ymin)
                & (mapped_center_y <= source_ymax)
            )
    # Keep the diagnostic count on-device until after ``nonzero`` has already
    # resolved the dynamic compacted shape. Reading it here would introduce a
    # separate host synchronization before scout dilation.
    selected_before_dilation_tensor = torch.count_nonzero(selected)
    if use_triton_scout and config.scout_dilation_cells <= 8:
        try:
            from .triton_scout import dilate_source_tiles_triton

            selected = dilate_source_tiles_triton(
                selected,
                config.scout_dilation_cells,
            )
        except Exception as error:
            if runtime.strict_backend:
                raise
            warn_backend_fallback("Triton IPM scout dilation", error)
            use_triton_scout = False
            selected = _dilate_mask(selected, config.scout_dilation_cells)
    else:
        selected = _dilate_mask(selected, config.scout_dilation_cells)
    selected_tiles = selected.nonzero(as_tuple=False)
    selected_before_dilation = int(selected_before_dilation_tensor.item())
    local_y, local_x = torch.meshgrid(
        torch.arange(ratio, device=runtime.device),
        torch.arange(ratio, device=runtime.device),
        indexing="ij",
    )
    if selected_tiles.numel() == 0:
        fine_linear = torch.empty(0, device=runtime.device, dtype=torch.long)
    else:
        fine_rows = selected_tiles[:, 0, None] * ratio + local_y.reshape(1, -1)
        fine_columns = selected_tiles[:, 1, None] * ratio + local_x.reshape(1, -1)
        fine_linear = (fine_rows * fine_nx + fine_columns).reshape(-1)
    metadata: dict[str, int | float | bool] = {
        "scout_ratio": ratio,
        "scout_grid_rows": coarse_ny,
        "scout_grid_columns": coarse_nx,
        "scout_corner_traces": corner_trace_count,
        "scout_center_traces": center_trace_count,
        "scout_trace_centers": bool(config.scout_trace_centers),
        "scout_selector": "triton_fused" if use_triton_scout else "torch",
        "scout_selected_before_dilation": selected_before_dilation,
        "scout_selected_tiles": int(selected_tiles.shape[0]),
        "selected_fine_cells": int(fine_linear.numel()),
        "selected_fine_fraction": float(fine_linear.numel() / (fine_ny * fine_nx)),
    }
    result = (fine_linear, fine_ny, fine_nx, metadata)
    return (*result, (corner_x, corner_y)) if _return_corners else result


@torch.no_grad()
def _source_scout_union_temporal(
    simulation: MicrolensingSimulation,
    far_fields,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    config: IPMConfig,
    *,
    _return_selected_masks: bool = False,
):
    """Select the union of several endpoint scouts in one CUDA query.

    The dynamic production scheduler needs only the union of its endpoint
    selections.  Tracing their shared corner lattice as one temporal queue and
    reducing the selected masks on the device avoids one Python launch and one
    compacted-index transfer per endpoint.  The selected cells are identical
    to the scalar :func:`_source_scout_cells` result.  Unsupported runtimes
    return ``None`` so the portable scheduler keeps its readable scalar path.
    """

    far_fields = tuple(far_fields)
    runtime = simulation.runtime
    if (
        not far_fields
        or any(item is None for item in far_fields)
        or config.scout_trace_centers
        or runtime.backend is not Backend.TRITON
        or runtime.device.type != "cuda"
        or runtime.dtype != torch.float32
    ):
        return None

    from .far_field import BatchedTaylorFarFieldApproximation
    from .triton_scout import (
        dilate_source_tiles_triton,
        select_source_tiles_triton,
    )

    ratio = int(config.scout_ratio)
    fine_ny, fine_nx = _ipm_cell_shape(config.rays, lens_region, ratio)
    coarse_ny = fine_ny // ratio
    coarse_nx = fine_nx // ratio
    lens_xmin, lens_xmax, lens_ymin, lens_ymax = lens_region.bounds_uas
    x_edges = torch.linspace(
        lens_xmin,
        lens_xmax,
        coarse_nx + 1,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    y_edges = torch.linspace(
        lens_ymin,
        lens_ymax,
        coarse_ny + 1,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    vertex_count = int((coarse_ny + 1) * (coarse_nx + 1))
    linear = torch.arange(vertex_count, device=runtime.device)
    row = torch.div(linear, coarse_nx + 1, rounding_mode="floor")
    column = linear - row * (coarse_nx + 1)
    query_x = x_edges[column]
    query_y = y_edges[row]
    batched = BatchedTaylorFarFieldApproximation(far_fields)
    corner_x, corner_y = batched.raytrace(query_x, query_y)
    corner_shape = (len(far_fields), coarse_ny + 1, coarse_nx + 1)
    corner_x = corner_x.reshape(corner_shape)
    corner_y = corner_y.reshape(corner_shape)

    source_xmin, source_xmax, source_ymin, source_ymax = source_grid.bounds_uas
    source_dy, source_dx = source_grid.pixel_scale_uas
    halo_x = float(config.scout_halo_pixels) * source_dx
    halo_y = float(config.scout_halo_pixels) * source_dy
    selected = select_source_tiles_triton(
        corner_x,
        corner_y,
        bounds=(
            source_xmin - halo_x,
            source_xmax + halo_x,
            source_ymin - halo_y,
            source_ymax + halo_y,
        ),
    )
    selected_before = selected.sum(dim=(-2, -1), dtype=torch.int64)
    selected = dilate_source_tiles_triton(
        selected,
        config.scout_dilation_cells,
    )
    selected_after = selected.sum(dim=(-2, -1), dtype=torch.int64)
    local_y, local_x = torch.meshgrid(
        torch.arange(ratio, device=runtime.device),
        torch.arange(ratio, device=runtime.device),
        indexing="ij",
    )
    local_y = local_y.reshape(1, -1)
    local_x = local_x.reshape(1, -1)

    def expand_mask(mask: torch.Tensor) -> torch.Tensor:
        tiles = mask.nonzero(as_tuple=False)
        fine_rows = tiles[:, 0, None] * ratio + local_y
        fine_columns = tiles[:, 1, None] * ratio + local_x
        return (fine_rows * fine_nx + fine_columns).reshape(-1)

    component_fine_linear = tuple(
        expand_mask(frame_selected) for frame_selected in selected
    )
    fine_linear = expand_mask(selected.any(dim=0))
    counts = torch.stack((selected_before, selected_after)).detach().cpu()
    metadata = [
        {
            "scout_ratio": ratio,
            "scout_grid_rows": coarse_ny,
            "scout_grid_columns": coarse_nx,
            "scout_corner_traces": vertex_count,
            "scout_center_traces": 0,
            "scout_trace_centers": False,
            "scout_selector": "triton_temporal_fused",
            "scout_selected_before_dilation": int(counts[0, frame]),
            "scout_selected_tiles": int(counts[1, frame]),
            "selected_fine_cells": int(counts[1, frame]) * ratio * ratio,
            "selected_fine_fraction": float(
                int(counts[1, frame]) * ratio * ratio / (fine_ny * fine_nx)
            ),
        }
        for frame in range(len(far_fields))
    ]
    result = (fine_linear, fine_ny, fine_nx, metadata, component_fine_linear)
    return (*result, selected) if _return_selected_masks else result


@torch.no_grad()
def dual_scout_scalar_correction(
    simulation: MicrolensingSimulation,
    far_field,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    config: IPMConfig,
    *,
    time_days: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return the reference-free scalar ``k=1 minus k=2`` correction.

    One dense ``k=1`` corner trace supplies both scouts. Only the signed
    symmetric-difference cells are rasterized into a single source-plane bin,
    yielding the constant magnification missing from the ordinary ``k=2``
    tiled map. This is a normalization correction, not a spatial residual-map
    reconstruction.
    """

    if not config.tiled or config.scout_ratio != 2:
        raise ValueError("dual-scout correction requires tiled k=2 IPM")
    fine_shape = _ipm_cell_shape(config.rays, lens_region, config.scout_ratio)
    common = replace(
        config,
        dual_scout_scalar_correction=False,
        scout_trace_centers=False,
    )
    dense_config = replace(common, scout_ratio=1)
    (
        k1_cells,
        fine_ny,
        fine_nx,
        k1_metadata,
        dense_corners,
    ) = _source_scout_cells(
        simulation,
        far_field,
        lens_region,
        source_grid,
        dense_config,
        time_days=time_days,
        _fine_shape=fine_shape,
        _return_corners=True,
    )
    dense_corner_x, dense_corner_y = dense_corners
    k2_cells, k2_ny, k2_nx, k2_metadata = _source_scout_cells(
        simulation,
        far_field,
        lens_region,
        source_grid,
        common,
        time_days=time_days,
        _fine_shape=fine_shape,
        _pretraced_corners=(
            dense_corner_x[::2, ::2],
            dense_corner_y[::2, ::2],
        ),
    )
    if (k2_ny, k2_nx) != (fine_ny, fine_nx):
        raise RuntimeError("dual scouts produced inconsistent fine grids")
    count = fine_ny * fine_nx
    k1_mask = torch.zeros(count, device=simulation.runtime.device, dtype=torch.bool)
    k2_mask = torch.zeros_like(k1_mask)
    k1_mask[k1_cells] = True
    k2_mask[k2_cells] = True
    k1_only = torch.nonzero(k1_mask & ~k2_mask, as_tuple=False).reshape(-1)
    k2_only = torch.nonzero(k2_mask & ~k1_mask, as_tuple=False).reshape(-1)
    one_pixel_grid = PlaneGrid(
        (1, 1),
        source_grid.field_of_view_uas,
        source_grid.center_uas,
    )
    correction = torch.zeros(
        (),
        device=simulation.runtime.device,
        dtype=simulation.runtime.dtype,
    )
    for sign, cells in ((1.0, k1_only), (-1.0, k2_only)):
        if cells.numel() == 0:
            continue
        contribution = full_field_ipm(
            simulation,
            lens_region,
            one_pixel_grid,
            common,
            time_days=time_days,
            _far_field=far_field,
            _selected_cell_indices=cells,
            _selected_cell_shape=(fine_ny, fine_nx),
            _selection_metadata={"dual_scout_symmetric_pass": True},
            _scalar_correction=0.0,
        )
        correction += sign * contribution.values[0, 0]
    diagnostics: dict[str, object] = {
        "dual_scout_k1_corner_traces": int(k1_metadata["scout_corner_traces"]),
        "dual_scout_k2_corner_traces": int(k2_metadata["scout_corner_traces"]),
        "dual_scout_k1_selected_cells": int(k1_cells.numel()),
        "dual_scout_k2_selected_cells": int(k2_cells.numel()),
        "dual_scout_k1_only_cells": int(k1_only.numel()),
        "dual_scout_k2_only_cells": int(k2_only.numel()),
        "dual_scout_symmetric_cells": int(k1_only.numel() + k2_only.numel()),
        "dual_scout_scalar_correction": float(correction.detach().cpu()),
    }
    return correction, diagnostics


def interpolated_nodes(
    node_x: torch.Tensor,
    node_y: torch.Tensor,
    *,
    virtual_refinement: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate a traced ``r`` lattice onto ``v`` polygon intervals.

    Inputs have shape ``[cell, r+1, r+1]``. A tensor-product degree-``r``
    barycentric Lagrange polynomial is sampled on ``v+1`` regular nodes. If
    ``v == r`` the traced lattice is returned unchanged. The common ``r=2``
    case is exactly the biquadratic reconstruction used by the paper method.
    Requiring ``v >= r`` prevents the virtual representation from discarding
    genuine lens-equation samples.
    """

    if node_x.shape != node_y.shape or node_x.ndim != 3:
        raise ValueError("node arrays must share shape [cell, r+1, r+1]")
    if node_x.shape[1] != node_x.shape[2] or node_x.shape[1] < 2:
        raise ValueError("node lattices must be square with at least two nodes")
    refinement = int(node_x.shape[1]) - 1
    side = int(virtual_refinement)
    if side < refinement:
        raise ValueError("virtual_refinement must be at least the traced refinement")
    if side == refinement:
        return node_x, node_y
    traced_coordinate = torch.linspace(
        0.0,
        1.0,
        refinement + 1,
        device=node_x.device,
        dtype=node_x.dtype,
    )
    virtual_coordinate = torch.linspace(
        0.0, 1.0, side + 1, device=node_x.device, dtype=node_x.dtype
    )
    differences = traced_coordinate[:, None] - traced_coordinate[None, :]
    differences.fill_diagonal_(1.0)
    barycentric_weights = 1.0 / torch.prod(differences, dim=1)
    delta = virtual_coordinate[:, None] - traced_coordinate[None, :]
    exact = delta == 0
    safe_delta = torch.where(exact, torch.ones_like(delta), delta)
    unnormalized = barycentric_weights[None, :] / safe_delta
    basis = unnormalized / torch.sum(unnormalized, dim=1, keepdim=True)
    if bool(exact.any()):
        exact_rows = exact.any(dim=1)
        basis[exact_rows] = exact[exact_rows].to(basis.dtype)
    basis_t = basis.transpose(0, 1)
    return (
        torch.matmul(torch.matmul(basis, node_x), basis_t),
        torch.matmul(torch.matmul(basis, node_y), basis_t),
    )


def triangles_from_node_lattices(
    node_x: torch.Tensor,
    node_y: torch.Tensor,
) -> torch.Tensor:
    """Split every mapped lattice subcell along one consistent diagonal."""

    if node_x.shape != node_y.shape or node_x.ndim != 3:
        raise ValueError("node arrays must share shape [cell, side+1, side+1]")
    if node_x.shape[1] != node_x.shape[2] or node_x.shape[1] < 2:
        raise ValueError("node lattices must be square with at least two nodes")
    p00 = torch.stack((node_x[:, :-1, :-1], node_y[:, :-1, :-1]), dim=-1)
    p10 = torch.stack((node_x[:, 1:, :-1], node_y[:, 1:, :-1]), dim=-1)
    p11 = torch.stack((node_x[:, 1:, 1:], node_y[:, 1:, 1:]), dim=-1)
    p01 = torch.stack((node_x[:, :-1, 1:], node_y[:, :-1, 1:]), dim=-1)
    first = torch.stack((p00, p10, p11), dim=-2)
    second = torch.stack((p00, p11, p01), dim=-2)
    return torch.stack((first, second), dim=-3).reshape(-1, 3, 2)


def _clip_axis(
    polygon: list[tuple[float, float]],
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    """Clip one polygon against one axis-aligned half plane."""

    if not polygon:
        return []
    result: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = (
        previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
    )
    for current in polygon:
        current_inside = (
            current[axis] >= boundary if keep_greater else current[axis] <= boundary
        )
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            fraction = (
                0.0 if denominator == 0.0 else (boundary - previous[axis]) / denominator
            )
            intersection = (
                previous[0] + fraction * (current[0] - previous[0]),
                previous[1] + fraction * (current[1] - previous[1]),
            )
            result.append(intersection)
        if current_inside:
            result.append(current)
        previous = current
        previous_inside = current_inside
    return result


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Return absolute shoelace area for a small Cartesian polygon."""

    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        total += point[0] * following[1] - point[1] * following[0]
    return 0.5 * abs(total)


def rasterize_triangles_exact_eager(
    triangles_uas: torch.Tensor,
    grid: PlaneGrid,
    *,
    lens_area_per_triangle_uas2: float,
) -> torch.Tensor:
    """Rasterize mapped triangles by exact rectangle-polygon clipping.

    This deliberately simple CPU implementation is a cross-platform
    correctness reference, not the production throughput path. Each mapped
    triangle carries a uniform lens-plane area and deposits its overlap with
    every intersected source pixel as absolute magnification.
    """

    triangles = torch.as_tensor(triangles_uas)
    if triangles.ndim != 3 or tuple(triangles.shape[1:]) != (3, 2):
        raise ValueError("triangles must have shape [triangle, 3, 2]")
    if lens_area_per_triangle_uas2 <= 0:
        raise ValueError("lens_area_per_triangle_uas2 must be positive")
    triangle_array = triangles.detach().to(device="cpu", dtype=torch.float64).numpy()
    ny, nx = grid.shape
    dy, dx = grid.pixel_scale_uas
    xmin, xmax, ymin, ymax = grid.bounds_uas
    pixel_area = dx * dy
    output = np.zeros((ny, nx), dtype=np.float64)
    for triangle in triangle_array:
        source_area = 0.5 * abs(
            (triangle[1, 0] - triangle[0, 0]) * (triangle[2, 1] - triangle[0, 1])
            - (triangle[1, 1] - triangle[0, 1]) * (triangle[2, 0] - triangle[0, 0])
        )
        if not math.isfinite(source_area) or source_area <= 0.0:
            continue
        triangle_xmin = max(xmin, float(np.min(triangle[:, 0])))
        triangle_xmax = min(xmax, float(np.max(triangle[:, 0])))
        triangle_ymin = max(ymin, float(np.min(triangle[:, 1])))
        triangle_ymax = min(ymax, float(np.max(triangle[:, 1])))
        if triangle_xmax <= triangle_xmin or triangle_ymax <= triangle_ymin:
            continue
        first_x = max(0, min(nx - 1, int(math.floor((triangle_xmin - xmin) / dx))))
        last_x = max(0, min(nx - 1, int(math.floor((triangle_xmax - xmin) / dx))))
        first_y = max(0, min(ny - 1, int(math.floor((triangle_ymin - ymin) / dy))))
        last_y = max(0, min(ny - 1, int(math.floor((triangle_ymax - ymin) / dy))))
        density = lens_area_per_triangle_uas2 / source_area / pixel_area
        base_polygon = [(float(x), float(y)) for x, y in triangle]
        for row in range(first_y, last_y + 1):
            pixel_y0 = ymin + row * dy
            pixel_y1 = pixel_y0 + dy
            for column in range(first_x, last_x + 1):
                pixel_x0 = xmin + column * dx
                pixel_x1 = pixel_x0 + dx
                polygon = _clip_axis(
                    base_polygon,
                    axis=0,
                    boundary=pixel_x0,
                    keep_greater=True,
                )
                polygon = _clip_axis(
                    polygon,
                    axis=0,
                    boundary=pixel_x1,
                    keep_greater=False,
                )
                polygon = _clip_axis(
                    polygon,
                    axis=1,
                    boundary=pixel_y0,
                    keep_greater=True,
                )
                polygon = _clip_axis(
                    polygon,
                    axis=1,
                    boundary=pixel_y1,
                    keep_greater=False,
                )
                output[row, column] += density * _polygon_area(polygon)
    return torch.as_tensor(output, device=triangles.device, dtype=triangles.dtype)


@torch.no_grad()
def full_field_ipm(
    simulation: MicrolensingSimulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    config: IPMConfig,
    *,
    time_days: float = 0.0,
    _far_field=None,
    _selected_cell_indices: torch.Tensor | None = None,
    _selected_cell_shape: tuple[int, int] | None = None,
    _selection_metadata: dict[str, object] | None = None,
    _scalar_correction: float | torch.Tensor | None = None,
) -> MagnificationMap:
    """Generate a full-field or nested-scout IPM magnification map.

    Underscored arguments are internal scheduling hooks. They allow dynamic
    calculations to reuse a validated scout selection and an already-built
    far-field approximation without changing the public static-map contract.
    """
    runtime = simulation.runtime
    runtime.synchronize(detailed=False)
    started = perf_counter()
    cell_ny, cell_nx = _ipm_cell_shape(
        config.rays,
        lens_region,
        config.scout_ratio,
    )
    actual_cells = cell_ny * cell_nx
    fov_y, fov_x = lens_region.field_of_view_uas
    xmin, _, ymin, _ = lens_region.bounds_uas
    far_field = _far_field
    if config.far_field_approx.enabled and far_field is None:
        from .far_field import TaylorFarFieldApproximation

        far_field = TaylorFarFieldApproximation(
            simulation,
            lens_region,
            config.far_field_approx,
            time_days=time_days,
        )
    built = perf_counter()
    # A dynamic scheduler may supply a conservative selection shared by several
    # epochs. Otherwise static tiled calls scout now, and full-field calls visit
    # every base cell. All three paths meet at the same mapping/rasterization loop.
    if _selected_cell_indices is not None:
        if _selected_cell_shape is None:
            raise ValueError("scheduled cell indices require their fine-grid shape")
        cell_ny, cell_nx = (int(value) for value in _selected_cell_shape)
        cell_indices = torch.as_tensor(
            _selected_cell_indices,
            device=runtime.device,
            dtype=torch.long,
        ).reshape(-1)
        scout_metadata = dict(_selection_metadata or {})
        scout_metadata.setdefault("scout_ratio", int(config.scout_ratio))
        scout_metadata.setdefault("selected_fine_cells", int(cell_indices.numel()))
        scout_metadata.setdefault(
            "selected_fine_fraction",
            float(cell_indices.numel() / max(cell_ny * cell_nx, 1)),
        )
        scout_metadata["scheduled_selection_reuse"] = True
    elif config.tiled:
        cell_indices, cell_ny, cell_nx, scout_metadata = _source_scout_cells(
            simulation,
            far_field,
            lens_region,
            source_grid,
            config,
            time_days=time_days,
        )
    else:
        cell_indices = torch.arange(
            actual_cells,
            device=runtime.device,
            dtype=torch.long,
        )
        scout_metadata = {
            "scout_ratio": int(config.scout_ratio),
            "selected_fine_cells": actual_cells,
            "selected_fine_fraction": 1.0,
            "scheduled_selection_reuse": False,
        }
    scouted = perf_counter()
    actual_cells = cell_ny * cell_nx
    cell_dx = fov_x / cell_nx
    cell_dy = fov_y / cell_ny
    side = int(config.virtual_refinement)
    triangle_lens_area = cell_dx * cell_dy / (2.0 * side * side)
    refinement = int(config.refinement)
    offsets = torch.linspace(
        0.0,
        1.0,
        refinement + 1,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    use_triton = bool(
        runtime.backend is Backend.TRITON
        and runtime.device.type == "cuda"
        and runtime.dtype == torch.float32
        and side <= 16
    )
    if use_triton:
        _warn_new_triton_ipm_specialization(
            runtime,
            temporal=False,
            frames=1,
            source_shape=source_grid.shape,
            refinement=refinement,
            virtual_refinement=side,
        )
    compact_sparse_nodes = bool(config.compact_sparse_nodes and config.tiled)
    compact_node_x = compact_node_y = compact_inverse = None
    if compact_sparse_nodes:
        # Neighboring active cells share refined vertices. Trace each unique
        # vertex once and use ``compact_inverse`` to reconstruct cell lattices.
        compact_node_x, compact_node_y, compact_inverse = _compact_sparse_node_plan(
            cell_indices,
            cell_ny=cell_ny,
            cell_nx=cell_nx,
            refinement=refinement,
            lens_region=lens_region,
            dtype=runtime.dtype,
        )
    trace_backends: set[str] = set()

    def _calculate_map(accelerated: bool) -> torch.Tensor:
        from .triton_ipm import (
            TritonRasterWorkspace,
            accumulate_cells_triton,
        )

        output = torch.zeros(
            source_grid.shape,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        workspace = (
            TritonRasterWorkspace.create(source_grid.shape, device=runtime.device)
            if accelerated
            else None
        )
        source_dy, source_dx = source_grid.pixel_scale_uas
        source_xmin, _, source_ymin, _ = source_grid.bounds_uas
        selected_count = int(cell_indices.numel())
        compact_source_x = compact_source_y = None
        if compact_sparse_nodes:
            if far_field is None:
                compact_source_x, compact_source_y, trace_diagnostics = (
                    simulation.raytrace_direct(
                        compact_node_x,
                        compact_node_y,
                        time_days=time_days,
                    )
                )
                trace_backends.add(trace_diagnostics.effective_backend)
            else:
                compact_source_x, compact_source_y = far_field.raytrace(
                    compact_node_x,
                    compact_node_y,
                )
                trace_backends.add(far_field.last_query_backend)
        for start in range(0, selected_count, int(config.cell_chunk_size)):
            # Each chunk follows the same IPM stages: trace base nodes, refine
            # their mapped polygons, then deposit conserved lens-plane area.
            stop = min(selected_count, start + int(config.cell_chunk_size))
            if compact_sparse_nodes:
                inverse = compact_inverse[start:stop]
                source_x = compact_source_x[inverse]
                source_y = compact_source_y[inverse]
            else:
                linear = cell_indices[start:stop]
                row = torch.div(linear, cell_nx, rounding_mode="floor")
                column = linear - row * cell_nx
                lens_x = (
                    xmin
                    + (column[:, None, None].to(runtime.dtype) + offsets[None, None, :])
                    * cell_dx
                )
                lens_y = (
                    ymin
                    + (row[:, None, None].to(runtime.dtype) + offsets[None, :, None])
                    * cell_dy
                )
                lens_x, lens_y = torch.broadcast_tensors(lens_x, lens_y)
                if far_field is None:
                    source_x, source_y, trace_diagnostics = simulation.raytrace_direct(
                        lens_x,
                        lens_y,
                        time_days=time_days,
                    )
                    trace_backends.add(trace_diagnostics.effective_backend)
                else:
                    source_x, source_y = far_field.raytrace(lens_x, lens_y)
                    trace_backends.add(far_field.last_query_backend)
            if accelerated and refinement == 2 and side == 4:
                from .triton_ipm import materialize_biquadratic_v4_triton

                virtual_x, virtual_y = materialize_biquadratic_v4_triton(
                    source_x, source_y
                )
            else:
                virtual_x, virtual_y = interpolated_nodes(
                    source_x,
                    source_y,
                    virtual_refinement=side,
                )
            if workspace is not None:
                accumulate_cells_triton(
                    workspace,
                    virtual_x,
                    virtual_y,
                    xmin=source_xmin,
                    ymin=source_ymin,
                    pixel_size_x=source_dx,
                    pixel_size_y=source_dy,
                    lens_area_per_triangle_uas2=triangle_lens_area,
                )
            else:
                triangles = triangles_from_node_lattices(virtual_x, virtual_y)
                output += rasterize_triangles_exact_eager(
                    triangles,
                    source_grid,
                    lens_area_per_triangle_uas2=triangle_lens_area,
                )
        return workspace.result() if workspace is not None else output

    try:
        output = _calculate_map(use_triton)
    except Exception as error:
        if not use_triton or runtime.strict_backend:
            raise
        warn_backend_fallback("Triton IPM rasterization", error)
        use_triton = False
        output = _calculate_map(False)
    runtime.synchronize()
    mapped = perf_counter()
    correction_diagnostics: dict[str, object] = {}
    scalar_correction = _scalar_correction
    if config.dual_scout_scalar_correction and scalar_correction is None:
        # The k=2 production scout omits a nearly uniform low-level contribution;
        # measure it once against k=1 and add only that scalar normalization.
        scalar_correction, correction_diagnostics = dual_scout_scalar_correction(
            simulation,
            far_field,
            lens_region,
            source_grid,
            config,
            time_days=time_days,
        )
    if scalar_correction is not None:
        correction_tensor = torch.as_tensor(
            scalar_correction,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        output = output + correction_tensor
        correction_diagnostics.setdefault(
            "dual_scout_scalar_correction",
            float(correction_tensor.detach().cpu()),
        )
    correction_diagnostics.setdefault(
        "dual_scout_scalar_correction_active",
        bool(config.dual_scout_scalar_correction or scalar_correction is not None),
    )
    runtime.synchronize(detailed=False)
    finished = perf_counter()
    elapsed = finished - started
    trace_backend = (
        "not-evaluated"
        if not trace_backends
        else (
            next(iter(trace_backends))
            if len(trace_backends) == 1
            else "mixed:" + ",".join(sorted(trace_backends))
        )
    )
    effective_backend = (
        "triton"
        if use_triton
        else (
            "torch-compile-partial"
            if trace_backend == "torch-compile"
            else "torch-eager"
        )
    )
    return MagnificationMap(
        output,
        source_grid,
        time_days=float(time_days),
        method=(
            "tiled_interpolated_ipm" if config.tiled else "full_field_interpolated_ipm"
        ),
        metadata={
            "requested_base_cells": int(config.rays),
            "actual_base_cells": actual_cells,
            "base_cell_grid_shape": [cell_ny, cell_nx],
            "refinement": refinement,
            "virtual_refinement": side,
            "compact_sparse_nodes": compact_sparse_nodes,
            "compact_unique_nodes": (
                int(compact_node_x.numel()) if compact_sparse_nodes else None
            ),
            "tiled": bool(config.tiled),
            "absolute_magnification": True,
            "requested_backend": runtime.backend.value,
            "effective_backend": effective_backend,
            "backend_components": {
                "raytrace": trace_backend,
                "far_field_coefficient_build": (
                    "disabled"
                    if far_field is None
                    else far_field.coefficient_build_backend
                ),
                "interpolation": (
                    "triton-biquadratic-v4"
                    if use_triton and refinement == 2 and side == 4
                    else "torch-eager"
                ),
                "rasterization": ("triton" if use_triton else "python-exact-reference"),
            },
            "rasterizer": (
                "triton_direct_cell_scanline"
                if use_triton
                else "exact_sutherland_hodgman_reference"
            ),
            **scout_metadata,
            **correction_diagnostics,
        },
        timing=TimingBreakdown(
            collected=runtime.profiling_enabled,
            steady_seconds=elapsed,
            component_seconds={
                "far_field_build": built - started,
                "source_scout": scouted - built,
                "trace_reconstruct_raster": mapped - scouted,
                "dual_scout_scalar_correction": finished - mapped,
            }
            if runtime.profiling.value == "detailed"
            else {},
        ),
    )


@torch.no_grad()
def temporal_batch_ipm(
    simulation: MicrolensingSimulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    config: IPMConfig,
    times_days,
    *,
    far_fields=None,
    selected_cell_indices: torch.Tensor | None = None,
    selected_cell_indices_by_frame=None,
    selected_cell_shape: tuple[int, int] | None = None,
    selection_metadata: dict[str, object] | None = None,
    scalar_correction: float | torch.Tensor | None = None,
    real_frame_count: int | None = None,
) -> tuple[MagnificationMap, ...]:
    """Generate a temporal IPM batch with shared or ragged cell queues.

    The CUDA production path stacks every frame's mapped node lattices and
    rasterizes them in one direct-cell Triton launch per cell chunk. Lens
    coordinates, selected-cell indices, interpolation work, and raster
    allocation are shared. Portable backends preserve the numerical contract
    with the exact eager rasterizer.

    A tiled calculation normally supplies one conservative cell selection valid
    for the complete temporal batch. Independent compatible systems may instead
    supply ``selected_cell_indices_by_frame``; those queues are concatenated
    with explicit output-frame tags, so no system traces another system's scout
    cells. ``real_frame_count`` permits a padded final batch while dropping the
    synthetic frames from the returned sequence.
    """

    runtime = simulation.runtime
    times = tuple(float(value) for value in times_days)
    if not times:
        return ()
    total_frames = len(times)
    real_frames = total_frames if real_frame_count is None else int(real_frame_count)
    if not 1 <= real_frames <= total_frames:
        raise ValueError("real_frame_count must lie within the temporal batch")
    runtime.synchronize(detailed=False)
    started = perf_counter()
    temporal_far_field_metadata: dict[str, object] = {}
    if far_fields is None:
        if config.far_field_approx.enabled:
            from .far_field import temporal_taylor_far_fields

            far_fields, temporal_far_field_metadata = temporal_taylor_far_fields(
                simulation,
                lens_region,
                config.far_field_approx,
                times,
            )
        else:
            far_fields = (None,) * total_frames
    else:
        far_fields = tuple(far_fields)
        if len(far_fields) != total_frames:
            raise ValueError("far_fields must match the padded temporal batch")
    batched_far_field = None
    if (
        total_frames > 1 or selected_cell_indices_by_frame is not None
    ) and all(item is not None for item in far_fields):
        from .far_field import BatchedTaylorFarFieldApproximation

        batched_far_field = BatchedTaylorFarFieldApproximation(far_fields)
    runtime.synchronize()
    far_field_prepared = perf_counter()

    cell_ny, cell_nx = _ipm_cell_shape(
        config.rays,
        lens_region,
        config.scout_ratio,
    )
    if selected_cell_indices is not None and selected_cell_indices_by_frame is not None:
        raise ValueError("supply shared or per-frame selected cells, not both")
    ragged_cells = selected_cell_indices_by_frame is not None
    cell_frame_index = None
    if ragged_cells:
        if selected_cell_shape is None:
            raise ValueError("selected_cell_shape is required with selected cells")
        cell_ny, cell_nx = (int(value) for value in selected_cell_shape)
        rows = tuple(
            torch.as_tensor(value, device=runtime.device, dtype=torch.long).reshape(-1)
            for value in selected_cell_indices_by_frame
        )
        if len(rows) != total_frames:
            raise ValueError("per-frame selected cells must match the temporal batch")
        lengths = tuple(int(value.numel()) for value in rows)
        cell_indices = torch.cat(rows) if rows else torch.empty(
            0, device=runtime.device, dtype=torch.long
        )
        cell_frame_index = torch.repeat_interleave(
            torch.arange(total_frames, device=runtime.device, dtype=torch.int32),
            torch.tensor(lengths, device=runtime.device, dtype=torch.int64),
            output_size=sum(lengths),
        )
        scout_metadata = dict(selection_metadata or {})
        scout_metadata.update(
            {
                "selected_fine_cells": int(cell_indices.numel()),
                "selected_fine_cells_per_frame": list(lengths),
                "selected_fine_fraction": float(
                    cell_indices.numel() / max(total_frames * cell_ny * cell_nx, 1)
                ),
                "scheduled_selection_reuse": True,
                "ragged_cell_queue": True,
            }
        )
    elif selected_cell_indices is None:
        if config.tiled:
            raise ValueError(
                "temporal tiled IPM requires a conservative shared selection"
            )
        cell_indices = torch.arange(
            cell_ny * cell_nx,
            device=runtime.device,
            dtype=torch.long,
        )
        scout_metadata: dict[str, object] = {
            "selected_fine_cells": int(cell_ny * cell_nx),
            "selected_fine_fraction": 1.0,
            "scheduled_selection_reuse": False,
        }
    else:
        if selected_cell_shape is None:
            raise ValueError("selected_cell_shape is required with selected cells")
        cell_ny, cell_nx = (int(value) for value in selected_cell_shape)
        cell_indices = torch.as_tensor(
            selected_cell_indices,
            device=runtime.device,
            dtype=torch.long,
        ).reshape(-1)
        scout_metadata = dict(selection_metadata or {})
        scout_metadata.setdefault("selected_fine_cells", int(cell_indices.numel()))
        scout_metadata.setdefault(
            "selected_fine_fraction",
            float(cell_indices.numel() / max(cell_ny * cell_nx, 1)),
        )
        scout_metadata["scheduled_selection_reuse"] = True

    use_triton = bool(
        runtime.backend is Backend.TRITON
        and runtime.device.type == "cuda"
        and runtime.dtype == torch.float32
        and int(config.virtual_refinement) <= 16
    )
    if use_triton:
        _warn_new_triton_ipm_specialization(
            runtime,
            temporal=True,
            frames=total_frames,
            source_shape=source_grid.shape,
            refinement=int(config.refinement),
            virtual_refinement=int(config.virtual_refinement),
        )
    trace_backends: set[str] = set()
    fov_y, fov_x = lens_region.field_of_view_uas
    lens_xmin, _, lens_ymin, _ = lens_region.bounds_uas
    cell_dx = fov_x / cell_nx
    cell_dy = fov_y / cell_ny
    refinement = int(config.refinement)
    virtual = int(config.virtual_refinement)
    offsets = torch.linspace(
        0.0,
        1.0,
        refinement + 1,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    triangle_lens_area = cell_dx * cell_dy / (2.0 * virtual * virtual)
    source_dy, source_dx = source_grid.pixel_scale_uas
    source_xmin, _, source_ymin, _ = source_grid.bounds_uas
    compact_sparse_nodes = bool(config.compact_sparse_nodes and config.tiled)
    compact_node_x = compact_node_y = compact_inverse = None
    compact_node_frame_index = None
    if compact_sparse_nodes:
        if ragged_cells:
            # Deduplicate independently inside each physical frame, then pack
            # those variable-length node queues with explicit ownership tags.
            # Nodes are never shared across frames because their lens states
            # differ even when their integer lattice coordinates coincide. The
            # scheduler reuses one selection tensor throughout each refresh
            # interval, so cache its integer topology instead of repeating the
            # relatively expensive unique/sort operation for every frame.
            plan_cache = {}
            plans = []
            for row in rows:
                key = (int(row.data_ptr()), int(row.numel()))
                plan = plan_cache.get(key)
                if plan is None:
                    plan = _compact_sparse_node_plan(
                        row,
                        cell_ny=cell_ny,
                        cell_nx=cell_nx,
                        refinement=refinement,
                        lens_region=lens_region,
                        dtype=runtime.dtype,
                    )
                    plan_cache[key] = plan
                plans.append(plan)
            plans = tuple(plans)
            node_lengths = tuple(int(plan[0].numel()) for plan in plans)
            compact_node_x = torch.cat(tuple(plan[0] for plan in plans))
            compact_node_y = torch.cat(tuple(plan[1] for plan in plans))
            node_offsets = []
            offset = 0
            for plan in plans:
                node_offsets.append(plan[2] + offset)
                offset += int(plan[0].numel())
            compact_inverse = torch.cat(tuple(node_offsets))
            compact_node_frame_index = torch.repeat_interleave(
                torch.arange(
                    total_frames,
                    device=runtime.device,
                    dtype=torch.int32,
                ),
                torch.tensor(
                    node_lengths,
                    device=runtime.device,
                    dtype=torch.int64,
                ),
                output_size=sum(node_lengths),
            )
        else:
            # The unique-node plan is spatial and shared across all temporal
            # frames; only mapped coordinates vary with the moving stars.
            compact_node_x, compact_node_y, compact_inverse = (
                _compact_sparse_node_plan(
                    cell_indices,
                    cell_ny=cell_ny,
                    cell_nx=cell_nx,
                    refinement=refinement,
                    lens_region=lens_region,
                    dtype=runtime.dtype,
                )
            )

    def calculate(accelerated: bool) -> torch.Tensor:
        from .triton_ipm import TritonRasterWorkspace, accumulate_cells_triton

        output = torch.zeros(
            (total_frames, *source_grid.shape),
            device=runtime.device,
            dtype=runtime.dtype,
        )
        workspace = (
            TritonRasterWorkspace.create(
                source_grid.shape,
                device=runtime.device,
                frames=total_frames,
            )
            if accelerated
            else None
        )
        selected_count = int(cell_indices.numel())
        compact_traced_x = compact_traced_y = None
        if compact_sparse_nodes:
            if batched_far_field is not None and ragged_cells:
                compact_traced_x, compact_traced_y = (
                    batched_far_field.raytrace_indexed_flat(
                        compact_node_x,
                        compact_node_y,
                        compact_node_frame_index,
                    )
                )
                trace_backends.update(item.last_query_backend for item in far_fields)
            elif batched_far_field is not None:
                compact_traced_x, compact_traced_y = batched_far_field.raytrace(
                    compact_node_x, compact_node_y
                )
                trace_backends.update(item.last_query_backend for item in far_fields)
            else:
                mapped_x = []
                mapped_y = []
                for time_days, far_field in zip(times, far_fields, strict=True):
                    if far_field is None:
                        source_x, source_y, trace_diagnostics = (
                            simulation.raytrace_direct(
                                compact_node_x,
                                compact_node_y,
                                time_days=time_days,
                            )
                        )
                        trace_backends.add(trace_diagnostics.effective_backend)
                    else:
                        source_x, source_y = far_field.raytrace(
                            compact_node_x,
                            compact_node_y,
                        )
                        trace_backends.add(far_field.last_query_backend)
                    mapped_x.append(source_x)
                    mapped_y.append(source_y)
                compact_traced_x = torch.stack(mapped_x)
                compact_traced_y = torch.stack(mapped_y)
        for start in range(0, selected_count, int(config.cell_chunk_size)):
            stop = min(selected_count, start + int(config.cell_chunk_size))
            if compact_sparse_nodes:
                inverse = compact_inverse[start:stop]
                chunk_frame_index = (
                    None if cell_frame_index is None else cell_frame_index[start:stop]
                )
                if ragged_cells:
                    traced_x = compact_traced_x[inverse]
                    traced_y = compact_traced_y[inverse]
                else:
                    traced_x = compact_traced_x[:, inverse]
                    traced_y = compact_traced_y[:, inverse]
            else:
                linear = cell_indices[start:stop]
                chunk_frame_index = (
                    None if cell_frame_index is None else cell_frame_index[start:stop]
                )
                row = torch.div(linear, cell_nx, rounding_mode="floor")
                column = linear - row * cell_nx
                lens_x = (
                    lens_xmin
                    + (column[:, None, None].to(runtime.dtype) + offsets[None, None, :])
                    * cell_dx
                )
                lens_y = (
                    lens_ymin
                    + (row[:, None, None].to(runtime.dtype) + offsets[None, :, None])
                    * cell_dy
                )
                lens_x, lens_y = torch.broadcast_tensors(lens_x, lens_y)
                if batched_far_field is not None and chunk_frame_index is not None:
                    point_frames = torch.repeat_interleave(
                        chunk_frame_index,
                        (refinement + 1) * (refinement + 1),
                    )
                    traced_x, traced_y = batched_far_field.raytrace_indexed_flat(
                        lens_x.reshape(-1), lens_y.reshape(-1), point_frames
                    )
                    traced_x = traced_x.reshape_as(lens_x)
                    traced_y = traced_y.reshape_as(lens_y)
                    trace_backends.update(
                        item.last_query_backend for item in far_fields
                    )
                elif batched_far_field is not None:
                    traced_x, traced_y = batched_far_field.raytrace(lens_x, lens_y)
                    trace_backends.update(
                        item.last_query_backend for item in far_fields
                    )
                else:
                    mapped_x = []
                    mapped_y = []
                    for time_days, far_field in zip(times, far_fields, strict=True):
                        if far_field is None:
                            source_x, source_y, trace_diagnostics = (
                                simulation.raytrace_direct(
                                    lens_x,
                                    lens_y,
                                    time_days=time_days,
                                )
                            )
                            trace_backends.add(trace_diagnostics.effective_backend)
                        else:
                            source_x, source_y = far_field.raytrace(
                                lens_x,
                                lens_y,
                            )
                            trace_backends.add(far_field.last_query_backend)
                        mapped_x.append(source_x)
                        mapped_y.append(source_y)
                    traced_x = torch.stack(mapped_x)
                    traced_y = torch.stack(mapped_y)
            chunk_cells = int(
                traced_x.shape[0] if ragged_cells else traced_x.shape[1]
            )
            raw_x = traced_x.reshape(-1, refinement + 1, refinement + 1)
            raw_y = traced_y.reshape(-1, refinement + 1, refinement + 1)
            if accelerated and refinement == 2 and virtual == 4:
                from .triton_ipm import materialize_biquadratic_v4_triton

                virtual_x, virtual_y = materialize_biquadratic_v4_triton(raw_x, raw_y)
            else:
                virtual_x, virtual_y = interpolated_nodes(
                    raw_x,
                    raw_y,
                    virtual_refinement=virtual,
                )
            virtual_x = virtual_x.reshape(
                *((chunk_cells,) if ragged_cells else (total_frames, chunk_cells)),
                virtual + 1,
                virtual + 1,
            )
            virtual_y = virtual_y.reshape_as(virtual_x)
            if workspace is not None:
                accumulate_cells_triton(
                    workspace,
                    virtual_x,
                    virtual_y,
                    xmin=source_xmin,
                    ymin=source_ymin,
                    pixel_size_x=source_dx,
                    pixel_size_y=source_dy,
                    lens_area_per_triangle_uas2=triangle_lens_area,
                    cell_frame_index=(
                        chunk_frame_index if ragged_cells else None
                    ),
                )
            else:
                for frame in range(total_frames):
                    frame_x = (
                        virtual_x[cell_frame_index[start:stop] == frame]
                        if ragged_cells
                        else virtual_x[frame]
                    )
                    frame_y = (
                        virtual_y[cell_frame_index[start:stop] == frame]
                        if ragged_cells
                        else virtual_y[frame]
                    )
                    output[frame] += rasterize_triangles_exact_eager(
                        triangles_from_node_lattices(
                            frame_x,
                            frame_y,
                        ),
                        source_grid,
                        lens_area_per_triangle_uas2=triangle_lens_area,
                    )
        if workspace is None:
            return output
        result = workspace.result()
        # The reusable raster workspace intentionally stores a single frame
        # as a 2D tensor for the ordinary static-map path. This temporal API
        # must retain its leading frame axis even when the requested batch is
        # one, otherwise ``output[0]`` is interpreted as a map row.
        return result.unsqueeze(0) if result.ndim == 2 else result

    try:
        output = calculate(use_triton)
    except Exception as error:
        if not use_triton or runtime.strict_backend:
            raise
        warn_backend_fallback("batched Triton IPM rasterization", error)
        use_triton = False
        output = calculate(False)
    if scalar_correction is not None:
        output += torch.as_tensor(
            scalar_correction,
            device=runtime.device,
            dtype=runtime.dtype,
        )
    runtime.synchronize(detailed=False)
    elapsed = perf_counter() - started
    per_frame_seconds = elapsed / real_frames
    far_field_seconds = (far_field_prepared - started) / real_frames
    solve_seconds = max(0.0, elapsed - (far_field_prepared - started)) / real_frames
    padding = total_frames - real_frames
    trace_backend = (
        "not-evaluated"
        if not trace_backends
        else (
            next(iter(trace_backends))
            if len(trace_backends) == 1
            else "mixed:" + ",".join(sorted(trace_backends))
        )
    )
    effective_backend = (
        "triton"
        if use_triton
        else (
            "torch-compile-partial"
            if trace_backend == "torch-compile"
            else "torch-eager"
        )
    )
    coefficient_backends = {
        item.coefficient_build_backend for item in far_fields if item is not None
    }
    far_field_build_backend = (
        "disabled"
        if not coefficient_backends
        else (
            next(iter(coefficient_backends))
            if len(coefficient_backends) == 1
            else "mixed:" + ",".join(sorted(coefficient_backends))
        )
    )
    maps = []
    for frame in range(real_frames):
        maps.append(
            MagnificationMap(
                output[frame],
                source_grid,
                time_days=times[frame],
                method=(
                    "tiled_interpolated_ipm"
                    if config.tiled
                    else "full_field_interpolated_ipm"
                ),
                metadata={
                    "requested_base_cells": int(config.rays),
                    "actual_base_cells": int(cell_ny * cell_nx),
                    "base_cell_grid_shape": [cell_ny, cell_nx],
                    "refinement": refinement,
                    "virtual_refinement": virtual,
                    "compact_sparse_nodes": compact_sparse_nodes,
                    "compact_unique_nodes": (
                        int(compact_node_x.numel()) if compact_sparse_nodes else None
                    ),
                    "tiled": bool(config.tiled),
                    "absolute_magnification": True,
                    "requested_backend": runtime.backend.value,
                    "effective_backend": effective_backend,
                    "backend_components": {
                        "raytrace": trace_backend,
                        "far_field_coefficient_build": far_field_build_backend,
                        "interpolation": (
                            "triton-biquadratic-v4"
                            if use_triton and refinement == 2 and virtual == 4
                            else "torch-eager"
                        ),
                        "rasterization": (
                            "triton" if use_triton else "python-exact-reference"
                        ),
                    },
                    "rasterizer": (
                        "triton_direct_cell_temporal_batch"
                        if use_triton
                        else "exact_sutherland_hodgman_temporal_reference"
                    ),
                    "temporal_solver_fused": bool(use_triton),
                    "temporal_far_field_query_fused": bool(
                        batched_far_field is not None
                        and batched_far_field._use_triton()
                    ),
                    "temporal_batch_frames": total_frames,
                    "temporal_batch_real_frames": real_frames,
                    "temporal_batch_padded_frames": padding,
                    "temporal_batch_total_seconds": elapsed,
                    "dual_scout_scalar_correction_active": (
                        scalar_correction is not None
                    ),
                    **temporal_far_field_metadata,
                    **scout_metadata,
                },
                timing=TimingBreakdown(
                    collected=runtime.profiling_enabled,
                    steady_seconds=per_frame_seconds,
                    component_seconds={
                        "temporal_far_field_preparation": far_field_seconds,
                        "shared_trace_interpolate_raster": solve_seconds,
                    }
                    if runtime.profiling.value == "detailed"
                    else {},
                ),
            )
        )
    return tuple(maps)
