"""Batched production critical curves, caustics, and source labels."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional

from ..config import CausticConfig
from ..geometry import PlaneGrid, PlaneRegion
from ..results import (
    CausticField,
    LabeledCausticFrame,
    LabeledLightCurve,
    LabeledMapFrame,
    TimingBreakdown,
)
from ..runtime import warn_backend_fallback
from ..solvers import (
    BatchedTaylorFarFieldApproximation,
    jacobian_determinant_direct,
    temporal_taylor_far_field_window,
    temporal_taylor_far_fields,
)
from .anchor_gauge import label_caustic_fields
from .marching import marching_squares_zero

if TYPE_CHECKING:
    from ..simulation import MicrolensingSimulation


def _marching_segments(
    field: torch.Tensor, grid: PlaneGrid
) -> tuple[torch.Tensor, str]:
    """Use compact Triton marching when available, else the portable reference."""

    if field.device.type == "cuda" and field.dtype == torch.float32:
        try:
            from .triton_caustics import (
                marching_squares_zero_triton,
                triton_caustics_available,
            )

            if triton_caustics_available():
                return marching_squares_zero_triton(field, grid), "triton_compact"
        except Exception as error:
            warn_backend_fallback("Triton marching squares", error)
    x, y = grid.mesh(device=field.device, dtype=field.dtype)
    return marching_squares_zero(field, x, y), "torch_portable"


def _boundary_component_mask(
    critical_segments: torch.Tensor,
    grid: PlaneGrid,
) -> torch.Tensor:
    """Mark complete critical-curve components clipped by the detA grid.

    Production labels must never interpret an open contour as a closed
    caustic. Endpoints shared by adjacent marching cells are canonicalized
    with a tolerance far below one grid pixel, then a small union-find marks
    every segment connected to the sampled lens-plane boundary. The geometry
    remains available for plotting. Only topological label queries ignore it.
    """

    count = int(critical_segments.shape[0])
    result = torch.zeros(count, device=critical_segments.device, dtype=torch.bool)
    if count == 0:
        return result
    dy, dx = grid.pixel_scale_uas
    xmin, xmax, ymin, ymax = grid.bounds_uas
    sampled_xmin = xmin + 0.5 * dx
    sampled_xmax = xmax - 0.5 * dx
    sampled_ymin = ymin + 0.5 * dy
    sampled_ymax = ymax - 0.5 * dy
    tolerance = max(min(dx, dy) * 1.0e-5, 1.0e-12)
    points = (
        critical_segments.detach().to(device="cpu", dtype=torch.float64).reshape(-1, 2)
    )
    on_boundary = (
        ((points[:, 0] - sampled_xmin).abs() <= tolerance)
        | ((points[:, 0] - sampled_xmax).abs() <= tolerance)
        | ((points[:, 1] - sampled_ymin).abs() <= tolerance)
        | ((points[:, 1] - sampled_ymax).abs() <= tolerance)
    )
    if not bool(on_boundary.any()):
        return result
    parent = list(range(2 * count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    endpoint_by_key: dict[tuple[int, int], int] = {}
    for endpoint, point in enumerate(points.tolist()):
        key = (round(point[0] / tolerance), round(point[1] / tolerance))
        previous = endpoint_by_key.get(key)
        if previous is None:
            endpoint_by_key[key] = endpoint
        else:
            union(endpoint, previous)
    for segment in range(count):
        union(2 * segment, 2 * segment + 1)
    invalid_roots = {
        find(endpoint)
        for endpoint, boundary in enumerate(on_boundary.tolist())
        if boundary
    }
    invalid = [find(2 * segment) in invalid_roots for segment in range(count)]
    return torch.tensor(invalid, device=critical_segments.device, dtype=torch.bool)


def _clean_small_sign_islands(
    determinant: torch.Tensor,
    minimum_pixels: int,
) -> torch.Tensor:
    """Remove unresolved one/few-pixel detA sign speckles locally."""

    if int(minimum_pixels) <= 1 or determinant.numel() == 0:
        return determinant
    finite = torch.isfinite(determinant)
    positive = (determinant > 0.0) & finite
    negative = (determinant < 0.0) & finite
    kernel = torch.ones(
        (1, 1, 3, 3),
        device=determinant.device,
        dtype=torch.float32,
    )

    def count(mask: torch.Tensor) -> torch.Tensor:
        return functional.conv2d(
            mask.to(torch.float32)[None, None],
            kernel,
            padding=1,
        )[0, 0]

    positive_count = count(positive)
    negative_count = count(negative)
    threshold = float(minimum_pixels)
    flip_positive = (
        positive & (positive_count < threshold) & (negative_count >= threshold)
    )
    flip_negative = (
        negative & (negative_count < threshold) & (positive_count >= threshold)
    )
    absolute = determinant.abs()
    cleaned = torch.where(flip_positive, -absolute, determinant)
    return torch.where(flip_negative, absolute, cleaned)


def _is_cuda_oom(error: BaseException) -> bool:
    out_of_memory = getattr(torch, "OutOfMemoryError", ())
    if out_of_memory and isinstance(error, out_of_memory):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _segments_intersecting_region(
    segments: torch.Tensor,
    region: PlaneRegion,
) -> torch.Tensor:
    """Return a conservative mask for linear segments touching a region."""

    if segments.numel() == 0:
        return torch.zeros(
            segments.shape[:-2], device=segments.device, dtype=torch.bool
        )
    xmin, xmax, ymin, ymax = region.bounds_uas
    x = segments[..., 0]
    y = segments[..., 1]
    return (
        (x.amin(dim=-1) <= xmax)
        & (x.amax(dim=-1) >= xmin)
        & (y.amin(dim=-1) <= ymax)
        & (y.amax(dim=-1) >= ymin)
    )


def _sparse_determinant_layout(
    lens_grid: PlaneGrid,
    selected_cell_indices: torch.Tensor,
    selected_cell_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map selected IPM cells onto active detA cells and unique vertices."""

    device = selected_cell_indices.device
    fine_ny, fine_nx = (int(value) for value in selected_cell_shape)
    linear = selected_cell_indices.to(device=device, dtype=torch.int64).reshape(-1)
    rows = torch.div(linear, fine_nx, rounding_mode="floor")
    columns = linear - rows * fine_nx
    ny, nx = lens_grid.shape
    output_ny, output_nx = ny - 1, nx - 1
    if linear.numel() == 0:
        empty = torch.empty((0, 2), device=device, dtype=torch.int64)
        return empty, empty

    # ``interpolate(..., mode='nearest')`` maps output index ``j`` to
    # floor(j * input/output).  Invert that relation only for selected input
    # cells instead of allocating dense 4096^2 and 8191^2 masks.  The direct
    # sparse expansion is exactly equivalent and removes hundreds of MB of
    # temporary traffic from each center-label batch.
    row_start = torch.div(
        rows * output_ny + fine_ny - 1,
        fine_ny,
        rounding_mode="floor",
    )
    row_stop = torch.div(
        (rows + 1) * output_ny + fine_ny - 1,
        fine_ny,
        rounding_mode="floor",
    )
    column_start = torch.div(
        columns * output_nx + fine_nx - 1,
        fine_nx,
        rounding_mode="floor",
    )
    column_stop = torch.div(
        (columns + 1) * output_nx + fine_nx - 1,
        fine_nx,
        rounding_mode="floor",
    )
    maximum_rows = max(1, (output_ny + fine_ny - 1) // fine_ny + 1)
    maximum_columns = max(1, (output_nx + fine_nx - 1) // fine_nx + 1)
    row_offsets = torch.arange(maximum_rows, device=device, dtype=torch.int64)
    column_offsets = torch.arange(maximum_columns, device=device, dtype=torch.int64)
    expanded_rows = row_start[:, None, None] + row_offsets[None, :, None]
    expanded_columns = column_start[:, None, None] + column_offsets[None, None, :]
    valid = (expanded_rows < row_stop[:, None, None]) & (
        expanded_columns < column_stop[:, None, None]
    )
    active_flat = (
        expanded_rows.expand(-1, maximum_rows, maximum_columns) * output_nx
        + expanded_columns.expand(-1, maximum_rows, maximum_columns)
    )[valid]
    active_flat = torch.unique(active_flat, sorted=True)
    active_rows = torch.div(active_flat, output_nx, rounding_mode="floor")
    active_columns = active_flat - active_rows * output_nx
    active_indices = torch.stack((active_rows, active_columns), dim=1)

    return active_indices, _vertices_for_active_cells(lens_grid, active_indices)


def _vertices_for_active_cells(
    lens_grid: PlaneGrid,
    active_indices: torch.Tensor,
) -> torch.Tensor:
    """Return the unique determinant vertices needed by active grid cells."""

    nx = int(lens_grid.shape[1])
    if active_indices.numel() == 0:
        return torch.empty((0, 2), device=active_indices.device, dtype=torch.int64)
    active_rows = active_indices[:, 0]
    active_columns = active_indices[:, 1]
    vertex_flat = torch.cat(
        (
            active_rows * nx + active_columns,
            active_rows * nx + active_columns + 1,
            (active_rows + 1) * nx + active_columns,
            (active_rows + 1) * nx + active_columns + 1,
        )
    )
    vertex_flat = torch.unique(vertex_flat, sorted=True)
    vertex_rows = torch.div(vertex_flat, nx, rounding_mode="floor")
    vertex_columns = vertex_flat - vertex_rows * nx
    return torch.stack((vertex_rows, vertex_columns), dim=1)


def _sparse_marching_geometry(
    lens_grid: PlaneGrid,
    active_indices: torch.Tensor,
    vertex_indices: torch.Tensor,
    dtype: torch.dtype,
):
    """Build reusable corner and boundary indices for one sparse cell layout."""

    ny, nx = lens_grid.shape
    row, column = active_indices.unbind(dim=1)
    vertex_flat = vertex_indices[:, 0] * nx + vertex_indices[:, 1]
    corner_flat = torch.stack(
        (
            row * nx + column,
            row * nx + column + 1,
            (row + 1) * nx + column + 1,
            (row + 1) * nx + column,
        )
    )
    corner_positions = torch.searchsorted(vertex_flat, corner_flat)
    dy, dx = lens_grid.pixel_scale_uas
    xmin, _, ymin, _ = lens_grid.bounds_uas
    x0 = xmin + (column.to(dtype) + 0.5) * dx
    x1 = x0 + dx
    y0 = ymin + (row.to(dtype) + 0.5) * dy
    y1 = y0 + dy
    active_flat = row * (nx - 1) + column

    def neighbor_active(neighbor_row, neighbor_column, valid):
        neighbor_flat = neighbor_row * (nx - 1) + neighbor_column
        locations = torch.searchsorted(active_flat, neighbor_flat)
        safe = locations.clamp_max(max(int(active_flat.numel()) - 1, 0))
        return (
            valid
            & (locations < active_flat.numel())
            & (active_flat[safe] == neighbor_flat)
        )

    boundary = torch.stack(
        (
            ~neighbor_active(row - 1, column, row > 0),
            ~neighbor_active(row, column + 1, column < nx - 2),
            ~neighbor_active(row + 1, column, row < ny - 2),
            ~neighbor_active(row, column - 1, column > 0),
        )
    )
    return corner_positions, x0, x1, y0, y1, boundary


def _critical_discovery_mask(
    simulation: MicrolensingSimulation,
    lens_grid: PlaneGrid,
    times: tuple[float, ...],
    far_fields: tuple,
    batched: BatchedTaylorFarFieldApproximation | None,
    config: CausticConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Find high-resolution detA cells worth evaluating from a coarse grid.

    A temporal batch uses the union of its per-epoch discovery masks. This is
    conservative across the batch and allows every epoch to share one compact
    sparse determinant layout.
    """

    runtime = simulation.runtime
    ny, nx = lens_grid.shape
    ratio = int(config.discovery_downsample_ratio)
    coarse_ny = max(2, int(round(ny / ratio)))
    coarse_nx = max(2, int(round(nx / ratio)))
    dy, dx = lens_grid.pixel_scale_uas
    xmin, xmax, ymin, ymax = lens_grid.bounds_uas
    x_axis = torch.linspace(
        xmin + 0.5 * dx,
        xmax - 0.5 * dx,
        coarse_nx,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    y_axis = torch.linspace(
        ymin + 0.5 * dy,
        ymax - 0.5 * dy,
        coarse_ny,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    y, x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    if batched is not None:
        determinant = batched.jacobian_determinant(flat_x, flat_y)
    else:
        values = []
        for time_days, far_field in zip(times, far_fields, strict=True):
            if far_field is None:
                value, _ = jacobian_determinant_direct(
                    simulation,
                    flat_x,
                    flat_y,
                    time_days=time_days,
                )
            else:
                value = far_field.jacobian_determinant(flat_x, flat_y)
            values.append(value)
        determinant = torch.stack(values)
    determinant = determinant.reshape(len(times), coarse_ny, coarse_nx)
    corners = (
        determinant[:, :-1, :-1],
        determinant[:, :-1, 1:],
        determinant[:, 1:, 1:],
        determinant[:, 1:, :-1],
    )
    cell_min = torch.stack(corners, dim=-1).amin(dim=-1)
    cell_max = torch.stack(corners, dim=-1).amax(dim=-1)
    candidate = (cell_min <= 0.0) & (cell_max >= 0.0)
    quantile = float(config.discovery_near_zero_quantile)
    if quantile > 0.0:
        minimum_absolute = torch.stack(
            tuple(value.abs() for value in corners), dim=-1
        ).amin(dim=-1)
        for frame in range(len(times)):
            finite = minimum_absolute[frame][torch.isfinite(minimum_absolute[frame])]
            if finite.numel():
                candidate[frame] |= minimum_absolute[frame] <= torch.quantile(
                    finite, quantile
                )
    union = candidate.any(dim=0)
    for _ in range(int(config.discovery_dilation_cells)):
        union = functional.max_pool2d(
            union[None, None].to(torch.float32), 3, stride=1, padding=1
        )[0, 0].to(torch.bool)
    fine = functional.interpolate(
        union[None, None].to(torch.float32),
        size=(ny - 1, nx - 1),
        mode="nearest",
    )[0, 0].to(torch.bool)
    return fine, {
        "critical_discovery_downsample_ratio": ratio,
        "critical_discovery_grid_shape": [coarse_ny, coarse_nx],
        "critical_discovery_candidate_fraction": float(
            union.to(torch.float32).mean().detach().cpu()
        ),
        "critical_discovery_temporal_union": len(times) > 1,
    }


def _sparse_marching_segments(
    determinant_at_vertices: torch.Tensor,
    active_indices: torch.Tensor,
    vertex_indices: torch.Tensor,
    lens_grid: PlaneGrid,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """March selected detA cells without materializing a full determinant."""

    if active_indices.numel() == 0:
        empty = determinant_at_vertices.new_empty((0, 2, 2))
        return (
            empty,
            torch.empty(
                (0, 2), device=determinant_at_vertices.device, dtype=torch.bool
            ),
            "empty",
        )
    ny, nx = lens_grid.shape
    row, column = active_indices.unbind(dim=1)
    vertex_flat = vertex_indices[:, 0] * nx + vertex_indices[:, 1]

    def corner_values(row_index, column_index):
        flat = row_index * nx + column_index
        return determinant_at_vertices[torch.searchsorted(vertex_flat, flat)]

    f0 = corner_values(row, column)
    f1 = corner_values(row, column + 1)
    f2 = corner_values(row + 1, column + 1)
    f3 = corner_values(row + 1, column)
    dy, dx = lens_grid.pixel_scale_uas
    xmin, _, ymin, _ = lens_grid.bounds_uas
    x0 = xmin + (column.to(f0.dtype) + 0.5) * dx
    x1 = x0 + dx
    y0 = ymin + (row.to(f0.dtype) + 0.5) * dy
    y1 = y0 + dy

    def interpolate(ax, ay, af, bx, by, bf):
        denominator = af - bf
        epsilon = torch.as_tensor(1.0e-12, device=af.device, dtype=af.dtype)
        safe = denominator + (denominator == 0).to(af.dtype) * epsilon
        fraction = af / safe
        return torch.stack(
            (ax + fraction * (bx - ax), ay + fraction * (by - ay)), dim=-1
        )

    p0 = torch.stack((x0, y0), dim=-1)
    p1 = torch.stack((x1, y0), dim=-1)
    p2 = torch.stack((x1, y1), dim=-1)
    p3 = torch.stack((x0, y1), dim=-1)
    edges = (
        interpolate(x0, y0, f0, x1, y0, f1),
        interpolate(x1, y0, f1, x1, y1, f2),
        interpolate(x1, y1, f2, x0, y1, f3),
        interpolate(x0, y1, f3, x0, y0, f0),
    )
    del p0, p1, p2, p3
    case = (
        (f0 > 0).to(torch.int16)
        + 2 * (f1 > 0).to(torch.int16)
        + 4 * (f2 > 0).to(torch.int16)
        + 8 * (f3 > 0).to(torch.int16)
    )

    active_flat = row * (nx - 1) + column

    def neighbor_active(neighbor_row, neighbor_column, valid):
        neighbor_flat = neighbor_row * (nx - 1) + neighbor_column
        locations = torch.searchsorted(active_flat, neighbor_flat)
        safe_locations = locations.clamp_max(max(int(active_flat.numel()) - 1, 0))
        return (
            valid
            & (locations < active_flat.numel())
            & (active_flat[safe_locations] == neighbor_flat)
        )

    edge_boundary = (
        ~neighbor_active(row - 1, column, row > 0),
        ~neighbor_active(row, column + 1, column < nx - 2),
        ~neighbor_active(row + 1, column, row < ny - 2),
        ~neighbor_active(row, column - 1, column > 0),
    )
    if (
        determinant_at_vertices.device.type == "cuda"
        and determinant_at_vertices.dtype == torch.float32
    ):
        try:
            from .triton_caustics import (
                sparse_marching_squares_zero_triton,
                triton_caustics_available,
            )

            if triton_caustics_available():
                segments, boundaries = sparse_marching_squares_zero_triton(
                    f0,
                    f1,
                    f2,
                    f3,
                    x0,
                    x1,
                    y0,
                    y1,
                    torch.stack(edge_boundary, dim=0),
                )
                return segments, boundaries, "triton_sparse_compact"
        except Exception as error:
            warn_backend_fallback("Triton sparse marching squares", error)
    pair_a = torch.tensor(
        (
            (0, 0),
            (3, 0),
            (0, 0),
            (3, 0),
            (1, 0),
            (0, 2),
            (0, 0),
            (3, 0),
            (2, 0),
            (2, 0),
            (0, 1),
            (2, 0),
            (1, 0),
            (1, 0),
            (0, 0),
            (0, 0),
        ),
        device=case.device,
        dtype=torch.int64,
    )[case.to(torch.int64)]
    pair_b = torch.tensor(
        (
            (0, 0),
            (0, 0),
            (1, 0),
            (1, 0),
            (2, 0),
            (1, 3),
            (2, 0),
            (2, 0),
            (3, 0),
            (0, 0),
            (3, 2),
            (1, 0),
            (3, 0),
            (0, 0),
            (3, 0),
            (0, 0),
        ),
        device=case.device,
        dtype=torch.int64,
    )[case.to(torch.int64)]
    center_positive = 0.25 * (f0 + f1 + f2 + f3) > 0
    use_positive = ((case == 5) & center_positive) | ((case == 10) & ~center_positive)
    use_negative = ((case == 5) & ~center_positive) | ((case == 10) & center_positive)
    positive_a = torch.tensor((0, 2), device=case.device)[None]
    positive_b = torch.tensor((1, 3), device=case.device)[None]
    negative_a = torch.tensor((0, 1), device=case.device)[None]
    negative_b = torch.tensor((3, 2), device=case.device)[None]
    pair_a = torch.where(use_positive[:, None], positive_a, pair_a)
    pair_b = torch.where(use_positive[:, None], positive_b, pair_b)
    pair_a = torch.where(use_negative[:, None], negative_a, pair_a)
    pair_b = torch.where(use_negative[:, None], negative_b, pair_b)
    counts = torch.where(
        (case == 0) | (case == 15),
        0,
        torch.where((case == 5) | (case == 10), 2, 1),
    )
    valid_pairs = torch.arange(2, device=case.device)[None] < counts[:, None]
    edge_points = torch.stack(edges, dim=1)
    gather_shape = (-1, -1, 2)
    first = edge_points.gather(1, pair_a[..., None].expand(*gather_shape))
    second = edge_points.gather(1, pair_b[..., None].expand(*gather_shape))
    segments = torch.stack((first, second), dim=2)[valid_pairs]
    boundary_values = torch.stack(edge_boundary, dim=1)
    first_boundary = boundary_values.gather(1, pair_a)
    second_boundary = boundary_values.gather(1, pair_b)
    boundaries = torch.stack((first_boundary, second_boundary), dim=2)[valid_pairs]
    return segments, boundaries, "torch_sparse_vectorized"


def _batched_sparse_marching_segments(
    determinant_at_vertices: torch.Tensor,
    active_indices: torch.Tensor,
    vertex_indices: torch.Tensor,
    lens_grid: PlaneGrid,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] | None:
    """March one shared selected-cell layout in a single temporal launch."""

    if (
        determinant_at_vertices.device.type != "cuda"
        or determinant_at_vertices.dtype != torch.float32
        or determinant_at_vertices.ndim != 2
        or active_indices.numel() == 0
    ):
        return None
    try:
        from .triton_caustics import (
            batched_sparse_marching_squares_zero_triton,
            triton_caustics_available,
        )

        if not triton_caustics_available():
            return None
        ny, nx = lens_grid.shape
        row, column = active_indices.unbind(dim=1)
        vertex_flat = vertex_indices[:, 0] * nx + vertex_indices[:, 1]

        def positions(row_index, column_index):
            flat = row_index * nx + column_index
            return torch.searchsorted(vertex_flat, flat)

        f0 = determinant_at_vertices[:, positions(row, column)]
        f1 = determinant_at_vertices[:, positions(row, column + 1)]
        f2 = determinant_at_vertices[:, positions(row + 1, column + 1)]
        f3 = determinant_at_vertices[:, positions(row + 1, column)]
        dy, dx = lens_grid.pixel_scale_uas
        xmin, _, ymin, _ = lens_grid.bounds_uas
        x0 = xmin + (column.to(f0.dtype) + 0.5) * dx
        x1 = x0 + dx
        y0 = ymin + (row.to(f0.dtype) + 0.5) * dy
        y1 = y0 + dy
        active_flat = row * (nx - 1) + column

        def neighbor_active(neighbor_row, neighbor_column, valid):
            neighbor_flat = neighbor_row * (nx - 1) + neighbor_column
            locations = torch.searchsorted(active_flat, neighbor_flat)
            locations_safe = locations.clamp_max(max(int(active_flat.numel()) - 1, 0))
            return (
                valid
                & (locations < active_flat.numel())
                & (active_flat[locations_safe] == neighbor_flat)
            )

        edge_boundary = torch.stack(
            (
                ~neighbor_active(row - 1, column, row > 0),
                ~neighbor_active(row, column + 1, column < nx - 2),
                ~neighbor_active(row + 1, column, row < ny - 2),
                ~neighbor_active(row, column - 1, column > 0),
            ),
            dim=0,
        )
        return batched_sparse_marching_squares_zero_triton(
            f0,
            f1,
            f2,
            f3,
            x0,
            x1,
            y0,
            y1,
            edge_boundary,
            return_flat=True,
        )
    except Exception as error:
        warn_backend_fallback("batched Triton sparse marching squares", error)
        return None


def _sparse_boundary_seed_mask(
    caustic_segments: torch.Tensor,
    endpoint_on_sparse_boundary: torch.Tensor,
    source_region: PlaneRegion,
) -> torch.Tensor:
    """Return sparse-boundary endpoints whose images threaten the source."""

    xmin, xmax, ymin, ymax = source_region.bounds_uas
    source_points = caustic_segments.reshape(-1, 2)
    threatens_source = (
        (source_points[:, 0] >= xmin)
        & (source_points[:, 0] <= xmax)
        & (source_points[:, 1] >= ymin)
        & (source_points[:, 1] <= ymax)
    )
    return endpoint_on_sparse_boundary.reshape(-1) & threatens_source


def _sparse_boundary_component_mask(
    critical_segments: torch.Tensor,
    caustic_segments: torch.Tensor,
    endpoint_on_sparse_boundary: torch.Tensor,
    source_region: PlaneRegion,
    lens_grid: PlaneGrid,
    *,
    has_threatening_seed: bool | None = None,
) -> torch.Tensor:
    """Invalidate sparse-boundary components that remain inside the source."""

    count = int(critical_segments.shape[0])
    if count == 0:
        return torch.zeros(0, device=critical_segments.device, dtype=torch.bool)
    seeds = _sparse_boundary_seed_mask(
        caustic_segments,
        endpoint_on_sparse_boundary,
        source_region,
    )
    if has_threatening_seed is None:
        has_threatening_seed = bool(seeds.any().detach().cpu())
    if not has_threatening_seed:
        return torch.zeros(count, device=critical_segments.device, dtype=torch.bool)

    points = (
        critical_segments.detach().to(device="cpu", dtype=torch.float64).reshape(-1, 2)
    )
    tolerance = max(min(lens_grid.pixel_scale_uas) * 1.0e-5, 1.0e-12)
    parent = list(range(2 * count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    endpoint_by_key = {}
    for endpoint, point in enumerate(points.tolist()):
        key = (round(point[0] / tolerance), round(point[1] / tolerance))
        previous = endpoint_by_key.get(key)
        if previous is None:
            endpoint_by_key[key] = endpoint
        else:
            union(endpoint, previous)
    for segment in range(count):
        union(2 * segment, 2 * segment + 1)
    invalid_roots = {
        find(endpoint)
        for endpoint, active in enumerate(seeds.detach().cpu().tolist())
        if active
    }
    return torch.tensor(
        [find(2 * segment) in invalid_roots for segment in range(count)],
        device=critical_segments.device,
        dtype=torch.bool,
    )


@torch.no_grad()
def caustic_fields_from_far_fields(
    simulation: MicrolensingSimulation,
    lens_grid: PlaneGrid,
    times_days,
    config: CausticConfig,
    *,
    far_fields=None,
    selected_cell_indices: torch.Tensor | None = None,
    selected_cell_indices_by_frame=None,
    selected_cell_shape: tuple[int, int] | None = None,
    source_region: PlaneRegion | None = None,
) -> tuple[CausticField, ...]:
    """Evaluate detA and extract a temporal caustic batch.

    When the fused tiled-IPM scheduler supplies its scout selection, detA is
    evaluated only at the unique vertices of active determinant cells while
    retaining the pixel scale of ``lens_grid``.
    """

    times = tuple(float(value) for value in times_days)
    if not times:
        return ()
    runtime = simulation.runtime
    requested_lens_grid = lens_grid
    runtime.synchronize(detailed=False)
    started = perf_counter()
    if far_fields is None:
        if config.far_field_approx.enabled:
            far_fields, temporal_metadata = temporal_taylor_far_fields(
                simulation,
                lens_grid.region,
                config.far_field_approx,
                times,
            )
        else:
            far_fields = (None,) * len(times)
            temporal_metadata = {"far_field_enabled": False}
    else:
        far_fields = tuple(far_fields)
        temporal_metadata = {"shared_far_fields": True}
    if len(far_fields) != len(times):
        raise ValueError("far_fields must match the caustic time batch")
    batched = (
        BatchedTaylorFarFieldApproximation(far_fields)
        if (len(times) > 1 or selected_cell_indices_by_frame is not None)
        and all(item is not None for item in far_fields)
        else None
    )
    runtime.synchronize()
    built = perf_counter()
    ny, nx = lens_grid.shape
    sparse_ragged = selected_cell_indices_by_frame is not None
    sparse = selected_cell_indices is not None or sparse_ragged
    active_indices = None
    vertex_indices = None
    discovery_metadata: dict[str, object] = {}
    if sparse:
        # Convert the source scout's active IPM cells to the unique determinant
        # vertices needed by marching squares, then add an independent coarse
        # discovery mask so distant critical structure cannot be missed.
        if selected_cell_shape is None:
            raise ValueError("selected_cell_shape is required with selected cells")
        selections = (
            tuple(selected_cell_indices_by_frame)
            if sparse_ragged
            else (selected_cell_indices,)
        )
        if sparse_ragged and len(selections) != len(times):
            raise ValueError("per-frame selected cells must match the time batch")
        layout_cache = {}
        layouts = []
        for selection in selections:
            selection = torch.as_tensor(
                selection,
                device=runtime.device,
                dtype=torch.int64,
            )
            key = (int(selection.data_ptr()), int(selection.numel()))
            layout = layout_cache.get(key)
            if layout is None:
                layout = _sparse_determinant_layout(
                    lens_grid,
                    selection,
                    selected_cell_shape,
                )
                layout_cache[key] = layout
            layouts.append(layout)
        active_rows, vertex_rows = zip(*layouts, strict=True)
        source_active_before = tuple(int(row.shape[0]) for row in active_rows)
        if sparse_ragged:
            # The per-frame source scouts are already conservative endpoint
            # unions.  This is the validated paper path: every selected cell
            # is marched directly, avoiding a second all-frame coarse detA
            # discovery pass that duplicates most of the Jacobian work.
            discovery_metadata = {"critical_discovery": "source_scout"}
        else:
            discovery_mask, discovery_metadata = _critical_discovery_mask(
                simulation,
                lens_grid,
                times,
                far_fields,
                batched,
                config,
            )
            filtered_layouts = []
            for row in active_rows:
                if row.numel():
                    keep = discovery_mask[row[:, 0], row[:, 1]]
                    row = row[keep]
                filtered_layouts.append(
                    (row, _vertices_for_active_cells(lens_grid, row))
                )
            active_rows, vertex_rows = zip(*filtered_layouts, strict=True)
        if sparse_ragged:
            active_indices = active_rows
            vertex_indices = vertex_rows
        else:
            active_indices = active_rows[0]
            vertex_indices = vertex_rows[0]
        discovery_metadata.update(
            {
                "critical_discovery_source_active_before": (
                    list(source_active_before)
                    if sparse_ragged
                    else source_active_before[0]
                ),
                "critical_discovery_source_active_after": (
                    [int(row.shape[0]) for row in active_rows]
                    if sparse_ragged
                    else int(active_indices.shape[0])
                ),
            }
        )
        point_count = (
            sum(int(row.shape[0]) for row in vertex_rows)
            if sparse_ragged
            else int(vertex_indices.shape[0])
        )
    else:
        point_count = ny * nx
    determinant = (
        None
        if sparse_ragged
        else torch.empty(
            (len(times), point_count),
            device=runtime.device,
            dtype=runtime.dtype,
        )
    )
    dy, dx = lens_grid.pixel_scale_uas
    xmin, _, ymin, _ = lens_grid.bounds_uas
    chunk_size = int(config.jacobian_chunk_size)
    if sparse_ragged:
        vertex_lengths = tuple(int(row.shape[0]) for row in vertex_indices)
        flat_vertices = torch.cat(vertex_indices)
        flat_frames = torch.repeat_interleave(
            torch.arange(len(times), device=runtime.device, dtype=torch.int32),
            torch.tensor(vertex_lengths, device=runtime.device, dtype=torch.int64),
            output_size=point_count,
        )
        flat_determinant = torch.empty(
            point_count, device=runtime.device, dtype=runtime.dtype
        )
    for start in range(0, point_count, chunk_size):
        stop = min(point_count, start + chunk_size)
        if sparse_ragged:
            row = flat_vertices[start:stop, 0]
            column = flat_vertices[start:stop, 1]
        elif sparse:
            row = vertex_indices[start:stop, 0]
            column = vertex_indices[start:stop, 1]
        else:
            linear = torch.arange(start, stop, device=runtime.device, dtype=torch.int64)
            row = torch.div(linear, nx, rounding_mode="floor")
            column = linear - row * nx
        x = xmin + (column.to(runtime.dtype) + 0.5) * dx
        y = ymin + (row.to(runtime.dtype) + 0.5) * dy
        if sparse_ragged and batched is not None:
            flat_determinant[start:stop] = batched.jacobian_determinant_indexed_flat(
                x,
                y,
                flat_frames[start:stop],
            )
        elif batched is not None:
            determinant[:, start:stop] = batched.jacobian_determinant(x, y)
        else:
            values = []
            for time_days, far_field in zip(times, far_fields, strict=True):
                if far_field is None:
                    direct, _ = jacobian_determinant_direct(
                        simulation,
                        x,
                        y,
                        time_days=time_days,
                    )
                    values.append(direct)
                else:
                    values.append(far_field.jacobian_determinant(x, y))
            determinant[:, start:stop] = torch.stack(values)
    if sparse_ragged:
        determinant = tuple(flat_determinant.split(vertex_lengths))
    elif not sparse:
        determinant = determinant.reshape(len(times), ny, nx)
    runtime.synchronize()
    evaluated = perf_counter()
    sparse_marching_started = perf_counter()
    if sparse_ragged:
        marching_inputs = []
        marching_geometry_cache = {}
        for det_row, active_row, vertex_row in zip(
            determinant, active_indices, vertex_indices, strict=True
        ):
            key = (int(active_row.data_ptr()), int(active_row.shape[0]))
            geometry = marching_geometry_cache.get(key)
            if geometry is None:
                geometry = _sparse_marching_geometry(
                    lens_grid,
                    active_row,
                    vertex_row,
                    runtime.dtype,
                )
                marching_geometry_cache[key] = geometry
            corner_positions, x0, x1, y0, y1, boundary = geometry
            values = tuple(det_row[position] for position in corner_positions)
            marching_inputs.append((*values, x0, x1, y0, y1, boundary))
        if runtime.device.type == "cuda" and runtime.dtype == torch.float32:
            from .triton_caustics import ragged_sparse_marching_squares_zero_triton

            cell_lengths = tuple(int(row.shape[0]) for row in active_indices)
            cell_frames = torch.repeat_interleave(
                torch.arange(len(times), device=runtime.device, dtype=torch.int32),
                torch.tensor(cell_lengths, device=runtime.device, dtype=torch.int64),
                output_size=sum(cell_lengths),
            )
            columns = tuple(torch.cat([row[index] for row in marching_inputs]) for index in range(8))
            boundaries = torch.cat([row[8] for row in marching_inputs], dim=1)
            sparse_marching_batch = ragged_sparse_marching_squares_zero_triton(
                *columns,
                boundaries,
                cell_frames,
                len(times),
            )
        else:
            rows = tuple(
                _sparse_marching_segments(det_row, active_row, vertex_row, lens_grid)
                for det_row, active_row, vertex_row in zip(
                    determinant, active_indices, vertex_indices, strict=True
                )
            )
            sparse_marching_batch = (
                tuple(row[0] for row in rows),
                tuple(row[1] for row in rows),
            )
    else:
        sparse_marching_batch = (
            _batched_sparse_marching_segments(
                determinant,
                active_indices,
                vertex_indices,
                lens_grid,
            )
            if sparse and len(times) > 1
            else None
        )
    runtime.synchronize()
    sparse_batch_marching_seconds = (
        perf_counter() - sparse_marching_started
        if sparse_marching_batch is not None
        else 0.0
    )
    dense_marching_batch = None
    dense_cleanup_seconds = 0.0
    dense_marching_seconds = 0.0
    if (
        not sparse
        and len(times) > 1
        and runtime.device.type == "cuda"
        and runtime.dtype == torch.float32
    ):
        from .triton_caustics import (
            batched_dense_marching_squares_zero_triton,
            triton_caustics_available,
        )

        if triton_caustics_available():
            # Dense CUDA batches can clean and march every frame together;
            # portable and single-frame paths retain the exact scalar fallback.
            phase_started = perf_counter()
            if config.minimum_determinant_sign_pixels > 0:
                for frame in range(len(times)):
                    determinant[frame] = _clean_small_sign_islands(
                        determinant[frame],
                        config.minimum_determinant_sign_pixels,
                    )
            runtime.synchronize()
            dense_cleanup_seconds = perf_counter() - phase_started
            phase_started = perf_counter()
            dense_marching_batch = batched_dense_marching_squares_zero_triton(
                determinant, lens_grid
            )
            runtime.synchronize()
            dense_marching_seconds = perf_counter() - phase_started
    ragged_caustics = None
    ragged_boundary_threats = None
    flat_source_filtered = False
    critical_rows = None
    boundary_rows = None
    ragged_endpoint_started = None
    if dense_marching_batch is not None and batched is not None:
        runtime.synchronize()
        ragged_endpoint_started = perf_counter()
        mapped_x, mapped_y = batched.raytrace_ragged(
            tuple(row[..., 0].reshape(-1) for row in dense_marching_batch),
            tuple(row[..., 1].reshape(-1) for row in dense_marching_batch),
        )
        ragged_caustics = tuple(
            torch.stack((x, y), dim=-1).reshape_as(critical)
            for critical, x, y in zip(
                dense_marching_batch, mapped_x, mapped_y, strict=True
            )
        )
    if sparse_marching_batch is not None and batched is not None:
        # Marching squares produces a different endpoint count in every
        # frame. Trace the compact queues together instead of launching and
        # synchronizing one Taylor evaluator per epoch. The indexed Triton
        # query carries a frame id for every endpoint and therefore requires
        # neither padding nor additional lens evaluations.
        ragged_endpoint_started = perf_counter()
        critical_rows = sparse_marching_batch[0]
        if len(sparse_marching_batch) > 2:
            flat_critical = sparse_marching_batch[2]
            flat_boundary = sparse_marching_batch[3]
            flat_frames = sparse_marching_batch[4]
            lengths = sparse_marching_batch[5]
            flat_x, flat_y = batched.raytrace_indexed_flat(
                flat_critical[..., 0].reshape(-1),
                flat_critical[..., 1].reshape(-1),
                flat_frames.repeat_interleave(2),
            )
            flat_caustic = torch.stack((flat_x, flat_y), dim=-1).reshape(-1, 2, 2)
            if source_region is not None:
                keep = _segments_intersecting_region(flat_caustic, source_region)
                flat_critical = flat_critical[keep]
                flat_boundary = flat_boundary[keep]
                flat_frames = flat_frames[keep]
                flat_caustic = flat_caustic[keep]
                kept_lengths = (
                    torch.bincount(flat_frames.to(torch.int64), minlength=len(times))
                    .detach()
                    .cpu()
                    .tolist()
                )
                flat_source_filtered = True
            else:
                kept_lengths = lengths
            critical_rows = tuple(flat_critical.split(tuple(map(int, kept_lengths))))
            boundary_rows = tuple(flat_boundary.split(tuple(map(int, kept_lengths))))
            ragged_caustics = tuple(flat_caustic.split(tuple(map(int, kept_lengths))))
            if source_region is not None:
                xmin, xmax, ymin, ymax = source_region.bounds_uas
                inside = (
                    (flat_caustic[..., 0] >= xmin)
                    & (flat_caustic[..., 0] <= xmax)
                    & (flat_caustic[..., 1] >= ymin)
                    & (flat_caustic[..., 1] <= ymax)
                )
                threatening = (flat_boundary & inside).any(dim=1).to(torch.int32)
                threat_counts = torch.zeros(
                    len(times), device=runtime.device, dtype=torch.int32
                )
                threat_counts.scatter_add_(0, flat_frames.to(torch.int64), threatening)
                ragged_boundary_threats = tuple(
                    bool(value) for value in (threat_counts > 0).detach().cpu().tolist()
                )
        else:
            mapped_x, mapped_y = batched.raytrace_ragged(
                tuple(row[..., 0].reshape(-1) for row in critical_rows),
                tuple(row[..., 1].reshape(-1) for row in critical_rows),
            )
        if ragged_caustics is None:
            ragged_caustics = tuple(
                torch.stack((source_x, source_y), dim=-1).reshape_as(critical)
                for critical, source_x, source_y in zip(
                    critical_rows,
                    mapped_x,
                    mapped_y,
                    strict=True,
                )
            )
        if source_region is not None and ragged_boundary_threats is None:
            ragged_boundary_threats = tuple(
                bool(value)
                for value in torch.stack(
                    tuple(
                        _sparse_boundary_seed_mask(
                            caustic,
                            boundary,
                            source_region,
                        ).any()
                        for caustic, boundary in zip(
                            ragged_caustics,
                            sparse_marching_batch[1],
                            strict=True,
                        )
                    )
                )
                .detach()
                .cpu()
                .tolist()
            )
    ragged_endpoint_seconds = 0.0
    if ragged_endpoint_started is not None:
        runtime.synchronize()
        ragged_endpoint_seconds = perf_counter() - ragged_endpoint_started
    fields = []
    cleanup_seconds = dense_cleanup_seconds
    marching_seconds = sparse_batch_marching_seconds + dense_marching_seconds
    endpoint_seconds = 0.0
    rasterizers = []
    boundary_count_tensors = []
    for frame, (time_days, far_field) in enumerate(zip(times, far_fields, strict=True)):
        phase = perf_counter()
        sparse_boundary = None
        frame_active_indices = (
            active_indices[frame] if sparse_ragged else active_indices
        )
        frame_vertex_indices = (
            vertex_indices[frame] if sparse_ragged else vertex_indices
        )
        if sparse:
            if sparse_marching_batch is None:
                critical, sparse_boundary, rasterizer = _sparse_marching_segments(
                    determinant[frame],
                    frame_active_indices,
                    frame_vertex_indices,
                    lens_grid,
                )
            else:
                critical = (
                    sparse_marching_batch[0][frame]
                    if critical_rows is None
                    else critical_rows[frame]
                )
                sparse_boundary = (
                    sparse_marching_batch[1][frame]
                    if boundary_rows is None
                    else boundary_rows[frame]
                )
                rasterizer = "triton_sparse_temporal_compact"
        elif dense_marching_batch is not None:
            critical = dense_marching_batch[frame]
            rasterizer = "triton_dense_temporal_compact"
            invalid_segments = _boundary_component_mask(critical, lens_grid)
        else:
            det_frame = determinant[frame]
            if config.minimum_determinant_sign_pixels > 0:
                det_frame = _clean_small_sign_islands(
                    det_frame,
                    config.minimum_determinant_sign_pixels,
                )
            runtime.synchronize()
            cleanup_seconds += perf_counter() - phase
            phase = perf_counter()
            critical, rasterizer = _marching_segments(det_frame, lens_grid)
            invalid_segments = _boundary_component_mask(critical, lens_grid)
        if sparse_marching_batch is None:
            runtime.synchronize()
            marching_seconds += perf_counter() - phase
        rasterizers.append(rasterizer)
        phase = perf_counter()
        if ragged_caustics is not None:
            caustic = ragged_caustics[frame]
        elif critical.numel():
            if far_field is None:
                source_x, source_y, _ = simulation.raytrace_direct(
                    critical[..., 0],
                    critical[..., 1],
                    time_days=time_days,
                )
            else:
                source_x, source_y = far_field.raytrace(
                    critical[..., 0],
                    critical[..., 1],
                )
            caustic = torch.stack((source_x, source_y), dim=-1)
        else:
            caustic = critical.clone()
        if sparse:
            if source_region is None:
                raise ValueError("sparse caustic extraction requires source_region")
            invalid_segments = _sparse_boundary_component_mask(
                critical,
                caustic,
                sparse_boundary,
                source_region,
                lens_grid,
                has_threatening_seed=(
                    None
                    if ragged_boundary_threats is None
                    else ragged_boundary_threats[frame]
                ),
            )
        if source_region is not None and caustic.numel() and not flat_source_filtered:
            keep = _segments_intersecting_region(caustic, source_region)
            critical = critical[keep]
            caustic = caustic[keep]
            invalid_segments = invalid_segments[keep]
        if ragged_caustics is None:
            runtime.synchronize()
            endpoint_seconds += perf_counter() - phase
        boundary_count_tensors.append(invalid_segments.sum())
        fields.append(
            CausticField(
                critical,
                caustic,
                lens_grid,
                time_days=time_days,
                invalid_segment_mask=invalid_segments,
                metadata={
                    "method": "production_analytic_far_field",
                    "determinant_grid_shape": list(lens_grid.shape),
                    "requested_determinant_grid_shape": list(requested_lens_grid.shape),
                    "scout_sparse_determinant": bool(sparse),
                    "determinant_grid_fraction": float(
                        1.0
                        if not sparse
                        else int(frame_active_indices.shape[0])
                        / max((ny - 1) * (nx - 1), 1)
                    ),
                    "sparse_active_cells": (
                        None if not sparse else int(frame_active_indices.shape[0])
                    ),
                    "sparse_unique_vertices": (
                        None if not sparse else int(frame_vertex_indices.shape[0])
                    ),
                    **discovery_metadata,
                    "source_region_filtered": bool(source_region is not None),
                    "marching_squares": rasterizer,
                    "determinant_cleanup": (
                        "local"
                        if not sparse and config.minimum_determinant_sign_pixels > 0
                        else ("none" if not sparse else "sparse_cell_topology")
                    ),
                    "minimum_determinant_sign_pixels": (
                        config.minimum_determinant_sign_pixels
                    ),
                    "segment_representation": "independent_linear_segments",
                    # Filled with one batched device transfer after all
                    # frames, avoiding a stream synchronization per epoch.
                    "boundary_components_excluded_from_labels": None,
                    **temporal_metadata,
                },
            )
        )
    endpoint_seconds += ragged_endpoint_seconds
    boundary_counts = (
        torch.stack(boundary_count_tensors).detach().cpu().tolist()
        if boundary_count_tensors
        else []
    )
    fields = [
        replace(
            field,
            metadata={
                **field.metadata,
                "boundary_components_excluded_from_labels": int(count),
            },
        )
        for field, count in zip(fields, boundary_counts, strict=True)
    ]
    runtime.synchronize(detailed=False)
    finished = perf_counter()
    frame_count = len(fields)
    total_seconds = finished - started
    component = {
        "far_field_preparation": (built - started) / frame_count,
        "analytic_jacobian": (evaluated - built) / frame_count,
        "determinant_cleanup": cleanup_seconds / frame_count,
        "marching_squares": marching_seconds / frame_count,
        "map_critical_segments": endpoint_seconds / frame_count,
    }
    return tuple(
        replace(
            field,
            timing=TimingBreakdown(
                collected=runtime.profiling_enabled,
                steady_seconds=total_seconds / frame_count,
                component_seconds=(
                    component if runtime.profiling.value == "detailed" else {}
                ),
                peak_device_memory_bytes=(
                    int(torch.cuda.max_memory_allocated(runtime.device))
                    if runtime.profiling.value == "detailed"
                    and runtime.device.type == "cuda"
                    else None
                ),
            ),
        )
        for field in fields
    )


@torch.no_grad()
def dynamic_labeled_caustics(
    simulation: MicrolensingSimulation,
    lens_grid: PlaneGrid,
    source_region: PlaneRegion,
    times_days,
    config: CausticConfig,
    *,
    diagnostic_grid: PlaneGrid | None = None,
    include_distance_map: bool = False,
) -> tuple[LabeledCausticFrame, ...]:
    """Return a temporally aligned caustic/label sequence."""

    times = tuple(float(value) for value in times_days)
    if not times:
        return ()
    tuning_result = None
    if config.tuning.enabled:
        from ..tuning import autotune_caustics

        tuning_result = autotune_caustics(
            simulation,
            lens_grid,
            source_region,
            times,
            config,
        )
        config = replace(
            config,
            temporal_batch_size=tuning_result.temporal_batch_size,
            jacobian_chunk_size=tuning_result.spatial_chunk_size,
            tuning=replace(config.tuning, enabled=False),
        )
    if config.temporal_batch_size is not None:
        batch_size = min(len(times), int(config.temporal_batch_size))
    else:
        batch_size = min(
            len(times), 40 if simulation.runtime.device.type == "cuda" else 1
        )
        if simulation.runtime.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(simulation.runtime.device)
            determinant_bytes = (
                lens_grid.shape[0]
                * lens_grid.shape[1]
                * torch.empty((), dtype=simulation.runtime.dtype).element_size()
            )
            memory_frames = max(1, int(0.35 * free_bytes) // determinant_bytes)
            batch_size = min(batch_size, memory_frames)
    outputs = []
    previous_gauges = None
    previous_gauge_distances = None
    previous_center = None
    previous_center_distance = None
    start = 0
    while start < len(times):
        stop = min(len(times), start + batch_size)
        try:
            shared_far_fields = None
            if config.far_field_approx.enabled:
                shared_far_fields, _ = temporal_taylor_far_field_window(
                    simulation,
                    lens_grid.region,
                    config.far_field_approx,
                    times,
                    range(start, stop),
                )
            fields = caustic_fields_from_far_fields(
                simulation,
                lens_grid,
                times[start:stop],
                config,
                far_fields=shared_far_fields,
            )
        except Exception as error:
            if not _is_cuda_oom(error) or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            continue
        (
            labeled,
            previous_gauges,
            previous_gauge_distances,
            previous_center,
            previous_center_distance,
        ) = label_caustic_fields(
            fields,
            source_region,
            config,
            previous_aligned_gauges=previous_gauges,
            previous_gauge_distances_uas=previous_gauge_distances,
            previous_center_label=previous_center,
            previous_center_distance_uas=previous_center_distance,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
        )
        outputs.extend(labeled)
        start = stop
    if tuning_result is not None:
        outputs = [
            replace(
                frame,
                caustics=replace(
                    frame.caustics,
                    metadata={
                        **frame.caustics.metadata,
                        **tuning_result.metadata(),
                    },
                ),
            )
            for frame in outputs
        ]
    return tuple(outputs)


@torch.no_grad()
def dynamic_labeled_maps(
    simulation: MicrolensingSimulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    lens_grid: PlaneGrid,
    times_days,
    *,
    method,
    map_schedule=None,
    caustic_config: CausticConfig | None = None,
    diagnostic_grid: PlaneGrid | None = None,
    include_distance_map: bool = False,
) -> Iterator[LabeledMapFrame]:
    """Stream maps and labels while sharing production temporal far-field approximations.

    The fused tiled-IPM path invokes caustic extraction on the already-built
    temporal far-field approximation batch. Other solvers retain identical results but use
    the standalone caustic scheduler because they do not expose a shared
    far-field approximation batch.
    """

    from ..config import DynamicConfig, IPMConfig
    from ..dynamic import dynamic_maps

    times = tuple(float(value) for value in times_days)
    schedule = DynamicConfig() if map_schedule is None else map_schedule
    config = CausticConfig() if caustic_config is None else caustic_config
    if config.temporal_batch_size is None:
        inherited_batch = schedule.temporal_batch_size
        if inherited_batch is None:
            inherited_batch = 49 if simulation.runtime.device.type == "cuda" else 1
        config = replace(config, temporal_batch_size=int(inherited_batch))
    caustic_tuning_result = None
    if config.tuning.enabled:
        from ..tuning import autotune_caustics

        caustic_tuning_result = autotune_caustics(
            simulation,
            lens_grid,
            source_grid.region,
            times,
            config,
        )
        config = replace(
            config,
            temporal_batch_size=caustic_tuning_result.temporal_batch_size,
            jacobian_chunk_size=caustic_tuning_result.spatial_chunk_size,
            tuning=replace(config.tuning, enabled=False),
        )
    shared = (
        isinstance(method, IPMConfig)
        and method.tiled
        and schedule.fused_temporal_ipm
        and method.far_field_approx == config.far_field_approx
        and simulation.point_masses.has_motion
    )
    labels_by_index: dict[int, LabeledCausticFrame] = {}
    previous_gauges = None
    previous_distances = None
    previous_center = None
    previous_center_distance = None

    def observe(
        indices,
        far_fields,
        selected_cell_indices=None,
        selected_cell_shape=None,
        selected_cell_indices_by_frame=None,
    ) -> None:
        nonlocal previous_gauges
        nonlocal previous_distances
        nonlocal previous_center
        nonlocal previous_center_distance
        fields = caustic_fields_from_far_fields(
            simulation,
            lens_grid,
            [times[index] for index in indices],
            config,
            far_fields=far_fields,
            selected_cell_indices=selected_cell_indices,
            selected_cell_indices_by_frame=selected_cell_indices_by_frame,
            selected_cell_shape=selected_cell_shape,
            source_region=source_grid.region,
        )
        (
            labeled,
            previous_gauges,
            previous_distances,
            previous_center,
            previous_center_distance,
        ) = label_caustic_fields(
            fields,
            source_grid.region,
            config,
            previous_aligned_gauges=previous_gauges,
            previous_gauge_distances_uas=previous_distances,
            previous_center_label=previous_center,
            previous_center_distance_uas=previous_center_distance,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
        )
        if caustic_tuning_result is not None:
            labeled = tuple(
                replace(
                    frame,
                    caustics=replace(
                        frame.caustics,
                        metadata={
                            **frame.caustics.metadata,
                            **caustic_tuning_result.metadata(),
                        },
                    ),
                )
                for frame in labeled
            )
        labels_by_index.update(zip(indices, labeled, strict=True))

    if not shared:
        labels = dynamic_labeled_caustics(
            simulation,
            lens_grid,
            source_grid.region,
            times,
            config,
            diagnostic_grid=diagnostic_grid,
            include_distance_map=include_distance_map,
        )
        if caustic_tuning_result is not None:
            labels = tuple(
                replace(
                    frame,
                    caustics=replace(
                        frame.caustics,
                        metadata={
                            **frame.caustics.metadata,
                            **caustic_tuning_result.metadata(),
                        },
                    ),
                )
                for frame in labels
            )
        labels_by_index.update(enumerate(labels))
    maps = dynamic_maps(
        simulation,
        lens_region,
        source_grid,
        times,
        method=method,
        config=schedule,
        _far_field_batch_observer=observe if shared else None,
    )
    for index, magnification_map in enumerate(maps):
        try:
            labeled = labels_by_index.pop(index)
        except KeyError as error:
            raise RuntimeError(
                "the shared far-field approximation observer did not produce this label frame"
            ) from error
        yield LabeledMapFrame(magnification_map, labeled)


@torch.no_grad()
def streaming_labeled_light_curve(
    simulation: MicrolensingSimulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    lens_grid: PlaneGrid,
    times_days,
    source,
    distances,
    *,
    method,
    trajectory=None,
    map_schedule=None,
    caustic_config: CausticConfig | None = None,
    strict_coverage: bool = True,
    diagnostic_grid: PlaneGrid | None = None,
    include_distance_map: bool = False,
    map_observer=None,
) -> LabeledLightCurve:
    """Generate a production LC and labels without retaining map tensors."""

    from ..photometry import LightCurveRequest, streaming_light_curves

    labeled_frames: list[LabeledCausticFrame] = []

    def map_iterator():
        for index, frame in enumerate(
            dynamic_labeled_maps(
                simulation,
                lens_region,
                source_grid,
                lens_grid,
                times_days,
                method=method,
                map_schedule=map_schedule,
                caustic_config=caustic_config,
                diagnostic_grid=diagnostic_grid,
                include_distance_map=include_distance_map,
            )
        ):
            labeled_frames.append(frame.caustics)
            if map_observer is not None:
                map_observer(index, frame)
            yield frame.magnification_map

    request = LightCurveRequest(
        source=source,
        distances=distances,
        trajectory=trajectory,
        strict_coverage=strict_coverage,
    )
    light_curve = streaming_light_curves(
        simulation,
        lens_region,
        source_grid,
        times_days,
        (request,),
        method=method,
        schedule=map_schedule,
        _map_iterator=map_iterator(),
    )[0]
    return LabeledLightCurve(light_curve, tuple(labeled_frames))


@torch.no_grad()
def multirate_labeled_light_curve(
    simulation: MicrolensingSimulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    lens_grid: PlaneGrid,
    map_times_days,
    flux_times_days,
    source,
    distances,
    *,
    method,
    trajectory=None,
    map_schedule=None,
    caustic_config: CausticConfig | None = None,
    strict_coverage: bool = True,
    diagnostic_grid: PlaneGrid | None = None,
    include_distance_map: bool = False,
    map_observer=None,
):
    """Generate fine-cadence photometry and sparse same-map caustic labels.

    Dynamic maps are evaluated only at ``map_times_days``. The exact same map
    stream supplies multirate photometry and, for fused tiled IPM, shares its
    temporal far-field approximation batches with the caustic/label pipeline. ``map_observer``
    receives each :class:`~microcaustics.LabeledMapFrame`, making animations
    and selective exports possible without retaining a full-resolution cube.
    """

    from ..photometry import multirate_streaming_light_curve
    from ..results import MultirateLabeledLightCurve

    labeled_frames: list[LabeledCausticFrame] = []

    def map_iterator():
        for index, frame in enumerate(
            dynamic_labeled_maps(
                simulation,
                lens_region,
                source_grid,
                lens_grid,
                map_times_days,
                method=method,
                map_schedule=map_schedule,
                caustic_config=caustic_config,
                diagnostic_grid=diagnostic_grid,
                include_distance_map=include_distance_map,
            )
        ):
            labeled_frames.append(frame.caustics)
            if map_observer is not None:
                map_observer(index, frame)
            yield frame.magnification_map

    light_curve = multirate_streaming_light_curve(
        simulation,
        lens_region,
        source_grid,
        map_times_days,
        flux_times_days,
        source,
        distances,
        method=method,
        trajectory=trajectory,
        schedule=map_schedule,
        strict_coverage=strict_coverage,
        _map_iterator=map_iterator(),
    )
    return MultirateLabeledLightCurve(light_curve, tuple(labeled_frames))
