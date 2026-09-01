"""Fused CUDA kernels for production caustic extraction and labels."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional backend
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _regular_grid_winding_updates_kernel(
        segments,
        y_axis,
        x_axis,
        difference,
        n_segments,
        ny,
        nx,
        BLOCK: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
    ):
        segment_block = tl.program_id(0)
        row = tl.program_id(1)
        segment = segment_block * BLOCK + tl.arange(0, BLOCK)
        active = segment < n_segments
        base = segment.to(tl.int64) * 4
        x0 = tl.load(segments + base, mask=active, other=0.0)
        y0 = tl.load(segments + base + 1, mask=active, other=0.0)
        x1 = tl.load(segments + base + 2, mask=active, other=0.0)
        y1 = tl.load(segments + base + 3, mask=active, other=0.0)
        query_y = tl.load(y_axis + row, mask=row < ny, other=0.0)
        upward = active & (y0 <= query_y) & (query_y < y1)
        downward = active & (y1 <= query_y) & (query_y < y0)
        crosses = upward | downward
        denominator = tl.where(crosses, y1 - y0, 1.0)
        intersection_x = x0 + (query_y - y0) * (x1 - x0) / denominator

        # First sampled x coordinate greater than or equal to the crossing.
        # Loading the actual axis preserves torch.linspace float32 rounding.
        low = tl.zeros((BLOCK,), tl.int32)
        high = tl.full((BLOCK,), nx, tl.int32)
        for _ in tl.static_range(SEARCH_STEPS):
            searching = crosses & (low < high)
            middle = (low + high) // 2
            sampled_x = tl.load(
                x_axis + tl.minimum(middle, nx - 1),
                mask=searching,
                other=0.0,
            )
            move_right = searching & (sampled_x < intersection_x)
            high = tl.where(searching & ~move_right, middle, high)
            low = tl.where(move_right, middle + 1, low)

        signed = upward.to(tl.int32) - downward.to(tl.int32)
        row_base = row.to(tl.int64) * (nx + 1)
        tl.atomic_add(difference + row_base, tl.sum(signed, axis=0))
        tl.atomic_add(
            difference + row_base + low,
            -signed,
            mask=crosses,
        )

    @triton.jit
    def _crossing_kernel(
        segments,
        valid,
        anchors,
        points,
        counts,
        n_segments,
        n_anchors,
        n_points,
        segment_blocks,
        BLOCK: tl.constexpr,
    ):
        program = tl.program_id(0)
        segment_block = program % segment_blocks
        task = program // segment_blocks
        anchor = task % n_anchors
        task = task // n_anchors
        point = task % n_points
        frame = task // n_points
        segment = segment_block * BLOCK + tl.arange(0, BLOCK)
        active = segment < n_segments
        keep = tl.load(
            valid + frame * n_segments + segment,
            mask=active,
            other=0,
        ).to(tl.int1)
        base = (frame * n_segments + segment).to(tl.int64) * 4
        ax = tl.load(segments + base, mask=active, other=0.0)
        ay = tl.load(segments + base + 1, mask=active, other=0.0)
        bx = tl.load(segments + base + 2, mask=active, other=0.0)
        by = tl.load(segments + base + 3, mask=active, other=0.0)
        anchor_x = tl.load(anchors + anchor * 2)
        anchor_y = tl.load(anchors + anchor * 2 + 1)
        point_x = tl.load(points + point * 2)
        point_y = tl.load(points + point * 2 + 1)
        ray_x = point_x - anchor_x
        ray_y = point_y - anchor_y
        segment_x = bx - ax
        segment_y = by - ay
        offset_ax = ax - anchor_x
        offset_ay = ay - anchor_y
        offset_bx = bx - anchor_x
        offset_by = by - anchor_y
        denominator = ray_x * segment_y - ray_y * segment_x
        non_parallel = tl.abs(denominator) > 1.0e-7
        safe = tl.where(non_parallel, denominator, 1.0)
        fraction = (offset_ax * segment_y - offset_ay * segment_x) / safe
        side_a = ray_x * offset_ay - ray_y * offset_ax
        side_b = ray_x * offset_by - ray_y * offset_bx
        intersects = (
            active
            & keep
            & non_parallel
            & (fraction > 1.0e-7)
            & (fraction < 1.0 - 1.0e-7)
            & ((side_a > 0.0) != (side_b > 0.0))
        )
        partial = tl.sum(intersects.to(tl.int32), axis=0)
        output = (frame * n_points + point) * n_anchors + anchor
        tl.atomic_add(counts + output, partial)


    @triton.jit
    def _distance_kernel(
        segments,
        valid,
        points,
        distance2,
        n_segments,
        n_points,
        segment_blocks,
        BLOCK: tl.constexpr,
    ):
        program = tl.program_id(0)
        segment_block = program % segment_blocks
        task = program // segment_blocks
        point = task % n_points
        frame = task // n_points
        segment = segment_block * BLOCK + tl.arange(0, BLOCK)
        active = segment < n_segments
        keep = tl.load(
            valid + frame * n_segments + segment,
            mask=active,
            other=0,
        ).to(tl.int1)
        base = (frame * n_segments + segment).to(tl.int64) * 4
        ax = tl.load(segments + base, mask=active, other=0.0)
        ay = tl.load(segments + base + 1, mask=active, other=0.0)
        bx = tl.load(segments + base + 2, mask=active, other=0.0)
        by = tl.load(segments + base + 3, mask=active, other=0.0)
        px = tl.load(points + point * 2)
        py = tl.load(points + point * 2 + 1)
        sx = bx - ax
        sy = by - ay
        length2 = tl.maximum(sx * sx + sy * sy, 1.0e-24)
        projection = ((px - ax) * sx + (py - ay) * sy) / length2
        projection = tl.maximum(0.0, tl.minimum(1.0, projection))
        dx = px - (ax + projection * sx)
        dy = py - (ay + projection * sy)
        candidate = tl.where(active & keep, dx * dx + dy * dy, float("inf"))
        partial = tl.min(candidate, axis=0)
        tl.atomic_min(distance2 + frame * n_points + point, partial)


    @triton.jit
    def _marching_counts(field, counts, ny, nx, cells_x, BLOCK: tl.constexpr):
        cell = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = cell < (ny - 1) * cells_x
        row = cell // cells_x
        column = cell - row * cells_x
        offset = row * nx + column
        f0 = tl.load(field + offset, mask=active, other=0.0)
        f1 = tl.load(field + offset + 1, mask=active, other=0.0)
        f2 = tl.load(field + offset + nx + 1, mask=active, other=0.0)
        f3 = tl.load(field + offset + nx, mask=active, other=0.0)
        case = (
            (f0 > 0.0).to(tl.int32)
            + 2 * (f1 > 0.0).to(tl.int32)
            + 4 * (f2 > 0.0).to(tl.int32)
            + 8 * (f3 > 0.0).to(tl.int32)
        )
        count = tl.where(
            (case == 0) | (case == 15),
            0,
            tl.where((case == 5) | (case == 10), 2, 1),
        )
        tl.store(counts + cell, count, mask=active)


    @triton.jit
    def _marching_write(
        field,
        offsets,
        pair_a,
        pair_b,
        case5_positive_a,
        case5_positive_b,
        case5_negative_a,
        case5_negative_b,
        case10_positive_a,
        case10_positive_b,
        case10_negative_a,
        case10_negative_b,
        segments,
        ny,
        nx,
        cells_x,
        xmin,
        ymin,
        dx,
        dy,
        BLOCK: tl.constexpr,
    ):
        cell = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = cell < (ny - 1) * cells_x
        row = cell // cells_x
        column = cell - row * cells_x
        field_offset = row * nx + column
        f0 = tl.load(field + field_offset, mask=active, other=0.0)
        f1 = tl.load(field + field_offset + 1, mask=active, other=0.0)
        f2 = tl.load(field + field_offset + nx + 1, mask=active, other=0.0)
        f3 = tl.load(field + field_offset + nx, mask=active, other=0.0)
        case = (
            (f0 > 0.0).to(tl.int32)
            + 2 * (f1 > 0.0).to(tl.int32)
            + 4 * (f2 > 0.0).to(tl.int32)
            + 8 * (f3 > 0.0).to(tl.int32)
        )
        count = tl.where(
            (case == 0) | (case == 15),
            0,
            tl.where((case == 5) | (case == 10), 2, 1),
        )
        output_start = tl.load(offsets + cell, mask=active, other=0).to(tl.int64)
        center_positive = 0.25 * (f0 + f1 + f2 + f3) > 0.0
        case5_positive = (case == 5) & center_positive
        case5_negative = (case == 5) & ~center_positive
        case10_positive = (case == 10) & center_positive
        case10_negative = (case == 10) & ~center_positive
        x0 = xmin + column * dx
        x1 = x0 + dx
        y0 = ymin + row * dy
        y1 = y0 + dy
        for pair in tl.static_range(2):
            valid_pair = active & (pair < count)
            table = case * 2 + pair
            edge_a = tl.load(pair_a + table, mask=active, other=0).to(tl.int32)
            edge_b = tl.load(pair_b + table, mask=active, other=0).to(tl.int32)
            edge_a = tl.where(case5_positive, tl.load(case5_positive_a + pair), edge_a)
            edge_b = tl.where(case5_positive, tl.load(case5_positive_b + pair), edge_b)
            edge_a = tl.where(case5_negative, tl.load(case5_negative_a + pair), edge_a)
            edge_b = tl.where(case5_negative, tl.load(case5_negative_b + pair), edge_b)
            edge_a = tl.where(case10_positive, tl.load(case10_positive_a + pair), edge_a)
            edge_b = tl.where(case10_positive, tl.load(case10_positive_b + pair), edge_b)
            edge_a = tl.where(case10_negative, tl.load(case10_negative_a + pair), edge_a)
            edge_b = tl.where(case10_negative, tl.load(case10_negative_b + pair), edge_b)
            output = output_start + pair
            for endpoint in tl.static_range(2):
                edge = tl.where(endpoint == 0, edge_a, edge_b)
                af = tl.where(edge == 0, f0, tl.where(edge == 1, f1, tl.where(edge == 2, f2, f3)))
                bf = tl.where(edge == 0, f1, tl.where(edge == 1, f2, tl.where(edge == 2, f3, f0)))
                ax = tl.where((edge == 0) | (edge == 3), x0, x1)
                ay = tl.where((edge == 0) | (edge == 1), y0, y1)
                bx = tl.where((edge == 0) | (edge == 1), x1, x0)
                by = tl.where((edge == 1) | (edge == 2), y1, y0)
                denominator = af - bf
                safe = denominator + (denominator == 0.0).to(tl.float32) * 1.0e-12
                fraction = af / safe
                base = output * 4 + endpoint * 2
                tl.store(segments + base, ax + fraction * (bx - ax), mask=valid_pair)
                tl.store(segments + base + 1, ay + fraction * (by - ay), mask=valid_pair)


    @triton.jit
    def _sparse_marching_counts(
        f0, f1, f2, f3, counts, n_cells, BLOCK: tl.constexpr
    ):
        cell = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = cell < n_cells
        value0 = tl.load(f0 + cell, mask=active, other=0.0)
        value1 = tl.load(f1 + cell, mask=active, other=0.0)
        value2 = tl.load(f2 + cell, mask=active, other=0.0)
        value3 = tl.load(f3 + cell, mask=active, other=0.0)
        case = (
            (value0 > 0.0).to(tl.int32)
            + 2 * (value1 > 0.0).to(tl.int32)
            + 4 * (value2 > 0.0).to(tl.int32)
            + 8 * (value3 > 0.0).to(tl.int32)
        )
        count = tl.where(
            (case == 0) | (case == 15),
            0,
            tl.where((case == 5) | (case == 10), 2, 1),
        )
        tl.store(counts + cell, count, mask=active)


    @triton.jit
    def _sparse_marching_write(
        f0_ptr,
        f1_ptr,
        f2_ptr,
        f3_ptr,
        x0_ptr,
        x1_ptr,
        y0_ptr,
        y1_ptr,
        edge_boundary_ptr,
        offsets_ptr,
        pair_a_ptr,
        pair_b_ptr,
        positive_a_ptr,
        positive_b_ptr,
        negative_a_ptr,
        negative_b_ptr,
        segments_ptr,
        boundaries_ptr,
        segment_frames_ptr,
        n_cells,
        cells_per_frame,
        STORE_FRAMES: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        cell = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = cell < n_cells
        local_cell = cell % cells_per_frame
        f0 = tl.load(f0_ptr + cell, mask=active, other=0.0)
        f1 = tl.load(f1_ptr + cell, mask=active, other=0.0)
        f2 = tl.load(f2_ptr + cell, mask=active, other=0.0)
        f3 = tl.load(f3_ptr + cell, mask=active, other=0.0)
        case = (
            (f0 > 0.0).to(tl.int32)
            + 2 * (f1 > 0.0).to(tl.int32)
            + 4 * (f2 > 0.0).to(tl.int32)
            + 8 * (f3 > 0.0).to(tl.int32)
        )
        count = tl.where(
            (case == 0) | (case == 15),
            0,
            tl.where((case == 5) | (case == 10), 2, 1),
        )
        output_start = tl.load(offsets_ptr + cell, mask=active, other=0).to(
            tl.int64
        )
        center_positive = 0.25 * (f0 + f1 + f2 + f3) > 0.0
        use_positive = ((case == 5) & center_positive) | (
            (case == 10) & ~center_positive
        )
        use_negative = ((case == 5) & ~center_positive) | (
            (case == 10) & center_positive
        )
        x0 = tl.load(x0_ptr + local_cell, mask=active, other=0.0)
        x1 = tl.load(x1_ptr + local_cell, mask=active, other=0.0)
        y0 = tl.load(y0_ptr + local_cell, mask=active, other=0.0)
        y1 = tl.load(y1_ptr + local_cell, mask=active, other=0.0)
        for pair in tl.static_range(2):
            valid_pair = active & (pair < count)
            table = case * 2 + pair
            edge_a = tl.load(pair_a_ptr + table, mask=active, other=0).to(
                tl.int32
            )
            edge_b = tl.load(pair_b_ptr + table, mask=active, other=0).to(
                tl.int32
            )
            edge_a = tl.where(
                use_positive,
                tl.load(positive_a_ptr + pair),
                tl.where(use_negative, tl.load(negative_a_ptr + pair), edge_a),
            )
            edge_b = tl.where(
                use_positive,
                tl.load(positive_b_ptr + pair),
                tl.where(use_negative, tl.load(negative_b_ptr + pair), edge_b),
            )
            output = output_start + pair
            if STORE_FRAMES:
                tl.store(
                    segment_frames_ptr + output,
                    cell // cells_per_frame,
                    mask=valid_pair,
                )
            for endpoint in tl.static_range(2):
                edge = tl.where(endpoint == 0, edge_a, edge_b)
                af = tl.where(
                    edge == 0,
                    f0,
                    tl.where(edge == 1, f1, tl.where(edge == 2, f2, f3)),
                )
                bf = tl.where(
                    edge == 0,
                    f1,
                    tl.where(edge == 1, f2, tl.where(edge == 2, f3, f0)),
                )
                ax = tl.where((edge == 0) | (edge == 3), x0, x1)
                ay = tl.where((edge == 0) | (edge == 1), y0, y1)
                bx = tl.where((edge == 0) | (edge == 1), x1, x0)
                by = tl.where((edge == 1) | (edge == 2), y1, y0)
                denominator = af - bf
                safe = denominator + (denominator == 0.0).to(tl.float32) * EPS
                fraction = af / safe
                base = output * 4 + endpoint * 2
                tl.store(
                    segments_ptr + base,
                    ax + fraction * (bx - ax),
                    mask=valid_pair,
                )
                tl.store(
                    segments_ptr + base + 1,
                    ay + fraction * (by - ay),
                    mask=valid_pair,
                )
                boundary = tl.load(
                    edge_boundary_ptr
                    + edge.to(tl.int64) * cells_per_frame
                    + local_cell,
                    mask=valid_pair,
                    other=0,
                )
                tl.store(
                    boundaries_ptr + output * 2 + endpoint,
                    boundary,
                    mask=valid_pair,
                )


def triton_caustics_available() -> bool:
    """Whether the optional CUDA kernels can be launched."""

    return triton is not None and torch.cuda.is_available()


def regular_grid_winding_number_triton(
    segments,
    y_axis,
    x_axis,
    *,
    block_segments: int = 1024,
    num_warps: int = 4,
):
    """Return exact signed winding numbers on a CUDA float32 regular grid."""

    if not triton_caustics_available() or segments.device.type != "cuda":
        raise RuntimeError("Triton winding maps are unavailable")
    if segments.dtype != torch.float32:
        raise ValueError("Triton winding maps require float32")
    segments = segments.contiguous()
    y_axis = y_axis.to(device=segments.device, dtype=segments.dtype).contiguous()
    x_axis = x_axis.to(device=segments.device, dtype=segments.dtype).contiguous()
    ny, nx = int(y_axis.numel()), int(x_axis.numel())
    if nx > 65_536:
        raise ValueError("Triton winding maps support at most 65536 columns")
    difference = torch.zeros(
        (ny, nx + 1),
        device=segments.device,
        dtype=torch.int32,
    )
    if segments.shape[0]:
        blocks = triton.cdiv(int(segments.shape[0]), int(block_segments))
        search_steps = max(1, math.ceil(math.log2(nx + 1)))
        _regular_grid_winding_updates_kernel[(blocks, ny)](
            segments,
            y_axis,
            x_axis,
            difference,
            int(segments.shape[0]),
            ny,
            nx,
            BLOCK=int(block_segments),
            SEARCH_STEPS=search_steps,
            num_warps=int(num_warps),
        )
    return torch.cumsum(difference[:, :nx], dim=1, dtype=torch.int64)


def batched_caustic_crossings_distances_triton(
    segments,
    valid,
    anchors,
    crossing_points,
    distance_points,
    *,
    block_segments: int = 256,
):
    """Fuse finite-path crossing counts and nearest-distance reductions."""

    if not triton_caustics_available() or segments.device.type != "cuda":
        raise RuntimeError("Triton caustic labels are unavailable")
    if segments.dtype != torch.float32:
        raise ValueError("Triton caustic labels require float32")
    segments = segments.contiguous()
    valid = valid.to(device=segments.device, dtype=torch.bool).contiguous()
    anchors = anchors.to(device=segments.device, dtype=segments.dtype).contiguous()
    crossing_points = crossing_points.to(device=segments.device, dtype=segments.dtype).contiguous()
    distance_points = distance_points.to(device=segments.device, dtype=segments.dtype).contiguous()
    frames, segment_count = map(int, segments.shape[:2])
    anchor_count = int(anchors.shape[0])
    point_count = int(crossing_points.shape[0])
    distance_count = int(distance_points.shape[0])
    counts = torch.zeros(
        (frames, point_count, anchor_count),
        device=segments.device,
        dtype=torch.int32,
    )
    distance2 = torch.full(
        (frames, distance_count),
        float("inf"),
        device=segments.device,
        dtype=segments.dtype,
    )
    if segment_count == 0:
        return counts, torch.sqrt(distance2)
    blocks = triton.cdiv(segment_count, int(block_segments))
    programs = frames * point_count * anchor_count * blocks
    if programs:
        _crossing_kernel[(programs,)](
            segments,
            valid,
            anchors,
            crossing_points,
            counts,
            segment_count,
            anchor_count,
            point_count,
            blocks,
            BLOCK=int(block_segments),
            num_warps=4,
        )
    distance_programs = frames * distance_count * blocks
    if distance_programs:
        _distance_kernel[(distance_programs,)](
            segments,
            valid,
            distance_points,
            distance2,
            segment_count,
            distance_count,
            blocks,
            BLOCK=int(block_segments),
            num_warps=4,
        )
    return counts, torch.sqrt(distance2)


def caustic_distances_triton(
    segments,
    points,
    *,
    block_segments: int = 256,
):
    """Return exact point-to-segment distances with the CUDA reduction kernel.

    This is the unbatched diagnostic-map counterpart of
    :func:`batched_caustic_crossings_distances_triton`.  It avoids allocating
    dummy anchors and crossing outputs when only nearest-caustic distances are
    requested.
    """

    if not triton_caustics_available() or segments.device.type != "cuda":
        raise RuntimeError("Triton caustic distances are unavailable")
    if segments.dtype != torch.float32:
        raise ValueError("Triton caustic distances require float32")
    segments = segments.contiguous()
    points = points.to(device=segments.device, dtype=segments.dtype).contiguous()
    segment_count = int(segments.shape[0])
    point_count = int(points.shape[0])
    distance2 = torch.full(
        (point_count,),
        float("inf"),
        device=segments.device,
        dtype=segments.dtype,
    )
    if segment_count == 0 or point_count == 0:
        return torch.sqrt(distance2)
    blocks = triton.cdiv(segment_count, int(block_segments))
    _distance_kernel[(point_count * blocks,)](
        segments,
        torch.ones(segment_count, device=segments.device, dtype=torch.bool),
        points,
        distance2,
        segment_count,
        point_count,
        blocks,
        BLOCK=int(block_segments),
        num_warps=4,
    )
    return torch.sqrt(distance2)


_PAIR_A = (
    (0, 0), (3, 0), (0, 0), (3, 0), (1, 0), (0, 2), (0, 0), (3, 0),
    (2, 0), (2, 0), (0, 1), (2, 0), (1, 0), (1, 0), (0, 0), (0, 0),
)
_PAIR_B = (
    (0, 0), (0, 0), (1, 0), (1, 0), (2, 0), (1, 3), (2, 0), (2, 0),
    (3, 0), (0, 0), (3, 2), (1, 0), (3, 0), (0, 0), (3, 0), (0, 0),
)

_MARCHING_TABLE_CACHE: dict[tuple[str, int | None], tuple[torch.Tensor, ...]] = {}


def _marching_tables(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Reuse the tiny immutable lookup tables across every temporal frame."""

    key = (device.type, device.index)
    cached = _MARCHING_TABLE_CACHE.get(key)
    if cached is None:
        cached = (
            torch.tensor(_PAIR_A, device=device, dtype=torch.int32).reshape(-1),
            torch.tensor(_PAIR_B, device=device, dtype=torch.int32).reshape(-1),
            torch.tensor((1, 3), device=device, dtype=torch.int32),
            torch.tensor((0, 2), device=device, dtype=torch.int32),
            torch.tensor((3, 1), device=device, dtype=torch.int32),
            torch.tensor((0, 2), device=device, dtype=torch.int32),
            torch.tensor((0, 2), device=device, dtype=torch.int32),
            torch.tensor((3, 1), device=device, dtype=torch.int32),
            torch.tensor((0, 2), device=device, dtype=torch.int32),
            torch.tensor((1, 3), device=device, dtype=torch.int32),
        )
        _MARCHING_TABLE_CACHE[key] = cached
    return cached


def marching_squares_zero_triton(field: torch.Tensor, grid) -> torch.Tensor:
    """Extract compact zero-level segments directly from a regular grid."""

    if not triton_caustics_available() or field.device.type != "cuda":
        raise RuntimeError("Triton marching squares is unavailable")
    if field.dtype != torch.float32 or field.ndim != 2:
        raise ValueError("Triton marching squares requires one float32 2D field")
    field = field.contiguous()
    ny, nx = map(int, field.shape)
    cells = (ny - 1) * (nx - 1)
    counts = torch.empty(cells, device=field.device, dtype=torch.int32)
    block = 256
    _marching_counts[(triton.cdiv(cells, block),)](
        field,
        counts,
        ny,
        nx,
        nx - 1,
        BLOCK=block,
        num_warps=8,
    )
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int64) - counts
    total = int(counts.sum().item())
    segments = torch.empty((total, 2, 2), device=field.device, dtype=field.dtype)
    if total == 0:
        return segments
    (
        pair_a,
        pair_b,
        case5_positive_a,
        case5_positive_b,
        case5_negative_a,
        case5_negative_b,
        case10_positive_a,
        case10_positive_b,
        case10_negative_a,
        case10_negative_b,
    ) = _marching_tables(field.device)
    dy, dx = grid.pixel_scale_uas
    xmin, _, ymin, _ = grid.bounds_uas
    _marching_write[(triton.cdiv(cells, block),)](
        field,
        offsets,
        pair_a,
        pair_b,
        case5_positive_a,
        case5_positive_b,
        case5_negative_a,
        case5_negative_b,
        case10_positive_a,
        case10_positive_b,
        case10_negative_a,
        case10_negative_b,
        segments,
        ny,
        nx,
        nx - 1,
        float(xmin + 0.5 * dx),
        float(ymin + 0.5 * dy),
        float(dx),
        float(dy),
        BLOCK=block,
        num_warps=8,
    )
    return segments


def sparse_marching_squares_zero_triton(
    f0: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    f3: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    y0: torch.Tensor,
    y1: torch.Tensor,
    edge_boundary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact selected-cell zero contours without padded Torch candidates.

    Parameters are the four determinant values, physical bounds, and four
    neighbor-boundary flags for each selected cell.  This is the production
    CUDA counterpart of the vectorized portable sparse marcher.
    """

    if not triton_caustics_available() or f0.device.type != "cuda":
        raise RuntimeError("Triton sparse marching squares is unavailable")
    if f0.dtype != torch.float32:
        raise ValueError("Triton sparse marching squares requires float32")
    values = tuple(item.contiguous() for item in (f0, f1, f2, f3))
    coordinates = tuple(item.contiguous() for item in (x0, x1, y0, y1))
    edge_boundary = edge_boundary.to(
        device=f0.device, dtype=torch.bool
    ).contiguous()
    cells = int(f0.numel())
    counts = torch.empty(cells, device=f0.device, dtype=torch.int32)
    block = 256
    if cells:
        _sparse_marching_counts[(triton.cdiv(cells, block),)](
            *values,
            counts,
            cells,
            BLOCK=block,
            num_warps=8,
        )
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int64) - counts
    total = int(counts.sum().item())
    segments = torch.empty((total, 2, 2), device=f0.device, dtype=f0.dtype)
    boundaries = torch.empty((total, 2), device=f0.device, dtype=torch.bool)
    if total == 0:
        return segments, boundaries
    tables = _marching_tables(f0.device)
    pair_a, pair_b = tables[:2]
    # Reuse the full-grid ambiguous-case arrays with endpoint order reversed.
    # Segment orientation is immaterial, while the connected edge pairs are
    # exactly (0,1)/(2,3) and (0,3)/(1,2), respectively.
    positive_a, positive_b = tables[3], tables[2]
    negative_a, negative_b = tables[5], tables[4]
    _sparse_marching_write[(triton.cdiv(cells, block),)](
        *values,
        *coordinates,
        edge_boundary,
        offsets,
        pair_a,
        pair_b,
        positive_a,
        positive_b,
        negative_a,
        negative_b,
        segments,
        boundaries,
        counts,
        cells,
        cells,
        EPS=1.0e-12,
        BLOCK=block,
        STORE_FRAMES=False,
        num_warps=8,
    )
    return segments, boundaries


def batched_sparse_marching_squares_zero_triton(
    f0: torch.Tensor,
    f1: torch.Tensor,
    f2: torch.Tensor,
    f3: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    y0: torch.Tensor,
    y1: torch.Tensor,
    edge_boundary: torch.Tensor,
    *,
    return_flat: bool = False,
):
    """Compact a temporal batch sharing one selected-cell geometry."""

    if not triton_caustics_available() or f0.device.type != "cuda":
        raise RuntimeError("batched Triton sparse marching squares is unavailable")
    if f0.dtype != torch.float32 or f0.ndim != 2:
        raise ValueError("batched sparse marching requires CUDA float32 [B,N] fields")
    values = tuple(item.contiguous() for item in (f0, f1, f2, f3))
    frames, cells_per_frame = map(int, f0.shape)
    total_cells = frames * cells_per_frame
    block = 256
    counts = torch.empty(total_cells, device=f0.device, dtype=torch.int32)
    if total_cells:
        _sparse_marching_counts[(triton.cdiv(total_cells, block),)](
            *(item.reshape(-1) for item in values),
            counts,
            total_cells,
            BLOCK=block,
            num_warps=8,
        )
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int64) - counts
    frame_totals = counts.reshape(frames, cells_per_frame).sum(dim=1)
    totals_cpu = frame_totals.detach().cpu().tolist()
    total = int(sum(totals_cpu))
    segments = torch.empty((total, 2, 2), device=f0.device, dtype=f0.dtype)
    boundaries = torch.empty((total, 2), device=f0.device, dtype=torch.bool)
    frame_indices = torch.empty(total, device=f0.device, dtype=torch.int32)
    if total:
        tables = _marching_tables(f0.device)
        pair_a, pair_b = tables[:2]
        positive_a, positive_b = tables[3], tables[2]
        negative_a, negative_b = tables[5], tables[4]
        _sparse_marching_write[(triton.cdiv(total_cells, block),)](
            *(item.reshape(-1) for item in values),
            x0.contiguous(),
            x1.contiguous(),
            y0.contiguous(),
            y1.contiguous(),
            edge_boundary.to(device=f0.device, dtype=torch.bool).contiguous(),
            offsets,
            pair_a,
            pair_b,
            positive_a,
            positive_b,
            negative_a,
            negative_b,
            segments,
            boundaries,
            frame_indices,
            total_cells,
            cells_per_frame,
            EPS=1.0e-12,
            BLOCK=block,
            STORE_FRAMES=bool(return_flat),
            num_warps=8,
        )
    segment_frames = []
    boundary_frames = []
    start = 0
    for frame_total in totals_cpu:
        stop = start + int(frame_total)
        segment_frames.append(segments[start:stop])
        boundary_frames.append(boundaries[start:stop])
        start = stop
    rows = (tuple(segment_frames), tuple(boundary_frames))
    if return_flat:
        return (*rows, segments, boundaries, frame_indices, tuple(map(int, totals_cpu)))
    return rows
