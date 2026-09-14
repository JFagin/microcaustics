"""Optional direct-cell Triton rasterization for mapped IPM cell lattices."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:  # Triton remains optional on CPU and Apple platforms.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on optional runtime
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _v4_basis(index, component: tl.constexpr):
        if component == 0:
            return tl.where(
                index == 0,
                1.0,
                tl.where(index == 1, 0.375, tl.where(index == 3, -0.125, 0.0)),
            )
        if component == 1:
            return tl.where(
                index == 1,
                0.75,
                tl.where(index == 2, 1.0, tl.where(index == 3, 0.75, 0.0)),
            )
        return tl.where(
            index == 3,
            0.375,
            tl.where(index == 4, 1.0, tl.where(index == 1, -0.125, 0.0)),
        )


    @triton.jit
    def _materialize_biquadratic_v4_kernel(
        raw_x,
        raw_y,
        out_x,
        out_y,
        n_cells,
        NODE_BLOCK: tl.constexpr,
    ):
        cell = tl.program_id(0)
        lane = tl.arange(0, NODE_BLOCK)
        valid = (cell < n_cells) & (lane < 25)
        node_i = lane // 5
        node_j = lane - node_i * 5
        value_x = 0.0
        value_y = 0.0
        for source_i in tl.static_range(3):
            weight_i = _v4_basis(node_i, component=source_i)
            for source_j in tl.static_range(3):
                weight = weight_i * _v4_basis(
                    node_j, component=source_j
                )
                source = cell * 9 + source_i * 3 + source_j
                value_x += weight * tl.load(raw_x + source)
                value_y += weight * tl.load(raw_y + source)
        output = cell * 25 + lane
        tl.store(out_x + output, value_x, mask=valid)
        tl.store(out_y + output, value_y, mask=valid)

    @triton.jit
    def _edge_x(x0, y0, x1, y1, y):
        dy = y1 - y0
        safe = tl.where(tl.abs(dy) > 1.1754944e-38, dy, 1.0)
        return x0 + (y - y0) * (x1 - x0) / safe


    @triton.jit
    def _pick_left(v0, v1, v2, a0, a1, a2):
        choose0 = (v0 <= v1) & (v0 <= v2)
        choose1 = (~choose0) & (v1 <= v2)
        return tl.where(choose0, a0, tl.where(choose1, a1, a2))


    @triton.jit
    def _pick_right(v0, v1, v2, a0, a1, a2):
        choose0 = (v0 >= v1) & (v0 >= v2)
        choose1 = (~choose0) & (v1 >= v2)
        return tl.where(choose0, a0, tl.where(choose1, a1, a2))


    @triton.jit
    def _positive_linear_integral(z0, z1, interval):
        both = (z0 >= 0.0) & (z1 >= 0.0)
        falling = (z0 > 0.0) & (z1 < 0.0)
        rising = (z0 < 0.0) & (z1 > 0.0)
        full = 0.5 * interval * (z0 + z1)
        fall = 0.5 * interval * z0 * z0 / tl.maximum(
            z0 - z1,
            1.1754944e-38,
        )
        rise = 0.5 * interval * z1 * z1 / tl.maximum(
            z1 - z0,
            1.1754944e-38,
        )
        return tl.where(
            both,
            full,
            tl.where(falling, fall, tl.where(rising, rise, 0.0)),
        )

    @triton.jit
    def _compact_active_triangles_kernel(
        node_x_ptr,
        node_y_ptr,
        active_cells_ptr,
        active_count_ptr,
        total_cells,
        XMIN: tl.constexpr,
        YMIN: tl.constexpr,
        INV_PIXEL_X: tl.constexpr,
        INV_PIXEL_Y: tl.constexpr,
        ROWS: tl.constexpr,
        COLUMNS: tl.constexpr,
        VIRTUAL_REFINEMENT: tl.constexpr,
        TRIANGLE_BLOCK: tl.constexpr,
    ):
        cell = tl.program_id(0)
        triangle = tl.arange(0, TRIANGLE_BLOCK)
        active_cell = cell < total_cells
        valid_triangle = triangle < 2 * VIRTUAL_REFINEMENT * VIRTUAL_REFINEMENT
        subcell = triangle // 2
        split = triangle - subcell * 2
        sub_i = subcell // VIRTUAL_REFINEMENT
        sub_j = subcell - sub_i * VIRTUAL_REFINEMENT
        node_side = VIRTUAL_REFINEMENT + 1
        base = cell * node_side * node_side
        index00 = base + sub_i * node_side + sub_j
        index10 = base + (sub_i + 1) * node_side + sub_j
        index11 = base + (sub_i + 1) * node_side + sub_j + 1
        index01 = base + sub_i * node_side + sub_j + 1
        ax = (tl.load(node_x_ptr + index00) - XMIN) * INV_PIXEL_X
        ay = (tl.load(node_y_ptr + index00) - YMIN) * INV_PIXEL_Y
        bx = (
            tl.where(
                split == 0,
                tl.load(node_x_ptr + index10),
                tl.load(node_x_ptr + index11),
            )
            - XMIN
        ) * INV_PIXEL_X
        by = (
            tl.where(
                split == 0,
                tl.load(node_y_ptr + index10),
                tl.load(node_y_ptr + index11),
            )
            - YMIN
        ) * INV_PIXEL_Y
        cx = (
            tl.where(
                split == 0,
                tl.load(node_x_ptr + index11),
                tl.load(node_x_ptr + index01),
            )
            - XMIN
        ) * INV_PIXEL_X
        cy = (
            tl.where(
                split == 0,
                tl.load(node_y_ptr + index11),
                tl.load(node_y_ptr + index01),
            )
            - YMIN
        ) * INV_PIXEL_Y
        twice_area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        minimum_x = tl.minimum(ax, tl.minimum(bx, cx))
        maximum_x = tl.maximum(ax, tl.maximum(bx, cx))
        minimum_y = tl.minimum(ay, tl.minimum(by, cy))
        maximum_y = tl.maximum(ay, tl.maximum(by, cy))
        finite = (
            (ax == ax)
            & (ay == ay)
            & (bx == bx)
            & (by == by)
            & (cx == cx)
            & (cy == cy)
            & (tl.abs(ax) < 3.4028235e38)
            & (tl.abs(ay) < 3.4028235e38)
            & (tl.abs(bx) < 3.4028235e38)
            & (tl.abs(by) < 3.4028235e38)
            & (tl.abs(cx) < 3.4028235e38)
            & (tl.abs(cy) < 3.4028235e38)
        )
        keep = (
            active_cell
            & valid_triangle
            & finite
            & (tl.abs(twice_area) > 1.1754944e-38)
            & (maximum_x > 0.0)
            & (minimum_x < COLUMNS)
            & (maximum_y > 0.0)
            & (minimum_y < ROWS)
        )
        has_source_triangle = tl.sum(keep.to(tl.int32), axis=0) > 0
        first = triangle == 0
        destination = tl.atomic_add(
            active_count_ptr + triangle * 0,
            1 + triangle * 0,
            mask=active_cell & has_source_triangle & first,
        )
        tl.store(
            active_cells_ptr + destination,
            cell.to(tl.int32),
            mask=active_cell & has_source_triangle & first,
        )


    @triton.jit
    def _direct_cell_kernel(
        node_x_ptr,
        node_y_ptr,
        active_cells_ptr,
        active_count_ptr,
        histogram_ptr,
        row_difference_ptr,
        total_cells,
        cells_per_frame,
        cell_frame_ptr,
        XMIN: tl.constexpr,
        YMIN: tl.constexpr,
        INV_PIXEL_X: tl.constexpr,
        INV_PIXEL_Y: tl.constexpr,
        ROWS: tl.constexpr,
        COLUMNS: tl.constexpr,
        TRIANGLE_MASS: tl.constexpr,
        TRIANGLE_BLOCK: tl.constexpr,
        VIRTUAL_REFINEMENT: tl.constexpr,
        COMPACT_ACTIVE: tl.constexpr,
        INDEXED_FRAMES: tl.constexpr,
    ):
        active_index = tl.program_id(0)
        if COMPACT_ACTIVE:
            active_limit = tl.load(active_count_ptr).to(tl.int32)
            active_cell = (active_index < total_cells) & (active_index < active_limit)
            cell = tl.load(
                active_cells_ptr + active_index,
                mask=active_cell,
                other=0,
            ).to(tl.int64)
        else:
            active_cell = active_index < total_cells
            cell = active_index
        if INDEXED_FRAMES:
            frame = tl.load(cell_frame_ptr + cell).to(tl.int64)
        else:
            frame = cell // cells_per_frame
        triangle = tl.arange(0, TRIANGLE_BLOCK)
        valid_triangle = triangle < 2 * VIRTUAL_REFINEMENT * VIRTUAL_REFINEMENT
        subcell = triangle // 2
        split = triangle - subcell * 2
        sub_i = subcell // VIRTUAL_REFINEMENT
        sub_j = subcell - sub_i * VIRTUAL_REFINEMENT
        node_side = VIRTUAL_REFINEMENT + 1
        base = cell * node_side * node_side
        index00 = base + sub_i * node_side + sub_j
        index10 = base + (sub_i + 1) * node_side + sub_j
        index11 = base + (sub_i + 1) * node_side + sub_j + 1
        index01 = base + sub_i * node_side + sub_j + 1
        ax = (tl.load(node_x_ptr + index00) - XMIN) * INV_PIXEL_X
        ay = (tl.load(node_y_ptr + index00) - YMIN) * INV_PIXEL_Y
        bx = (
            tl.where(
                split == 0,
                tl.load(node_x_ptr + index10),
                tl.load(node_x_ptr + index11),
            )
            - XMIN
        ) * INV_PIXEL_X
        by = (
            tl.where(
                split == 0,
                tl.load(node_y_ptr + index10),
                tl.load(node_y_ptr + index11),
            )
            - YMIN
        ) * INV_PIXEL_Y
        cx = (
            tl.where(
                split == 0,
                tl.load(node_x_ptr + index11),
                tl.load(node_x_ptr + index01),
            )
            - XMIN
        ) * INV_PIXEL_X
        cy = (
            tl.where(
                split == 0,
                tl.load(node_y_ptr + index11),
                tl.load(node_y_ptr + index01),
            )
            - YMIN
        ) * INV_PIXEL_Y

        twice_area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        minimum_x = tl.minimum(ax, tl.minimum(bx, cx))
        maximum_x = tl.maximum(ax, tl.maximum(bx, cx))
        minimum_y = tl.minimum(ay, tl.minimum(by, cy))
        maximum_y = tl.maximum(ay, tl.maximum(by, cy))
        finite = (
            (ax == ax)
            & (ay == ay)
            & (bx == bx)
            & (by == by)
            & (cx == cx)
            & (cy == cy)
            & (tl.abs(ax) < 3.4028235e38)
            & (tl.abs(ay) < 3.4028235e38)
            & (tl.abs(bx) < 3.4028235e38)
            & (tl.abs(by) < 3.4028235e38)
            & (tl.abs(cx) < 3.4028235e38)
            & (tl.abs(cy) < 3.4028235e38)
        )
        keep = (
            active_cell
            & valid_triangle
            & finite
            & (tl.abs(twice_area) > 1.1754944e-38)
            & (maximum_x > 0.0)
            & (minimum_x < COLUMNS)
            & (maximum_y > 0.0)
            & (minimum_y < ROWS)
        )
        density = TRIANGLE_MASS / tl.maximum(
            0.5 * tl.abs(twice_area),
            1.1754944e-38,
        )
        first_row = tl.maximum(
            0,
            tl.minimum(ROWS - 1, tl.floor(minimum_y).to(tl.int32)),
        )
        last_row = tl.maximum(
            0,
            tl.minimum(ROWS - 1, tl.floor(maximum_y).to(tl.int32)),
        )
        row_count = tl.where(keep, last_row - first_row + 1, 0)
        maximum_rows = tl.max(row_count, axis=0)

        for local_row in tl.range(0, maximum_rows, num_stages=1):
            row = first_row + local_row
            active_row = keep & (local_row < row_count)
            row0 = row.to(tl.float32)
            row1 = row0 + 1.0
            pair_low = tl.minimum(ay, by)
            pair_high = tl.maximum(ay, by)
            sorted_low = tl.minimum(pair_low, cy)
            sorted_high = tl.maximum(pair_high, cy)
            sorted_mid = tl.maximum(pair_low, tl.minimum(pair_high, cy))
            lower = tl.maximum(row0, sorted_low)
            upper = tl.minimum(row1, sorted_high)
            middle = tl.minimum(upper, tl.maximum(lower, sorted_mid))
            segment0_y0 = lower
            segment0_y1 = middle
            segment1_y0 = middle
            segment1_y1 = upper
            epsilon = 1.9073486328125e-6
            valid0 = active_row & (segment0_y1 > segment0_y0 + epsilon)
            valid1 = active_row & (segment1_y1 > segment1_y0 + epsilon)

            dy01 = by - ay
            dy12 = cy - by
            dy20 = ay - cy
            edge01 = tl.abs(dy01) > 1.1754944e-38
            edge12 = tl.abs(dy12) > 1.1754944e-38
            edge20 = tl.abs(dy20) > 1.1754944e-38
            min01, max01 = tl.minimum(ay, by), tl.maximum(ay, by)
            min12, max12 = tl.minimum(by, cy), tl.maximum(by, cy)
            min20, max20 = tl.minimum(cy, ay), tl.maximum(cy, ay)

            mid0 = 0.5 * (segment0_y0 + segment0_y1)
            e00 = edge01 & (mid0 >= min01) & (mid0 <= max01)
            e01 = edge12 & (mid0 >= min12) & (mid0 <= max12)
            e02 = edge20 & (mid0 >= min20) & (mid0 <= max20)
            x00a = _edge_x(ax, ay, bx, by, segment0_y0)
            x00b = _edge_x(ax, ay, bx, by, segment0_y1)
            x01a = _edge_x(bx, by, cx, cy, segment0_y0)
            x01b = _edge_x(bx, by, cx, cy, segment0_y1)
            x02a = _edge_x(cx, cy, ax, ay, segment0_y0)
            x02b = _edge_x(cx, cy, ax, ay, segment0_y1)
            xm00 = tl.where(e00, 0.5 * (x00a + x00b), float("inf"))
            xm01 = tl.where(e01, 0.5 * (x01a + x01b), float("inf"))
            xm02 = tl.where(e02, 0.5 * (x02a + x02b), float("inf"))
            xr00 = tl.where(e00, 0.5 * (x00a + x00b), -float("inf"))
            xr01 = tl.where(e01, 0.5 * (x01a + x01b), -float("inf"))
            xr02 = tl.where(e02, 0.5 * (x02a + x02b), -float("inf"))
            left00 = _pick_left(xm00, xm01, xm02, x00a, x01a, x02a)
            left01 = _pick_left(xm00, xm01, xm02, x00b, x01b, x02b)
            right00 = _pick_right(xr00, xr01, xr02, x00a, x01a, x02a)
            right01 = _pick_right(xr00, xr01, xr02, x00b, x01b, x02b)
            valid0 &= (e00.to(tl.int32) + e01.to(tl.int32) + e02.to(tl.int32)) >= 2

            mid1 = 0.5 * (segment1_y0 + segment1_y1)
            e10 = edge01 & (mid1 >= min01) & (mid1 <= max01)
            e11 = edge12 & (mid1 >= min12) & (mid1 <= max12)
            e12 = edge20 & (mid1 >= min20) & (mid1 <= max20)
            x10a = _edge_x(ax, ay, bx, by, segment1_y0)
            x10b = _edge_x(ax, ay, bx, by, segment1_y1)
            x11a = _edge_x(bx, by, cx, cy, segment1_y0)
            x11b = _edge_x(bx, by, cx, cy, segment1_y1)
            x12a = _edge_x(cx, cy, ax, ay, segment1_y0)
            x12b = _edge_x(cx, cy, ax, ay, segment1_y1)
            xm10 = tl.where(e10, 0.5 * (x10a + x10b), float("inf"))
            xm11 = tl.where(e11, 0.5 * (x11a + x11b), float("inf"))
            xm12 = tl.where(e12, 0.5 * (x12a + x12b), float("inf"))
            xr10 = tl.where(e10, 0.5 * (x10a + x10b), -float("inf"))
            xr11 = tl.where(e11, 0.5 * (x11a + x11b), -float("inf"))
            xr12 = tl.where(e12, 0.5 * (x12a + x12b), -float("inf"))
            left10 = _pick_left(xm10, xm11, xm12, x10a, x11a, x12a)
            left11 = _pick_left(xm10, xm11, xm12, x10b, x11b, x12b)
            right10 = _pick_right(xr10, xr11, xr12, x10a, x11a, x12a)
            right11 = _pick_right(xr10, xr11, xr12, x10b, x11b, x12b)
            valid1 &= (e10.to(tl.int32) + e11.to(tl.int32) + e12.to(tl.int32)) >= 2

            strip_min = tl.minimum(
                tl.where(valid0, tl.minimum(left00, left01), float("inf")),
                tl.where(valid1, tl.minimum(left10, left11), float("inf")),
            )
            strip_max = tl.maximum(
                tl.where(valid0, tl.maximum(right00, right01), -float("inf")),
                tl.where(valid1, tl.maximum(right10, right11), -float("inf")),
            )
            valid_row = (
                active_row
                & (valid0 | valid1)
                & (strip_max >= 0.0)
                & (strip_min < COLUMNS)
            )
            safe_min = tl.where(
                valid_row,
                tl.minimum(float(COLUMNS), tl.maximum(0.0, strip_min)),
                0.0,
            )
            safe_max = tl.where(
                valid_row,
                tl.minimum(float(COLUMNS), tl.maximum(0.0, strip_max)),
                0.0,
            )
            bbox0 = tl.maximum(
                0,
                tl.minimum(COLUMNS - 1, tl.floor(safe_min).to(tl.int32)),
            )
            bbox1 = tl.maximum(
                0,
                tl.minimum(COLUMNS - 1, tl.floor(safe_max).to(tl.int32)),
            )
            covered0 = tl.minimum(
                tl.where(valid0, segment0_y0, float("inf")),
                tl.where(valid1, segment1_y0, float("inf")),
            )
            covered1 = tl.maximum(
                tl.where(valid0, segment0_y1, -float("inf")),
                tl.where(valid1, segment1_y1, -float("inf")),
            )
            left_limit = tl.maximum(
                tl.where(valid0, tl.maximum(left00, left01), -float("inf")),
                tl.where(valid1, tl.maximum(left10, left11), -float("inf")),
            )
            right_limit = tl.minimum(
                tl.where(valid0, tl.minimum(right00, right01), float("inf")),
                tl.where(valid1, tl.minimum(right10, right11), float("inf")),
            )
            full_epsilon = 3.814697265625e-6
            raw_first = tl.ceil(left_limit - full_epsilon).to(tl.int32)
            raw_last = tl.floor(right_limit + full_epsilon).to(tl.int32) - 1
            full0 = tl.maximum(
                bbox0,
                tl.maximum(0, tl.minimum(COLUMNS - 1, raw_first)),
            )
            full1 = tl.minimum(
                bbox1,
                tl.maximum(0, tl.minimum(COLUMNS - 1, raw_last)),
            )
            full = (
                valid_row
                & (covered0 <= row0 + full_epsilon)
                & (covered1 >= row1 - full_epsilon)
                & (right_limit > left_limit)
                & (raw_first <= bbox1)
                & (raw_last >= bbox0)
                & (full1 >= full0)
            )
            full_count = tl.where(full, full1 - full0 + 1, 0)
            boundary_count = tl.where(
                valid_row,
                bbox1 - bbox0 + 1 - full_count,
                0,
            )
            row_base = (
                frame.to(tl.int64) * ROWS * (COLUMNS + 1)
                + row.to(tl.int64) * (COLUMNS + 1)
            )
            tl.atomic_add(
                row_difference_ptr + row_base + full0,
                density,
                mask=full,
            )
            tl.atomic_add(
                row_difference_ptr + row_base + full1 + 1,
                -density,
                mask=full,
            )
            left_count = tl.where(full, full0 - bbox0, bbox1 - bbox0 + 1)
            maximum_boundary = tl.max(boundary_count, axis=0)
            for boundary_lane in tl.range(0, maximum_boundary, num_stages=1):
                column = tl.where(
                    boundary_lane < left_count,
                    bbox0 + boundary_lane,
                    full1 + 1 + boundary_lane - left_count,
                )
                column = tl.where(full, column, bbox0 + boundary_lane)
                xlo = column.to(tl.float32)
                xhi = xlo + 1.0
                interval0 = tl.maximum(0.0, segment0_y1 - segment0_y0)
                interval1 = tl.maximum(0.0, segment1_y1 - segment1_y0)
                area0 = (
                    _positive_linear_integral(right00 - xlo, right01 - xlo, interval0)
                    - _positive_linear_integral(right00 - xhi, right01 - xhi, interval0)
                    - _positive_linear_integral(left00 - xlo, left01 - xlo, interval0)
                    + _positive_linear_integral(left00 - xhi, left01 - xhi, interval0)
                )
                area1 = (
                    _positive_linear_integral(right10 - xlo, right11 - xlo, interval1)
                    - _positive_linear_integral(right10 - xhi, right11 - xhi, interval1)
                    - _positive_linear_integral(left10 - xlo, left11 - xlo, interval1)
                    + _positive_linear_integral(left10 - xhi, left11 - xhi, interval1)
                )
                area = tl.where(valid0, area0, 0.0) + tl.where(valid1, area1, 0.0)
                histogram_index = (
                    frame.to(tl.int64) * ROWS * COLUMNS
                    + row.to(tl.int64) * COLUMNS
                    + column.to(tl.int64)
                )
                tl.atomic_add(
                    histogram_ptr + histogram_index,
                    density * tl.maximum(area, 0.0),
                    mask=valid_row & (boundary_lane < boundary_count),
                )


def triton_ipm_available() -> bool:
    """Whether the CUDA float32 direct-cell rasterizer can be launched."""

    return bool(triton is not None and torch.cuda.is_available())


def materialize_biquadratic_v4_triton(
    node_x: torch.Tensor,
    node_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a 5×5 quadratic lattice from mapped 3×3 r=2 nodes."""

    if not triton_ipm_available():
        raise RuntimeError("Triton IPM interpolation is unavailable")
    if node_x.shape != node_y.shape or tuple(node_x.shape[-2:]) != (3, 3):
        raise ValueError("biquadratic v=4 interpolation requires 3×3 node grids")
    if node_x.device.type != "cuda" or node_x.dtype != torch.float32:
        raise ValueError("Triton IPM interpolation requires CUDA float32")
    leading = node_x.shape[:-2]
    n_cells = int(node_x.numel() // 9)
    raw_x = node_x.reshape(n_cells, 3, 3).contiguous()
    raw_y = node_y.reshape(n_cells, 3, 3).contiguous()
    output_x = torch.empty(
        (n_cells, 5, 5), dtype=node_x.dtype, device=node_x.device
    )
    output_y = torch.empty_like(output_x)
    if n_cells:
        _materialize_biquadratic_v4_kernel[(n_cells,)](
            raw_x,
            raw_y,
            output_x,
            output_y,
            n_cells,
            NODE_BLOCK=32,
            num_warps=1,
            num_stages=1,
        )
    return (
        output_x.reshape(*leading, 5, 5),
        output_y.reshape(*leading, 5, 5),
    )


@dataclass
class TritonRasterWorkspace:
    """Persistent accumulation buffers for a chunked Triton IPM map.

    Reusing these buffers across cell chunks avoids materializing triangle
    queues and avoids a full-map prefix sum after every chunk.
    """

    histogram: torch.Tensor
    row_difference: torch.Tensor

    @classmethod
    def create(
        cls,
        shape: tuple[int, int],
        *,
        device: torch.device,
        frames: int = 1,
    ) -> TritonRasterWorkspace:
        """Allocate zeroed float32 buffers for one source-plane map."""

        rows, columns = (int(value) for value in shape)
        if rows < 1 or columns < 1:
            raise ValueError("map shape must be positive")
        frames = int(frames)
        if frames < 1:
            raise ValueError("frames must be positive")
        leading = () if frames == 1 else (frames,)
        return cls(
            histogram=torch.zeros(
                (*leading, rows, columns), device=device, dtype=torch.float32
            ),
            row_difference=torch.zeros(
                (*leading, rows, columns + 1), device=device, dtype=torch.float32
            ),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the source-map shape represented by this workspace."""

        return tuple(int(value) for value in self.histogram.shape[-2:])

    @property
    def frames(self) -> int:
        """Number of independent temporal maps in the workspace."""

        return 1 if self.histogram.ndim == 2 else int(self.histogram.shape[0])

    def result(self) -> torch.Tensor:
        """Materialize the accumulated map by applying row prefix sums."""

        return self.histogram + torch.cumsum(
            self.row_difference[..., :-1],
            dim=-1,
        )


def accumulate_cells_triton(
    workspace: TritonRasterWorkspace,
    node_x: torch.Tensor,
    node_y: torch.Tensor,
    *,
    xmin: float,
    ymin: float,
    pixel_size_x: float,
    pixel_size_y: float,
    lens_area_per_triangle_uas2: float,
    cell_frame_index: torch.Tensor | None = None,
) -> None:
    """Add one chunk of mapped cell lattices to a persistent workspace."""

    if not triton_ipm_available():
        raise RuntimeError("Triton IPM rasterization is unavailable")
    if node_x.device.type != "cuda" or node_x.dtype != torch.float32:
        raise ValueError("Triton IPM rasterization requires CUDA float32")
    if node_x.shape != node_y.shape or node_x.ndim not in (3, 4):
        raise ValueError(
            "node arrays must share shape [cell, v+1, v+1] or "
            "[frame, cell, v+1, v+1]"
        )
    if node_x.shape[-2] != node_x.shape[-1] or node_x.shape[-1] < 2:
        raise ValueError("node lattices must be square with at least two nodes")
    if workspace.histogram.device != node_x.device:
        raise ValueError("workspace and mapped nodes must share a CUDA device")
    temporal_frames = 1 if node_x.ndim == 3 else int(node_x.shape[0])
    cells_per_frame = int(node_x.shape[-3])
    indexed_frames = cell_frame_index is not None
    if indexed_frames:
        if node_x.ndim != 3:
            raise ValueError("indexed raster queues must have shape [cell, v+1, v+1]")
        cell_frame_index = torch.as_tensor(
            cell_frame_index, device=node_x.device, dtype=torch.int32
        ).reshape(-1).contiguous()
        if cell_frame_index.numel() != cells_per_frame:
            raise ValueError("cell_frame_index must contain one entry per cell")
        if cell_frame_index.numel() and bool(
            torch.any((cell_frame_index < 0) | (cell_frame_index >= workspace.frames))
        ):
            raise ValueError("cell_frame_index entries are outside the workspace")
        temporal_frames = workspace.frames
    elif workspace.frames != temporal_frames:
        raise ValueError("workspace and mapped nodes must have the same frame count")
    virtual_refinement = int(node_x.shape[-1]) - 1
    if virtual_refinement > 16:
        raise ValueError("the Triton direct-cell kernel currently supports v <= 16")
    if float(pixel_size_x) <= 0 or float(pixel_size_y) <= 0:
        raise ValueError("pixel sizes must be positive")
    node_x = node_x.contiguous()
    node_y = node_y.contiguous()
    rows, columns = workspace.shape
    source_pixel_area = float(pixel_size_x) * float(pixel_size_y)
    triangle_count = 2 * virtual_refinement * virtual_refinement
    triangle_block = triton.next_power_of_2(triangle_count)
    total_cells = cells_per_frame if indexed_frames else temporal_frames * cells_per_frame
    frame_ptr = (
        cell_frame_index
        if indexed_frames
        else torch.empty(1, device=node_x.device, dtype=torch.int32)
    )
    active_cells = torch.empty(
        total_cells, device=node_x.device, dtype=torch.int32
    )
    active_count = torch.zeros((), device=node_x.device, dtype=torch.int32)
    _compact_active_triangles_kernel[(total_cells,)](
        node_x,
        node_y,
        active_cells,
        active_count,
        total_cells,
        XMIN=float(xmin),
        YMIN=float(ymin),
        INV_PIXEL_X=1.0 / float(pixel_size_x),
        INV_PIXEL_Y=1.0 / float(pixel_size_y),
        ROWS=rows,
        COLUMNS=columns,
        VIRTUAL_REFINEMENT=virtual_refinement,
        TRIANGLE_BLOCK=triangle_block,
        num_warps=1,
        num_stages=1,
    )
    _direct_cell_kernel[(total_cells,)](
        node_x,
        node_y,
        active_cells,
        active_count,
        workspace.histogram,
        workspace.row_difference,
        total_cells,
        cells_per_frame,
        frame_ptr,
        XMIN=float(xmin),
        YMIN=float(ymin),
        INV_PIXEL_X=1.0 / float(pixel_size_x),
        INV_PIXEL_Y=1.0 / float(pixel_size_y),
        ROWS=rows,
        COLUMNS=columns,
        TRIANGLE_MASS=float(lens_area_per_triangle_uas2) / source_pixel_area,
        TRIANGLE_BLOCK=triangle_block,
        VIRTUAL_REFINEMENT=virtual_refinement,
        COMPACT_ACTIVE=True,
        INDEXED_FRAMES=indexed_frames,
        num_warps=max(1, min(8, triangle_block // 32)),
        num_stages=1,
    )


def rasterize_cells_triton(
    node_x: torch.Tensor,
    node_y: torch.Tensor,
    *,
    xmin: float,
    ymin: float,
    pixel_size_x: float,
    pixel_size_y: float,
    shape: tuple[int, int],
    lens_area_per_triangle_uas2: float,
) -> torch.Tensor:
    """Rasterize mapped ``(v+1)×(v+1)`` cell lattices into a map.

    The virtual refinement ``v`` is inferred from the node shape. Current
    Triton launch geometry accelerates ``1 <= v <= 16``. Larger lattices remain
    supported by the portable rasterizer used by the public IPM dispatcher.
    """

    workspace = TritonRasterWorkspace.create(
        shape,
        device=node_x.device,
    )
    accumulate_cells_triton(
        workspace,
        node_x,
        node_y,
        xmin=xmin,
        ymin=ymin,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        lens_area_per_triangle_uas2=lens_area_per_triangle_uas2,
    )
    return workspace.result()
