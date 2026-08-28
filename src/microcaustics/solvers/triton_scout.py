"""Optional fused CUDA kernels for nested IPM source-region scouting."""

from __future__ import annotations

import torch

try:  # Triton is optional on CPU and Apple platforms.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on optional runtime
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _source_box_selection_kernel(
        corner_x_ptr,
        corner_y_ptr,
        center_x_ptr,
        center_y_ptr,
        selected_ptr,
        n_cells,
        COLUMNS: tl.constexpr,
        XMIN: tl.constexpr,
        XMAX: tl.constexpr,
        YMIN: tl.constexpr,
        YMAX: tl.constexpr,
        HAS_CENTER: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = linear < n_cells
        row = linear // COLUMNS
        column = linear - row * COLUMNS
        vertex_columns = COLUMNS + 1
        index00 = row * vertex_columns + column
        index01 = index00 + 1
        index10 = index00 + vertex_columns
        index11 = index10 + 1
        x00 = tl.load(corner_x_ptr + index00, mask=valid, other=0.0)
        x01 = tl.load(corner_x_ptr + index01, mask=valid, other=0.0)
        x10 = tl.load(corner_x_ptr + index10, mask=valid, other=0.0)
        x11 = tl.load(corner_x_ptr + index11, mask=valid, other=0.0)
        y00 = tl.load(corner_y_ptr + index00, mask=valid, other=0.0)
        y01 = tl.load(corner_y_ptr + index01, mask=valid, other=0.0)
        y10 = tl.load(corner_y_ptr + index10, mask=valid, other=0.0)
        y11 = tl.load(corner_y_ptr + index11, mask=valid, other=0.0)
        finite = (
            (x00 == x00)
            & (x01 == x01)
            & (x10 == x10)
            & (x11 == x11)
            & (y00 == y00)
            & (y01 == y01)
            & (y10 == y10)
            & (y11 == y11)
            & (tl.abs(x00) < 3.4028235e38)
            & (tl.abs(x01) < 3.4028235e38)
            & (tl.abs(x10) < 3.4028235e38)
            & (tl.abs(x11) < 3.4028235e38)
            & (tl.abs(y00) < 3.4028235e38)
            & (tl.abs(y01) < 3.4028235e38)
            & (tl.abs(y10) < 3.4028235e38)
            & (tl.abs(y11) < 3.4028235e38)
        )
        minimum_x = tl.minimum(tl.minimum(x00, x01), tl.minimum(x10, x11))
        maximum_x = tl.maximum(tl.maximum(x00, x01), tl.maximum(x10, x11))
        minimum_y = tl.minimum(tl.minimum(y00, y01), tl.minimum(y10, y11))
        maximum_y = tl.maximum(tl.maximum(y00, y01), tl.maximum(y10, y11))
        selected = (
            (minimum_x <= XMAX)
            & (maximum_x >= XMIN)
            & (minimum_y <= YMAX)
            & (maximum_y >= YMIN)
        )
        # A non-finite corner is retained conservatively. Such a cell contains
        # or directly neighbors a singular lens position and must not be
        # silently discarded by source-region selection.
        selected |= ~finite
        if HAS_CENTER:
            center_x = tl.load(center_x_ptr + linear, mask=valid, other=0.0)
            center_y = tl.load(center_y_ptr + linear, mask=valid, other=0.0)
            center_inside = (
                (center_x >= XMIN)
                & (center_x <= XMAX)
                & (center_y >= YMIN)
                & (center_y <= YMAX)
            )
            selected |= center_inside
        tl.store(selected_ptr + linear, selected, mask=valid)


    @triton.jit
    def _dilate_selection_kernel(
        input_ptr,
        output_ptr,
        n_cells,
        ROWS: tl.constexpr,
        COLUMNS: tl.constexpr,
        RADIUS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = linear < n_cells
        row = linear // COLUMNS
        column = linear - row * COLUMNS
        retained = tl.zeros((BLOCK,), tl.int1)
        for row_offset in range(-RADIUS, RADIUS + 1):
            neighbor_row = row + row_offset
            row_valid = (neighbor_row >= 0) & (neighbor_row < ROWS)
            for column_offset in range(-RADIUS, RADIUS + 1):
                neighbor_column = column + column_offset
                neighbor_valid = (
                    valid
                    & row_valid
                    & (neighbor_column >= 0)
                    & (neighbor_column < COLUMNS)
                )
                neighbor = neighbor_row * COLUMNS + neighbor_column
                retained |= tl.load(
                    input_ptr + neighbor,
                    mask=neighbor_valid,
                    other=0,
                ).to(tl.int1)
        tl.store(output_ptr + linear, retained, mask=valid)


def triton_scout_available() -> bool:
    """Whether fused CUDA source-scout kernels can be launched."""

    return bool(triton is not None and torch.cuda.is_available())


def select_source_tiles_triton(
    corner_x: torch.Tensor,
    corner_y: torch.Tensor,
    *,
    bounds: tuple[float, float, float, float],
    center_x: torch.Tensor | None = None,
    center_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select mapped cells overlapping a rectangular source-plane field."""

    if not triton_scout_available():
        raise RuntimeError("Triton source scouting is unavailable")
    if corner_x.shape != corner_y.shape or corner_x.ndim != 2:
        raise ValueError("corner arrays must share shape [rows+1, columns+1]")
    if corner_x.device.type != "cuda" or corner_x.dtype != torch.float32:
        raise ValueError("Triton source scouting requires CUDA float32")
    has_center = center_x is not None or center_y is not None
    if has_center and (center_x is None or center_y is None):
        raise ValueError("both center arrays must be supplied together")
    rows = int(corner_x.shape[0]) - 1
    columns = int(corner_x.shape[1]) - 1
    if rows < 1 or columns < 1:
        raise ValueError("corner arrays must describe at least one cell")
    if has_center and (
        center_x.shape != (rows, columns)
        or center_y.shape != (rows, columns)
    ):
        raise ValueError("center arrays must have shape [rows, columns]")
    corner_x = corner_x.contiguous()
    corner_y = corner_y.contiguous()
    placeholder = corner_x
    if has_center:
        center_x = center_x.contiguous()
        center_y = center_y.contiguous()
    else:
        center_x = placeholder
        center_y = placeholder
    selected = torch.empty((rows, columns), device=corner_x.device, dtype=torch.bool)
    count = rows * columns
    block = 256
    xmin, xmax, ymin, ymax = (float(value) for value in bounds)
    _source_box_selection_kernel[(triton.cdiv(count, block),)](
        corner_x,
        corner_y,
        center_x,
        center_y,
        selected,
        count,
        COLUMNS=columns,
        XMIN=xmin,
        XMAX=xmax,
        YMIN=ymin,
        YMAX=ymax,
        HAS_CENTER=has_center,
        BLOCK=block,
        num_warps=4,
    )
    return selected


def dilate_source_tiles_triton(mask: torch.Tensor, cells: int) -> torch.Tensor:
    """Dilate a CUDA boolean scout mask without a float pooling workspace."""

    cells = int(cells)
    if cells <= 0:
        return mask
    if cells > 8:
        raise ValueError("the fused Triton dilation supports at most eight cells")
    if mask.device.type != "cuda" or mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError("mask must be a 2D CUDA boolean tensor")
    rows, columns = (int(value) for value in mask.shape)
    result = torch.empty_like(mask)
    count = rows * columns
    block = 256
    _dilate_selection_kernel[(triton.cdiv(count, block),)](
        mask,
        result,
        count,
        ROWS=rows,
        COLUMNS=columns,
        RADIUS=cells,
        BLOCK=block,
        num_warps=4,
    )
    return result
