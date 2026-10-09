"""Device-portable direct-cell IPM rasterization in batched PyTorch.

This is a tensor port of the Triton direct-cell kernel in ``triton_ipm`` for
accelerators without Triton, such as Apple MPS. Mapped triangles are expanded
into (triangle, row) strips. Columns a strip covers completely are written as
two entries of a row-difference buffer, and only boundary pixels evaluate the
exact strip-pixel overlap. A final row prefix sum materializes the map, so
large caustic-stretched triangles cost a few entries per row rather than one
per covered pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_TINY = 1.1754944e-38
_SEGMENT_EPSILON = 1.9073486328125e-6
_FULL_EPSILON = 3.814697265625e-6


@dataclass
class TorchRasterWorkspace:
    """Persistent accumulation buffers for a chunked tensor IPM map."""

    histogram: torch.Tensor
    row_difference: torch.Tensor

    @classmethod
    def create(
        cls,
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        frames: int = 1,
    ) -> TorchRasterWorkspace:
        """Allocate zeroed buffers for one map, or ``frames`` temporal maps."""

        rows, columns = (int(value) for value in shape)
        if rows < 1 or columns < 1:
            raise ValueError("map shape must be positive")
        frames = int(frames)
        if frames < 1:
            raise ValueError("frames must be positive")
        leading = () if frames == 1 else (frames,)
        return cls(
            histogram=torch.zeros(
                (*leading, rows, columns), device=device, dtype=dtype
            ),
            row_difference=torch.zeros(
                (*leading, rows, columns + 1), device=device, dtype=dtype
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

        return self.histogram + torch.cumsum(self.row_difference[..., :-1], dim=-1)


def _edge_x(x0, y0, x1, y1, y):
    dy = y1 - y0
    safe = torch.where(dy.abs() > _TINY, dy, torch.ones_like(dy))
    return x0 + (y - y0) * (x1 - x0) / safe


def _pick_left(v0, v1, v2, a0, a1, a2):
    choose0 = (v0 <= v1) & (v0 <= v2)
    choose1 = (~choose0) & (v1 <= v2)
    return torch.where(choose0, a0, torch.where(choose1, a1, a2))


def _pick_right(v0, v1, v2, a0, a1, a2):
    choose0 = (v0 >= v1) & (v0 >= v2)
    choose1 = (~choose0) & (v1 >= v2)
    return torch.where(choose0, a0, torch.where(choose1, a1, a2))


def _positive_linear_integral(z0, z1, interval):
    """Integrate ``max(0, z)`` for ``z`` linear from ``z0`` to ``z1``."""

    both = (z0 >= 0.0) & (z1 >= 0.0)
    falling = (z0 > 0.0) & (z1 < 0.0)
    rising = (z0 < 0.0) & (z1 > 0.0)
    full = 0.5 * interval * (z0 + z1)
    fall = 0.5 * interval * z0 * z0 / (z0 - z1).clamp_min(_TINY)
    rise = 0.5 * interval * z1 * z1 / (z1 - z0).clamp_min(_TINY)
    zero = torch.zeros_like(full)
    return torch.where(
        both, full, torch.where(falling, fall, torch.where(rising, rise, zero))
    )


def _segment_edges(ax, ay, bx, by, cx, cy, y0, y1, valid):
    """Left and right triangle boundaries at both ends of one row segment."""

    infinity = torch.full_like(y0, float("inf"))
    middle = 0.5 * (y0 + y1)
    present = []
    lower_x = []
    upper_x = []
    for x0, ey0, x1, ey1 in ((ax, ay, bx, by), (bx, by, cx, cy), (cx, cy, ax, ay)):
        edge = (
            ((ey1 - ey0).abs() > _TINY)
            & (middle >= torch.minimum(ey0, ey1))
            & (middle <= torch.maximum(ey0, ey1))
        )
        present.append(edge)
        lower_x.append(_edge_x(x0, ey0, x1, ey1, y0))
        upper_x.append(_edge_x(x0, ey0, x1, ey1, y1))
    centers = [0.5 * (low + high) for low, high in zip(lower_x, upper_x, strict=True)]
    left_key = [torch.where(e, c, infinity) for e, c in zip(present, centers, strict=True)]
    right_key = [
        torch.where(e, c, -infinity) for e, c in zip(present, centers, strict=True)
    ]
    left0 = _pick_left(*left_key, *lower_x)
    left1 = _pick_left(*left_key, *upper_x)
    right0 = _pick_right(*right_key, *lower_x)
    right1 = _pick_right(*right_key, *upper_x)
    edges = present[0].int() + present[1].int() + present[2].int()
    return left0, left1, right0, right1, valid & (edges >= 2)


def _strip_geometry(row, ax, ay, bx, by, cx, cy, columns: int):
    """Elementwise strip geometry: segments, edges, bounding and full columns."""

    row0 = row.to(ax.dtype)
    row1 = row0 + 1.0
    pair_low = torch.minimum(ay, by)
    pair_high = torch.maximum(ay, by)
    sorted_low = torch.minimum(pair_low, cy)
    sorted_high = torch.maximum(pair_high, cy)
    sorted_mid = torch.maximum(pair_low, torch.minimum(pair_high, cy))
    lower = torch.maximum(row0, sorted_low)
    upper = torch.minimum(row1, sorted_high)
    middle = torch.minimum(upper, torch.maximum(lower, sorted_mid))
    valid0 = middle > lower + _SEGMENT_EPSILON
    valid1 = upper > middle + _SEGMENT_EPSILON
    left00, left01, right00, right01, valid0 = _segment_edges(
        ax, ay, bx, by, cx, cy, lower, middle, valid0
    )
    left10, left11, right10, right11, valid1 = _segment_edges(
        ax, ay, bx, by, cx, cy, middle, upper, valid1
    )

    infinity = torch.full_like(row0, float("inf"))
    strip_min = torch.minimum(
        torch.where(valid0, torch.minimum(left00, left01), infinity),
        torch.where(valid1, torch.minimum(left10, left11), infinity),
    )
    strip_max = torch.maximum(
        torch.where(valid0, torch.maximum(right00, right01), -infinity),
        torch.where(valid1, torch.maximum(right10, right11), -infinity),
    )
    valid_row = (valid0 | valid1) & (strip_max >= 0.0) & (strip_min < columns)
    zero = torch.zeros_like(row0)
    safe_min = torch.where(valid_row, strip_min.clamp(0.0, float(columns)), zero)
    safe_max = torch.where(valid_row, strip_max.clamp(0.0, float(columns)), zero)
    bbox0 = torch.floor(safe_min).long().clamp(0, columns - 1)
    bbox1 = torch.floor(safe_max).long().clamp(0, columns - 1)
    covered0 = torch.minimum(
        torch.where(valid0, lower, infinity), torch.where(valid1, middle, infinity)
    )
    covered1 = torch.maximum(
        torch.where(valid0, middle, -infinity), torch.where(valid1, upper, -infinity)
    )
    left_limit = torch.maximum(
        torch.where(valid0, torch.maximum(left00, left01), -infinity),
        torch.where(valid1, torch.maximum(left10, left11), -infinity),
    )
    right_limit = torch.minimum(
        torch.where(valid0, torch.minimum(right00, right01), infinity),
        torch.where(valid1, torch.minimum(right10, right11), infinity),
    )
    # Infinite limits only occur on rows without a valid segment; clamp them
    # before the integer conversion and let ``full`` reject those rows.
    raw_first = torch.ceil(
        (left_limit - _FULL_EPSILON).clamp(-1.0, columns + 1.0)
    ).long()
    raw_last = (
        torch.floor((right_limit + _FULL_EPSILON).clamp(-1.0, columns + 1.0)).long()
        - 1
    )
    full0 = torch.maximum(bbox0, raw_first.clamp(0, columns - 1))
    full1 = torch.minimum(bbox1, raw_last.clamp(0, columns - 1))
    full = (
        valid_row
        & (covered0 <= row0 + _FULL_EPSILON)
        & (covered1 >= row1 - _FULL_EPSILON)
        & (right_limit > left_limit)
        & (raw_first <= bbox1)
        & (raw_last >= bbox0)
        & (full1 >= full0)
    )
    full_count = torch.where(full, full1 - full0 + 1, torch.zeros_like(full0))
    boundary_count = torch.where(
        valid_row, bbox1 - bbox0 + 1 - full_count, torch.zeros_like(full0)
    )
    # Columns left of the full run come first, then those right of it.
    left_count = torch.where(full, full0 - bbox0, bbox1 - bbox0 + 1)
    first_segment = torch.stack(
        (left00, left01, right00, right01, (middle - lower).clamp_min(0.0))
    )
    second_segment = torch.stack(
        (left10, left11, right10, right11, (upper - middle).clamp_min(0.0))
    )
    return (
        first_segment,
        second_segment,
        valid0,
        valid1,
        full,
        full0,
        full1,
        bbox0,
        boundary_count,
        left_count,
    )


def _boundary_area(lane, left_count, bbox0, full, full1, first, second, valid0, valid1):
    """Exact overlap of one strip with one boundary pixel, and that pixel's column."""

    column = torch.where(
        full & (lane >= left_count), full1 + 1 + lane - left_count, bbox0 + lane
    )
    xlo = column.to(first.dtype)
    xhi = xlo + 1.0

    def overlap(segment, valid):
        left0, left1, right0, right1, interval = segment
        area = (
            _positive_linear_integral(right0 - xlo, right1 - xlo, interval)
            - _positive_linear_integral(right0 - xhi, right1 - xhi, interval)
            - _positive_linear_integral(left0 - xlo, left1 - xlo, interval)
            + _positive_linear_integral(left0 - xhi, left1 - xhi, interval)
        )
        return torch.where(valid, area, torch.zeros_like(area))

    return column, (overlap(first, valid0) + overlap(second, valid1)).clamp_min(0.0)


_COMPILED: dict[str, object] = {}


def _kernel(name: str, function, device: torch.device):
    """Fuse the elementwise strip math with ``torch.compile`` on accelerators.

    Dozens of small elementwise launches dominate otherwise. A failed
    compilation falls back to eager PyTorch for the rest of the process.
    """

    if device.type == "cpu":
        return function
    compiled = _COMPILED.get(name)
    if compiled is None:
        compiled = torch.compile(function, dynamic=True)
        _COMPILED[name] = compiled
    if compiled is function:
        return function

    def run(*arguments):
        try:
            return compiled(*arguments)
        except Exception as error:  # pragma: no cover - depends on the toolchain
            from ..runtime import warn_backend_fallback

            warn_backend_fallback(f"compiled tensor IPM {name}", error)
            _COMPILED[name] = function
            return function(*arguments)

    return run


def _rasterize_strips(
    histogram: torch.Tensor,
    row_difference: torch.Tensor,
    frame: torch.Tensor,
    ax, ay, bx, by, cx, cy,
    density: torch.Tensor,
    first_row: torch.Tensor,
    row_count: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> None:
    """Deposit kept triangles, already in pixel units, one row strip at a time."""

    device = ax.device
    owner = torch.repeat_interleave(
        torch.arange(ax.numel(), device=device), row_count
    )
    if owner.numel() == 0:
        return
    starts = torch.cumsum(row_count, 0) - row_count
    row = first_row[owner] + (
        torch.arange(owner.numel(), device=device) - starts[owner]
    )
    density = density[owner]
    frame = frame[owner]
    (
        first,
        second,
        valid0,
        valid1,
        full,
        full0,
        full1,
        bbox0,
        boundary_count,
        left_count,
    ) = _kernel("strip geometry", _strip_geometry, device)(
        row, *(value[owner] for value in (ax, ay, bx, by, cx, cy)), columns
    )

    difference = row_difference.reshape(-1)
    row_base = (frame * rows + row) * (columns + 1)
    full_density = density[full]
    difference.index_add_(0, (row_base + full0)[full], full_density)
    difference.index_add_(0, (row_base + full1 + 1)[full], -full_density)

    strip = torch.repeat_interleave(
        torch.arange(row.numel(), device=device), boundary_count
    )
    if strip.numel() == 0:
        return
    lane_starts = torch.cumsum(boundary_count, 0) - boundary_count
    lane = torch.arange(strip.numel(), device=device) - lane_starts[strip]
    column, area = _kernel("boundary area", _boundary_area, device)(
        lane,
        left_count[strip],
        bbox0[strip],
        full[strip],
        full1[strip],
        first[:, strip],
        second[:, strip],
        valid0[strip],
        valid1[strip],
    )
    histogram.reshape(-1).index_add_(
        0,
        (frame[strip] * rows + row[strip]) * columns + column,
        density[strip] * area,
    )


def accumulate_cells_torch(
    workspace: TorchRasterWorkspace,
    node_x: torch.Tensor,
    node_y: torch.Tensor,
    *,
    xmin: float,
    ymin: float,
    pixel_size_x: float,
    pixel_size_y: float,
    lens_area_per_triangle_uas2: float,
    cell_frame_index: torch.Tensor | None = None,
    max_strips: int = 1 << 20,
) -> None:
    """Add one chunk of mapped cell lattices to a persistent workspace.

    Arguments match :func:`accumulate_cells_triton`. ``max_strips`` bounds the
    temporary (triangle, row) expansion to keep peak memory predictable.
    """

    if node_x.shape != node_y.shape or node_x.ndim not in (3, 4):
        raise ValueError(
            "node arrays must share shape [cell, v+1, v+1] or "
            "[frame, cell, v+1, v+1]"
        )
    if node_x.shape[-2] != node_x.shape[-1] or node_x.shape[-1] < 2:
        raise ValueError("node lattices must be square with at least two nodes")
    if float(pixel_size_x) <= 0 or float(pixel_size_y) <= 0:
        raise ValueError("pixel sizes must be positive")
    device = workspace.histogram.device
    if node_x.device != device:
        raise ValueError("workspace and mapped nodes must share a device")
    rows, columns = workspace.shape
    side = int(node_x.shape[-1])
    if node_x.ndim == 4:
        if cell_frame_index is not None:
            raise ValueError("indexed raster queues must have shape [cell, v+1, v+1]")
        if workspace.frames != int(node_x.shape[0]):
            raise ValueError("workspace and mapped nodes must have the same frame count")
        cell_frame = torch.arange(
            node_x.shape[0], device=device
        ).repeat_interleave(int(node_x.shape[1]))
    elif cell_frame_index is not None:
        cell_frame = torch.as_tensor(cell_frame_index, device=device).reshape(-1).long()
        if cell_frame.numel() != node_x.shape[0]:
            raise ValueError("cell_frame_index must contain one entry per cell")
        if cell_frame.numel() and bool(
            torch.any((cell_frame < 0) | (cell_frame >= workspace.frames))
        ):
            raise ValueError("cell_frame_index entries are outside the workspace")
    else:
        if workspace.frames != 1:
            raise ValueError("workspace and mapped nodes must have the same frame count")
        cell_frame = torch.zeros(node_x.shape[0], device=device, dtype=torch.long)
    dtype = workspace.histogram.dtype
    x = ((node_x.reshape(-1, side, side).to(dtype) - xmin) * (1.0 / pixel_size_x))
    y = ((node_y.reshape(-1, side, side).to(dtype) - ymin) * (1.0 / pixel_size_y))

    # Each virtual subcell splits into (00, 10, 11) and (00, 11, 01).
    def corners(values):
        v00 = values[:, :-1, :-1]
        v10 = values[:, 1:, :-1]
        v11 = values[:, 1:, 1:]
        v01 = values[:, :-1, 1:]
        return (
            torch.stack((v00, v00), dim=-1).reshape(-1),
            torch.stack((v10, v11), dim=-1).reshape(-1),
            torch.stack((v11, v01), dim=-1).reshape(-1),
        )

    ax, bx, cx = corners(x)
    ay, by, cy = corners(y)
    triangles_per_cell = 2 * (side - 1) * (side - 1)
    frame = cell_frame.repeat_interleave(triangles_per_cell)

    twice_area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    stacked_x = torch.stack((ax, bx, cx))
    stacked_y = torch.stack((ay, by, cy))
    minimum_x, maximum_x = stacked_x.amin(0), stacked_x.amax(0)
    minimum_y, maximum_y = stacked_y.amin(0), stacked_y.amax(0)
    keep = (
        torch.isfinite(stacked_x).all(0)
        & torch.isfinite(stacked_y).all(0)
        & (twice_area.abs() > _TINY)
        & (maximum_x > 0.0)
        & (minimum_x < columns)
        & (maximum_y > 0.0)
        & (minimum_y < rows)
    )
    kept = torch.nonzero(keep).reshape(-1)
    if kept.numel() == 0:
        return
    ax, ay, bx, by, cx, cy, frame = (
        value[kept] for value in (ax, ay, bx, by, cx, cy, frame)
    )
    minimum_y, maximum_y, twice_area = (
        value[kept] for value in (minimum_y, maximum_y, twice_area)
    )
    mass = float(lens_area_per_triangle_uas2) / (
        float(pixel_size_x) * float(pixel_size_y)
    )
    density = mass / (0.5 * twice_area.abs()).clamp_min(_TINY)
    first_row = torch.floor(minimum_y.clamp(0.0, rows - 1.0)).long()
    last_row = torch.floor(maximum_y.clamp(0.0, rows - 1.0)).long()
    row_count = last_row - first_row + 1

    # Split the triangles so no chunk expands into more than ``max_strips``
    # strips; one triangle never spans more than ``rows`` strips.
    budget = max(int(max_strips), rows)
    cumulative = torch.cumsum(row_count, 0)
    total = int(cumulative[-1])
    boundaries = [0]
    if total > budget:
        targets = torch.arange(budget, total, budget, device=device)
        boundaries += torch.searchsorted(cumulative, targets).tolist()
    boundaries.append(kept.numel())
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if stop <= start:
            continue
        part = slice(start, stop)
        _rasterize_strips(
            workspace.histogram,
            workspace.row_difference,
            frame[part],
            ax[part], ay[part], bx[part], by[part], cx[part], cy[part],
            density[part],
            first_row[part],
            row_count[part],
            rows=rows,
            columns=columns,
        )
