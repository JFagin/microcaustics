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

from ..compile import run_tensor_kernel
from ..runtime import ResolvedRuntime

_TINY = 1.1754944e-38
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
    # A thin mapped triangle can still carry a full lens-plane triangle's
    # flux. Reject only zero-height segments, not small positive intervals.
    valid0 = middle > lower
    valid1 = upper > middle
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


def _kernel(name: str, function, runtime: ResolvedRuntime | None):
    """Use the caller's compilation, warning, and fallback settings."""

    if runtime is None:
        return function

    def run(*arguments):
        result, _ = run_tensor_kernel(
            runtime, f"tensor IPM {name}", function, *arguments
        )
        return result

    return run


def _expanded_chunks(counts: torch.Tensor, budget: int):
    """Yield bounded owner/lane arrays without first expanding the full queue."""

    ends = torch.cumsum(counts, 0)
    total = int(ends[-1]) if ends.numel() else 0
    starts = ends - counts
    for start in range(0, total, budget):
        entry = torch.arange(start, min(start + budget, total), device=counts.device)
        if total <= budget:
            owner = torch.repeat_interleave(
                torch.arange(counts.numel(), device=counts.device), counts
            )
        else:
            owner = torch.searchsorted(ends, entry, right=True)
        yield owner, entry - starts[owner]


def _difference_of_products(a, b, c, d):
    """Compensated determinant for nearly collinear float32 edges."""

    ab, cd = a * b, c * d

    def product_error(x, y, product):
        split = 4097.0 if x.dtype == torch.float32 else 134217729.0
        sx, sy = split * x, split * y
        hx, hy = sx - (sx - x), sy - (sy - y)
        lx, ly = x - hx, y - hy
        return ((hx * hy - product) + hx * ly + lx * hy) + lx * ly

    return (ab - cd) + (product_error(a, b, ab) - product_error(c, d, cd))


def _triangle_pixel_fraction(ax, ay, bx, by, cx, cy, xlo, xhi, ylo, yhi):
    """Clip in unit-triangle coordinates, avoiding division by a tiny area.

    A point is A + u(B-A) + v(C-A). Clipping this unit triangle against the
    pixel's four half-planes gives the fraction of the original triangle's
    flux directly, even when its mapped area is very small.
    """

    u = torch.zeros((ax.numel(), 8), device=ax.device, dtype=ax.dtype)
    v = torch.zeros_like(u)
    u[:, 1] = 1.0
    v[:, 2] = 1.0
    count = torch.full_like(ax, 3, dtype=torch.long)
    slot = torch.arange(8, device=ax.device)[None, :]
    for origin, edge_b, edge_c, boundary, sign in (
        (ax, bx - ax, cx - ax, xlo, 1.0),
        (ax, bx - ax, cx - ax, xhi, -1.0),
        (ay, by - ay, cy - ay, ylo, 1.0),
        (ay, by - ay, cy - ay, yhi, -1.0),
    ):
        previous = torch.where(slot == 0, (count - 1)[:, None], slot - 1).clamp_min(0)
        pu, pv = u.gather(1, previous), v.gather(1, previous)
        distance = sign * ((origin - boundary)[:, None] + u * edge_b[:, None] + v * edge_c[:, None])
        prior_distance = distance.gather(1, previous)
        active = slot < count[:, None]
        inside = distance >= 0.0
        crossing = active & (inside != (prior_distance >= 0.0))
        denominator = prior_distance - distance
        fraction = prior_distance / torch.where(crossing, denominator, torch.ones_like(denominator))
        iu = pu + fraction * (u - pu)
        iv = pv + fraction * (v - pv)
        emit = torch.stack((crossing, active & inside), dim=-1).flatten(1)
        out_slot = (emit.long().cumsum(1) - 1).clamp(0, 7)
        candidate_u = torch.stack((iu, u), dim=-1).flatten(1)
        candidate_v = torch.stack((iv, v), dim=-1).flatten(1)
        u = torch.zeros_like(u).scatter_add(1, out_slot, torch.where(emit, candidate_u, 0.0))
        v = torch.zeros_like(v).scatter_add(1, out_slot, torch.where(emit, candidate_v, 0.0))
        count = emit.sum(1)
    following = torch.where(slot + 1 < count[:, None], slot + 1, 0)
    cross = u * v.gather(1, following) - v * u.gather(1, following)
    # The unit triangle has area 1/2, so its doubled clipped area is the fraction.
    return torch.where(slot < count[:, None], cross, 0.0).sum(1).abs().clamp(0.0, 1.0)


def _rasterize_thin_triangles(
    workspace, raw, frame, *, xmin, ymin, pixel_size_x, pixel_size_y,
    mass, budget, runtime,
    source_pixels=None, cell_flux=None, cell_id=None,
):
    """Use bounded pixel clipping only for triangles ill-conditioned in scanlines."""

    ax, ay, bx, by, cx, cy = raw
    rows, columns = workspace.shape
    xs, ys = torch.stack((ax, bx, cx)), torch.stack((ay, by, cy))

    def pixel_index(value, lower, step, size):
        index = torch.floor((value - lower) / step).long()
        # A global pixel coordinate can round onto the next integer. Compare
        # in native coordinates before choosing which side of that edge to use.
        edge = lower + index.to(value.dtype) * step
        next_edge = lower + (index + 1).to(value.dtype) * step
        index = index - (value < edge).long() + (value >= next_edge).long()
        return index.clamp(0, size - 1)

    col0 = pixel_index(xs.amin(0), xmin, pixel_size_x, columns)
    col1 = pixel_index(xs.amax(0), xmin, pixel_size_x, columns)
    row0 = pixel_index(ys.amin(0), ymin, pixel_size_y, rows)
    row1 = pixel_index(ys.amax(0), ymin, pixel_size_y, rows)
    width = col1 - col0 + 1
    counts = width * (row1 - row0 + 1)
    contained = (
        (counts == 1)
        & (xs.amin(0) >= xmin) & (xs.amax(0) <= xmin + columns * pixel_size_x)
        & (ys.amin(0) >= ymin) & (ys.amax(0) <= ymin + rows * pixel_size_y)
    )
    # Tiny triangles usually lie in one pixel. Deposit their entire mass
    # directly, reserving clipping for those that actually cross a boundary.
    workspace.histogram.reshape(-1).index_add_(
        0, ((frame * rows + row0) * columns + col0)[contained],
        torch.full_like(ax[contained], mass),
    )
    if cell_flux is not None:
        cell_flux.index_add_(0, cell_id[contained],
                            mass * source_pixels[row0[contained], col0[contained]])
    counts = torch.where(contained, 0, counts)
    kernel = _kernel("thin triangle clipping", _triangle_pixel_fraction, runtime)
    for owner, lane in _expanded_chunks(counts, budget):
        column = col0[owner] + lane % width[owner]
        row = row0[owner] + lane // width[owner]
        xlo = xmin + column.to(ax.dtype) * pixel_size_x
        xhi = xmin + (column + 1).to(ax.dtype) * pixel_size_x
        ylo = ymin + row.to(ay.dtype) * pixel_size_y
        yhi = ymin + (row + 1).to(ay.dtype) * pixel_size_y
        fraction = kernel(*(value[owner] for value in raw), xlo, xhi, ylo, yhi)
        workspace.histogram.reshape(-1).index_add_(
            0, (frame[owner] * rows + row) * columns + column, mass * fraction,
        )
        if cell_flux is not None:
            cell_flux.index_add_(0, cell_id[owner],
                                (mass * fraction)[:, None] * source_pixels[row, column])


def _rasterize_strips(
    histogram: torch.Tensor,
    row_difference: torch.Tensor,
    frame: torch.Tensor,
    ax, ay, bx, by, cx, cy,
    density: torch.Tensor,
    owner: torch.Tensor,
    row: torch.Tensor,
    *,
    rows: int,
    columns: int,
    max_boundary_entries: int,
    runtime: ResolvedRuntime | None,
    source_pixels=None, source_prefix=None, cell_flux=None, cell_id=None,
) -> None:
    """Deposit kept triangles, already in pixel units, one row strip at a time."""

    if owner.numel() == 0:
        return
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
    ) = _kernel("strip geometry", _strip_geometry, runtime)(
        row, *(value[owner] for value in (ax, ay, bx, by, cx, cy)), columns
    )

    difference = row_difference.reshape(-1)
    row_base = (frame * rows + row) * (columns + 1)
    full_density = density[full]
    difference.index_add_(0, (row_base + full0)[full], full_density)
    difference.index_add_(0, (row_base + full1 + 1)[full], -full_density)
    if cell_flux is not None:
        strip_cells = cell_id[owner]
        cell_flux.index_add_(0, strip_cells[full], full_density[:, None] * (
            source_prefix[row[full], full1[full] + 1] - source_prefix[row[full], full0[full]]
        ))

    # One shallow strip can touch thousands of boundary pixels. Bound this
    # expansion separately from the triangle-row queue, including a single
    # strip whose boundary alone exceeds the budget.
    boundary_kernel = _kernel("boundary area", _boundary_area, runtime)
    for strip, lane in _expanded_chunks(boundary_count, max_boundary_entries):
        column, area = boundary_kernel(
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
        if cell_flux is not None:
            cell_flux.index_add_(0, strip_cells[strip],
                                (density[strip] * area)[:, None] * source_pixels[row[strip], column])


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
    max_boundary_entries: int = 1 << 20,
    runtime: ResolvedRuntime | None = None,
    source_pixels: torch.Tensor | None = None,
    source_prefix: torch.Tensor | None = None,
    cell_flux: torch.Tensor | None = None,
) -> None:
    """Add one chunk of mapped cell lattices to a persistent workspace.

    Arguments match :func:`accumulate_cells_triton`. ``max_strips`` bounds the
    temporary (triangle, row) expansion to keep peak memory predictable.
    ``max_boundary_entries`` separately bounds the boundary-pixel expansion.
    Without a runtime, this low-level function uses eager PyTorch.
    """

    if node_x.shape != node_y.shape or node_x.ndim not in (3, 4):
        raise ValueError(
            "node arrays must share shape [cell, v+1, v+1] or "
            "[frame, cell, v+1, v+1]"
        )
    if source_pixels is not None and (node_x.ndim != 3 or cell_frame_index is not None or workspace.frames != 1):
        raise ValueError("Microimages currently use single-frame rasterization")
    if source_pixels is not None or source_prefix is not None or cell_flux is not None:
        from ..microimages import validate_render_tensors

        validate_render_tensors(source_pixels, source_prefix, cell_flux, workspace.shape,
                                node_x.shape[0], workspace.histogram.device, workspace.histogram.dtype)
    if node_x.shape[-2] != node_x.shape[-1] or node_x.shape[-1] < 2:
        raise ValueError("node lattices must be square with at least two nodes")
    if float(pixel_size_x) <= 0 or float(pixel_size_y) <= 0:
        raise ValueError("pixel sizes must be positive")
    if int(max_strips) < 1 or int(max_boundary_entries) < 1:
        raise ValueError("rasterization chunk limits must be positive")
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
    raw_ax, raw_bx, raw_cx = corners(node_x.reshape(-1, side, side).to(dtype))
    raw_ay, raw_by, raw_cy = corners(node_y.reshape(-1, side, side).to(dtype))
    triangles_per_cell = 2 * (side - 1) * (side - 1)
    frame = cell_frame.repeat_interleave(triangles_per_cell)
    cell_id = torch.arange(cell_frame.numel(), device=device).repeat_interleave(triangles_per_cell) if cell_flux is not None else None

    twice_area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    ebx, eby = raw_bx - raw_ax, raw_by - raw_ay
    ecx, ecy = raw_cx - raw_ax, raw_cy - raw_ay
    product0, product1 = ebx * ecy, eby * ecx
    native_area = product0 - product1
    uncertain = torch.nonzero(
        native_area.abs() <= 8 * torch.finfo(dtype).eps * (product0.abs() + product1.abs())
    ).reshape(-1)
    if uncertain.numel():
        native_area[uncertain] = _difference_of_products(
            ebx[uncertain], ecy[uncertain], eby[uncertain], ecx[uncertain]
        )
    native_area = native_area / (pixel_size_x * pixel_size_y)
    stacked_x = torch.stack((ax, bx, cx))
    stacked_y = torch.stack((ay, by, cy))
    minimum_x, maximum_x = stacked_x.amin(0), stacked_x.amax(0)
    minimum_y, maximum_y = stacked_y.amin(0), stacked_y.amax(0)
    keep = (
        torch.isfinite(stacked_x).all(0)
        & torch.isfinite(stacked_y).all(0)
        & (native_area.abs() > _TINY)
        & (torch.maximum(torch.maximum(raw_ax, raw_bx), raw_cx) > xmin)
        & (torch.minimum(torch.minimum(raw_ax, raw_bx), raw_cx) < xmin + columns * pixel_size_x)
        & (torch.maximum(torch.maximum(raw_ay, raw_by), raw_cy) > ymin)
        & (torch.minimum(torch.minimum(raw_ay, raw_by), raw_cy) < ymin + rows * pixel_size_y)
    )
    mass = float(lens_area_per_triangle_uas2) / (
        float(pixel_size_x) * float(pixel_size_y)
    )
    # Very narrow triangles amplify scanline subtraction error. Compare area
    # with the squared longest extent, and also catch precision lost when
    # converting physical coordinates to global pixel coordinates.
    extent = torch.maximum(maximum_x - minimum_x, maximum_y - minimum_y)
    thin = keep & (
        (native_area.abs() < 1e-3 * extent.square())
        | ((twice_area - native_area).abs() > 1e-3 * native_area.abs())
    )
    thin_index = torch.nonzero(thin).reshape(-1)
    if thin_index.numel():
        _rasterize_thin_triangles(
            workspace,
            tuple(value[thin_index] for value in (raw_ax, raw_ay, raw_bx, raw_by, raw_cx, raw_cy)),
            frame[thin_index], xmin=xmin, ymin=ymin,
            pixel_size_x=pixel_size_x, pixel_size_y=pixel_size_y, mass=mass,
            budget=min(int(max_boundary_entries), 1 << 15), runtime=runtime,
            source_pixels=source_pixels, cell_flux=cell_flux,
            cell_id=None if cell_id is None else cell_id[thin_index],
        )
    keep = keep & ~thin
    kept = torch.nonzero(keep).reshape(-1)
    if kept.numel() == 0:
        return
    ax, ay, bx, by, cx, cy, frame = (
        value[kept] for value in (ax, ay, bx, by, cx, cy, frame)
    )
    minimum_y, maximum_y, twice_area = (
        value[kept] for value in (minimum_y, maximum_y, twice_area)
    )
    density = mass / (0.5 * twice_area.abs()).clamp_min(_TINY)
    first_row = torch.floor(minimum_y.clamp(0.0, rows - 1.0)).long()
    last_row = torch.floor(maximum_y.clamp(0.0, rows - 1.0)).long()
    row_count = last_row - first_row + 1

    # Bound the expanded row queue itself, even if a single triangle spans
    # more rows than the requested budget.
    for owner, lane in _expanded_chunks(row_count, int(max_strips)):
        _rasterize_strips(
            workspace.histogram,
            workspace.row_difference,
            frame,
            ax, ay, bx, by, cx, cy,
            density,
            owner,
            first_row[owner] + lane,
            rows=rows,
            columns=columns,
            max_boundary_entries=int(max_boundary_entries),
            runtime=runtime,
            source_pixels=source_pixels, source_prefix=source_prefix, cell_flux=cell_flux,
            cell_id=None if cell_id is None else cell_id[kept],
        )
