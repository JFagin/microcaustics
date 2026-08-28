"""Device-independent marching squares for zero-level segments."""

from __future__ import annotations

import torch


@torch.no_grad()
def marching_squares_zero(
    field: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    *,
    epsilon: float | None = None,
) -> torch.Tensor:
    """Extract line segments where a scalar field crosses zero.

    Parameters
    ----------
    field, x_grid, y_grid:
        Matching two-dimensional arrays. Coordinates may be non-square but
        must describe a logically rectangular grid.
    epsilon:
        Denominator guard for edge interpolation. The dtype machine epsilon is
        used by default.

    Returns
    -------
    torch.Tensor
        Segment endpoints with shape ``[segment, 2, 2]`` and Cartesian
        coordinate order ``(x, y)``.

    Notes
    -----
    Ambiguous saddle cases use the sign of the bilinear cell-center estimate
    to select connectivity. Exact zeros are assigned to the non-positive side
    consistently, preventing a shared grid vertex from changing case between
    adjacent cells.
    """

    field = torch.as_tensor(field)
    x_grid = torch.as_tensor(x_grid, device=field.device, dtype=field.dtype)
    y_grid = torch.as_tensor(y_grid, device=field.device, dtype=field.dtype)
    if field.ndim != 2 or x_grid.shape != field.shape or y_grid.shape != field.shape:
        raise ValueError("field and coordinate grids must have one matching 2D shape")
    if min(field.shape) < 2:
        return field.new_empty((0, 2, 2))
    eps = torch.finfo(field.dtype).eps if epsilon is None else float(epsilon)
    f0, f1 = field[:-1, :-1], field[:-1, 1:]
    f2, f3 = field[1:, 1:], field[1:, :-1]
    x0, x1 = x_grid[:-1, :-1], x_grid[:-1, 1:]
    x2, x3 = x_grid[1:, 1:], x_grid[1:, :-1]
    y0, y1 = y_grid[:-1, :-1], y_grid[:-1, 1:]
    y2, y3 = y_grid[1:, 1:], y_grid[1:, :-1]
    case = (
        (f0 > 0).to(torch.int16)
        + 2 * (f1 > 0).to(torch.int16)
        + 4 * (f2 > 0).to(torch.int16)
        + 8 * (f3 > 0).to(torch.int16)
    )

    def _interpolate(ax, ay, af, bx, by, bf):
        denominator = af - bf
        safe = torch.where(
            denominator.abs() > eps,
            denominator,
            torch.full_like(denominator, eps),
        )
        fraction = af / safe
        return torch.stack(
            (ax + fraction * (bx - ax), ay + fraction * (by - ay)),
            dim=-1,
        )

    edges = (
        _interpolate(x0, y0, f0, x1, y1, f1),
        _interpolate(x1, y1, f1, x2, y2, f2),
        _interpolate(x2, y2, f2, x3, y3, f3),
        _interpolate(x3, y3, f3, x0, y0, f0),
    )
    segments: list[torch.Tensor] = []

    def _append(mask: torch.Tensor, edge_a: int, edge_b: int) -> None:
        if bool(mask.any()):
            segments.append(
                torch.stack((edges[edge_a][mask], edges[edge_b][mask]), dim=1)
            )

    # Orient every segment so that the positive side of ``field`` lies to
    # its right.  Complementary marching cases must therefore traverse the
    # same geometric segment in opposite directions.  This orientation is
    # immaterial for parity labels, but is required for a genuine signed
    # winding number after the critical curves are mapped to caustics.
    mapping = {
        1: (3, 0),
        2: (0, 1),
        3: (3, 1),
        4: (1, 2),
        6: (0, 2),
        7: (3, 2),
        8: (2, 3),
        9: (2, 0),
        11: (2, 1),
        12: (1, 3),
        13: (1, 0),
        14: (0, 3),
    }
    for case_value, (edge_a, edge_b) in mapping.items():
        _append(case == case_value, edge_a, edge_b)
    center_positive = 0.25 * (f0 + f1 + f2 + f3) > 0
    case_five = case == 5
    _append(case_five & center_positive, 1, 0)
    _append(case_five & center_positive, 3, 2)
    _append(case_five & ~center_positive, 3, 0)
    _append(case_five & ~center_positive, 1, 2)
    case_ten = case == 10
    _append(case_ten & center_positive, 0, 3)
    _append(case_ten & center_positive, 2, 1)
    _append(case_ten & ~center_positive, 0, 1)
    _append(case_ten & ~center_positive, 2, 3)
    return torch.cat(segments, dim=0) if segments else field.new_empty((0, 2, 2))
