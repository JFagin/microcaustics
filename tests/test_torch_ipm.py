"""Batched tensor IPM rasterization and far-field coefficients for portable accelerators."""

from __future__ import annotations

import pytest
import torch

import microcaustics as mc
from microcaustics.solvers.ipm import (
    rasterize_triangles_exact_eager,
    triangles_from_node_lattices,
)
from microcaustics.solvers.taylor import (
    complex_taylor_coefficients,
    masked_complex_taylor_coefficients,
)
from microcaustics.solvers.torch_ipm import TorchRasterWorkspace, accumulate_cells_torch

DEVICES = ["cpu"] + (
    ["mps"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else []
)


def _lattices(cells=300, side=5, seed=0):
    """Distorted cell lattices, a tenth of them stretched across many pixels."""

    generator = torch.Generator().manual_seed(seed)
    base = torch.linspace(0.0, 1.0, side, dtype=torch.float64)
    center_x = torch.rand(cells, 1, 1, generator=generator, dtype=torch.float64) * 30 - 5
    center_y = torch.rand(cells, 1, 1, generator=generator, dtype=torch.float64) * 30 - 5
    stretched = torch.rand(cells, 1, 1, generator=generator) < 0.1
    scale = torch.where(stretched, 20.0, 1.5).to(torch.float64)
    jitter = 0.03 * torch.randn(2, cells, side, side, generator=generator, dtype=torch.float64)
    x = center_x + scale * (base[None, None, :] + jitter[0])
    y = center_y + scale * (base[None, :, None] + jitter[1])
    return x, y


def _rasterize(x, y, grid, *, device, dtype, frames=None, max_strips=1 << 22):
    xmin, _, ymin, _ = grid.bounds_uas
    dy, dx = grid.pixel_scale_uas
    workspace = TorchRasterWorkspace.create(
        grid.shape, device=torch.device(device), dtype=dtype, frames=frames or 1
    )
    accumulate_cells_torch(
        workspace,
        x.to(device=device, dtype=dtype),
        y.to(device=device, dtype=dtype),
        xmin=xmin,
        ymin=ymin,
        pixel_size_x=dx,
        pixel_size_y=dy,
        lens_area_per_triangle_uas2=0.3,
        cell_frame_index=(
            None if frames is None else torch.arange(x.shape[0], device=device) % frames
        ),
        max_strips=max_strips,
    )
    return workspace.result().cpu().to(torch.float64)


def _reference(x, y, grid):
    return rasterize_triangles_exact_eager(
        triangles_from_node_lattices(x, y), grid, lens_area_per_triangle_uas2=0.3
    )


def test_tensor_rasterizer_matches_exact_reference_in_float64():
    x, y = _lattices()
    grid = mc.PlaneGrid((24, 20), (20.0, 24.0))
    expected = _reference(x, y, grid)
    assert float(expected.sum()) > 0.0
    for max_strips in (1 << 22, 97):
        actual = _rasterize(x, y, grid, device="cpu", dtype=torch.float64, max_strips=max_strips)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_rasterizer_float32_matches_reference(device):
    x, y = _lattices(seed=1)
    grid = mc.PlaneGrid((24, 20), (20.0, 24.0))
    expected = _reference(x, y, grid)
    actual = _rasterize(x, y, grid, device=device, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-5 * float(expected.max()))


def test_tensor_rasterizer_indexed_frames_and_degenerate_triangles():
    x, y = _lattices(cells=40, seed=2)
    x[0] = float("nan")
    y[1] = y[1, :1, :1]  # every node on one row: zero-area triangles
    x[2] = x[2] + 1.0e6  # entirely outside the map
    grid = mc.PlaneGrid((16, 16), (20.0, 20.0))
    actual = _rasterize(x, y, grid, device="cpu", dtype=torch.float64, frames=3)
    for frame in range(3):
        keep = torch.arange(x.shape[0]) % 3 == frame
        keep[0] = False
        expected = _reference(x[keep], y[keep], grid)
        torch.testing.assert_close(actual[frame], expected, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_masked_taylor_coefficients_match_per_center_gathers(device):
    generator = torch.Generator().manual_seed(3)
    stars = torch.rand(3, 500, generator=generator, dtype=torch.float64)
    star_x, star_y = stars[0] * 40 - 20, stars[1] * 40 - 20
    mass = stars[2] * 0.5
    center_x = torch.linspace(-15.0, 15.0, 6, dtype=torch.float64).repeat(6)
    center_y = torch.linspace(-15.0, 15.0, 6, dtype=torch.float64).repeat_interleave(6)
    local = (center_x[:, None] - star_x).square() + (center_y[:, None] - star_y).square() < 9.0
    expected = [
        complex_taylor_coefficients(
            center_x[index : index + 1],
            center_y[index : index + 1],
            star_x[~local[index]],
            star_y[~local[index]],
            mass[~local[index]],
            order=6,
            star_chunk_size=128,
        )
        for index in range(center_x.numel())
    ]
    dtype = torch.float64 if device == "cpu" else torch.float32
    real, imag = masked_complex_taylor_coefficients(
        *(value.to(device=device, dtype=dtype) for value in (center_x, center_y, star_x, star_y, mass)),
        local.to(device),
        order=6,
        star_chunk_size=128,
    )
    tolerance = 1e-12 if dtype == torch.float64 else 2e-5
    for actual, reference in zip(
        (real, imag),
        (torch.cat([item[0] for item in expected]), torch.cat([item[1] for item in expected])),
        strict=True,
    ):
        scale = reference.abs().amax(dim=0)
        torch.testing.assert_close(
            actual.cpu().double() / scale, reference / scale, rtol=0.0, atol=tolerance
        )
