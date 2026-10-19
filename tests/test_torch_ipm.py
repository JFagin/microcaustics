"""Batched tensor IPM rasterization and far-field coefficients for portable accelerators."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import microcaustics as mc
from microcaustics.compile import clear_compiled_kernel_cache
from microcaustics.solvers.ipm import (
    full_field_ipm,
    rasterize_triangles_exact_eager,
    temporal_batch_ipm,
    triangles_from_node_lattices,
)
from microcaustics.solvers.taylor import (
    complex_taylor_coefficients,
    masked_complex_taylor_coefficients,
)
from microcaustics.solvers.torch_ipm import (
    TorchRasterWorkspace,
    _boundary_area,
    _expanded_chunks,
    _kernel,
    _strip_geometry,
    _triangle_pixel_fraction,
    accumulate_cells_torch,
)

DEVICES = ["cpu"] + (
    [pytest.param("mps", marks=pytest.mark.mps)]
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("height", [1e-6, 1e-7])
def test_tensor_rasterizer_preserves_thin_cell_flux(dtype, height):
    x = torch.tensor([[[0.1, 0.9], [0.1, 0.9]]], dtype=dtype)
    y = torch.tensor([[[0.1, 0.1], [0.1 + height, 0.1 + height]]], dtype=dtype)
    grid = mc.PlaneGrid((1, 1), (1.0, 1.0), center_uas=(0.5, 0.5))
    actual = _rasterize(x, y, grid, device="cpu", dtype=dtype)
    expected = _reference(x.double(), y.double(), grid)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-8)
    assert float(actual.sum()) == pytest.approx(0.6, rel=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("straddle", [False, True])
def test_tensor_rasterizer_retains_cells_below_global_pixel_precision(device, straddle):
    lower = -4e-8 if straddle else 0.0
    x = torch.tensor([[[lower, 4e-8], [lower, 4e-8]]], device=device)
    y = x.transpose(-1, -2).contiguous()
    workspace = TorchRasterWorkspace.create((128, 128), device=torch.device(device))
    accumulate_cells_torch(
        workspace, x, y, xmin=-2.0, ymin=-2.0,
        pixel_size_x=1 / 32, pixel_size_y=1 / 32,
        lens_area_per_triangle_uas2=0.3 / 32**2,
    )
    result = workspace.result().cpu()
    assert float(result.sum()) == pytest.approx(0.6, rel=2e-6)
    if straddle:
        torch.testing.assert_close(result[63:65, 63:65], torch.full((2, 2), 0.15))
    else:
        assert float(result[64, 64]) == pytest.approx(0.6, rel=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_rasterizer_near_critical_cell_matches_clipping(device):
    points = torch.tensor([
        [(0.0715685561299324, -0.06438595056533813), (0.09734947234392166, -0.06900149583816528)],
        [(0.0715685561299324, -0.06438595056533813), (0.09279758483171463, -0.06818687915802002)],
    ], dtype=torch.float32)
    x, y = points[..., 0][None], points[..., 1][None]
    grid = mc.PlaneGrid((128, 128), (0.384, 0.384), center_uas=(0.092, 0.092))
    actual = _rasterize(x, y, grid, device=device, dtype=torch.float32)
    expected = _reference(x.double(), y.double(), grid)
    mass = 0.3 / 0.003**2
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=mass * 2e-6)
    assert float(actual.sum()) == pytest.approx(mass, rel=2e-6)


def test_unit_triangle_clipping_handles_partial_and_empty_pixels():
    # Triangle (0,0),(1,0),(0,1), first pixel has exactly half its area.
    bounds = [torch.tensor(values) for values in (
        [0.0, 2.0, -1.0], [0.5, 3.0, 2.0],
        [0.0, 2.0, -1.0], [0.5, 3.0, 2.0],
    )]
    z, o = torch.zeros(3), torch.ones(3)
    result = _triangle_pixel_fraction(z, z, o, z, z, o, *bounds)
    torch.testing.assert_close(result, torch.tensor([0.5, 0.0, 1.0]))


@pytest.mark.parametrize("device", DEVICES)
def test_near_collinear_native_area_cancellation_retains_flux(device):
    a = torch.tensor([0.1, 0.2], device=device)
    b = torch.tensor([0.7, 0.8], device=device)
    c = torch.tensor([0.3, 0.4], device=device)
    # Specify the neighbouring float32 value without requiring MPS nextafter.
    c[1] = 0.40000003576278687
    points = torch.stack((torch.stack((a, a)), torch.stack((b, c))))
    workspace = TorchRasterWorkspace.create((1, 1), device=torch.device(device))
    accumulate_cells_torch(
        workspace, points[None, ..., 0], points[None, ..., 1],
        xmin=0.0, ymin=0.0, pixel_size_x=1.0, pixel_size_y=1.0,
        lens_area_per_triangle_uas2=0.3,
    )
    torch.testing.assert_close(workspace.result().cpu(), torch.tensor([[0.3]]), rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_tiny_cells_at_upper_map_edge_are_not_discarded(device):
    x = torch.tensor([[[-4e-8, -2e-8], [-4e-8, -2e-8]]], device=device)
    y = torch.tensor([[[0.0, 0.0], [4e-8, 4e-8]]], device=device)
    workspace = TorchRasterWorkspace.create((128, 128), device=torch.device(device))
    accumulate_cells_torch(
        workspace, x, y, xmin=-4.0, ymin=-2.0,
        pixel_size_x=1 / 32, pixel_size_y=1 / 32,
        lens_area_per_triangle_uas2=0.3 / 32**2,
    )
    expected = torch.zeros((128, 128))
    expected[64, 127] = 0.6
    torch.testing.assert_close(workspace.result().cpu(), expected, rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("budget", [1, 7, 100])
def test_expanded_chunks_bound_even_a_single_long_strip(budget):
    counts = torch.tensor([0, 3, 0, 19, 2, 0])
    chunks = list(_expanded_chunks(counts, budget))
    assert all(owner.numel() <= budget for owner, _ in chunks)
    expected = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    torch.testing.assert_close(torch.cat([owner for owner, _ in chunks]), expected)
    starts = counts.cumsum(0) - counts
    torch.testing.assert_close(
        torch.cat([lane for _, lane in chunks]), torch.arange(24) - starts[expected]
    )
    assert list(_expanded_chunks(torch.zeros(2, dtype=torch.long), budget)) == []


def test_boundary_expansion_is_bounded_and_matches_exact_reference():
    # One triangle-row strip touches more boundary pixels than the budget.
    x = torch.tensor([[[0.1, 127.9], [0.1, 127.9]]], dtype=torch.float64)
    y = torch.tensor([[[0.1, 0.1], [0.9, 0.9]]], dtype=torch.float64)
    grid = mc.PlaneGrid((1, 128), (1.0, 128.0), center_uas=(0.5, 64.0))
    workspace = TorchRasterWorkspace.create((1, 128), device=torch.device("cpu"), dtype=torch.float64)
    with patch("microcaustics.solvers.torch_ipm._boundary_area", wraps=_boundary_area) as boundary:
        accumulate_cells_torch(
            workspace, x, y, xmin=0.0, ymin=0.0,
            pixel_size_x=1.0, pixel_size_y=1.0,
            lens_area_per_triangle_uas2=0.3, max_boundary_entries=7,
        )
    assert boundary.call_count > 1
    assert all(call.args[0].numel() <= 7 for call in boundary.call_args_list)
    torch.testing.assert_close(workspace.result(), _reference(x, y, grid), rtol=1e-9, atol=1e-12)


def test_row_expansion_is_bounded_even_for_one_tall_triangle():
    x = torch.tensor([[[0.1, 7.9], [0.1, 7.9]]], dtype=torch.float64)
    y = torch.tensor([[[0.1, 0.1], [31.9, 31.9]]], dtype=torch.float64)
    grid = mc.PlaneGrid((32, 8), (32.0, 8.0), center_uas=(16.0, 4.0))
    workspace = TorchRasterWorkspace.create(grid.shape, device=torch.device("cpu"), dtype=torch.float64)
    with patch("microcaustics.solvers.torch_ipm._strip_geometry", wraps=_strip_geometry) as geometry:
        accumulate_cells_torch(
            workspace, x, y, xmin=0.0, ymin=0.0,
            pixel_size_x=1.0, pixel_size_y=1.0,
            lens_area_per_triangle_uas2=0.3, max_strips=7, max_boundary_entries=5,
        )
    assert geometry.call_count > 1
    assert all(call.args[0].numel() <= 7 for call in geometry.call_args_list)
    torch.testing.assert_close(workspace.result(), _reference(x, y, grid), rtol=1e-9, atol=1e-12)


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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", [10, 16])
@pytest.mark.parametrize("star_chunk_size", [1, 3])
def test_masked_near_center_stars_do_not_overflow(device, order, star_chunk_size):
    # The first two stars are local, including one exactly at the center.
    # At order ten the unmasked inverse powers overflow in float32.
    center = torch.tensor([0.0], device=device)
    star_x = torch.tensor([0.0, 1e-4, 10.0], device=device)
    star_y = torch.tensor([0.0, -1e-4, 3.0], device=device)
    mass = torch.ones_like(star_x)
    local = torch.tensor([[True, True, False]], device=device)
    actual = masked_complex_taylor_coefficients(
        center, center, star_x, star_y, mass, local,
        order=order, star_chunk_size=star_chunk_size,
    )
    expected = complex_taylor_coefficients(
        center, center, star_x[-1:], star_y[-1:], mass[-1:], order=order,
    )
    for value, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, reference, rtol=2e-6, atol=0.0)


def test_tensor_rasterizer_respects_eager_runtime():
    runtime = mc.resolve_runtime(mc.RuntimeConfig(device="cpu", backend="torch-eager"))
    x, y = _lattices(cells=3)
    workspace = TorchRasterWorkspace.create((8, 8), device=torch.device("cpu"))
    with patch("torch.compile", side_effect=AssertionError("must remain eager")):
        accumulate_cells_torch(
            workspace, x, y, xmin=0.0, ymin=0.0,
            pixel_size_x=3.0, pixel_size_y=3.0,
            lens_area_per_triangle_uas2=0.3, runtime=runtime,
        )
    assert torch.isfinite(workspace.result()).all()


@pytest.mark.parametrize("strict", [False, True])
def test_tensor_kernel_respects_compilation_failure_policy(strict):
    runtime = replace(
        mc.resolve_runtime(mc.RuntimeConfig(device="cpu", backend="torch-eager")),
        backend=mc.Backend.TORCH_COMPILE, strict_backend=strict,
        warn_on_compile=False, torch_compile_mode="reduce-overhead",
    )
    clear_compiled_kernel_cache()
    try:
        with patch("torch.compile", side_effect=RuntimeError("test compiler failure")) as compile_mock:
            kernel = _kernel("policy regression", torch.neg, runtime)
            if strict:
                with pytest.raises(RuntimeError, match="test compiler failure"):
                    kernel(torch.tensor([2.0]))
            else:
                with pytest.warns(RuntimeWarning, match="test compiler failure"):
                    actual = kernel(torch.tensor([2.0]))
                torch.testing.assert_close(actual, torch.tensor([-2.0]))
            compile_mock.assert_called_once_with(
                torch.neg, fullgraph=True, dynamic=True, mode="reduce-overhead"
            )
    finally:
        clear_compiled_kernel_cache()


@pytest.mark.parametrize("temporal", [False, True])
def test_map_dispatch_forwards_runtime_to_tensor_rasterizer(temporal):
    """Exercise the portable-accelerator route without requiring Apple hardware."""
    simulation = mc.MicrolensingSimulation.create(
        mc.MacroLens(0.0, 0.0),
        mc.PointMassField._from_einstein_radii(
            torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
        ),
        runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager", strict_backend=True),
    )
    region = mc.PlaneRegion((2.0, 3.0))
    grid = mc.PlaneGrid((7, 11), (2.0, 3.0))
    config = mc.IPMConfig(
        rays=35, refinement=2, virtual_refinement=4, tiled=False,
        cell_chunk_size=9, far_field_approx=mc.FarFieldApproxConfig(enabled=False),
    )
    with (
        patch("microcaustics.solvers.ipm._tensor_rasterizer", return_value=True),
        patch("microcaustics.solvers.torch_ipm.accumulate_cells_torch", wraps=accumulate_cells_torch) as accumulate,
        patch("torch.compile", side_effect=AssertionError("must remain eager")),
    ):
        if temporal:
            maps = temporal_batch_ipm(simulation, region, grid, config, [0.0, 1.0])
        else:
            maps = (full_field_ipm(simulation, region, grid, config),)
    assert accumulate.call_count > 0
    assert all(call.kwargs["runtime"] is simulation.runtime for call in accumulate.call_args_list)
    for result in maps:
        assert result.metadata["rasterizer"] == (
            "torch_direct_cell_temporal_batch" if temporal else "torch_direct_cell_scanline"
        )
        torch.testing.assert_close(result.values, torch.ones_like(result.values), rtol=2e-5, atol=2e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_tensor_rasterizer_matches_eager():
    """Check the portable tensor kernels with a real compiler as well as mocks."""
    runtime = mc.resolve_runtime(mc.RuntimeConfig(
        device="cuda", backend="torch-compile", strict_backend=True, warn_on_compile=False,
    ))
    x, y = (value.cuda().float() for value in _lattices(cells=8, side=2))
    # Exercise the rare thin-triangle clipping kernel under compilation too.
    x = torch.cat((x, torch.tensor([[[0.1, 6.9], [0.1, 6.9]]], device="cuda")))
    y = torch.cat((y, torch.tensor([[[0.1, 0.1], [0.100001, 0.100001]]], device="cuda")))
    maps = []
    for selected_runtime in (replace(runtime, backend=mc.Backend.TORCH_EAGER), runtime):
        workspace = TorchRasterWorkspace.create((8, 8), device=torch.device("cuda"))
        accumulate_cells_torch(
            workspace, x, y, xmin=0.0, ymin=0.0,
            pixel_size_x=3.0, pixel_size_y=3.0,
            lens_area_per_triangle_uas2=0.3, runtime=selected_runtime,
        )
        maps.append(workspace.result())
    torch.testing.assert_close(maps[1], maps[0], rtol=2e-4, atol=2e-5)
