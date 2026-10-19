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
    _kernel,
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
        torch.testing.assert_close(result.values, torch.ones_like(result.values), rtol=2e-5, atol=2e-5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_tensor_rasterizer_matches_eager():
    """Check the portable tensor kernels with a real compiler as well as mocks."""
    runtime = mc.resolve_runtime(mc.RuntimeConfig(
        device="cuda", backend="torch-compile", strict_backend=True, warn_on_compile=False,
    ))
    x, y = (value.cuda().float() for value in _lattices(cells=8))
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
