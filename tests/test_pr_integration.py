"""Regression checks for changing IPM geometry and the lamp parameter convention."""

from __future__ import annotations

import pytest
import torch

import microcaustics as mc
from microcaustics.solvers.ipm import (
    rasterize_triangles_exact_eager,
    triangles_from_node_lattices,
)
from microcaustics.solvers.triton_ipm import (
    rasterize_cells_triton,
    triton_ipm_available,
)
from microcaustics.sources.thin_disk import _thin_disk_temperature4_coefficient


@pytest.mark.cuda
@pytest.mark.skipif(not triton_ipm_available(), reason="CUDA Triton is unavailable")
def test_triton_runtime_geometry_and_mass_change_between_calls():
    """The same kernels must use new bounds, scales, and weights on every call."""
    # Includes a cell well outside the first grid but inside the shifted grid,
    # exercising active-cell compaction as well as deposition.
    x = torch.tensor([
        [[-0.8, 0.4], [-0.5, 0.7]],
        [[3.1, 4.3], [3.4, 4.0]],
    ], device="cuda", dtype=torch.float32)
    y = torch.tensor([
        [[-0.9, -0.6], [0.7, 1.1]],
        [[2.1, 2.4], [3.2, 3.5]],
    ], device="cuda", dtype=torch.float32)
    configurations = [
        (mc.PlaneGrid((12, 16), (4.0, 4.0)), 0.3),
        (mc.PlaneGrid((12, 16), (4.0, 4.0)), 0.7),
        (mc.PlaneGrid((12, 16), (5.3, 7.1), center_uas=(1.8, 2.0)), 0.3),
        (mc.PlaneGrid((12, 16), (4.0, 4.0)), 0.3),
    ]
    outputs = []
    for grid, mass in configurations:
        xmin, _, ymin, _ = grid.bounds_uas
        dy, dx = grid.pixel_scale_uas
        actual = rasterize_cells_triton(
            x, y, xmin=xmin, ymin=ymin, pixel_size_x=dx, pixel_size_y=dy,
            shape=grid.shape, lens_area_per_triangle_uas2=mass,
        ).cpu().double()
        expected = rasterize_triangles_exact_eager(
            triangles_from_node_lattices(x.cpu().double(), y.cpu().double()),
            grid, lens_area_per_triangle_uas2=mass,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        outputs.append(actual)
    torch.testing.assert_close(outputs[1], outputs[0] * (0.7 / 0.3))
    torch.testing.assert_close(outputs[3], outputs[0])


@pytest.mark.parametrize("profile", ["novikov-thorne", "shakura-sunyaev"])
def test_fixed_eddington_lamp_fraction_keeps_absolute_heating_fixed(profile):
    """The inverse Eddington ratio cancels the accretion-rate scale."""
    spin = torch.tensor(0.4, dtype=torch.float64)
    heating = []
    for ratio in (0.03, 0.1, 0.34, 1.0):
        scale, _ = _thin_disk_temperature4_coefficient(
            1e8, ratio, spin, viscous_flux_profile=profile,
            device="cpu", dtype=torch.float64,
        )
        efficiency = mc.lamppost_irradiation_efficiency(
            0.1, ratio, spin, viscous_flux_profile=profile,
        )
        heating.append(scale * efficiency)
    values = torch.stack(heating)
    torch.testing.assert_close(values, values[0].expand_as(values), rtol=2e-14, atol=0.0)
