"""Regression tests for thin mapped triangles on high-resolution IPM grids."""

from __future__ import annotations

import unittest

import torch

import microcaustics as mc
from microcaustics.solvers import (
    rasterize_triangles_exact_eager,
    triangles_from_node_lattices,
)
from microcaustics.solvers.triton_ipm import (
    TritonRasterWorkspace,
    accumulate_cells_triton,
    rasterize_cells_triton,
    triton_ipm_available,
)


def thin_cell(device):
    """Mapped float32 vertices of the isolated near-critical triangle."""
    a = (0.0715685561299324, -0.06438595056533813)
    b = (0.09279758483171463, -0.06818687915802002)
    c = (0.09734947234392166, -0.06900149583816528)
    points = torch.tensor([[a, c], [a, b]], dtype=torch.float32, device=device)
    return (
        points[..., 0].unsqueeze(0).contiguous(),
        points[..., 1].unsqueeze(0).contiguous(),
    )


class PortableIPMPrecisionTests(unittest.TestCase):
    """CPU and non-Triton GPU IPM share the float64 clipping reference."""

    def check_portable(self, device):
        x, y = thin_cell(device)
        grid = mc.PlaneGrid((4096, 4096), (10.76080777812415, 10.76080777812415))
        area = 8.884576280266643e-5
        result = rasterize_triangles_exact_eager(
            triangles_from_node_lattices(x, y),
            grid,
            lens_area_per_triangle_uas2=area,
        )
        self.assertEqual(result.device, x.device)
        self.assertEqual(result.dtype, torch.float32)
        mass = area / (grid.pixel_scale_uas[0] * grid.pixel_scale_uas[1])
        self.assertAlmostEqual(float(result.sum()), mass, delta=mass * 2e-6)
        self.assertAlmostEqual(float(result[2023, 2076]), 0.408464956, delta=2e-6)
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_cpu_portable_ipm_preserves_thin_triangle(self):
        self.check_portable("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_non_triton_cuda_ipm_preserves_thin_triangle(self):
        self.check_portable("cuda")


@unittest.skipUnless(triton_ipm_available(), "CUDA Triton IPM is unavailable")
class ThinTrianglePrecisionTests(unittest.TestCase):
    """Preserve skinny polygons instead of amplifying global-coordinate rounding."""

    @staticmethod
    def fixture():
        # One actual near-critical triangle. The first half of this cell is
        # degenerate, so the second half alone carries one lens-area quantum.
        # Its width is ~1e-4 pixel at a global pixel coordinate near 2000.
        node_x, node_y = thin_cell("cuda")
        grid = mc.PlaneGrid((4096, 4096), (10.76080777812415, 10.76080777812415))
        row, column = 2019, 2073
        shape = (8, 16)
        dy, dx = grid.pixel_scale_uas
        xmin, _, ymin, _ = grid.bounds_uas
        crop = mc.PlaneGrid(
            shape,
            (shape[0] * dy, shape[1] * dx),
            (ymin + (row + shape[0] / 2) * dy, xmin + (column + shape[1] / 2) * dx),
        )
        area = 8.884576280266643e-5
        reference = rasterize_triangles_exact_eager(
            triangles_from_node_lattices(node_x.cpu().double(), node_y.cpu().double()),
            crop,
            lens_area_per_triangle_uas2=area,
        )
        return node_x, node_y, grid, crop, (row, column), area, reference

    @staticmethod
    def kwargs(grid, area):
        dy, dx = grid.pixel_scale_uas
        xmin, _, ymin, _ = grid.bounds_uas
        return dict(
            xmin=xmin,
            ymin=ymin,
            pixel_size_x=dx,
            pixel_size_y=dy,
            lens_area_per_triangle_uas2=area,
        )

    @staticmethod
    def assert_projection_close(actual, expected, mass):
        # A ~1e-4-pixel-wide triangle has finite float32 edge-rounding error.
        # Bound its absolute per-pixel error by 0.1% of the conserved quantum,
        # rather than requiring relative precision in almost-empty pixels.
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=mass * 1e-3)

    def test_thin_triangle_matches_float64_clipping_and_conserves_area(self):
        x, y, grid, _, (row, column), area, reference = self.fixture()
        actual = rasterize_cells_triton(
            x, y, shape=grid.shape, **self.kwargs(grid, area)
        )
        cropped = actual[row : row + 8, column : column + 16].cpu().double()
        expected_mass = area / (grid.pixel_scale_uas[0] * grid.pixel_scale_uas[1])
        self.assert_projection_close(cropped, reference, expected_mass)
        self.assertAlmostEqual(
            float(actual.sum()), expected_mass, delta=expected_mass * 2e-3
        )
        # Even a near-degenerate triangle cannot deposit more than its entire
        # conserved mass in a single pixel. The original kernel violated this.
        self.assertLessEqual(float(actual.max()), expected_mass * 1.002)
        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_grid_padding_does_not_change_thin_triangle_photometry(self):
        x, y, grid, crop, (row, column), area, _ = self.fixture()
        padded = rasterize_cells_triton(
            x, y, shape=grid.shape, **self.kwargs(grid, area)
        )
        local = rasterize_cells_triton(
            x, y, shape=crop.shape, **self.kwargs(crop, area)
        )
        mass = area / (grid.pixel_scale_uas[0] * grid.pixel_scale_uas[1])
        self.assert_projection_close(
            padded[row : row + 8, column : column + 16], local, mass
        )

    def test_temporal_and_indexed_thin_triangles_match_clipping(self):
        x, y, grid, _, (row, column), area, reference = self.fixture()
        for indexed in (False, True):
            with self.subTest(indexed=indexed):
                workspace = TritonRasterWorkspace.create(
                    grid.shape, device=x.device, frames=2
                )
                if indexed:
                    xx, yy = x.repeat(2, 1, 1), y.repeat(2, 1, 1)
                    kwargs = dict(
                        cell_frame_index=torch.tensor([0, 1], device=x.device)
                    )
                else:
                    xx, yy = (
                        x.unsqueeze(0).repeat(2, 1, 1, 1),
                        y.unsqueeze(0).repeat(2, 1, 1, 1),
                    )
                    kwargs = {}
                accumulate_cells_triton(
                    workspace, xx, yy, **self.kwargs(grid, area), **kwargs
                )
                result = (
                    workspace.result()[:, row : row + 8, column : column + 16]
                    .cpu()
                    .double()
                )
                mass = area / (grid.pixel_scale_uas[0] * grid.pixel_scale_uas[1])
                self.assert_projection_close(result, reference.repeat(2, 1, 1), mass)

    def test_compaction_retains_triangles_smaller_than_global_coordinate_ulp(self):
        grid = mc.PlaneGrid((4096, 4096), (10.76080777812415, 10.76080777812415))
        x = torch.tensor([[[0.0, 4e-8], [0.0, 4e-8]]], device="cuda")
        y = torch.tensor([[[0.0, 0.0], [4e-8, 4e-8]]], device="cuda")
        dy, dx = grid.pixel_scale_uas
        actual = rasterize_cells_triton(
            x, y, shape=grid.shape, **self.kwargs(grid, dx * dy)
        )
        # Both nonzero triangles lie inside the source and each carries unit
        # mass. A global-coordinate area test rounded both triangles to zero.
        self.assertAlmostEqual(float(actual.sum()), 2.0, delta=2e-3)

    def test_degenerate_cells_do_not_deposit_mass(self):
        grid = mc.PlaneGrid((16, 16), (4.0, 4.0))
        for value in (0.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                x = torch.full((1, 2, 2), value, device="cuda")
                y = torch.zeros_like(x)
                actual = rasterize_cells_triton(
                    x, y, shape=grid.shape, **self.kwargs(grid, 0.01)
                )
                torch.testing.assert_close(
                    actual, torch.zeros_like(actual), rtol=0.0, atol=0.0
                )


if __name__ == "__main__":
    unittest.main()
