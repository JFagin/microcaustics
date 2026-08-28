"""CUDA/Triton parity tests skipped automatically on portable CPU installs."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch

os.environ.setdefault(
    "TRITON_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "microcaustics-triton-tests"),
)

import microcaustics as mc
from microcaustics.solvers import (
    BatchedTaylorFarFieldApproximation,
    TaylorFarFieldApproximation,
)
from microcaustics.solvers.triton_ipm import (
    rasterize_cells_triton,
    triton_ipm_available,
)
from microcaustics.solvers.triton_scout import (
    dilate_source_tiles_triton,
    select_source_tiles_triton,
    triton_scout_available,
)
from microcaustics.solvers.triton_taylor import triton_taylor_available


@unittest.skipUnless(triton_taylor_available(), "CUDA Triton is unavailable")
class TritonTaylorTests(unittest.TestCase):
    """Compare fused float32 kernels with the readable eager implementation."""

    def test_compact_caustic_kernels_match_portable_predicates(self) -> None:
        """Exercise all marching cases and the shared-vertex crossing rule."""

        from microcaustics.caustics.anchor_gauge import (
            _crossing_counts_and_distances_portable,
        )
        from microcaustics.caustics.marching import marching_squares_zero
        from microcaustics.caustics.triton_caustics import (
            batched_caustic_crossings_distances_triton,
            marching_squares_zero_triton,
            sparse_marching_squares_zero_triton,
        )

        grid = mc.PlaneGrid((2, 2), (2.0, 2.0))
        x_grid, y_grid = grid.mesh(device="cuda", dtype=torch.float32)

        def canonical(segments: torch.Tensor) -> torch.Tensor:
            values = segments.detach().cpu()
            first = torch.minimum(values[:, 0], values[:, 1])
            second = torch.maximum(values[:, 0], values[:, 1])
            values = torch.cat((first, second), dim=1)
            key = values[:, 0] * 1000.0 + values[:, 1]
            return values[torch.argsort(key)]

        for case in range(16):
            corners = torch.tensor(
                [1.0 if case & (1 << index) else -1.0 for index in range(4)],
                device="cuda",
            )
            field = torch.stack(
                (
                    torch.stack((corners[0], corners[1])),
                    torch.stack((corners[3], corners[2])),
                )
            )
            portable = marching_squares_zero(field, x_grid, y_grid)
            fused = marching_squares_zero_triton(field, grid)
            sparse, sparse_boundary = sparse_marching_squares_zero_triton(
                corners[0:1],
                corners[1:2],
                corners[2:3],
                corners[3:4],
                torch.tensor([-0.5], device="cuda"),
                torch.tensor([0.5], device="cuda"),
                torch.tensor([-0.5], device="cuda"),
                torch.tensor([0.5], device="cuda"),
                torch.zeros((4, 1), device="cuda", dtype=torch.bool),
            )
            self.assertEqual(portable.shape, fused.shape)
            torch.testing.assert_close(
                canonical(fused),
                canonical(portable),
                rtol=0.0,
                atol=1.0e-6,
            )
            torch.testing.assert_close(
                canonical(sparse),
                canonical(portable),
                rtol=0.0,
                atol=1.0e-6,
            )
            self.assertEqual(tuple(sparse_boundary.shape), (len(sparse), 2))
            self.assertFalse(bool(sparse_boundary.any()))

        segments = torch.tensor(
            [[
                [[-1.0, -1.0], [1.0, -1.0]],
                [[1.0, -1.0], [1.0, 1.0]],
                [[1.0, 1.0], [-1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, -1.0]],
            ]],
            device="cuda",
        )
        valid = torch.ones((1, 4), device="cuda", dtype=torch.bool)
        anchors = torch.tensor([[-2.0, 0.0], [-2.0, 1.0]], device="cuda")
        points = torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
            device="cuda",
        )
        fused_counts, fused_distances = (
            batched_caustic_crossings_distances_triton(
                segments,
                valid,
                anchors,
                points,
                points,
                block_segments=64,
            )
        )
        portable_counts, portable_distances = (
            _crossing_counts_and_distances_portable(
                segments,
                valid,
                anchors,
                points,
                points,
                point_chunk_size=3,
                segment_chunk_size=4,
            )
        )
        torch.testing.assert_close(fused_counts, portable_counts)
        torch.testing.assert_close(fused_distances, portable_distances)

    def test_far_field_coefficients_rays_and_jacobian_match_eager(self) -> None:
        """Exercise local packs, far coefficients, ray tracing, and det A."""

        generator = torch.Generator(device="cuda").manual_seed(123)
        far_x = torch.rand(48, generator=generator, device="cuda") * 4.0 + 4.0
        far_y = torch.rand(48, generator=generator, device="cuda") * 12.0 - 6.0
        local_x = torch.tensor([-0.7, 0.4, 1.1], device="cuda")
        local_y = torch.tensor([0.6, -0.8, 0.3], device="cuda")
        x = torch.cat((far_x, local_x))
        y = torch.cat((far_y, local_y))
        radii = torch.rand(51, generator=generator, device="cuda") * 0.03 + 0.02
        field = mc.PointMassField(x, y, radii)
        macro = mc.MacroLens(
            0.32,
            0.17,
            shear_angle_rad=0.23,
            smooth_matter_fraction=0.35,
        )
        config = mc.FarFieldApproxConfig(
            cells_per_axis=4,
            nodes_per_cell_axis=8,
            exact_radius_cells=1.0,
            taylor_order=4,
            center_translation_order=10,
        )
        eager = TaylorFarFieldApproximation(
            mc.MicrolensingSimulation.create(
                macro,
                field,
                runtime=mc.RuntimeConfig(device="cuda", backend="torch-eager"),
            ),
            mc.PlaneRegion((4.0, 4.0)),
            config,
        )
        fused = TaylorFarFieldApproximation(
            mc.MicrolensingSimulation.create(
                macro,
                field,
                runtime=mc.RuntimeConfig(
                    device="cuda",
                    backend="triton",
                    strict_backend=True,
                ),
            ),
            mc.PlaneRegion((4.0, 4.0)),
            config,
        )
        torch.testing.assert_close(
            fused.coefficient_real,
            eager.coefficient_real,
            rtol=2.0e-5,
            atol=8.0e-8,
        )
        torch.testing.assert_close(
            fused.coefficient_imag,
            eager.coefficient_imag,
            rtol=2.0e-5,
            atol=8.0e-8,
        )
        query_x = torch.linspace(-1.91, 1.89, 1025, device="cuda")
        query_y = torch.sin(query_x * 2.3) * 1.7
        eager_x, eager_y = eager.raytrace(query_x, query_y)
        fused_x, fused_y = fused.raytrace(query_x, query_y)
        torch.testing.assert_close(fused_x, eager_x, rtol=3.0e-6, atol=5.0e-7)
        torch.testing.assert_close(fused_y, eager_y, rtol=3.0e-6, atol=5.0e-7)
        eager_det = eager.jacobian_determinant(query_x, query_y)
        fused_det = fused.jacobian_determinant(query_x, query_y)
        torch.testing.assert_close(
            fused_det,
            eager_det,
            rtol=3.0e-6,
            atol=2.0e-5,
        )

    def test_temporal_far_field_query_matches_independent_frames(self) -> None:
        """The unified query queue preserves each frame's Taylor solution."""

        field = mc.PointMassField(
            torch.tensor([-1.2, -0.4, 0.7, 1.5], device="cuda"),
            torch.tensor([0.8, -1.1, 0.3, -0.5], device="cuda"),
            torch.tensor([0.12, 0.08, 0.1, 0.07], device="cuda"),
            torch.tensor([0.002, -0.001, 0.0015, -0.002], device="cuda"),
            torch.tensor([-0.001, 0.002, -0.0015, 0.001], device="cuda"),
        )
        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(
                0.25,
                0.12,
                shear_angle_rad=0.17,
                smooth_matter_fraction=0.2,
            ),
            field,
            runtime=mc.RuntimeConfig(
                device="cuda",
                backend="triton",
                strict_backend=True,
            ),
        )
        config = mc.FarFieldApproxConfig(
            cells_per_axis=4,
            nodes_per_cell_axis=8,
            exact_radius_cells=1.0,
            taylor_order=4,
            center_translation_order=10,
        )
        region = mc.PlaneRegion((4.0, 4.0))
        frames = tuple(
            TaylorFarFieldApproximation(simulation, region, config, time_days=value)
            for value in (0.0, 2.0, 4.0)
        )
        batch = BatchedTaylorFarFieldApproximation(frames)
        x = torch.linspace(-1.8, 1.8, 513, device="cuda")
        y = 1.6 * torch.sin(x * 1.3)
        expected_x, expected_y = zip(
            *(item.raytrace(x, y) for item in frames),
            strict=True,
        )
        actual_x, actual_y = batch.raytrace(x, y)
        torch.testing.assert_close(actual_x, torch.stack(expected_x), rtol=0, atol=2e-6)
        torch.testing.assert_close(actual_y, torch.stack(expected_y), rtol=0, atol=2e-6)
        ragged_x = (x[:127], x[:0], x[127:])
        ragged_y = (y[:127], y[:0], y[127:])
        expected_ragged = tuple(
            item.raytrace(row_x, row_y)
            for item, row_x, row_y in zip(
                frames,
                ragged_x,
                ragged_y,
                strict=True,
            )
        )
        actual_ragged_x, actual_ragged_y = batch.raytrace_ragged(
            ragged_x,
            ragged_y,
        )
        for frame in range(len(frames)):
            torch.testing.assert_close(
                actual_ragged_x[frame],
                expected_ragged[frame][0],
                rtol=0,
                atol=2e-6,
            )
            torch.testing.assert_close(
                actual_ragged_y[frame],
                expected_ragged[frame][1],
                rtol=0,
                atol=2e-6,
            )
        expected_det = torch.stack([item.jacobian_determinant(x, y) for item in frames])
        torch.testing.assert_close(
            batch.jacobian_determinant(x, y),
            expected_det,
            rtol=0,
            atol=3e-5,
        )
        from microcaustics.solvers import temporal_taylor_far_fields

        batch_config = mc.FarFieldApproxConfig(
            cells_per_axis=4,
            nodes_per_cell_axis=8,
            exact_radius_cells=1.0,
            taylor_order=4,
            center_translation_order=10,
        )
        exact_batched, metadata = temporal_taylor_far_fields(
            simulation,
            region,
            batch_config,
            [0.0, 2.0, 4.0],
        )
        self.assertTrue(metadata["far_field_batched_accumulator"])
        self.assertTrue(metadata["far_field_exact_each_frame"])
        torch.testing.assert_close(
            exact_batched[0].coefficient_real,
            frames[0].coefficient_real,
            rtol=2e-5,
            atol=1e-7,
        )
        torch.testing.assert_close(
            exact_batched[2].coefficient_real,
            frames[2].coefficient_real,
            rtol=2e-5,
            atol=1e-7,
        )
        torch.testing.assert_close(
            exact_batched[1].coefficient_real,
            frames[1].coefficient_real,
            rtol=2e-5,
            atol=1e-7,
        )

    @unittest.skipUnless(triton_ipm_available(), "CUDA Triton IPM is unavailable")
    def test_public_dynamic_ipm_fuses_tree_queries_and_rasterization(self) -> None:
        """Exercise the complete production temporal dispatcher on CUDA."""

        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(0.1, 0.05),
            mc.PointMassField(
                torch.tensor([-0.6, 0.5], device="cuda"),
                torch.tensor([0.4, -0.3], device="cuda"),
                torch.tensor([0.18, 0.16], device="cuda"),
                torch.tensor([0.001, -0.0015], device="cuda"),
                torch.tensor([-0.001, 0.0005], device="cuda"),
            ),
            runtime=mc.RuntimeConfig(
                device="cuda",
                backend="triton",
                strict_backend=True,
            ),
        )
        maps = list(
            simulation.dynamic_maps(
                mc.PlaneRegion((3.0, 3.0)),
                mc.PlaneGrid((8, 9), (1.5, 1.6)),
                [0.0, 1.0, 2.0],
                method=mc.IPMConfig(
                    rays=144,
                    scout_ratio=2,
                    refinement=2,
                    virtual_refinement=4,
                    tiled=True,
                    cell_chunk_size=64,
                    far_field_approx=mc.FarFieldApproxConfig(
                        cells_per_axis=4,
                        nodes_per_cell_axis=8,
                    ),
                ),
                schedule=mc.DynamicConfig(
                    temporal_batch_size=2,
                    scout_refresh_frames=3,
                    endpoint_union=True,
                    fused_temporal_ipm=True,
                    pad_temporal_batches=True,
                ),
            )
        )
        self.assertEqual(len(maps), 3)
        self.assertTrue(all(item.metadata["dynamic_temporal_solver_fused"] for item in maps))
        self.assertTrue(all(item.metadata["temporal_far_field_query_fused"] for item in maps))
        self.assertTrue(all(item.metadata["far_field_exact_each_frame"] for item in maps))
        self.assertTrue(maps[0].metadata["far_field_batched_accumulator"])
        self.assertTrue(maps[1].metadata["far_field_batched_accumulator"])
        # The padded tail reuses the last real frame's complete far-field approximation.
        # it does not need a multi-anchor coefficient build of its own.
        self.assertFalse(maps[2].metadata["far_field_batched_accumulator"])
        self.assertEqual(maps[-1].metadata["temporal_batch_padded_frames"], 1)

    @unittest.skipUnless(triton_ipm_available(), "CUDA Triton IPM is unavailable")
    def test_public_padded_batch_and_cell_chunk_are_numerically_invariant(self) -> None:
        """Static launch shapes may not alter any real dynamic map frame."""

        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(
                0.14,
                0.07,
                shear_angle_rad=0.2,
                smooth_matter_fraction=0.25,
            ),
            mc.PointMassField(
                torch.tensor([-0.7, 0.25, 0.9], device="cuda"),
                torch.tensor([0.45, -0.55, 0.3], device="cuda"),
                torch.tensor([0.19, 0.15, 0.12], device="cuda"),
                torch.tensor([0.002, -0.001, 0.0015], device="cuda"),
                torch.tensor([-0.001, 0.0015, -0.002], device="cuda"),
            ),
            runtime=mc.RuntimeConfig(
                device="cuda",
                backend="triton",
                strict_backend=True,
            ),
        )
        region = mc.PlaneRegion((3.0, 3.0))
        grid = mc.PlaneGrid((9, 11), (1.6, 1.8))
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        outputs = []
        for chunk, batch, pad in ((17, 1, False), (31, 3, True), (128, 4, True)):
            outputs.append(
                tuple(
                    simulation.dynamic_maps(
                        region,
                        grid,
                        times,
                        method=mc.IPMConfig(
                            rays=100,
                            refinement=2,
                            virtual_refinement=4,
                            tiled=False,
                            cell_chunk_size=chunk,
                            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
                        ),
                        schedule=mc.DynamicConfig(
                            temporal_batch_size=batch,
                            fused_temporal_ipm=True,
                            pad_temporal_batches=pad,
                            scout_refresh_frames=1,
                        ),
                    )
                )
            )
        for candidate in outputs[1:]:
            for actual, reference in zip(candidate, outputs[0], strict=True):
                torch.testing.assert_close(
                    actual.values,
                    reference.values,
        # Atomic accumulation order changes with launch shape.
                    # the largest observed difference is a few float32 ulps.
                    rtol=1.0e-5,
                    atol=3.0e-5,
                )
        self.assertEqual(outputs[1][-1].metadata["temporal_batch_padded_frames"], 1)
        self.assertEqual(outputs[2][-1].metadata["temporal_batch_padded_frames"], 3)

    @unittest.skipUnless(triton_ipm_available(), "CUDA Triton IPM is unavailable")
    def test_direct_cell_rasterizer_matches_exact_reference(self) -> None:
        """Check several virtual refinements on a rectangular source grid."""

        from microcaustics.solvers import (
            rasterize_triangles_exact_eager,
            triangles_from_node_lattices,
        )

        grid = mc.PlaneGrid((19, 23), (3.8, 4.6))
        dy, dx = grid.pixel_scale_uas
        xmin, _, ymin, _ = grid.bounds_uas
        for virtual in (1, 2, 3, 4, 8):
            coordinate = torch.linspace(0.0, 1.0, virtual + 1, device="cuda")
            yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
            node_x = torch.stack(
                (
                    -1.7 + 1.2 * xx + 0.08 * xx * yy,
                    0.15 + 1.1 * xx - 0.05 * yy.square(),
                )
            )
            node_y = torch.stack(
                (
                    -1.5 + 1.25 * yy + 0.04 * xx.square(),
                    0.1 + 1.0 * yy + 0.06 * xx * yy,
                )
            )
            lens_area = 0.037 / (2.0 * virtual * virtual)
            expected = rasterize_triangles_exact_eager(
                triangles_from_node_lattices(node_x, node_y),
                grid,
                lens_area_per_triangle_uas2=lens_area,
            )
            actual = rasterize_cells_triton(
                node_x,
                node_y,
                xmin=xmin,
                ymin=ymin,
                pixel_size_x=dx,
                pixel_size_y=dy,
                shape=grid.shape,
                lens_area_per_triangle_uas2=lens_area,
            )
            torch.testing.assert_close(actual, expected, rtol=3.0e-4, atol=2.0e-6)

    @unittest.skipUnless(triton_ipm_available(), "CUDA Triton IPM is unavailable")
    def test_temporal_direct_cell_raster_matches_independent_launches(self) -> None:
        """A fused temporal workspace must equal one launch per map."""

        from microcaustics.solvers.triton_ipm import (
            TritonRasterWorkspace,
            accumulate_cells_triton,
        )

        grid = mc.PlaneGrid((17, 21), (3.4, 4.2))
        dy, dx = grid.pixel_scale_uas
        xmin, _, ymin, _ = grid.bounds_uas
        coordinate = torch.linspace(0.0, 1.0, 5, device="cuda")
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        first_x = torch.stack((-1.8 + xx, 0.1 + 0.8 * xx + 0.03 * yy))
        first_y = torch.stack((-1.4 + yy, 0.2 + 0.9 * yy - 0.02 * xx))
        node_x = torch.stack((first_x, first_x + 0.07, first_x - 0.04))
        node_y = torch.stack((first_y, first_y - 0.03, first_y + 0.05))
        kwargs = dict(
            xmin=xmin,
            ymin=ymin,
            pixel_size_x=dx,
            pixel_size_y=dy,
            lens_area_per_triangle_uas2=0.002,
        )
        expected = torch.stack(
            [
                rasterize_cells_triton(
                    node_x[index],
                    node_y[index],
                    shape=grid.shape,
                    **kwargs,
                )
                for index in range(3)
            ]
        )
        workspace = TritonRasterWorkspace.create(
            grid.shape,
            device=torch.device("cuda"),
            frames=3,
        )
        accumulate_cells_triton(workspace, node_x, node_y, **kwargs)
        torch.testing.assert_close(
            workspace.result(),
            expected,
            rtol=0.0,
            atol=2.0e-6,
        )

    @unittest.skipUnless(triton_ipm_available(), "CUDA Triton IPM is unavailable")
    def test_full_field_triton_ipm_supports_general_refinement(self) -> None:
        """Exercise the public dispatcher beyond the production r=2, v=4 case."""

        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(0.0, 0.0),
            mc.PointMassField(torch.empty(0), torch.empty(0), torch.empty(0)),
            runtime=mc.RuntimeConfig(
                device="cuda",
                backend="triton",
                strict_backend=True,
            ),
        )
        for refinement, virtual in ((1, 1), (2, 4), (3, 5)):
            with self.subTest(refinement=refinement, virtual=virtual):
                result = simulation.magnification_map(
                    mc.PlaneRegion((2.0, 3.0)),
                    mc.PlaneGrid((7, 11), (2.0, 3.0)),
                    method=mc.IPMConfig(
                        rays=35,
                        refinement=refinement,
                        virtual_refinement=virtual,
                        tiled=False,
                        cell_chunk_size=9,
                        far_field_approx=mc.FarFieldApproxConfig(enabled=False),
                    ),
                )
                torch.testing.assert_close(
                    result.values,
                    torch.ones_like(result.values),
                    rtol=2.0e-5,
                    atol=2.0e-5,
                )
                self.assertEqual(
                    result.metadata["rasterizer"],
                    "triton_direct_cell_scanline",
                )

    @unittest.skipUnless(triton_scout_available(), "CUDA Triton scout is unavailable")
    def test_fused_source_scout_matches_torch(self) -> None:
        """Compare corner/center selection and dilation on a general grid."""

        generator = torch.Generator(device="cuda").manual_seed(987)
        corner_x = torch.randn((18, 23), generator=generator, device="cuda")
        corner_y = torch.randn((18, 23), generator=generator, device="cuda")
        center_x = torch.randn((17, 22), generator=generator, device="cuda")
        center_y = torch.randn((17, 22), generator=generator, device="cuda")
        bounds = (-0.4, 0.7, -0.8, 0.2)
        xcorners = torch.stack(
            (
                corner_x[:-1, :-1],
                corner_x[:-1, 1:],
                corner_x[1:, :-1],
                corner_x[1:, 1:],
            ),
            dim=-1,
        )
        ycorners = torch.stack(
            (
                corner_y[:-1, :-1],
                corner_y[:-1, 1:],
                corner_y[1:, :-1],
                corner_y[1:, 1:],
            ),
            dim=-1,
        )
        xmin, xmax, ymin, ymax = bounds
        expected = (
            (xcorners.amin(dim=-1) <= xmax)
            & (xcorners.amax(dim=-1) >= xmin)
            & (ycorners.amin(dim=-1) <= ymax)
            & (ycorners.amax(dim=-1) >= ymin)
        ) | (
            (center_x >= xmin)
            & (center_x <= xmax)
            & (center_y >= ymin)
            & (center_y <= ymax)
        )
        actual = select_source_tiles_triton(
            corner_x,
            corner_y,
            bounds=bounds,
            center_x=center_x,
            center_y=center_y,
        )
        self.assertTrue(torch.equal(actual, expected))
        for radius in (1, 2, 4):
            expected_dilated = torch.nn.functional.max_pool2d(
                expected[None, None].float(),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )[0, 0].bool()
            actual_dilated = dilate_source_tiles_triton(actual, radius)
            self.assertTrue(torch.equal(actual_dilated, expected_dilated))

    @unittest.skipUnless(triton_scout_available(), "CUDA Triton scout is unavailable")
    def test_public_tiled_ipm_uses_fused_scout(self) -> None:
        """Exercise the fused scout and rasterizer through the public API."""

        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(0.0, 0.0),
            mc.PointMassField(torch.empty(0), torch.empty(0), torch.empty(0)),
            runtime=mc.RuntimeConfig(
                device="cuda",
                backend="triton",
                strict_backend=True,
            ),
        )
        result = simulation.magnification_map(
            mc.PlaneRegion((4.0, 5.0)),
            mc.PlaneGrid((8, 10), (2.0, 2.5)),
            method=mc.IPMConfig(
                rays=240,
                scout_ratio=3,
                refinement=2,
                virtual_refinement=4,
                tiled=True,
                cell_chunk_size=64,
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            ),
        )
        self.assertEqual(result.metadata["scout_selector"], "triton_fused")
        self.assertEqual(result.metadata["rasterizer"], "triton_direct_cell_scanline")


if __name__ == "__main__":
    unittest.main()
