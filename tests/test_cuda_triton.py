"""CUDA/Triton parity tests skipped automatically on portable CPU installs."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
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
    materialize_biquadratic_v4_triton,
    rasterize_cells_triton,
    triton_ipm_available,
)
from microcaustics.solvers.triton_scout import (
    dilate_source_tiles_triton,
    select_source_tiles_triton,
    triton_scout_available,
)
from microcaustics.solvers.triton_taylor import triton_taylor_available


def _small_cuda_system(*, backend: str, offset: float = 0.0):
    """Build a small physical system for complete CUDA backend tests."""

    distances = mc.LensingDistances(1.0e25, 2.0e25, 1.2e25)
    source_grid = mc.PlaneGrid((9, 11), (1.8, 2.0))
    pixel_scale_m = distances.uas_to_source_length(
        source_grid.pixel_scale_uas,
        dtype=torch.float64,
    )
    source = mc.StaticSource(
        torch.ones((9, 11, 1), dtype=torch.float32),
        mc.SourceGeometry(
            shape=source_grid.shape,
            pixel_scale_m=(float(pixel_scale_m[0]), float(pixel_scale_m[1])),
            wavelengths_angstrom=(6_000.0,),
            band_names=("optical",),
        ),
    )
    stars = mc.PointMassField._from_einstein_radii(
        torch.tensor([-0.65 + offset, 0.35 + offset, 0.9 + offset]),
        torch.tensor([0.4, -0.5, 0.25]),
        velocity_x_uas_per_day=torch.tensor([0.002, -0.001, 0.0015]),
        velocity_y_uas_per_day=torch.tensor([-0.001, 0.0015, -0.002]),
        einstein_radius_uas=torch.tensor([0.18, 0.15, 0.12]),
    )
    return mc.MicrolensingSystem(
        macro=mc.MacroLens(0.14, 0.07, shear_angle_deg=11.0),
        distances=distances,
        source=source,
        source_grid=source_grid,
        stars=stars,
        integration_domain="scout",
        lens_region=mc.PlaneRegion((3.0, 3.0)),
        caustic_grid_shape=17,
        runtime=mc.RuntimeConfig(
            device="cuda",
            backend=backend,
            strict_backend=True,
        ),
    )


def _small_production_method(*, compact_sparse_nodes: bool = True):
    return mc.IPMConfig(
        rays=144,
        scout_ratio=2,
        refinement=2,
        virtual_refinement=4,
        tiled=True,
        cell_chunk_size=64,
        compact_sparse_nodes=compact_sparse_nodes,
        far_field_approx=mc.FarFieldApproxConfig(
            cells_per_axis=4,
            nodes_per_cell_axis=8,
            exact_radius_cells=1.0,
        ),
    )


def _small_dynamic_schedule():
    return mc.DynamicConfig(
        temporal_batch_size=2,
        scout_refresh_frames=2,
        fused_temporal_ipm=True,
    )


def _small_caustic_config():
    return mc.CausticConfig(
        far_field_approx=mc.FarFieldApproxConfig(
            cells_per_axis=4,
            nodes_per_cell_axis=8,
            exact_radius_cells=1.0,
        ),
        temporal_batch_size=2,
        jacobian_chunk_size=257,
        minimum_determinant_sign_pixels=1,
        anchor_count=3,
        gauge_count=3,
        minimum_alignment_gauges=1,
    )


@unittest.skipUnless(triton_taylor_available(), "CUDA Triton is unavailable")
class TritonTaylorTests(unittest.TestCase):
    """Compare fused float32 kernels with the readable eager implementation."""

    def test_cross_disk_kerr_setup_matches_serial(self):
        runtime = mc.RuntimeConfig(
            device="cuda",
            backend="triton",
            strict_backend=True,
            warn_on_compile=False,
        )
        distances = mc.LensingDistances.from_redshifts(0.0395, 1.695)
        driver = mc.TabulatedDrivingSignal([-100.0, 100.0], [1.0, 1.0])
        base = mc.KerrDiskModel(
            black_hole_mass_solar=10.0**9.08,
            eddington_ratio=0.34,
            bands_angstrom={"g": 4827},
            source_redshift=1.695,
            source_grid_shape=64,
            observer_coordinate_chunk_size=4096,
            lamppost_nalpha=32,
            lamppost_radial_bins=32,
            driving_signal=driver,
            compile_solver=False,
            warn_on_compile=False,
        )
        models = (
            replace(base, spin=0.74, inclination_deg=10.0),
            replace(base, spin=0.31, inclination_deg=41.0),
        )
        pooled = mc.batched_pixelate_sources(
            models, distances, batch_size=2, runtime=runtime
        )
        serial = tuple(model.pixelate(distances, runtime=runtime) for model in models)
        self.assertTrue(
            all(
                source.transfer.metadata["coordinate_cross_disk_pooled"]
                for source in pooled
            )
        )
        for actual, expected in zip(pooled, serial, strict=True):
            torch.testing.assert_close(
                actual.delay_days,
                expected.delay_days,
                equal_nan=True,
                rtol=2.0e-5,
                atol=2.0e-3,
            )

    def test_dense_temporal_marching_matches_scalar(self):
        from microcaustics.caustics.triton_caustics import (
            batched_dense_marching_squares_zero_triton,
            marching_squares_zero_triton,
        )

        grid = mc.PlaneGrid((37, 43), (3.0, 4.0))
        generator = torch.Generator(device="cuda").manual_seed(0)
        fields = torch.randn((3, *grid.shape), device="cuda", generator=generator)
        fields[0].fill_(1)
        actual = batched_dense_marching_squares_zero_triton(fields, grid)
        for field, row in zip(fields, actual, strict=True):
            expected = marching_squares_zero_triton(field, grid)

            # The sparse marcher reverses the endpoints of ambiguous cells.
            def canonical(segments):
                a, b = segments.unbind(dim=1)
                swap = (a[:, 0] > b[:, 0]) | (
                    (a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1])
                )
                return torch.stack(
                    (
                        torch.where(swap[:, None], b, a),
                        torch.where(swap[:, None], a, b),
                    ),
                    dim=1,
                )

            torch.testing.assert_close(
                canonical(row), canonical(expected), atol=2e-6, rtol=2e-6
            )

    def test_dense_temporal_caustics_batch_endpoint_queries(self):
        from dataclasses import replace
        from unittest.mock import patch

        from microcaustics.caustics.production import caustic_fields_from_far_fields

        system = _small_cuda_system(backend="triton")
        simulation = system.realize().simulation
        grid = mc.PlaneGrid((129, 128), (3.0, 3.0))
        config = _small_caustic_config()
        times = (0.0, 1.0)
        original = BatchedTaylorFarFieldApproximation.raytrace_ragged
        with patch.object(
            BatchedTaylorFarFieldApproximation,
            "raytrace_ragged",
            autospec=True,
            side_effect=original,
        ) as query:
            actual = caustic_fields_from_far_fields(simulation, grid, times, config)
        self.assertEqual(query.call_count, 1)
        expected = tuple(
            caustic_fields_from_far_fields(
                simulation, grid, (time,), replace(config, temporal_batch_size=1)
            )[0]
            for time in times
        )
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                a.critical_segments_uas, b.critical_segments_uas, atol=3e-6, rtol=3e-6
            )
            torch.testing.assert_close(
                a.caustic_segments_uas, b.caustic_segments_uas, atol=3e-6, rtol=3e-6
            )
            torch.testing.assert_close(a.invalid_segment_mask, b.invalid_segment_mask)

    def test_compact_caustic_kernels_match_portable_predicates(self) -> None:
        """Exercise all marching cases and the shared-vertex crossing rule."""

        from microcaustics.caustics.anchor_gauge import (
            _crossing_counts_and_distances_portable,
        )
        from microcaustics.caustics.marching import marching_squares_zero
        from microcaustics.caustics.triton_caustics import (
            batched_caustic_crossings_distances_triton,
            batched_sparse_marching_squares_zero_triton,
            marching_squares_zero_triton,
            ragged_sparse_marching_squares_zero_triton,
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

        # The production temporal handoff keeps the compact marching output
        # flat and records its owning frame without changing any geometry.
        signs = torch.tensor(
            [[1.0, -1.0, -1.0, -1.0], [-1.0, 1.0, -1.0, -1.0]],
            device="cuda",
        )
        flat_result = batched_sparse_marching_squares_zero_triton(
            signs[:, 0:1],
            signs[:, 1:2],
            signs[:, 2:3],
            signs[:, 3:4],
            torch.tensor([-0.5], device="cuda"),
            torch.tensor([0.5], device="cuda"),
            torch.tensor([-0.5], device="cuda"),
            torch.tensor([0.5], device="cuda"),
            torch.zeros((4, 1), device="cuda", dtype=torch.bool),
            return_flat=True,
        )
        rows, boundary_rows, flat_segments, flat_boundaries, frames, lengths = (
            flat_result
        )
        torch.testing.assert_close(flat_segments, torch.cat(rows))
        torch.testing.assert_close(flat_boundaries, torch.cat(boundary_rows))
        self.assertEqual(lengths, tuple(map(len, rows)))
        torch.testing.assert_close(
            frames,
            torch.repeat_interleave(
                torch.arange(2, device="cuda", dtype=torch.int32),
                torch.tensor(lengths, device="cuda"),
            ),
        )

        # Ragged temporal queues preserve the same per-frame geometry without
        # padding either frame to the other's active-cell count.
        ragged_signs = torch.tensor(
            [
                [1.0, -1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, -1.0],
            ],
            device="cuda",
        )
        ragged = ragged_sparse_marching_squares_zero_triton(
            ragged_signs[:, 0],
            ragged_signs[:, 1],
            ragged_signs[:, 2],
            ragged_signs[:, 3],
            torch.full((3,), -0.5, device="cuda"),
            torch.full((3,), 0.5, device="cuda"),
            torch.full((3,), -0.5, device="cuda"),
            torch.full((3,), 0.5, device="cuda"),
            torch.zeros((4, 3), device="cuda", dtype=torch.bool),
            torch.tensor([0, 1, 1], device="cuda"),
            2,
        )
        ragged_rows, _, ragged_flat, _, ragged_frames, ragged_lengths = ragged
        self.assertEqual(ragged_lengths, (1, 2))
        torch.testing.assert_close(ragged_flat, torch.cat(ragged_rows))
        torch.testing.assert_close(
            ragged_frames,
            torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32),
        )
        for index, corners in enumerate(ragged_signs):
            expected_row, _ = sparse_marching_squares_zero_triton(
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
            frame = 0 if index == 0 else 1
            offset = 0 if index < 2 else 1
            torch.testing.assert_close(
                canonical(ragged_rows[frame][offset : offset + 1]),
                canonical(expected_row),
            )

        segments = torch.tensor(
            [
                [
                    [[-1.0, -1.0], [1.0, -1.0]],
                    [[1.0, -1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [-1.0, 1.0]],
                    [[-1.0, 1.0], [-1.0, -1.0]],
                ]
            ],
            device="cuda",
        )
        valid = torch.ones((1, 4), device="cuda", dtype=torch.bool)
        anchors = torch.tensor([[-2.0, 0.0], [-2.0, 1.0]], device="cuda")
        points = torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
            device="cuda",
        )
        fused_counts, fused_distances = batched_caustic_crossings_distances_triton(
            segments,
            valid,
            anchors,
            points,
            points,
            block_segments=64,
        )
        portable_counts, portable_distances = _crossing_counts_and_distances_portable(
            segments,
            valid,
            anchors,
            points,
            points,
            point_chunk_size=3,
            segment_chunk_size=4,
        )
        torch.testing.assert_close(fused_counts, portable_counts)
        torch.testing.assert_close(fused_distances, portable_distances)

    def test_diagnostic_maps_match_portable_float32_results(self) -> None:
        """Exercise full-grid winding, anchor/gauge, and distance products."""

        from microcaustics.caustics import label_caustic_fields
        from microcaustics.caustics.labels import distance_to_segments
        from microcaustics.results import CausticField

        generator = torch.Generator().manual_seed(314159)
        cpu_segments = (
            4.0 * torch.rand((193, 2, 2), generator=generator, dtype=torch.float32)
            - 2.0
        )
        grid = mc.PlaneGrid((37, 41), (4.5, 4.25), (0.11, -0.19))
        gpu_field = CausticField(
            cpu_segments.cuda(),
            cpu_segments.cuda(),
            grid,
        )
        cpu_field = CausticField(cpu_segments, cpu_segments, grid)

        x, y = grid.mesh(dtype=torch.float32)
        cpu_points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
        expected_distance = distance_to_segments(
            cpu_segments,
            cpu_points,
            point_chunk_size=113,
            segment_chunk_size=47,
        ).reshape(grid.shape)
        actual_distance = gpu_field.distance_map(
            grid,
            point_chunk_size=113,
            segment_chunk_size=47,
        ).values_uas
        torch.testing.assert_close(
            actual_distance.cpu(),
            expected_distance,
            rtol=2.0e-6,
            atol=2.0e-6,
        )

        expected_winding = cpu_field.winding_map(grid).values
        actual_winding = gpu_field.winding_map(grid).values
        torch.testing.assert_close(actual_winding.cpu(), expected_winding)
        torch.testing.assert_close(
            gpu_field.label_map(grid).values.cpu(),
            cpu_field.label_map(grid).values,
        )

        config = mc.CausticConfig(
            anchor_count=3,
            gauge_count=3,
            minimum_alignment_gauges=1,
            point_chunk_size=127,
            segment_chunk_size=47,
            triton_segment_block=64,
        )
        cpu_labeled = label_caustic_fields(
            (cpu_field,),
            grid.region,
            config,
            diagnostic_grid=grid,
            include_distance_map=True,
        )[0][0]
        gpu_labeled = label_caustic_fields(
            (gpu_field,),
            grid.region,
            config,
            diagnostic_grid=grid,
            include_distance_map=True,
        )[0][0]
        self.assertTrue(gpu_labeled.label_map.metadata["triton_grid_query"])
        torch.testing.assert_close(
            gpu_labeled.label_map.values.cpu(),
            cpu_labeled.label_map.values,
        )
        torch.testing.assert_close(
            gpu_labeled.distance_map.values_uas.cpu(),
            cpu_labeled.distance_map.values_uas,
            rtol=2.0e-6,
            atol=2.0e-6,
        )

    def test_complete_far_field_caustics_are_chunk_invariant(self) -> None:
        """Complete winding fields retain their geometry across query chunks."""

        system = _small_cuda_system(backend="triton")
        simulation = system.realize().simulation
        grid = mc.PlaneGrid((65, 67), (3.0, 3.0))
        config = mc.FarFieldApproxConfig(
            cells_per_axis=4,
            nodes_per_cell_axis=8,
            exact_radius_cells=1.0,
        )
        small_chunks = simulation.caustics(
            grid,
            far_field_approx=config,
            ray_chunk_size=127,
        )
        one_chunk = simulation.caustics(
            grid,
            far_field_approx=config,
            ray_chunk_size=grid.shape[0] * grid.shape[1],
        )
        automatic = simulation.caustics(grid, far_field_approx=config)
        self.assertEqual(small_chunks.metadata["marching_squares"], "triton_compact")
        self.assertEqual(
            automatic.metadata["regular_grid_determinant"], "triton_indexed"
        )
        self.assertEqual(automatic.metadata["determinant_ray_chunks"], 1)
        self.assertEqual(small_chunks.segment_count, one_chunk.segment_count)
        torch.testing.assert_close(
            small_chunks.critical_segments_uas,
            one_chunk.critical_segments_uas,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            small_chunks.caustic_segments_uas,
            one_chunk.caustic_segments_uas,
            rtol=2.0e-6,
            atol=2.0e-6,
        )
        torch.testing.assert_close(
            automatic.caustic_segments_uas,
            one_chunk.caustic_segments_uas,
            rtol=0.0,
            atol=0.0,
        )

    def test_regular_grid_determinant_matches_coordinate_query(self) -> None:
        """The coordinate-free full-grid kernel preserves float32 det A."""

        from microcaustics.solvers.triton_taylor import (
            evaluate_far_field_p4_regular_grid_jacobian_triton,
        )

        system = _small_cuda_system(backend="triton")
        simulation = system.realize().simulation
        grid = mc.PlaneGrid((71, 67), (3.1, 2.9), (0.13, -0.21))
        approximation = TaylorFarFieldApproximation(
            simulation,
            grid.region,
            mc.FarFieldApproxConfig(
                cells_per_axis=4,
                nodes_per_cell_axis=8,
                exact_radius_cells=1.0,
            ),
        )
        x, y = grid.mesh(device="cuda", dtype=torch.float32)
        expected = approximation.jacobian_determinant(x, y)
        actual = torch.empty_like(expected)
        split = 1_337
        evaluate_far_field_p4_regular_grid_jacobian_triton(
            approximation,
            grid,
            actual,
            stop=split,
        )
        evaluate_far_field_p4_regular_grid_jacobian_triton(
            approximation,
            grid,
            actual,
            start=split,
        )
        torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=2.0e-6)

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
        field = mc.PointMassField._from_einstein_radii(x, y, einstein_radius_uas=radii)
        macro = mc.MacroLens(
            0.32,
            0.17,
            shear_angle_deg=13.178029288008934,
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

        field = mc.PointMassField._from_einstein_radii(
            torch.tensor([-1.2, -0.4, 0.7, 1.5], device="cuda"),
            torch.tensor([0.8, -1.1, 0.3, -0.5], device="cuda"),
            velocity_x_uas_per_day=torch.tensor(
                [0.002, -0.001, 0.0015, -0.002], device="cuda"
            ),
            velocity_y_uas_per_day=torch.tensor(
                [-0.001, 0.002, -0.0015, 0.001], device="cuda"
            ),
            einstein_radius_uas=torch.tensor([0.12, 0.08, 0.1, 0.07], device="cuda"),
        )
        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(
                0.25,
                0.12,
                shear_angle_deg=9.740282517223996,
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
        indexed_x = torch.cat((x[:127], x[127:301], x[301:]))
        indexed_y = torch.cat((y[:127], y[127:301], y[301:]))
        indexed_frames = torch.cat(
            (
                torch.zeros(127, device="cuda", dtype=torch.int32),
                torch.ones(174, device="cuda", dtype=torch.int32),
                torch.full((212,), 2, device="cuda", dtype=torch.int32),
            )
        )
        expected_indexed_det = torch.cat(
            (
                expected_det[0, :127],
                expected_det[1, 127:301],
                expected_det[2, 301:],
            )
        )
        torch.testing.assert_close(
            batch.jacobian_determinant_indexed_flat(
                indexed_x,
                indexed_y,
                indexed_frames,
            ),
            expected_indexed_det,
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
        for actual, expected in zip(exact_batched, frames, strict=True):
            torch.testing.assert_close(
                actual.coefficient_imag,
                expected.coefficient_imag,
                rtol=2e-5,
                atol=1e-7,
            )
        for actual, expected in zip(exact_batched, frames, strict=True):
            torch.testing.assert_close(actual.local_x, expected.local_x)
            torch.testing.assert_close(actual.local_y, expected.local_y)
            torch.testing.assert_close(actual.local_mass, expected.local_mass)
            actual_x, actual_y = actual.raytrace(x, y)
            expected_x, expected_y = expected.raytrace(x, y)
            torch.testing.assert_close(actual_x, expected_x, rtol=0, atol=2e-6)
            torch.testing.assert_close(actual_y, expected_y, rtol=0, atol=2e-6)

    @unittest.skipUnless(triton_ipm_available(), "CUDA Triton IPM is unavailable")
    def test_public_dynamic_ipm_fuses_tree_queries_and_rasterization(self) -> None:
        """Exercise the complete production temporal dispatcher on CUDA."""

        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(0.1, 0.05),
            mc.PointMassField._from_einstein_radii(
                torch.tensor([-0.6, 0.5], device="cuda"),
                torch.tensor([0.4, -0.3], device="cuda"),
                velocity_x_uas_per_day=torch.tensor([0.001, -0.0015], device="cuda"),
                velocity_y_uas_per_day=torch.tensor([-0.001, 0.0005], device="cuda"),
                einstein_radius_uas=torch.tensor([0.18, 0.16], device="cuda"),
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
                    compact_sparse_nodes=True,
                    far_field_approx=mc.FarFieldApproxConfig(
                        cells_per_axis=4,
                        nodes_per_cell_axis=8,
                    ),
                ),
                schedule=mc.DynamicConfig(
                    temporal_batch_size=2,
                    scout_refresh_frames=3,
                    fused_temporal_ipm=True,
                ),
            )
        )
        self.assertEqual(len(maps), 3)
        self.assertTrue(
            all(item.metadata["dynamic_temporal_solver_fused"] for item in maps)
        )
        self.assertTrue(
            all(item.metadata["temporal_far_field_query_fused"] for item in maps)
        )
        self.assertTrue(
            all(item.metadata["far_field_exact_each_frame"] for item in maps)
        )
        self.assertTrue(all(item.metadata["compact_sparse_nodes"] for item in maps))
        self.assertTrue(maps[0].metadata["far_field_batched_accumulator"])
        self.assertTrue(maps[1].metadata["far_field_batched_accumulator"])
        self.assertTrue(maps[0].metadata["far_field_batched_local_pack"])
        self.assertTrue(maps[1].metadata["far_field_batched_local_pack"])
        # The final real frame shares its exact coefficient construction with
        # the refresh-interval endpoint used by the conservative scout union.
        self.assertTrue(maps[2].metadata["far_field_batched_accumulator"])
        self.assertEqual(maps[-1].metadata["temporal_batch_padded_frames"], 1)

    @unittest.skipUnless(triton_ipm_available(), "CUDA Triton IPM is unavailable")
    def test_public_padded_batch_and_cell_chunk_are_numerically_invariant(self) -> None:
        """Static launch shapes may not alter any real dynamic map frame."""

        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(
                0.14,
                0.07,
                shear_angle_deg=11.459155902616466,
                smooth_matter_fraction=0.25,
            ),
            mc.PointMassField._from_einstein_radii(
                torch.tensor([-0.7, 0.25, 0.9], device="cuda"),
                torch.tensor([0.45, -0.55, 0.3], device="cuda"),
                velocity_x_uas_per_day=torch.tensor(
                    [0.002, -0.001, 0.0015], device="cuda"
                ),
                velocity_y_uas_per_day=torch.tensor(
                    [-0.001, 0.0015, -0.002], device="cuda"
                ),
                einstein_radius_uas=torch.tensor([0.19, 0.15, 0.12], device="cuda"),
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
        for chunk, batch in ((17, 1), (31, 3), (128, 4)):
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
    def test_biquadratic_v4_materializer_matches_portable_interpolation(self) -> None:
        """The compact Triton interpolation must preserve mapped r=2 geometry."""

        from microcaustics.solvers import interpolated_nodes

        generator = torch.Generator(device="cuda").manual_seed(31)
        node_x = torch.randn((37, 3, 3), device="cuda", generator=generator)
        node_y = torch.randn((37, 3, 3), device="cuda", generator=generator)
        expected_x, expected_y = interpolated_nodes(
            node_x,
            node_y,
            virtual_refinement=4,
        )
        actual_x, actual_y = materialize_biquadratic_v4_triton(node_x, node_y)
        torch.testing.assert_close(actual_x, expected_x, rtol=2e-6, atol=5e-7)
        torch.testing.assert_close(actual_y, expected_y, rtol=2e-6, atol=5e-7)

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
    def test_indexed_direct_cell_raster_matches_ragged_launches(self) -> None:
        """Ragged cell queues must accumulate into their tagged maps only."""

        from microcaustics.solvers.triton_ipm import (
            TritonRasterWorkspace,
            accumulate_cells_triton,
        )

        grid = mc.PlaneGrid((17, 21), (3.4, 4.2))
        dy, dx = grid.pixel_scale_uas
        xmin, _, ymin, _ = grid.bounds_uas
        coordinate = torch.linspace(0.0, 1.0, 5, device="cuda")
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        base_x = torch.stack((-1.8 + xx, 0.1 + 0.8 * xx + 0.03 * yy))
        base_y = torch.stack((-1.4 + yy, 0.2 + 0.9 * yy - 0.02 * xx))
        rows_x = (base_x[:1], base_x + 0.07, base_x[:0], base_x - 0.04)
        rows_y = (base_y[:1], base_y - 0.03, base_y[:0], base_y + 0.05)
        kwargs = dict(
            xmin=xmin,
            ymin=ymin,
            pixel_size_x=dx,
            pixel_size_y=dy,
            lens_area_per_triangle_uas2=0.002,
        )
        expected = torch.stack(
            [
                rasterize_cells_triton(x, y, shape=grid.shape, **kwargs)
                if x.numel()
                else torch.zeros(grid.shape, device="cuda")
                for x, y in zip(rows_x, rows_y, strict=True)
            ]
        )
        flat_x = torch.cat(rows_x)
        flat_y = torch.cat(rows_y)
        lengths = [len(row) for row in rows_x]
        frames = torch.repeat_interleave(
            torch.arange(4, device="cuda", dtype=torch.int32),
            torch.tensor(lengths, device="cuda"),
        )
        workspace = TritonRasterWorkspace.create(
            grid.shape, device=torch.device("cuda"), frames=4
        )
        accumulate_cells_triton(
            workspace, flat_x, flat_y, cell_frame_index=frames, **kwargs
        )
        torch.testing.assert_close(workspace.result(), expected, rtol=0.0, atol=2e-6)

    @unittest.skipUnless(triton_ipm_available(), "CUDA Triton IPM is unavailable")
    def test_all_cell_controls_batch_maps_not_only_far_field(self) -> None:
        """Square and rectangular all-cell controls fuse the actual IPM maps."""
        stars = mc.PointMassField._from_einstein_radii(
            torch.tensor([-0.6, 0.5]),
            torch.tensor([0.4, -0.3]),
            einstein_radius_uas=torch.tensor([0.18, 0.16]),
            velocity_x_uas_per_day=torch.tensor([0.01, -0.015]),
            velocity_y_uas_per_day=torch.tensor([-0.01, 0.005]),
        )
        macro = mc.MacroLens(0.1, 0.05)
        grid = mc.PlaneGrid((8, 9), (1.5, 1.6))
        method = mc.IPMConfig(
            rays=144,
            refinement=2,
            virtual_refinement=4,
            tiled=False,
            cell_chunk_size=32,
            far_field_approx=mc.FarFieldApproxConfig(
                cells_per_axis=4,
                nodes_per_cell_axis=8,
            ),
        )
        for size in ((3.0, 3.0), (3.0, 2.0)):
            with self.subTest(lens_region=size):
                region = mc.PlaneRegion(size)
                results = []
                for backend, batch in (
                    ("triton", 2),
                    ("triton", 1),
                    ("torch-eager", 2),
                ):
                    simulation = mc.MicrolensingSimulation.create(
                        macro,
                        stars,
                        runtime=mc.RuntimeConfig(
                            device="cuda", backend=backend, strict_backend=True
                        ),
                    )
                    maps = list(
                        simulation.dynamic_maps(
                            region,
                            grid,
                            [0.0, 1.0, 2.0],
                            method=method,
                            schedule=mc.DynamicConfig(
                                temporal_batch_size=batch, fused_temporal_ipm=True
                            ),
                        )
                    )
                    if backend == "triton":
                        self.assertTrue(
                            all(
                                item.metadata["dynamic_temporal_solver_fused"]
                                for item in maps
                            )
                        )
                        if batch > 1:
                            self.assertTrue(
                                maps[0].metadata["temporal_far_field_query_fused"]
                            )
                    results.append(torch.stack([item.values for item in maps]))
                for comparison in results[1:]:
                    torch.testing.assert_close(
                        results[0], comparison, rtol=3e-4, atol=3e-5
                    )
                self.assertGreater(
                    float((results[0][0] - results[0][-1]).abs().max()), 1e-4
                )

    @unittest.skipUnless(triton_ipm_available(), "CUDA Triton IPM is unavailable")
    def test_full_field_triton_ipm_supports_general_refinement(self) -> None:
        """Exercise the public dispatcher beyond the production r=2, v=4 case."""

        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(0.0, 0.0),
            mc.PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
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
        batched_corner_x = torch.stack((corner_x, corner_x + 0.13))
        batched_corner_y = torch.stack((corner_y, corner_y - 0.07))
        batched_center_x = torch.stack((center_x, center_x + 0.13))
        batched_center_y = torch.stack((center_y, center_y - 0.07))
        batched = select_source_tiles_triton(
            batched_corner_x,
            batched_corner_y,
            bounds=bounds,
            center_x=batched_center_x,
            center_y=batched_center_y,
        )
        scalar = torch.stack(
            tuple(
                select_source_tiles_triton(
                    batched_corner_x[index],
                    batched_corner_y[index],
                    bounds=bounds,
                    center_x=batched_center_x[index],
                    center_y=batched_center_y[index],
                )
                for index in range(2)
            )
        )
        self.assertTrue(torch.equal(batched, scalar))
        for radius in (1, 2, 4):
            expected_dilated = torch.nn.functional.max_pool2d(
                expected[None, None].float(),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )[0, 0].bool()
            actual_dilated = dilate_source_tiles_triton(actual, radius)
            self.assertTrue(torch.equal(actual_dilated, expected_dilated))
            batched_dilated = dilate_source_tiles_triton(batched, radius)
            scalar_dilated = torch.stack(
                tuple(
                    dilate_source_tiles_triton(batched[index], radius)
                    for index in range(2)
                )
            )
            self.assertTrue(torch.equal(batched_dilated, scalar_dilated))

    @unittest.skipUnless(triton_scout_available(), "CUDA Triton scout is unavailable")
    def test_public_tiled_ipm_uses_fused_scout(self) -> None:
        """Exercise the fused scout and rasterizer through the public API."""

        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(0.0, 0.0),
            mc.PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
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

    def test_complete_production_outputs_match_torch_backend(self) -> None:
        """Compare maps, fluxes, and center labels through both backends."""

        times = (0.0, 1.0, 2.0)
        method = _small_production_method()
        schedule = _small_dynamic_schedule()
        caustics = _small_caustic_config()
        outputs = []
        for backend in ("torch-eager", "triton"):
            outputs.append(
                _small_cuda_system(backend=backend).light_curve(
                    times,
                    include_labels=True,
                    method=method,
                    schedule=schedule,
                    caustics=caustics,
                    keep_maps_at_days=times,
                )
            )
        portable, fused = outputs
        torch.testing.assert_close(
            fused.flux,
            portable.flux,
            rtol=5.0e-5,
            atol=5.0e-6,
        )
        torch.testing.assert_close(
            fused.labels.crossing_labels,
            portable.labels.crossing_labels,
            rtol=0.0,
            atol=0.0,
        )
        for index in range(len(times)):
            torch.testing.assert_close(
                fused.maps[index].values,
                portable.maps[index].values,
                # Scanline atomics do not have a fixed accumulation order.
                rtol=1.0e-4,
                atol=2.0e-4,
            )

    def test_source_owned_driver_matches_eager_compiled_and_triton(self) -> None:
        """Daily source driving must agree across all CUDA map backends."""

        outputs = []
        driver = mc.TabulatedDrivingSignal(
            torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0]),
            torch.tensor([1.0, 0.8, 1.3, 0.9, 1.0]),
        )
        for backend in ("torch-eager", "torch-compile", "triton"):
            base = _small_cuda_system(backend=backend)
            system = base.with_source(mc.ModulatedSource(base.source, driver))
            options = dict(
                source_cadence_days=0.5,
                include_labels=True,
                method=_small_production_method(),
                schedule=_small_dynamic_schedule(),
                caustics=_small_caustic_config(),
            )
            off = system.light_curve((0, 1, 2), apply_driving_signal=False, **options)
            on = system.light_curve((0, 1, 2), apply_driving_signal=True, **options)
            amplitudes = driver.amplitudes(
                on.times_days, bands=1, device=on.flux.device, dtype=on.flux.dtype
            )
            torch.testing.assert_close(
                on.flux, off.flux * amplitudes, rtol=5e-5, atol=0
            )
            torch.testing.assert_close(
                on.labels.crossing_labels, off.labels.crossing_labels
            )
            outputs.append(on)
        for actual in outputs[1:]:
            torch.testing.assert_close(actual.flux, outputs[0].flux, rtol=5e-5, atol=0)
            torch.testing.assert_close(
                actual.labels.crossing_labels, outputs[0].labels.crossing_labels
            )

    def test_compact_sparse_nodes_match_complete_cuda_path(self) -> None:
        """Compact node preparation must preserve the public Triton map."""

        system = _small_cuda_system(backend="triton")
        outputs = []
        for compact in (False, True):
            outputs.append(
                system.magnification_map(
                    method=_small_production_method(compact_sparse_nodes=compact)
                )
            )
        torch.testing.assert_close(
            outputs[1].values,
            outputs[0].values,
            rtol=2.0e-5,
            atol=3.0e-5,
        )
        self.assertFalse(outputs[0].metadata["compact_sparse_nodes"])
        self.assertTrue(outputs[1].metadata["compact_sparse_nodes"])

    def test_independent_cuda_curve_batch_matches_serial_with_labels(self) -> None:
        """Concurrent independent systems must match serial production calls."""

        systems = tuple(
            _small_cuda_system(backend="triton", offset=offset)
            for offset in (0.0, 0.04, -0.03)
        )
        common = dict(
            include_labels=True,
            method=replace(
                _small_production_method(), scout_trace_centers=False
            ),
            schedule=_small_dynamic_schedule(),
            caustics=_small_caustic_config(),
        )
        serial = mc.batched_system_light_curves(
            systems,
            (0.0, 1.0, 2.0),
            curves_per_batch=1,
            **common,
        )
        concurrent = mc.batched_system_light_curves(
            systems,
            (0.0, 1.0, 2.0),
            curves_per_batch=3,
            **common,
        )
        self.assertEqual(concurrent.executed_batch_sizes, (3,))
        self.assertTrue(
            all(
                curve.metadata["cross_system_solver_fused"]
                and curve.metadata["cross_system_compact_sparse_nodes"]
                for curve in concurrent.light_curves
            )
        )
        for expected, actual in zip(
            serial.light_curves,
            concurrent.light_curves,
            strict=True,
        ):
            torch.testing.assert_close(
                actual.flux,
                expected.flux,
                rtol=5.0e-5,
                atol=5.0e-6,
            )
            torch.testing.assert_close(
                actual.labels.crossing_labels,
                expected.labels.crossing_labels,
                rtol=0.0,
                atol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
