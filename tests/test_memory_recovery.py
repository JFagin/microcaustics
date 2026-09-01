"""Lossless simulated OOM recovery without allocating excessive memory."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import microcaustics as mc
from microcaustics.dynamic import DynamicMapScheduler


def _simulation() -> mc.MicrolensingSimulation:
    empty = torch.empty(0)
    return mc.MicrolensingSimulation.create(
        mc.MacroLens(0.0, 0.0),
        mc.PointMassField._from_einstein_radii(empty, empty, einstein_radius_uas=empty),
        runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
    )


class MemoryRecoveryTests(unittest.TestCase):
    grid = mc.PlaneGrid((3, 4), (1.0, 1.2))
    region = mc.PlaneRegion((2.0, 2.0))

    def test_spatial_oom_halves_only_cell_chunk_until_success(self) -> None:
        observed_chunks: list[int] = []

        def fake_full_field(simulation, lens_region, source_grid, config, **kwargs):
            del simulation, lens_region, kwargs
            observed_chunks.append(config.cell_chunk_size)
            if config.cell_chunk_size > 16:
                raise RuntimeError("CUDA out of memory in simulated raster workspace")
            return mc.MagnificationMap(
                torch.ones(source_grid.shape),
                source_grid,
                method="simulated",
                metadata={"cell_chunk_size": config.cell_chunk_size},
            )

        scheduler = DynamicMapScheduler(
            _simulation(),
            self.region,
            self.grid,
            [0.0],
            mc.IPMConfig(
                rays=16,
                refinement=1,
                virtual_refinement=1,
                tiled=False,
                cell_chunk_size=64,
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            ),
            mc.DynamicConfig(minimum_cell_chunk_size=8),
        )
        with patch("microcaustics.solvers.ipm.full_field_ipm", fake_full_field):
            result, reductions = scheduler._calculate_with_backoff(
                scheduler.method,
                time_days=0.0,
            )
        self.assertEqual(observed_chunks, [64, 32, 16])
        self.assertEqual(reductions, 2)
        self.assertEqual(result.metadata["cell_chunk_size"], 16)

    def test_temporal_oom_reduces_chunk_then_splits_real_frames(self) -> None:
        calls: list[tuple[int, int, int]] = []

        def fake_temporal(
            simulation,
            lens_region,
            source_grid,
            config,
            times,
            **kwargs,
        ):
            del simulation, lens_region
            real = int(kwargs.get("real_frame_count", len(times)))
            calls.append((config.cell_chunk_size, len(times), real))
            if config.cell_chunk_size > 8 or real > 2:
                raise RuntimeError("simulated CUDA out of memory")
            return tuple(
                mc.MagnificationMap(
                    torch.full(source_grid.shape, float(index + 1)),
                    source_grid,
                    time_days=float(time),
                    method="simulated_temporal",
                )
                for index, time in enumerate(times[:real])
            )

        scheduler = DynamicMapScheduler(
            _simulation(),
            self.region,
            self.grid,
            [0.0, 1.0, 2.0, 3.0],
            mc.IPMConfig(
                rays=16,
                refinement=1,
                virtual_refinement=1,
                tiled=False,
                cell_chunk_size=32,
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            ),
            mc.DynamicConfig(minimum_cell_chunk_size=8),
        )
        with patch("microcaustics.solvers.ipm.temporal_batch_ipm", fake_temporal):
            maps, reductions, splits = scheduler._calculate_ipm_batch_with_backoff(
                scheduler.method,
                scheduler.times,
            )
        self.assertEqual(len(maps), 4)
        self.assertEqual([item.time_days for item in maps], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(reductions, 2)
        self.assertEqual(splits, 1)
        self.assertEqual(calls[:3], [(32, 4, 4), (16, 4, 4), (8, 4, 4)])
        self.assertIn((8, 2, 2), calls)

    def test_non_oom_errors_are_never_masked_or_retried(self) -> None:
        scheduler = DynamicMapScheduler(
            _simulation(),
            self.region,
            self.grid,
            [0.0],
            mc.IPMConfig(
                rays=16,
                refinement=1,
                virtual_refinement=1,
                tiled=False,
                cell_chunk_size=64,
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            ),
            mc.DynamicConfig(minimum_cell_chunk_size=8),
        )
        with (
            patch(
                "microcaustics.solvers.ipm.full_field_ipm",
                side_effect=ValueError("invalid scientific configuration"),
            ) as calculation,
            self.assertRaisesRegex(ValueError, "invalid scientific configuration"),
        ):
            scheduler._calculate_with_backoff(scheduler.method, time_days=0.0)
        calculation.assert_called_once()


if __name__ == "__main__":
    unittest.main()
