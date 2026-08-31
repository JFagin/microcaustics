"""CPU contracts for optional production autotuning."""

from __future__ import annotations

import unittest

import torch

import microcaustics as mc
from microcaustics.solvers import (
    temporal_taylor_far_field_window,
    temporal_taylor_far_fields,
)


def _moving_simulation(dtype: str = "float32") -> mc.MicrolensingSimulation:
    return mc.MicrolensingSimulation.create(
        mc.MacroLens(convergence=0.12, shear=0.06),
        mc.PointMassField(
            torch.tensor([-0.45, 0.35]),
            torch.tensor([0.25, -0.3]),
            torch.tensor([0.22, 0.17]),
            torch.tensor([0.002, -0.001]),
            torch.tensor([-0.001, 0.0015]),
        ),
        runtime=mc.RuntimeConfig(
            device="cpu",
            backend=mc.Backend.TORCH_EAGER,
            dtype=dtype,
        ),
    )


class AutoTuningTests(unittest.TestCase):
    def setUp(self) -> None:
        mc.clear_tuning_cache()

    def test_temporal_far_field_windows_match_complete_exact_sequence(self) -> None:
        simulation = _moving_simulation("float64")
        config = mc.FarFieldApproxConfig(
            cells_per_axis=3,
            nodes_per_cell_axis=3,
            exact_radius_cells=0.8,
            taylor_order=3,
            center_translation_order=6,
        )
        region = mc.PlaneRegion((3.0, 3.0))
        times = tuple(float(index) for index in range(6))
        complete, _ = temporal_taylor_far_fields(
            simulation,
            region,
            config,
            times,
        )
        first, first_metadata = temporal_taylor_far_field_window(
            simulation,
            region,
            config,
            times,
            [0, 1, 2],
        )
        second, second_metadata = temporal_taylor_far_field_window(
            simulation,
            region,
            config,
            times,
            [3, 4, 5],
        )
        windowed = (*first, *second)
        for actual, expected in zip(windowed, complete, strict=True):
            torch.testing.assert_close(actual.coefficient_real, expected.coefficient_real)
            torch.testing.assert_close(actual.coefficient_imag, expected.coefficient_imag)
            torch.testing.assert_close(actual.local_x, expected.local_x)
            torch.testing.assert_close(actual.local_y, expected.local_y)
        self.assertEqual(first_metadata["far_field_requested_frames"], [0, 1, 2])
        self.assertEqual(second_metadata["far_field_requested_frames"], [3, 4, 5])
        self.assertTrue(first_metadata["far_field_exact_each_frame"])
        self.assertTrue(second_metadata["far_field_exact_each_frame"])

    def test_dynamic_ipm_autotuning_selects_records_and_caches(self) -> None:
        simulation = _moving_simulation()
        method = mc.IPMConfig(
            rays=36,
            scout_ratio=2,
            refinement=1,
            virtual_refinement=1,
            tiled=True,
            cell_chunk_size=8,
            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
        )
        tuning = mc.AutoTuningConfig(
            enabled=True,
            temporal_candidates=(1, 2),
            spatial_candidates=(8, 16),
            warmup_runs=0,
            benchmark_runs=1,
            maximum_trial_frames=2,
        )
        schedule = mc.DynamicConfig(
            temporal_batch_size=1,
            scout_refresh_frames=2,
            tuning=tuning,
        )
        request = dict(
            lens_region=mc.PlaneRegion((3.0, 3.0)),
            source_grid=mc.PlaneGrid((4, 4), (1.0, 1.0)),
            times_days=[0.0, 1.0],
            method=method,
            schedule=schedule,
        )
        first = list(simulation.dynamic_maps(**request))
        second = list(simulation.dynamic_maps(**request))
        self.assertEqual(len(first), 2)
        self.assertIn(first[0].metadata["autotune_temporal_batch_size"], (1, 2))
        self.assertIn(first[0].metadata["autotune_spatial_chunk_size"], (8, 16))
        self.assertGreaterEqual(first[0].metadata["autotune_accepted_trials"], 2)
        self.assertFalse(first[0].metadata["autotune_cache_hit"])
        self.assertTrue(second[0].metadata["autotune_cache_hit"])
        self.assertEqual(second[0].metadata["autotune_seconds_excluded"], 0.0)
        for actual, expected in zip(first, second, strict=True):
            torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)

    def test_caustic_autotuning_records_selected_work_sizes(self) -> None:
        simulation = _moving_simulation("float64")
        tuning = mc.AutoTuningConfig(
            enabled=True,
            temporal_candidates=(1, 2),
            spatial_candidates=(64, 128),
            warmup_runs=0,
            benchmark_runs=1,
            maximum_trial_frames=2,
        )
        frames = simulation.dynamic_labeled_caustics(
            mc.PlaneGrid((17, 17), (3.0, 3.0)),
            mc.PlaneRegion((0.8, 0.8)),
            [0.0, 1.0],
            config=mc.CausticConfig(
                far_field_approx=mc.FarFieldApproxConfig(
                    cells_per_axis=3,
                    nodes_per_cell_axis=3,
                    exact_radius_cells=1.0,
                ),
                temporal_batch_size=1,
                jacobian_chunk_size=64,
                anchor_count=5,
                gauge_count=7,
                minimum_alignment_gauges=3,
                tuning=tuning,
            ),
        )
        self.assertEqual(len(frames), 2)
        metadata = frames[0].caustics.metadata
        self.assertIn(metadata["autotune_temporal_batch_size"], (1, 2))
        self.assertIn(metadata["autotune_spatial_chunk_size"], (64, 128))
        self.assertEqual(frames[0].labels.anchor_points_uas.shape[0], 5)
        self.assertEqual(frames[0].labels.gauge_labels.numel(), 7)

    def test_short_sequence_never_reports_an_unmeasured_batch(self) -> None:
        simulation = _moving_simulation()
        result = mc.autotune_dynamic_ipm(
            simulation,
            mc.PlaneRegion((3.0, 3.0)),
            mc.PlaneGrid((3, 3), (1.0, 1.0)),
            [0.0, 1.0, 2.0],
            mc.IPMConfig(
                rays=25,
                refinement=1,
                virtual_refinement=1,
                cell_chunk_size=8,
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            ),
            mc.DynamicConfig(
                temporal_batch_size=1,
                tuning=mc.AutoTuningConfig(
                    enabled=True,
                    temporal_candidates=(1, 4, 8),
                    spatial_candidates=(8,),
                    warmup_runs=0,
                    benchmark_runs=1,
                    maximum_trial_frames=3,
                ),
            ),
        )
        self.assertIn(result.temporal_batch_size, (1, 3))

    def test_dynamic_irs_autotuning_selects_ray_chunk_only(self) -> None:
        simulation = _moving_simulation()
        method = mc.IRSConfig(
            rays=64,
            ray_chunk_size=16,
            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
        )
        schedule = mc.DynamicConfig(
            temporal_batch_size=2,
            tuning=mc.AutoTuningConfig(
                enabled=True,
                temporal_candidates=(1, 2),
                spatial_candidates=(8, 16),
                warmup_runs=0,
                benchmark_runs=1,
                maximum_trial_frames=2,
            ),
        )
        direct = mc.autotune_dynamic_maps(
            simulation,
            mc.PlaneRegion((2.0, 2.0)),
            mc.PlaneGrid((4, 4), (1.0, 1.0)),
            [0.0, 1.0],
            method,
            schedule,
        )
        self.assertEqual(direct.operation, "dynamic_irs")
        self.assertIn(direct.spatial_chunk_size, (8, 16))
        maps = list(
            simulation.dynamic_maps(
                mc.PlaneRegion((2.0, 2.0)),
                mc.PlaneGrid((4, 4), (1.0, 1.0)),
                [0.0, 1.0],
                method=method,
                schedule=schedule,
            )
        )
        self.assertEqual(len(maps), 2)
        self.assertEqual(maps[0].metadata["autotune_operation"], "dynamic_irs")
        self.assertEqual(maps[0].metadata["autotune_temporal_batch_size"], 2)
        self.assertIn(maps[0].metadata["autotune_spatial_chunk_size"], (8, 16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_tuning_records_peak_memory_below_budget(self) -> None:
        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(convergence=0.0, shear=0.0),
            mc.PointMassField(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                torch.tensor([0.2]),
                torch.tensor([0.001]),
                torch.tensor([0.0]),
            ),
            runtime=mc.RuntimeConfig(
                device="cuda",
                backend=mc.Backend.TORCH_EAGER,
                memory_fraction=0.8,
            ),
        )
        result = mc.autotune_dynamic_ipm(
            simulation,
            mc.PlaneRegion((2.0, 2.0)),
            mc.PlaneGrid((3, 3), (1.0, 1.0)),
            [0.0],
            mc.IPMConfig(
                rays=25,
                refinement=1,
                virtual_refinement=1,
                tiled=False,
                cell_chunk_size=8,
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            ),
            mc.DynamicConfig(
                temporal_batch_size=1,
                tuning=mc.AutoTuningConfig(
                    enabled=True,
                    temporal_candidates=(1,),
                    spatial_candidates=(8, 16),
                    warmup_runs=0,
                    benchmark_runs=1,
                    maximum_trial_frames=1,
                ),
            ),
        )
        accepted = [trial for trial in result.trials if trial.accepted]
        self.assertTrue(accepted)
        self.assertIsNotNone(result.memory_budget_bytes)
        self.assertTrue(
            all(trial.peak_device_memory_bytes is not None for trial in accepted)
        )
        self.assertTrue(
            all(
                trial.peak_device_memory_bytes <= result.memory_budget_bytes
                for trial in accepted
            )
        )


if __name__ == "__main__":
    unittest.main()
