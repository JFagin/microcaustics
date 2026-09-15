"""Tests for concurrent, scientifically independent light curves."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

import microcaustics as mc


class _FakeRealization:
    def __init__(self, value: float) -> None:
        runtime = SimpleNamespace(device=torch.device("cpu"))
        self.simulation = SimpleNamespace(runtime=runtime)
        self.source = None
        self.value = float(value)

    def light_curve(self, times, **_kwargs):
        count = len(tuple(times))
        return mc.LightCurve(
            torch.as_tensor(times), torch.full((count, 1), self.value), ("g",)
        )

    def multirate_light_curve(self, _map_times, flux_times, **_kwargs):
        count = len(tuple(flux_times))
        return mc.LightCurve(
            torch.as_tensor(flux_times), torch.full((count, 1), self.value), ("g",)
        )


class IndependentLightCurveBatchTests(unittest.TestCase):
    def test_mixed_macroimage_batch_fuses_matching_cross_system_groups(self) -> None:
        """Interleaved quad-like jobs should fuse by macroimage contract."""

        import microcaustics.batching as batching

        runtime = SimpleNamespace(device=torch.device("cpu"))
        jobs = tuple(
            SimpleNamespace(
                value=value,
                signature=signature,
                realization=SimpleNamespace(
                    simulation=SimpleNamespace(runtime=runtime)
                ),
            )
            for value, signature in (("A0", "A"), ("B0", "B"), ("A1", "A"), ("B1", "B"))
        )

        def fused(group):
            if len(group) == len(jobs):
                return None
            return tuple(item.value for item in group)

        with (
            patch(
                "microcaustics.batching._cross_system_signature",
                side_effect=lambda job: job.signature,
            ),
            patch(
                "microcaustics.batching._run_cross_system_ipm_group",
                side_effect=fused,
            ) as calculate,
        ):
            result = batching._run_independent_group(jobs)
        self.assertEqual(result, ("A0", "B0", "A1", "B1"))
        self.assertEqual(calculate.call_count, 3)

    def test_cuda_stream_pool_is_reused_across_groups(self) -> None:
        import microcaustics.batching as batching

        runtime = SimpleNamespace(device=torch.device("cuda:0"))
        jobs = tuple(
            SimpleNamespace(
                realization=SimpleNamespace(
                    simulation=SimpleNamespace(runtime=runtime)
                )
            )
            for _ in range(2)
        )
        streams = (Mock(), Mock())
        batching._INDEPENDENT_STREAM_STATE.pools = {}
        with (
            patch("torch.cuda.Stream", side_effect=streams) as constructor,
            patch("torch.cuda.current_stream", return_value=Mock()),
            patch(
                "microcaustics.batching._run_independent_curve",
                side_effect=lambda job, _stream: job,
            ),
        ):
            first = batching._run_independent_group(jobs)
            second = batching._run_independent_group(jobs)
        self.assertEqual(first, jobs)
        self.assertEqual(second, jobs)
        self.assertEqual(constructor.call_count, 2)

    def test_independent_batch_preserves_order_and_multirate_axis(self) -> None:
        systems = tuple(_FakeRealization(value) for value in (1.0, 2.0, 3.0))
        result = mc.batched_system_light_curves(
            systems,
            (0.0, 25.0),
            (0.0, 1.0, 2.0),
            curves_per_batch=2,
        )
        self.assertEqual(result.executed_batch_sizes, (2, 1))
        self.assertEqual(result.requested_curves_per_batch, 2)
        self.assertEqual(result.oom_reductions, 0)
        self.assertEqual(
            [float(curve.flux[0, 0]) for curve in result.light_curves],
            [1.0, 2.0, 3.0],
        )
        self.assertTrue(
            all(curve.flux.shape == (3, 1) for curve in result.light_curves)
        )

    def test_source_setup_batch_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_setup_batch_size"):
            mc.batched_system_light_curves(
                (_FakeRealization(1.0),),
                (0.0,),
                source_setup_batch_size=0,
            )

    def test_cuda_oom_backoff_retries_smaller_groups(self) -> None:
        systems = tuple(_FakeRealization(value) for value in (1.0, 2.0, 3.0))

        def execute(group, *_args, **_kwargs):
            if len(group) > 1:
                raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")
            return (group[0].realization.light_curve((0.0,)),)

        with patch(
            "microcaustics.batching._run_independent_group", side_effect=execute
        ):
            result = mc.batched_system_light_curves(
                systems,
                (0.0,),
                curves_per_batch=3,
            )
        self.assertEqual(result.executed_batch_sizes, (1, 1, 1))
        self.assertEqual(result.oom_reductions, 1)

    def test_tuner_returns_a_verified_explicit_candidate(self) -> None:
        systems = tuple(_FakeRealization(value) for value in (1.0, 2.0))
        result = mc.tune_system_light_curve_batch(
            systems,
            (0.0, 1.0),
            candidates=(1, 2),
        )
        self.assertIn(result.curves_per_batch, (1, 2))
        self.assertTrue(all(trial.accepted for trial in result.trials))

    def test_flat_output_is_disk_backed_and_round_trips(self) -> None:
        systems = tuple(_FakeRealization(value) for value in (1.0, 2.0))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "curves"
            result = mc.batched_system_light_curves(
                systems, (0.0, 1.0), curves_per_batch=2, output_path=output
            )
            self.assertEqual(result.light_curves, ())
            self.assertEqual(result.storage_mode, "flat-npz")
            self.assertEqual(result.completed_systems, 2)
            self.assertEqual(result.completed_light_curves, 2)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertEqual(len(tuple(output.glob("system_*.npz"))), 2)
            loaded = result.load_system(1)
            torch.testing.assert_close(loaded.flux, torch.full((2, 1), 2.0))

    def test_combined_output_uses_one_numpy_archive(self) -> None:
        systems = tuple(_FakeRealization(value) for value in (3.0, 4.0))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "curves.npz"
            result = mc.batched_system_light_curves(
                systems,
                (0.0,),
                curves_per_batch=2,
                output_path=output,
                compression=False,
            )
            self.assertEqual(result.storage_mode, "combined-npz")
            self.assertEqual(tuple(output.parent.glob("*.npz")), (output,))
            with np.load(output, allow_pickle=False) as stored:
                self.assertIn("manifest_json", stored.files)
            loaded = result.load_system(0)
            torch.testing.assert_close(loaded.flux, torch.full((1, 1), 3.0))

    def test_flat_output_can_resume_completed_systems(self) -> None:
        systems = tuple(_FakeRealization(value) for value in (1.0, 2.0))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "curves"
            mc.batched_system_light_curves(systems, (0.0,), output_path=output)
            with patch(
                "microcaustics.batching._run_independent_group",
                side_effect=AssertionError("completed curves should be reused"),
            ):
                resumed = mc.batched_system_light_curves(
                    systems, (0.0,), output_path=output, resume=True
                )
            self.assertEqual(resumed.executed_batch_sizes, ())
            self.assertEqual(resumed.completed_systems, 2)
            torch.testing.assert_close(
                resumed.load_system(1).flux, torch.full((1, 1), 2.0)
            )

    def test_disk_output_rejects_retained_maps(self) -> None:
        with (
            TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "do not serialize retained maps"),
        ):
            mc.batched_system_light_curves(
                (_FakeRealization(1.0),),
                (0.0,),
                output_path=Path(directory) / "curves",
                keep_maps_at_days=(0.0,),
            )


if __name__ == "__main__":
    unittest.main()
