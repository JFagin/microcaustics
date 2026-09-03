"""Tests for concurrent, scientifically independent light curves."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_cuda_oom_backoff_retries_smaller_groups(self) -> None:
        systems = tuple(_FakeRealization(value) for value in (1.0, 2.0, 3.0))

        def execute(group, *_args, **_kwargs):
            if len(group) > 1:
                raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")
            return (group[0].light_curve((0.0,)),)

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


if __name__ == "__main__":
    unittest.main()
