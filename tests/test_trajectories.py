"""Source-trajectory contracts and validation."""

from __future__ import annotations

import unittest

import torch

import microcaustics as mc


class TrajectoryTests(unittest.TestCase):
    def test_linear_trajectory_obeys_reference_epoch_and_dtype(self) -> None:
        trajectory = mc.LinearTrajectory(
            initial_position_uas=(1.0, -2.0),
            velocity_uas_per_day=(0.5, 0.25),
            reference_time_days=2.0,
        )
        actual = trajectory.position_uas(
            [0.0, 2.0, 4.0],
            dtype=torch.float64,
        )
        expected = torch.tensor(
            [[0.0, -2.5], [1.0, -2.0], [2.0, -1.5]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual, expected)

    def test_tabulated_trajectory_requires_the_registered_time_axis(self) -> None:
        times = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64)
        positions = torch.tensor(
            [[0.0, 0.0], [0.5, -0.2], [0.7, -0.4]],
            dtype=torch.float64,
        )
        trajectory = mc.TabulatedTrajectory(times, positions)
        torch.testing.assert_close(
            trajectory.position_uas(times, dtype=torch.float32),
            positions.float(),
        )
        with self.assertRaisesRegex(ValueError, "same ordered time grid"):
            trajectory.position_uas([0.0, 2.0, 4.0])

    def test_invalid_trajectory_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contain x and y"):
            mc.LinearTrajectory(initial_position_uas=(0.0,))
        with self.assertRaisesRegex(ValueError, "shape"):
            mc.TabulatedTrajectory(torch.tensor([0.0, 1.0]), torch.zeros(2, 3))


if __name__ == "__main__":
    unittest.main()
