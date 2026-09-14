"""Mass-function and reproducible point-population contracts."""

from __future__ import annotations

import importlib.util
import math
import unittest
from itertools import pairwise

import torch

import microcaustics as mc


class LensPopulationTests(unittest.TestCase):
    @staticmethod
    def _sample_field(seed: int | None) -> mc.PointMassField:
        return mc.sample_uniform_point_masses(
            mc.PlaneRegion((8.0, 10.0), center_uas=(1.0, -2.0)),
            0.2,
            mc.LensingDistances(1.0e25, 2.0e25, 1.2e25),
            mc.PowerLawMassFunction(0.2, 0.8, 2.0),
            count=37,
            velocity_dispersion_uas_per_day=(0.01, 0.02),
            seed=seed,
            dtype=torch.float64,
        )

    def test_power_law_sampling_matches_bounds_and_analytic_mean(self) -> None:
        distribution = mc.PowerLawMassFunction(0.1, 10.0, slope=1.0)
        generator = torch.Generator().manual_seed(123)
        values = distribution.sample(
            100_000,
            generator=generator,
            dtype=torch.float64,
        )
        self.assertGreaterEqual(float(values.min()), 0.1)
        self.assertLessEqual(float(values.max()), 10.0)
        self.assertAlmostEqual(
            float(values.mean()),
            distribution.mean_mass(),
            delta=0.02 * distribution.mean_mass(),
        )
        self.assertAlmostEqual(
            float(values.square().mean()),
            distribution.second_moment(),
            delta=0.04 * distribution.second_moment(),
        )

    def test_broken_power_law_is_reproducible_and_samples_every_segment(self) -> None:
        distribution = mc.BrokenPowerLawMassFunction(
            edges=(0.05, 0.2, 1.0, 5.0),
            slopes=(0.3, 1.3, 2.3),
        )
        first = distribution.sample(
            20_000,
            generator=torch.Generator().manual_seed(17),
            dtype=torch.float64,
        )
        second = distribution.sample(
            20_000,
            generator=torch.Generator().manual_seed(17),
            dtype=torch.float64,
        )
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        for lower, upper in pairwise(distribution.edges):
            self.assertTrue(bool(torch.any((first >= lower) & (first < upper))))
        self.assertAlmostEqual(
            float(first.mean()),
            distribution.mean_mass(),
            delta=0.04 * distribution.mean_mass(),
        )

    def test_kroupa_constructor_keeps_each_requested_interval(self) -> None:
        distribution = mc.kroupa_mass_function(0.08, 10.0)
        self.assertEqual(distribution.edges, (0.08, 0.5, 10.0))
        self.assertEqual(distribution.slopes, (1.3, 2.3))

    def test_public_population_builder_respects_count_region_and_motion(self) -> None:
        region = mc.PlaneRegion((8.0, 10.0), center_uas=(1.0, -2.0))
        field = self._sample_field(9)
        self.assertEqual(len(field), 37)
        self.assertTrue(field.has_motion)
        xmin, xmax, ymin, ymax = region.bounds_uas
        self.assertTrue(bool(torch.all((field.x_uas >= xmin) & (field.x_uas < xmax))))
        self.assertTrue(bool(torch.all((field.y_uas >= ymin) & (field.y_uas < ymax))))
        self.assertGreater(mc.compact_convergence(field, region), 0.0)

    def test_population_seed_reproduces_complete_realization(self) -> None:
        first = self._sample_field(1234)
        second = self._sample_field(1234)
        for name in (
            "x_uas",
            "y_uas",
            "einstein_radius_uas",
            "velocity_x_uas_per_day",
            "velocity_y_uas_per_day",
        ):
            torch.testing.assert_close(
                getattr(first, name),
                getattr(second, name),
                rtol=0.0,
                atol=0.0,
            )

    def test_population_without_seed_advances_global_stream(self) -> None:
        with torch.random.fork_rng():
            torch.manual_seed(811)
            first = self._sample_field(None)
            second = self._sample_field(None)
        self.assertFalse(torch.equal(first.x_uas, second.x_uas))
        self.assertFalse(
            torch.equal(first.einstein_radius_uas, second.einstein_radius_uas)
        )

    def test_population_builder_rejects_ambiguous_random_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most one"):
            mc.sample_uniform_point_masses(
                mc.PlaneRegion((1.0, 1.0)),
                0.1,
                mc.LensingDistances(1.0e25, 2.0e25, 1.2e25),
                mc.PowerLawMassFunction(0.1, 1.0, 2.0),
                count=1,
                seed=1,
                generator=torch.Generator(),
            )

    def test_population_builder_adds_requested_bulk_velocity(self) -> None:
        field = mc.sample_uniform_circular_point_masses(
            3.0,
            0.2,
            mc.LensingDistances(1.0e25, 2.0e25, 1.2e25),
            mc.PowerLawMassFunction(0.1, 1.0, 2.0),
            count=8,
            velocity_mean_uas_per_day=(0.003, -0.002),
            seed=17,
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            field.velocity_x_uas_per_day,
            torch.full((8,), 0.003, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            field.velocity_y_uas_per_day,
            torch.full((8,), -0.002, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_light_loss_rectangle_handles_rotated_shear_coordinates(self) -> None:
        macro = mc.MacroLens(0.31, 0.22, shear_angle_deg=math.degrees(0.37))
        source = mc.PlaneRegion((1.4, 2.2), center_uas=(0.13, -0.27))
        distances = mc.LensingDistances(1.0e25, 2.0e25, 1.2e25)
        mass_function = mc.PowerLawMassFunction(0.1, 1.0, 2.0)
        region = mc.rectangular_lens_region(
            macro,
            source,
            distances,
            mass_function,
            light_loss=0.02,
        )

        angle = 2.0 * macro.shear_angle_rad
        gamma_1 = macro.shear * math.cos(angle)
        gamma_2 = macro.shear * math.sin(angle)
        macro_matrix = torch.tensor(
            [
                [1.0 - macro.convergence - gamma_1, -gamma_2],
                [-gamma_2, 1.0 - macro.convergence + gamma_1],
            ],
            dtype=torch.float64,
        )
        inverse = torch.linalg.inv(macro_matrix)
        mean_mass = mass_function.mean_mass()
        margin = float(distances.einstein_radius_uas(mean_mass, dtype=torch.float64))
        margin *= math.sqrt(
            macro.compact_convergence
            * mass_function.second_moment()
            / mean_mass**2
            / 0.02
        )
        source_half_y = 0.5 * source.field_of_view_uas[0] + margin
        source_half_x = 0.5 * source.field_of_view_uas[1] + margin
        corners = torch.tensor(
            [
                [
                    source.center_uas[1] + sx * source_half_x,
                    source.center_uas[0] + sy * source_half_y,
                ]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ],
            dtype=torch.float64,
        )
        lens_corners = corners @ inverse.T
        xmin, xmax, ymin, ymax = region.bounds_uas
        tolerance = 1.0e-12
        self.assertTrue(bool(torch.all(lens_corners[:, 0] >= xmin - tolerance)))
        self.assertTrue(bool(torch.all(lens_corners[:, 0] <= xmax + tolerance)))
        self.assertTrue(bool(torch.all(lens_corners[:, 1] >= ymin - tolerance)))
        self.assertTrue(bool(torch.all(lens_corners[:, 1] <= ymax + tolerance)))

    @unittest.skipUnless(importlib.util.find_spec("astropy"), "requires astropy")
    def test_q2237b_light_loss_rectangle_matches_paper_geometry(self) -> None:
        from astropy.cosmology import FlatLambdaCDM

        unit_distribution = mc.salpeter_mass_function(1.0, 100.0)
        minimum_mass = 0.3 / unit_distribution.mean_mass()
        distribution = mc.salpeter_mass_function(
            minimum_mass,
            100.0 * minimum_mass,
        )
        rectangle = mc.rectangular_lens_region(
            mc.MacroLens(0.391, 0.391),
            mc.PlaneRegion((9.38191344959511, 9.38191344959511)),
            mc.LensingDistances.from_redshifts(
                0.0395,
                1.695,
                cosmology=FlatLambdaCDM(H0=70.0, Om0=0.3),
            ),
            distribution,
            light_loss=0.01,
        )
        fov_y, fov_x = rectangle.field_of_view_uas
        self.assertAlmostEqual(fov_y, 106.41557, delta=0.03)
        self.assertAlmostEqual(fov_x, 488.14490, delta=0.15)


if __name__ == "__main__":
    unittest.main()
