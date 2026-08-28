"""Seeded property checks over many valid numerical configurations."""

from __future__ import annotations

import math
import unittest

import torch

import microcaustics as mc


class RandomizedNumericalProperties(unittest.TestCase):
    def test_direct_raytrace_matches_independent_sum_over_seeded_cases(self) -> None:
        generator = torch.Generator().manual_seed(20260823)
        for case in range(12):
            stars = case % 6
            dtype = torch.float64
            star_x = torch.rand(stars, generator=generator, dtype=dtype) * 5.0 - 2.5
            star_y = torch.rand(stars, generator=generator, dtype=dtype) * 5.0 - 2.5
            radii = torch.rand(stars, generator=generator, dtype=dtype) * 0.2 + 0.03
            convergence = float(torch.rand((), generator=generator) * 0.5)
            shear = float(torch.rand((), generator=generator) * 0.25)
            angle = float(torch.rand((), generator=generator) * math.pi - math.pi / 2)
            smooth_fraction = float(torch.rand((), generator=generator))
            macro = mc.MacroLens(
                convergence,
                shear,
                shear_angle_rad=angle,
                smooth_matter_fraction=smooth_fraction,
            )
            simulation = mc.MicrolensingSimulation.create(
                macro,
                mc.PointMassField(star_x, star_y, radii),
                runtime=mc.RuntimeConfig(
                    device="cpu",
                    backend="torch-eager",
                    dtype=dtype,
                ),
            )
            x = torch.rand(31, generator=generator, dtype=dtype) * 4.0 - 2.0
            y = torch.rand(31, generator=generator, dtype=dtype) * 4.0 - 2.0
            actual_x, actual_y, _ = simulation.raytrace_direct(
                x,
                y,
                star_chunk_size=max(1, 1 + case % 3),
                ray_chunk_size=4 + case % 7,
            )

            alpha_x = torch.zeros_like(x)
            alpha_y = torch.zeros_like(y)
            if stars:
                dx = x[:, None] - star_x[None]
                dy = y[:, None] - star_y[None]
                denominator = dx.square() + dy.square()
                weight = radii.square()[None] / denominator
                alpha_x += (dx * weight).sum(dim=1)
                alpha_y += (dy * weight).sum(dim=1)
            gamma1 = shear * math.cos(2.0 * angle)
            gamma2 = shear * math.sin(2.0 * angle)
            sheet = macro.smooth_convergence
            alpha_x += (sheet + gamma1) * x + gamma2 * y
            alpha_y += gamma2 * x + (sheet - gamma1) * y
            torch.testing.assert_close(actual_x, x - alpha_x, rtol=2e-14, atol=2e-14)
            torch.testing.assert_close(actual_y, y - alpha_y, rtol=2e-14, atol=2e-14)

    def test_direct_lens_equation_is_translation_covariant_without_macro_terms(self) -> None:
        generator = torch.Generator().manual_seed(74021)
        star_x = torch.rand(9, generator=generator, dtype=torch.float64) * 3.0 - 1.5
        star_y = torch.rand(9, generator=generator, dtype=torch.float64) * 3.0 - 1.5
        radii = torch.rand(9, generator=generator, dtype=torch.float64) * 0.1 + 0.05
        x = torch.rand(43, generator=generator, dtype=torch.float64) * 3.0 - 1.5
        y = torch.rand(43, generator=generator, dtype=torch.float64) * 3.0 - 1.5
        shift_x, shift_y = 3.7, -2.4

        def trace(offset_x: float, offset_y: float):
            simulation = mc.MicrolensingSimulation.create(
                mc.MacroLens(0.0, 0.0),
                mc.PointMassField(star_x + offset_x, star_y + offset_y, radii),
                runtime=mc.RuntimeConfig(
                    device="cpu",
                    backend="torch-eager",
                    dtype=torch.float64,
                ),
            )
            return simulation.raytrace_direct(x + offset_x, y + offset_y)[:2]

        original_x, original_y = trace(0.0, 0.0)
        shifted_x, shifted_y = trace(shift_x, shift_y)
        torch.testing.assert_close(shifted_x, original_x + shift_x, rtol=0, atol=2e-14)
        torch.testing.assert_close(shifted_y, original_y + shift_y, rtol=0, atol=2e-14)

    def test_identity_ipm_on_random_rectangles_and_refinements(self) -> None:
        generator = torch.Generator().manual_seed(99117)
        empty = torch.empty(0, dtype=torch.float64)
        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(0.0, 0.0),
            mc.PointMassField(empty, empty, empty),
            runtime=mc.RuntimeConfig(
                device="cpu",
                backend="torch-eager",
                dtype=torch.float64,
            ),
        )
        for case in range(10):
            rows = 3 + case % 5
            columns = 5 + (2 * case) % 7
            fov_y = 1.0 + float(torch.rand((), generator=generator)) * 2.0
            fov_x = 1.0 + float(torch.rand((), generator=generator)) * 3.0
            refinement = 1 + case % 4
            virtual = refinement + case % 3
            result = simulation.magnification_map(
                mc.PlaneRegion((fov_y, fov_x)),
                mc.PlaneGrid((rows, columns), (fov_y, fov_x)),
                method=mc.IPMConfig(
                    rays=(rows + case % 3) * (columns + (case + 1) % 3),
                    scout_ratio=1 + case % 4,
                    refinement=refinement,
                    virtual_refinement=virtual,
                    tiled=bool(case % 2),
                    scout_dilation_cells=1,
                    cell_chunk_size=3 + case,
                    far_field_approx=mc.FarFieldApproxConfig(enabled=False),
                ),
            )
            torch.testing.assert_close(
                result.values,
                torch.ones_like(result.values),
                rtol=3e-12,
                atol=3e-12,
            )


if __name__ == "__main__":
    unittest.main()
