"""Validation of the thin caustics macro-image adapter."""

from __future__ import annotations

import importlib.util
import math
import unittest

import torch

import microcaustics as mc

CAUSTICS_AVAILABLE = importlib.util.find_spec("caustics") is not None


class RenderingUtilityTests(unittest.TestCase):
    def test_gaussian_psf_and_noise_helpers_are_normalized_and_seeded(self) -> None:
        kernel = mc.gaussian_psf_kernel(
            0.08,
            0.02,
            size=17,
            oversample_factor=2,
            dtype=torch.float64,
        )
        self.assertEqual(kernel.shape, (17, 17))
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=14)
        self.assertAlmostEqual(float(kernel[8, 7]), float(kernel[8, 9]), places=14)

        model = mc.PeakScaledPoissonReadNoise(
            peak_electrons=5000.0,
            read_noise_fraction=0.004,
        )
        image = torch.ones((5, 5, 2), dtype=torch.float64)
        first = model(image, generator=torch.Generator().manual_seed(8))
        second = model(image, generator=torch.Generator().manual_seed(8))
        torch.testing.assert_close(first[0], second[0], rtol=0.0, atol=0.0)
        torch.testing.assert_close(first[1], second[1], rtol=0.0, atol=0.0)
        self.assertEqual(first[2]["peak_electrons"], 5000.0)


@unittest.skipUnless(CAUSTICS_AVAILABLE, "requires caustics")
class CausticsRenderingTests(unittest.TestCase):
    @staticmethod
    def lens():
        import caustics

        return caustics.SIS(
            cosmology=caustics.FlatLambdaCDM(),
            z_l=0.5,
            z_s=1.0,
            x0=0.0,
            y0=0.0,
            Rein=0.7,
            s=0.01,
        )

    @staticmethod
    def source(intensity, name):
        import caustics

        return caustics.Sersic(
            x0=0.05,
            y0=0.02,
            q=0.8,
            phi=0.2,
            n=1.0,
            Re=0.15,
            Ie=float(intensity),
            name=name,
        )

    def test_multiband_result_matches_caustics_lens_source(self) -> None:
        import caustics

        lens = self.lens()
        sources = (self.source(1.0, "blue_source"), self.source(2.0, "red_source"))
        grid = mc.ImagePlaneGrid(
            shape=(24, 32),
            field_of_view_arcsec=(1.2, 1.6),
            center_arcsec=(-0.1, 0.1),
        )
        renderer = mc.CausticsMacroImageRenderer(
            lens,
            grid,
            band_names=("blue", "red"),
            wavelengths_angstrom=(5_000.0, 7_000.0),
            dtype=torch.float64,
        )
        actual = renderer.render(12.0, sources=sources)
        expected = []
        for index, source in enumerate(sources):
            direct = caustics.LensSource(
                lens,
                source,
                pixelscale=0.05,
                pixels_x=32,
                pixels_y=24,
                psf=[[1.0]],
                x0=0.1,
                y0=-0.1,
                name=f"direct_{index}",
            ).to(dtype=torch.float64)
            expected.append(direct())
        torch.testing.assert_close(actual.values, torch.stack(expected, dim=-1))
        self.assertEqual(actual.metadata["renderer"], "caustics.LensSource")

    def test_caustics_handles_lens_light_psf_and_retained_components(self) -> None:
        lens = self.lens()
        source = self.source(1.0, "source")
        lens_light = self.source(0.3, "lens_light")
        grid = mc.ImagePlaneGrid((25, 25), (1.25, 1.25))
        axis = torch.arange(-2, 3, dtype=torch.float64)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        psf = torch.exp(-(xx.square() + yy.square()) / 2.0)
        renderer = mc.CausticsMacroImageRenderer(
            lens,
            grid,
            band_names=("band",),
            wavelengths_angstrom=(6_000.0,),
            psf_mode="conv2d",
            dtype=torch.float64,
        )
        result = renderer.render(
            0.0,
            sources=source,
            lens_light=lens_light,
            psf=psf,
            retain_components=True,
        )
        reconstructed = (
            result.component_values["lensed_source"]
            + result.component_values["lens_light"]
        )
        torch.testing.assert_close(result.noiseless_values, reconstructed)
        self.assertGreater(float(result.values.sum()), 0.0)

    def test_package_source_converts_to_caustics_pixelated_bands(self) -> None:
        distance = 1.0e25
        meters_per_arcsec = distance * math.pi / (180.0 * 3_600.0)
        geometry = mc.SourceGeometry(
            shape=(9, 9),
            pixel_scale_m=(0.02 * meters_per_arcsec, 0.02 * meters_per_arcsec),
            wavelengths_angstrom=(5_000.0, 7_000.0),
            band_names=("blue", "red"),
        )
        image = torch.zeros(9, 9, 2)
        image[3:6, 3:6] = torch.tensor([1.0, 2.0])
        source = mc.StaticSource(image, geometry)
        grid = mc.ImagePlaneGrid((32, 32), (1.6, 1.6))
        bands = mc.caustics_pixelated_sources(
            source,
            0.0,
            distance,
            grid,
            convert_to_pixel_flux=False,
        )
        renderer = mc.CausticsMacroImageRenderer(
            self.lens(),
            grid,
            band_names=geometry.band_names,
            wavelengths_angstrom=geometry.wavelengths_angstrom,
        )
        result = renderer.render(0.0, sources=bands)
        self.assertEqual(result.values.shape, (32, 32, 2))
        self.assertTrue(torch.isfinite(result.values).all())
        self.assertGreater(float(result.values.sum()), 0.0)

    def test_observation_callback_is_instrument_independent(self) -> None:
        renderer = mc.CausticsMacroImageRenderer(
            self.lens(),
            mc.ImagePlaneGrid((16, 16), (0.8, 0.8)),
            band_names=("band",),
            wavelengths_angstrom=(6_000.0,),
        )

        def observation(noiseless, **context):
            self.assertEqual(context["band_names"], ("band",))
            variance = torch.full_like(noiseless, 0.25)
            return noiseless + 1.0, variance, {"name": "test detector"}

        result = renderer.render(
            3.0,
            sources=self.source(1.0, "source"),
            observation_model=observation,
        )
        torch.testing.assert_close(result.values, result.noiseless_values + 1.0)
        torch.testing.assert_close(
            result.variance,
            torch.full_like(result.values, 0.25),
        )
        self.assertEqual(result.metadata["observation"]["name"], "test detector")

    def test_static_caustics_macro_model_does_not_misroute_empty_parameters(self) -> None:
        model = mc.CausticsMacroModel(self.lens(), parameters=torch.empty(0))
        x = torch.tensor([0.8], dtype=torch.float64)
        y = torch.tensor([0.1], dtype=torch.float64)
        beta_x, beta_y = model.raytrace(x, y)
        jacobian = model.jacobian_lens_equation(x, y)
        delay = model.time_delay_days(x, y)
        self.assertEqual(beta_x.shape, x.shape)
        self.assertEqual(beta_y.shape, y.shape)
        self.assertEqual(jacobian.shape, (1, 2, 2))
        self.assertEqual(delay.shape, x.shape)
        self.assertTrue(torch.isfinite(jacobian).all())

    def test_caustics_macroimage_solver_recovers_sis_images_and_delays(self) -> None:
        solutions = mc.solve_caustics_macroimages(
            self.lens(),
            0.1,
            0.0,
            parameters=torch.empty(0),
            initial_grid_size=31,
            field_of_view_arcsec=1.2,
        )
        self.assertEqual(len(solutions), 2)
        positions = sorted(solution.x_arcsec for solution in solutions)
        self.assertAlmostEqual(positions[0], -0.5883, places=3)
        self.assertAlmostEqual(positions[1], 0.7913, places=3)
        self.assertEqual(min(item.arrival_time_delay_days for item in solutions), 0.0)
        self.assertTrue(all(math.isfinite(item.shear) for item in solutions))

    def test_epl_shear_convenience_model_has_no_unfilled_parameters(self) -> None:
        """Protect the static SinglePlane redshift-parameter contract."""

        solutions = mc.solve_epl_shear_macroimages(
            mc.EPLShearConfig(
                lens_redshift=0.5,
                source_redshift=1.0,
                einstein_radius_arcsec=0.7,
            ),
            0.1,
            0.0,
            initial_grid_size=31,
            field_of_view_arcsec=1.2,
        )
        self.assertEqual(len(solutions), 2)
        positions = sorted(solution.x_arcsec for solution in solutions)
        self.assertAlmostEqual(positions[0], -0.6, places=3)
        self.assertAlmostEqual(positions[1], 0.8, places=3)


if __name__ == "__main__":
    unittest.main()
