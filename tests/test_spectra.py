"""Bandpass integration and empirical quasar-spectrum tests."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

import microcaustics as mc
from microcaustics.spectral_photometry import (
    _log_flux_interpolate,
    finish_spectral_light_curve,
    spectral_sampling_plan,
)


class BandpassTests(unittest.TestCase):
    def test_flat_fnu_is_preserved(self) -> None:
        bandpass = mc.Bandpass(
            [4000.0, 4500.0, 5500.0, 6000.0],
            [0.0, 1.0, 1.0, 0.0],
            "test",
        )
        nodes, weights = bandpass.quadrature(32)
        self.assertEqual(nodes.shape, (32,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=14)
        self.assertAlmostEqual(float(np.sum(3.7 * weights)), 3.7, places=13)

    def test_bundled_lsst_filters_and_grid(self) -> None:
        bandpasses = mc.BandpassSet.lsst()
        self.assertEqual(bandpasses.names, tuple("ugrizy"))
        grid = bandpasses.grid(8)
        self.assertEqual(grid.weights.shape, (48, 6))
        integrated = grid.integrate(torch.full((3, 48), 2.5))
        torch.testing.assert_close(integrated, torch.full((3, 6), 2.5))

    def test_custom_bandpass_set(self) -> None:
        custom = mc.BandpassSet(
            {
                "blue": mc.Bandpass([4000, 4500, 5000], [0, 1, 0], "blue"),
                "red": mc.Bandpass([6000, 6500, 7000], [0, 1, 0], "red"),
            }
        )
        self.assertEqual(custom.names, ("blue", "red"))

    def test_lsst_integration_matches_speclite(self) -> None:
        try:
            import astropy.units as units
            from speclite.filters import load_filters
        except ImportError:
            self.skipTest("speclite is an external test dependency")
        grid = mc.BandpassSet.lsst().grid(32)
        wavelength = np.asarray(grid.wavelengths_angstrom)
        flux_nu = 2.3 * (wavelength / 6000.0) ** 0.7
        actual = grid.integrate(torch.from_numpy(flux_nu)).numpy()
        expected = []
        for response in load_filters("lsst2023-*"):
            response_wavelength = response.wavelength * units.AA
            spectrum_nu = 2.3 * units.Jy * (
                response.wavelength / 6000.0
            ) ** 0.7
            spectrum_lambda = spectrum_nu.to(
                units.erg / units.s / units.cm**2 / units.AA,
                equivalencies=units.spectral_density(response_wavelength),
            )
            expected.append(
                float(
                    response.get_ab_maggies(
                        spectrum_lambda,
                        response_wavelength,
                    )
                    * 3631.0
                )
            )
        difference_mag = -2.5 * np.log10(actual / np.asarray(expected))
        self.assertLess(float(np.max(np.abs(difference_mag))), 0.003)


class QuasarSpectrumTests(unittest.TestCase):
    def test_population_sampling_is_reproducible_and_isolated(self) -> None:
        np.random.seed(91)
        expected = np.random.random(4)
        np.random.seed(91)
        first = mc.QuasarSpectrumPopulation().sample(seed=17)
        actual = np.random.random(4)
        second = mc.QuasarSpectrumPopulation().sample(seed=17)
        different = mc.QuasarSpectrumPopulation().sample(seed=18)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_fixed_population_parameters_remain_fixed(self) -> None:
        population = mc.QuasarSpectrumPopulation(
            ebv=0.0,
            emission_line_scale=1.2,
            host_fraction=0.0,
            absolute_i_magnitude_scatter=0.0,
            global_magnitude_scatter=0.0,
            color_tilt_scatter=0.0,
            amplitude_color_scatter=0.0,
        )
        spectrum = population.sample(seed=3)
        self.assertEqual(spectrum.ebv, 0.0)
        self.assertEqual(spectrum.emission_line_scale, 1.2)
        self.assertEqual(spectrum.host_fraction, 0.0)

    def test_unseeded_population_draw_is_deterministic_after_sampling(self) -> None:
        spectrum = mc.QuasarSpectrumPopulation().sample()
        wavelengths = torch.linspace(3500.0, 10500.0, 6)
        first = spectrum.band_offsets(tuple("ugrizy"), wavelengths)
        second = spectrum.band_offsets(tuple("ugrizy"), wavelengths)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_continuum_only_model_is_identity(self) -> None:
        model = mc.QuasarSpectrum(
            ebv=0.0,
            include_emission_lines=False,
            include_host=False,
            include_igm_absorption=False,
        )
        wavelength = torch.linspace(3000.0, 11000.0, 100, dtype=torch.float64)
        continuum = torch.linspace(1.0, 2.0, 100, dtype=torch.float64)
        components = model.components(
            wavelength,
            continuum,
            source_redshift=1.0,
        )
        torch.testing.assert_close(components.continuum_fnu, continuum)
        torch.testing.assert_close(
            components.emission_line_fnu, torch.zeros_like(continuum)
        )
        torch.testing.assert_close(components.host_fnu, torch.zeros_like(continuum))

    def test_host_requires_normalization_wavelengths(self) -> None:
        model = mc.QuasarSpectrum(include_emission_lines=False)
        wavelength = torch.linspace(3000.0, 9000.0, 32)
        with self.assertRaisesRegex(ValueError, "4000-5000"):
            model.components(wavelength, torch.ones(32), source_redshift=2.0)
        required = model.required_wavelengths(2.0)
        self.assertGreaterEqual(len(required), 32)
        self.assertIn(12000.0, required)

    def test_dense_returned_spectrum_does_not_add_spatial_disk_channels(self) -> None:
        bandpasses = mc.BandpassSet(
            (mc.Bandpass([4000, 5000, 6000], [0, 1, 0], "b"),)
        )
        coarse = spectral_sampling_plan(
            bandpasses,
            continuum_samples=8,
            spectrum_wavelengths=np.linspace(4100.0, 5900.0, 1000),
            return_spectrum=True,
        )
        self.assertEqual(len(coarse.wavelengths_angstrom), 8)
        self.assertEqual(len(coarse.output_wavelengths_angstrom), 1000)
        default_output = spectral_sampling_plan(
            mc.BandpassSet.lsst(),
            continuum_samples=4,
            spectrum_wavelengths=None,
            return_spectrum=True,
        )
        # Four physical disk wavelengths span the union of all filter supports.
        self.assertEqual(len(default_output.wavelengths_angstrom), 4)
        self.assertEqual(len(default_output.output_wavelengths_angstrom), 1000)

    def test_default_log_grid_and_dense_bandpass_are_decoupled(self) -> None:
        bandpasses = mc.BandpassSet.lsst()
        plan = spectral_sampling_plan(
            bandpasses,
            continuum_samples=32,
            spectrum_wavelengths=None,
            return_spectrum=False,
        )
        physical = np.asarray(plan.wavelengths_angstrom)
        np.testing.assert_allclose(np.diff(np.log(physical)), np.diff(np.log(physical))[0])
        self.assertEqual(len(physical), 32)
        self.assertGreater(len(plan.band_grid.wavelengths_angstrom), 32)
        np.testing.assert_allclose(plan.band_grid.weights.sum(dim=0), 1.0, atol=1e-13)

    def test_lensed_spectrum_reuses_in_range_normalization_wavelengths(self) -> None:
        model = replace(mc.QuasarSpectrum(), include_host=False)
        required = model.required_wavelengths(1.695)
        bandpasses = mc.BandpassSet.lsst()
        plan = spectral_sampling_plan(
            bandpasses,
            continuum_samples=32,
            spectrum_wavelengths=None,
            return_spectrum=False,
            required_wavelengths=required,
        )
        self.assertLess(len(plan.wavelengths_angstrom), 40)
        lower = min(band.support_angstrom[0] for band in bandpasses.bandpasses)
        upper = max(band.support_angstrom[1] for band in bandpasses.bandpasses)
        self.assertTrue(
            all(
                node in plan.wavelengths_angstrom
                for node in required
                if node < lower or node > upper
            )
        )

    def test_log_flux_interpolation_preserves_power_law_and_dark_pixels(self) -> None:
        physical = torch.as_tensor(np.geomspace(3200.0, 11000.0, 32))
        query = torch.linspace(3200.0, 11000.0, 1024, dtype=torch.float64)
        power_law = 1e-6 * (physical / 6000.0) ** -0.7
        values = torch.stack((power_law, torch.zeros_like(power_law)))
        actual = _log_flux_interpolate(physical, values, query)
        torch.testing.assert_close(
            actual[0], 1e-6 * (query / 6000.0) ** -0.7, rtol=1e-12, atol=0
        )
        torch.testing.assert_close(actual[1], torch.zeros_like(query))

    def test_log_flux_range_tolerates_float32_endpoint_rounding_only(self) -> None:
        wavelengths = torch.tensor([2461.80810546875, 5000.0], dtype=torch.float32)
        values = torch.tensor([1.0, 2.0], dtype=torch.float32)
        rounded_query = torch.tensor([2461.807861328125, 4000.0])
        result = _log_flux_interpolate(
            wavelengths, values, rounded_query, check_range=True
        )
        self.assertEqual(float(result[0]), 1.0)
        with self.assertRaisesRegex(ValueError, "normalization grid"):
            _log_flux_interpolate(
                wavelengths,
                values,
                torch.tensor([2460.0, 4000.0]),
                check_range=True,
            )

    def test_log_flux_interpolation_keeps_source_gradients(self) -> None:
        physical = torch.as_tensor(np.geomspace(3200.0, 11000.0, 32))
        query = torch.linspace(3200.0, 11000.0, 100, dtype=torch.float64)
        values = (
            1e-6 * (physical / 6000.0) ** -0.7
        ).requires_grad_()
        _log_flux_interpolate(physical, values, query).sum().backward()
        self.assertTrue(bool(torch.isfinite(values.grad).all()))
        self.assertTrue(bool((values.grad > 0).all()))

    def test_reconstructed_lsst_photometry_matches_dense_power_law(self) -> None:
        bandpasses = mc.BandpassSet.lsst()
        plan = spectral_sampling_plan(
            bandpasses,
            continuum_samples=32,
            spectrum_wavelengths=None,
            return_spectrum=True,
        )
        physical = torch.tensor(plan.wavelengths_angstrom, dtype=torch.float64)
        mean = 1e-6 * (physical / 6000.0) ** -0.7
        names = tuple(plan.bands_angstrom)
        raw = mc.LightCurve(
            torch.tensor([0.0], dtype=torch.float64),
            mean.unsqueeze(0),
            names,
            unlensed_flux=mean.unsqueeze(0),
        )
        result = finish_spectral_light_curve(
            raw, raw, plan,
            spectral_model=None,
            source_redshift=1.0,
            macro_magnification=1.0,
            include_microlensing_only=True,
            return_spectrum=True,
            host_lensing="omit",
            bandpass_version=bandpasses.version,
        )
        dense = bandpasses.dense_grid()
        dense_wavelengths = torch.tensor(dense.wavelengths_angstrom, dtype=torch.float64)
        expected = dense.integrate(1e-6 * (dense_wavelengths / 6000.0) ** -0.7)
        error_mmag = (-2500.0 * torch.log10(result.flux[0] / expected)).abs()
        self.assertLess(float(error_mmag.max()), 0.05)
        self.assertEqual(result.metadata["continuum_sampling"], "log_wavelength_log_flux")

    def test_lines_host_and_extinction_match_dense_model_integration(self) -> None:
        bandpasses = mc.BandpassSet.lsst()
        model = mc.QuasarSpectrum(
            absolute_i_magnitude=-25.0, ebv=0.02, host_fraction=0.2
        )
        plan = spectral_sampling_plan(
            bandpasses,
            continuum_samples=32,
            spectrum_wavelengths=None,
            return_spectrum=False,
        )
        physical = torch.tensor(plan.wavelengths_angstrom, dtype=torch.float64)
        continuum = 1e-6 * (physical / 6000.0) ** -0.7
        raw = mc.LightCurve(
            torch.tensor([0.0], dtype=torch.float64),
            continuum.unsqueeze(0),
            tuple(plan.bands_angstrom),
            unlensed_flux=continuum.unsqueeze(0),
        )
        result = finish_spectral_light_curve(
            raw, raw, plan,
            spectral_model=model,
            source_redshift=1.0,
            macro_magnification=1.0,
            include_microlensing_only=False,
            return_spectrum=False,
            host_lensing="macro",
            bandpass_version=bandpasses.version,
        )
        dense = bandpasses.dense_grid()
        dense_wavelengths = torch.tensor(dense.wavelengths_angstrom, dtype=torch.float64)
        dense_continuum = 1e-6 * (dense_wavelengths / 6000.0) ** -0.7
        expected = dense.integrate(
            model.components(
                dense_wavelengths, dense_continuum, source_redshift=1.0
            ).total_fnu
        )
        error_mmag = (-2500.0 * torch.log10(result.flux[0] / expected)).abs()
        self.assertLess(float(error_mmag.max()), 0.1)

    def test_retained_spectrum_rejects_continuum_extrapolation(self) -> None:
        with self.assertRaisesRegex(ValueError, "sampled continuum range"):
            spectral_sampling_plan(
                mc.BandpassSet.lsst(),
                continuum_samples=4,
                spectrum_wavelengths=(3_000.0, 8_000.0),
                return_spectrum=True,
            )

    def test_physical_absolute_magnitude_uses_the_sparse_continuum(self) -> None:
        model = mc.QuasarSpectrum(include_host=False)
        redshift = 1.5
        wavelengths = torch.as_tensor(
            model.required_wavelengths(redshift), dtype=torch.float64
        )
        continuum_jy = torch.full_like(wavelengths, 1.0e-6)
        distance_m = 1.0e26
        actual = model.infer_absolute_i_magnitude(
            wavelengths,
            continuum_jy,
            source_redshift=redshift,
            luminosity_distance_m=distance_m,
        )
        ten_parsec_m = 10.0 * 3.085677581491367e16
        expected_flux = (
            1.0e-6 * (distance_m / ten_parsec_m) ** 2 * 3.0 / (1.0 + redshift)
        )
        expected = -2.5 * np.log10(expected_flux / 3631.0)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_shared_postprocessing_returns_all_products(self) -> None:
        bandpasses = mc.BandpassSet(
            (mc.Bandpass([4000, 5000, 6000], [0, 1, 0], "b"),)
        )
        requested = np.linspace(4200.0, 5800.0, 5)
        plan = spectral_sampling_plan(
            bandpasses,
            continuum_samples=8,
            spectrum_wavelengths=requested,
            return_spectrum=True,
        )
        channels = len(plan.wavelengths_angstrom)
        times = torch.tensor([0.0, 1.0])
        variable = torch.stack(
            (torch.ones(channels), torch.full((channels,), 1.2))
        )
        mean = torch.ones_like(variable)
        raw = mc.LightCurve(
            times,
            variable,
            tuple(f"n{i}" for i in range(channels)),
            unlensed_flux=0.5 * variable,
        )
        raw_mean = mc.LightCurve(
            times,
            mean,
            tuple(f"n{i}" for i in range(channels)),
            unlensed_flux=0.5 * mean,
        )
        result = finish_spectral_light_curve(
            raw,
            raw_mean,
            plan,
            spectral_model=mc.QuasarSpectrum(
                ebv=0.0,
                include_emission_lines=False,
                include_host=False,
                include_igm_absorption=False,
            ),
            source_redshift=1.0,
            macro_magnification=2.0,
            include_microlensing_only=True,
            return_spectrum=True,
            host_lensing="omit",
            bandpass_version="test",
        )
        self.assertEqual(result.band_names, ("b",))
        self.assertEqual(result.flux.shape, (2, 1))
        self.assertEqual(result.microlensing_only_flux.shape, (2, 1))
        self.assertIsNotNone(result.spectrum)
        self.assertEqual(result.spectrum.total_flux.shape, (2, 5))
        self.assertEqual(
            result.spectrum.microlensing_only_continuum_flux.shape, (2, 5)
        )
        self.assertEqual(result.component_flux["continuum"].shape, (2, 1))
        with TemporaryDirectory() as directory:
            restored = mc.load_light_curve(
                mc.save_light_curve(result, Path(directory) / "spectrum.npz")
            )
        torch.testing.assert_close(restored.flux, result.flux)
        torch.testing.assert_close(
            restored.spectrum.total_flux,
            result.spectrum.total_flux,
        )
        self.assertEqual(tuple(restored.component_flux), tuple(result.component_flux))

    def test_default_lensed_spectrum_uses_physical_luminosity_and_omits_host(
        self,
    ) -> None:
        model = mc.QuasarSpectrum()
        redshift = 1.5
        lensed_model = replace(model, include_host=False)
        plan = spectral_sampling_plan(
            mc.BandpassSet.lsst(),
            continuum_samples=16,
            spectrum_wavelengths=np.linspace(3200.0, 10900.0, 128),
            return_spectrum=True,
            required_wavelengths=lensed_model.required_wavelengths(redshift),
        )
        channels = len(plan.wavelengths_angstrom)
        raw = mc.LightCurve(
            torch.tensor([0.0]),
            torch.full((1, channels), 1.0e-6),
            tuple(f"n{i}" for i in range(channels)),
            unlensed_flux=torch.full((1, channels), 5.0e-7),
        )
        result = finish_spectral_light_curve(
            raw,
            raw,
            plan,
            spectral_model=model,
            source_redshift=redshift,
            luminosity_distance_m=1.0e26,
            macro_magnification=2.0,
            include_microlensing_only=False,
            return_spectrum=True,
            host_lensing="omit",
            bandpass_version="test",
        )
        self.assertTrue(bool(torch.isfinite(result.flux).all()))
        self.assertTrue(bool(torch.any(result.component_flux["emission_lines"] != 0)))
        self.assertTrue(bool(torch.all(result.component_flux["host"] == 0)))
        self.assertIsInstance(
            result.metadata["spectral_model"]["absolute_i_magnitude"], float
        )


if __name__ == "__main__":
    unittest.main()
