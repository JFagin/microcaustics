"""CPU validation of configurable expanding-photosphere sources."""

from __future__ import annotations

import math
import unittest

import torch

import microcaustics as mc


def simple_evolution() -> mc.PowerLawExponentialPhotosphere:
    return mc.PowerLawExponentialPhotosphere(
        peak_time_rest_days=15.0,
        peak_luminosity_watts=1.0e36,
        rise_power=2.0,
        decline_time_rest_days=25.0,
        photosphere_velocity_km_s=10_000.0,
        initial_radius_m=1.0e9,
        temperature_floor_k=3_000.0,
        temperature_ceiling_k=15_000.0,
    )


class SupernovaSourceTests(unittest.TestCase):
    def test_band_mapping_replaces_parallel_name_and_wavelength_inputs(self) -> None:
        source = mc.ExpandingPhotosphereSource(
            redshift=0.5,
            bands={"blue": 4_500.0, "red": 7_500.0},
            maximum_observer_time_days=100.0,
            evolution=simple_evolution(),
            resolution=16,
            luminosity_distance_m=1.0e25,
        )
        self.assertEqual(source.band_names, ("blue", "red"))
        self.assertEqual(source.wavelengths_angstrom, (4_500.0, 7_500.0))

    def test_vectorized_brightness_and_exact_zero_before_explosion(self) -> None:
        source = mc.ExpandingPhotosphereSource(
            redshift=0.5,
            wavelengths_angstrom=(4_500.0, 7_500.0),
            band_names=("blue", "red"),
            maximum_observer_time_days=100.0,
            evolution=simple_evolution(),
            resolution=32,
            explosion_time_days=5.0,
            luminosity_distance_m=1.0e25,
        )
        brightness = source.brightness(
            torch.tensor([0.0, 5.0, 20.0, 60.0], dtype=torch.float64)
        )
        self.assertEqual(brightness.shape, (4, 32, 32, 2))
        self.assertTrue(torch.isfinite(brightness).all())
        self.assertTrue(torch.equal(brightness[:2], torch.zeros_like(brightness[:2])))
        self.assertGreater(float(brightness[2:].sum()), 0.0)
        self.assertIsInstance(source, mc.PixelatedSource)

    def test_default_float32_does_not_underflow_at_cosmological_distance(self) -> None:
        source = mc.ExpandingPhotosphereSource(
            redshift=0.65,
            wavelengths_angstrom=(7_500.0,),
            maximum_observer_time_days=100.0,
            evolution=simple_evolution(),
            resolution=32,
        )
        brightness = source.brightness(torch.tensor([30.0], dtype=torch.float32))
        self.assertEqual(brightness.dtype, torch.float32)
        self.assertTrue(torch.isfinite(brightness).all())
        self.assertGreater(float(brightness.sum()), 0.0)

    def test_fixed_field_contains_maximum_radius_with_requested_margin(self) -> None:
        source = mc.ExpandingPhotosphereSource(
            redshift=0.2,
            wavelengths_angstrom=(6_000.0,),
            maximum_observer_time_days=80.0,
            evolution=simple_evolution(),
            resolution=40,
            source_fov_margin=1.1,
            luminosity_distance_m=1.0e25,
        )
        field_half_width = (
            source.geometry.shape[1] * source.geometry.pixel_scale_m[1] / 2.0
        )
        self.assertAlmostEqual(
            field_half_width / source.maximum_photosphere_radius_m,
            1.1,
            places=12,
        )

    def test_custom_profile_and_spectral_modifier_are_supported(self) -> None:
        kwargs = dict(
            redshift=0.3,
            wavelengths_angstrom=(5_000.0, 8_000.0),
            maximum_observer_time_days=60.0,
            evolution=simple_evolution(),
            resolution=24,
            luminosity_distance_m=1.0e25,
            appearance=mc.PhotosphereAppearance(limb_darkening=0.0),
        )
        baseline = mc.ExpandingPhotosphereSource(**kwargs)

        def uniform_profile(x, y, radius, times, wavelengths):
            del times
            rho2 = (x[None] / radius[:, None, None]).square() + (
                y[None] / radius[:, None, None]
            ).square()
            return (rho2 <= 1.0)[..., None].expand(-1, -1, -1, len(wavelengths))

        def double_spectrum(times, wavelengths):
            return torch.full(
                (len(times), len(wavelengths)),
                2.0,
                dtype=times.dtype,
                device=times.device,
            )

        customized = mc.ExpandingPhotosphereSource(
            **kwargs,
            spatial_profile=uniform_profile,
            spectral_modifier=double_spectrum,
        )
        time = torch.tensor([30.0], dtype=torch.float64)
        pixel_area = math.prod(baseline.geometry.pixel_scale_m)
        baseline_flux = baseline.brightness(time).sum((1, 2)) * pixel_area
        custom_flux = customized.brightness(time).sum((1, 2)) * pixel_area
        torch.testing.assert_close(custom_flux, 2.0 * baseline_flux)

    def test_paper_preset_reproduces_peak_i_band_scale(self) -> None:
        source = mc.paper_type_ia_supernova_source(
            redshift=0.658,
            wavelengths_angstrom=(3_671.0, 4_827.0, 6_223.0, 7_546.0),
            band_names=("u", "g", "r", "i"),
            maximum_observer_time_days=200.0,
            resolution=128,
        )
        peak_observer_days = 18.0 * (1.0 + source.redshift)
        brightness = source.brightness(
            torch.tensor([peak_observer_days], dtype=torch.float64)
        )
        pixel_area = math.prod(source.geometry.pixel_scale_m)
        i_flux_jy = brightness[0, ..., 3].sum() * pixel_area
        i_magnitude_ab = -2.5 * torch.log10(i_flux_jy / 3_631.0)
        self.assertAlmostEqual(float(i_magnitude_ab), 23.55, delta=0.15)

    def test_evolution_rises_then_declines_without_late_rebrightening(self) -> None:
        evolution = simple_evolution()
        times = torch.tensor([0.0, 7.5, 15.0, 40.0, 100.0])
        luminosity = evolution.luminosity(times)
        self.assertEqual(float(luminosity[0]), 0.0)
        self.assertLess(float(luminosity[1]), float(luminosity[2]))
        self.assertGreater(float(luminosity[2]), float(luminosity[3]))
        self.assertGreater(float(luminosity[3]), float(luminosity[4]))


if __name__ == "__main__":
    unittest.main()
