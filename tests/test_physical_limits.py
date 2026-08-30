"""Analytic and statistical limits for source variability and GR products."""

from __future__ import annotations

import math
import unittest

import torch

import microcaustics as mc


class VariabilityLimitTests(unittest.TestCase):
    def test_broken_power_law_psd_matches_formula_and_asymptotic_slopes(self) -> None:
        psd = mc.BrokenPowerLawPSD(
            break_timescale_days=100.0,
            low_frequency_slope=0.5,
            high_frequency_slope=2.5,
        )
        frequency = torch.logspace(-6.0, 2.0, 4097, dtype=torch.float64)
        break_frequency = 1.0 / 100.0
        expected = frequency.pow(-0.5) / (
            1.0 + (frequency / break_frequency).pow(2.0)
        )
        torch.testing.assert_close(psd(frequency), expected, rtol=2.0e-15, atol=0.0)

        log_frequency = torch.log(frequency)
        log_power = torch.log(psd(frequency))
        local_slope = (log_power[1:] - log_power[:-1]) / (
            log_frequency[1:] - log_frequency[:-1]
        )
        low = frequency[:-1] < break_frequency / 100.0
        high = frequency[:-1] > break_frequency * 100.0
        self.assertAlmostEqual(float(local_slope[low].median()), -0.5, places=4)
        self.assertAlmostEqual(float(local_slope[high].median()), -2.5, places=4)

    def test_random_phase_synthesis_recovers_requested_psd_shape(self) -> None:
        sample_count = 4096
        times = torch.arange(sample_count, dtype=torch.float64)
        psd = mc.BrokenPowerLawPSD(
            break_timescale_days=64.0,
            low_frequency_slope=0.0,
            high_frequency_slope=2.0,
        )
        signal = mc.driving_signal_from_psd(
            times,
            psd,
            padding_factor=1,
            fourier_sampling="random_phase",
            amplitude_transform="lognormal",
            seed=31415,
            dtype=torch.float64,
        )

        # The logarithm removes the deterministic lognormal transform. With
        # random-phase sampling and no crop, every non-Nyquist Fourier modulus
        # is proportional to sqrt(PSD), up to one global standardization.
        latent = torch.log(signal.values[:, 0])
        latent = latent - latent.mean()
        frequency = torch.fft.rfftfreq(sample_count, d=1.0, dtype=torch.float64)
        measured = torch.fft.rfft(latent).abs().square()[1:-1]
        expected = psd(frequency[1:-1])
        ratio = measured / expected
        normalized_ratio = ratio / ratio.median()
        torch.testing.assert_close(
            normalized_ratio,
            torch.ones_like(normalized_ratio),
            rtol=5.0e-13,
            atol=5.0e-13,
        )

    def test_damped_random_walk_recovers_analytic_log_structure_function(self) -> None:
        damping_time = 200.0
        asymptotic_log_std = 0.2
        times = torch.arange(20_000, dtype=torch.float64)
        lags = (1, 10, 50, 200)
        ensemble = torch.zeros(len(lags), dtype=torch.float64)
        for seed in range(8):
            signal = mc.lognormal_damped_random_walk(
                times,
                damping_timescale_days=damping_time,
                asymptotic_log_std=asymptotic_log_std,
                seed=seed,
            )
            log_flux = torch.log(signal.values[:, 0])
            ensemble += torch.stack(
                [
                    (log_flux[lag:] - log_flux[:-lag]).square().mean()
                    for lag in lags
                ]
            )
        ensemble /= 8.0

        for lag, measured in zip(lags, ensemble, strict=True):
            expected = 2.0 * asymptotic_log_std**2 * (
                1.0 - math.exp(-lag / damping_time)
            )
            self.assertAlmostEqual(
                float(measured / expected),
                1.0,
                delta=0.04,
                msg=f"structure-function mismatch at lag {lag} days",
            )

    def test_lognormal_driver_recovers_requested_mean_and_variance(self) -> None:
        sample_count = 65_536
        requested_mean = 2.0
        requested_std = 0.5
        signal = mc.driving_signal_from_psd(
            torch.arange(sample_count, dtype=torch.float64),
            lambda frequency: torch.ones_like(frequency),
            mean_amplitude=requested_mean,
            standard_deviation=requested_std,
            padding_factor=1,
            fourier_sampling="gaussian",
            amplitude_transform="lognormal",
            seed=99,
            dtype=torch.float64,
        )
        values = signal.values[:, 0]
        self.assertAlmostEqual(
            float(values.mean()), requested_mean, delta=0.005 * requested_mean
        )
        self.assertAlmostEqual(
            float(values.var(unbiased=True)),
            requested_std**2,
            delta=0.01 * requested_std**2,
        )


class TransferFunctionLimitTests(unittest.TestCase):
    def test_generated_newtonian_face_on_lamppost_transfer(self) -> None:
        from microcaustics.sources import ThermalReprocessingSource

        dtype = torch.float64
        mass_solar = 1.0e8
        lamp_height_rg = 12.0
        disk_outer_rg = 80.0
        light_days_per_rg = mc.gravitational_radius_m(mass_solar) / (
            299_792_458.0 * 86_400.0
        )

        geometry = mc.SourceGeometry(
            shape=(129, 129),
            pixel_scale_m=(1.0, 1.0),
            wavelengths_angstrom=(5000.0,),
            band_names=("band",),
        )
        axis = torch.linspace(-disk_outer_rg, disk_outer_rg, 129, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        hit = (radius >= 6.0) & (radius <= disk_outer_rg)
        lamp_distance = torch.sqrt(radius.square() + lamp_height_rg**2)
        analytic_delay = (lamp_distance + lamp_height_rg) * light_days_per_rg
        response_temperature4 = lamp_height_rg / lamp_distance.pow(3)
        transfer = mc.ObserverTransfer(
            radius,
            torch.ones_like(radius),
            torch.ones_like(radius),
            hit,
            relative_delay_days=analytic_delay,
            emission_azimuth_rad=torch.atan2(yy, xx),
            metadata={"backend": "analytic_newtonian_face_on_lamppost"},
        )
        source = ThermalReprocessingSource(
            geometry,
            transfer,
            mc.CallableDrivingSignal(lambda time: torch.ones_like(time)),
            response_temperature4,
            analytic_delay,
            black_hole_mass_solar=mass_solar,
            eddington_ratio=0.1,
            spin=0.0,
            source_redshift=0.0,
        )
        minimum_delay = analytic_delay[hit].min()
        maximum_delay = analytic_delay[hit].max()
        edges = torch.linspace(
            float(minimum_delay),
            float(maximum_delay) * (1.0 + 1.0e-12),
            257,
            dtype=dtype,
        )
        product = mc.steady_transfer_function(source, edges)

        self.assertAlmostEqual(float(product.values[:, 0].sum()), 1.0, places=14)
        nonzero = torch.nonzero(product.values[:, 0] > 0.0).reshape(-1)
        self.assertEqual(int(nonzero[0]), 0)
        self.assertEqual(int(nonzero[-1]), product.values.shape[0] - 1)

        weights = source.linear_response_weights(dtype=dtype)[..., 0]
        direct_mean = (weights[hit] * analytic_delay[hit]).sum() / weights[hit].sum()
        bin_width = edges[1] - edges[0]
        self.assertLess(
            abs(float(product.mean_delays_days[0] - direct_mean)),
            0.5 * float(bin_width),
        )
        self.assertTrue(
            bool(
                torch.allclose(
                    analytic_delay,
                    torch.rot90(analytic_delay, 1, (0, 1)),
                    rtol=0.0,
                    atol=2.0e-15,
                )
            )
        )


class GeneralRelativityLimitTests(unittest.TestCase):
    @staticmethod
    def _trace(dtype: torch.dtype):
        screen = mc.ObserverScreen.uniform(
            (32, 32),
            (60.0, 60.0),
            gravitational_radius_m=1.0,
            observer_distance_m=1.0e6,
            dtype=dtype,
        )
        primary = mc.trace_primary_equatorial(
            screen,
            spin=0.7,
            inclination_deg=53.0,
            disk_outer_rg=50.0,
        )
        return mc.add_observer_coordinates(
            primary,
            screen,
            black_hole_mass_solar=1.0e9,
            spin=0.7,
            inclination_deg=53.0,
            coordinate_dtype=dtype,
        ).transfer

    def test_float32_gr_transfer_matches_float64_on_same_screen(self) -> None:
        transfer32 = self._trace(torch.float32)
        transfer64 = self._trace(torch.float64)
        torch.testing.assert_close(transfer32.hit, transfer64.hit, rtol=0, atol=0)
        hit = transfer64.hit

        torch.testing.assert_close(
            transfer32.radius_rg[hit].double(),
            transfer64.radius_rg[hit],
            rtol=2.0e-6,
            atol=4.0e-5,
        )
        torch.testing.assert_close(
            transfer32.gfactor[hit].double(),
            transfer64.gfactor[hit],
            rtol=2.0e-6,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            transfer32.emission_azimuth_rad[hit].double(),
            transfer64.emission_azimuth_rad[hit],
            rtol=2.0e-5,
            atol=6.0e-5,
        )
        delay_error = (
            transfer32.relative_delay_days[hit].double()
            - transfer64.relative_delay_days[hit]
        )
        self.assertLess(float(delay_error.square().mean().sqrt()), 0.01)
        self.assertLess(float(delay_error.abs().max()), 0.03)

    def test_nearly_face_on_schwarzschild_transfer_is_rotationally_symmetric(
        self,
    ) -> None:
        screen = mc.ObserverScreen.uniform(
            (64, 64),
            (80.0, 80.0),
            gravitational_radius_m=1.0,
            observer_distance_m=1.0e7,
            dtype=torch.float64,
        )
        primary = mc.trace_primary_equatorial(
            screen,
            spin=0.0,
            inclination_deg=1.0e-5,
            disk_outer_rg=35.0,
        )
        transfer = mc.add_observer_coordinates(
            primary,
            screen,
            black_hole_mass_solar=1.0e8,
            spin=0.0,
            inclination_deg=1.0e-5,
            coordinate_dtype=torch.float64,
        ).transfer
        rotated_hit = torch.rot90(transfer.hit, 1, (0, 1))
        torch.testing.assert_close(transfer.hit, rotated_hit, rtol=0, atol=0)
        hit = transfer.hit & rotated_hit
        for field_name, tolerance in (
            ("radius_rg", 6.0e-7),
            ("gfactor", 1.0e-7),
            ("relative_delay_days", 6.0e-8),
        ):
            values = getattr(transfer, field_name)
            rotated = torch.rot90(values, 1, (0, 1))
            self.assertLess(float((values[hit] - rotated[hit]).abs().max()), tolerance)

    def test_schwarzschild_transfer_approaches_weak_field_limit(self) -> None:
        screen = mc.ObserverScreen.uniform(
            (64, 64),
            (1000.0, 1000.0),
            gravitational_radius_m=1.0,
            observer_distance_m=1.0e9,
            dtype=torch.float64,
        )
        primary = mc.trace_primary_equatorial(
            screen,
            spin=0.0,
            inclination_deg=1.0e-5,
            disk_outer_rg=700.0,
        )
        transfer = mc.add_observer_coordinates(
            primary,
            screen,
            black_hole_mass_solar=1.0e8,
            spin=0.0,
            inclination_deg=1.0e-5,
            coordinate_dtype=torch.float64,
        ).transfer
        distant = transfer.hit & (transfer.radius_rg > 300.0)
        self.assertGreater(int(distant.sum()), 1000)
        self.assertLess(float((transfer.gfactor[distant] - 1.0).abs().max()), 0.006)

        screen_radius = torch.sqrt(screen.x_rg.square() + screen.y_rg.square())
        fractional_radius_error = (
            (transfer.radius_rg - screen_radius).abs() / screen_radius.clamp_min(1.0)
        )
        self.assertLess(float(fractional_radius_error[distant].max()), 0.004)

    def test_gr_lamppost_approaches_euclidean_delay_and_illumination(self) -> None:
        source_height_rg = 1000.0
        profile = mc.axis_lamppost_profile(
            spin=0.0,
            source_height_rg=source_height_rg,
            disk_outer_rg=5000.0,
            nalpha=1024,
            radial_bins=256,
            dtype=torch.float64,
        )
        radius = profile.radius_rg
        euclidean_delay = torch.sqrt(radius.square() + source_height_rg**2)
        euclidean_delay = euclidean_delay - euclidean_delay.min()
        euclidean_illumination = source_height_rg / (
            radius.square() + source_height_rg**2
        ).pow(1.5)
        weak_field = radius > 2000.0
        self.assertGreater(int(weak_field.sum()), 100)

        relative_delay_error = (
            (profile.lamp_delay_rg - euclidean_delay).abs()
            / euclidean_delay.clamp_min(1.0)
        )
        self.assertLess(float(relative_delay_error[weak_field].max()), 0.005)
        illumination_ratio = (
            profile.illumination[weak_field]
            / euclidean_illumination[weak_field]
        )
        normalized_illumination_ratio = illumination_ratio / illumination_ratio.median()
        self.assertLess(
            float((normalized_illumination_ratio - 1.0).abs().max()), 0.004
        )
        self.assertLess(
            float(
                (profile.gfactor_lamp_to_disk[weak_field] - 1.0)
                .abs()
                .max()
            ),
            0.001,
        )


if __name__ == "__main__":
    unittest.main()
