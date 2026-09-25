"""Survey-observation and stochastic-driver contracts."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import microcaustics as mc


class ObservationTests(unittest.TestCase):
    def test_random_wfd_cadence_reuses_index_without_reopening_sqlite(self) -> None:
        """Exercise deterministic sky selection, filtering, and time ordering."""

        with tempfile.TemporaryDirectory(prefix="microcaustics-opsim-") as temporary:
            path = Path(temporary) / "opsim.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE observations ("
                "fieldRA REAL, fieldDec REAL, observationStartMJD REAL, "
                "band TEXT, fiveSigmaDepth REAL, seeingFwhmEff REAL, "
                "science_program TEXT)"
            )
            rows = [
                (15.0, -8.0, 60000.0 + day, band, 24.5, 0.75, "WFD")
                for day, band in zip(
                    (4.0, 0.0, 2.0, 1.0, 3.0, 5.0),
                    ("u", "g", "r", "i", "z", "y"),
                    strict=True,
                )
            ]
            connection.executemany(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            connection.execute(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?)",
                (15.0, -8.0, 60001.5, "g", 26.0, 0.4, "DD"),
            )
            connection.commit()
            connection.close()

            index = mc.RubinOpSimCadenceIndex.from_database(path)
            first = index.sample(
                seed=42,
                survey="wfd",
                radius_deg=0.5,
                min_visits=4,
                max_visits=10,
                duration_days=4.0,
            )
            with patch(
                "microcaustics.observations.sqlite3.connect",
                side_effect=AssertionError("cached sampling reopened SQLite"),
            ):
                second = index.sample(
                    seed=42,
                    survey="wfd",
                    radius_deg=0.5,
                    min_visits=4,
                    max_visits=10,
                    duration_days=4.0,
                )
            np.testing.assert_array_equal(first.time_days, second.time_days)
            np.testing.assert_array_equal(first.mjd, second.mjd)
            self.assertTrue(np.all(np.diff(first.time_days) >= 0.0))
            self.assertLessEqual(float(first.time_days[-1]), 4.0)
            self.assertNotIn("DD", first.band_names)
            self.assertEqual(first.metadata["survey"], "Rubin WFD")

    def test_index_supports_sky_position_wfd_and_named_or_random_ddf(self) -> None:
        with tempfile.TemporaryDirectory(prefix="microcaustics-opsim-") as temporary:
            path = Path(temporary) / "opsim.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE observations ("
                "fieldRA REAL, fieldDec REAL, observationStartMJD REAL, "
                "band TEXT, fiveSigmaDepth REAL, seeingFwhmEff REAL, "
                "science_program TEXT, target_name TEXT)"
            )
            rows = [
                (10.0, 0.0, 60000.0, "u", 23.5, 0.9, "", "WFD"),
                (359.8, -10.0, 60001.0, "g", 24.5, 0.8, "", "WFD"),
                (0.2, -10.0, 60003.0, "r", 24.2, 0.7, "", "WFD"),
                (150.0, 2.2, 60002.0, "g", 25.5, 0.6, "DD", "DD:COSMOS, lowdust"),
                (150.1, 2.2, 60004.0, "i", 25.0, 0.6, "DD", "DD:COSMOS, lowdust"),
                (150.0, 2.2, 60005.0, "r", 24.5, 0.7, "", "WFD"),
                (53.0, -28.1, 60006.0, "z", 24.0, 0.8, "DD", "DD:ECDFS"),
            ]
            connection.executemany(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            connection.commit()
            connection.close()

            index = mc.RubinOpSimCadenceIndex.from_database(path, bin_size_deg=0.5)
            self.assertIs(
                index,
                mc.RubinOpSimCadenceIndex.from_database(path, bin_size_deg=0.5),
            )
            self.assertEqual(set(index.ddf_fields), {"COSMOS", "ECDFS"})

            wrapped = index.at_sky_position(
                359.9,
                -10.0,
                survey="wfd",
                radius_deg=0.5,
                duration_days=None,
            )
            np.testing.assert_array_equal(wrapped.mjd, [60001.0, 60003.0])
            np.testing.assert_array_equal(wrapped.time_days, [1.0, 3.0])

            cosmos = index.sample(
                seed=3,
                survey="ddf",
                field="cosmos",
                radius_deg=0.5,
                duration_days=None,
            )
            self.assertEqual(cosmos.metadata["ddf_field"], "COSMOS")
            self.assertEqual(cosmos.band_names, ("g", "i"))
            cosmos_all = index.sample(
                seed=3,
                survey="ddf",
                field="COSMOS",
                include_wfd=True,
                radius_deg=0.5,
                duration_days=None,
            )
            self.assertEqual(cosmos_all.band_names, ("g", "i", "r"))
            random_ddf = index.sample(
                seed=4,
                survey="ddf",
                radius_deg=0.5,
                duration_days=None,
            )
            self.assertIn(random_ddf.metadata["ddf_field"], index.ddf_fields)
            with self.assertRaisesRegex(ValueError, "only valid"):
                index.sample(seed=0, survey="wfd", field="COSMOS")

    def test_rubin_uncertainty_matches_m5_definition(self) -> None:
        magnitude = torch.tensor([24.0, 24.0], dtype=torch.float64)
        uncertainty = mc.rubin_magnitude_uncertainty(
            magnitude,
            magnitude,
            ("u", "i"),
        )
        torch.testing.assert_close(
            uncertainty,
            torch.full_like(uncertainty, np.sqrt(0.04 + 0.005**2)),
        )

    def test_resolved_light_curves_are_sampled_in_each_visit_band(self) -> None:
        times = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        curve_a = mc.LightCurve(
            times,
            torch.tensor([[10.0, 20.0], [20.0, 40.0], [30.0, 60.0]]),
            ("g", "r"),
        )
        curve_b = mc.LightCurve(
            times,
            torch.tensor([[5.0, 10.0], [10.0, 20.0], [15.0, 30.0]]),
            ("g", "r"),
        )
        curves = mc.MultiImageLightCurves(
            (
                mc.MacroImageLightCurve("A", 0.0, curve_a),
                mc.MacroImageLightCurve("B", 1.0, curve_b),
            )
        )
        cadence = mc.SurveyCadence(
            np.array([0.5, 1.5]),
            ("g", "r"),
            np.array([24.5, 24.0]),
        )
        observed = mc.observe_multi_image_light_curves(
            curves,
            cadence,
            zero_point_flux=10.0,
            add_noise=False,
        )
        expected_flux = torch.tensor(
            [[15.0, 7.5], [50.0, 25.0]],
            dtype=torch.float64,
        )
        expected = -2.5 * torch.log10(expected_flux / 10.0)
        torch.testing.assert_close(observed.magnitude, expected)
        torch.testing.assert_close(observed.magnitude, observed.noiseless_magnitude)
        self.assertEqual(observed.band_names, ("g", "r"))
        self.assertTrue(bool(torch.all(observed.magnitude_error > 0)))

        single = mc.observe_light_curve(
            curve_a,
            cadence,
            image_name="A",
            zero_point_flux=10.0,
            add_noise=False,
            gamma_by_band={"g": 0.02, "r": 0.03},
            systematic_floor_mag=0.01,
        )
        torch.testing.assert_close(single.magnitude, expected[:, 0])
        torch.testing.assert_close(single.magnitude, single.noiseless_magnitude)
        self.assertEqual(tuple(single.magnitude.shape), (cadence.visit_count,))
        self.assertEqual(single.image_name, "A")
        self.assertEqual(single.metadata["systematic_floor_mag"], 0.01)

        noisy_first = mc.observe_multi_image_light_curves(
            curves,
            cadence,
            zero_point_flux=10.0,
            seed=17,
            gamma_by_band={"g": 0.02, "r": 0.03},
            systematic_floor_mag=0.01,
        )
        noisy_second = mc.observe_multi_image_light_curves(
            curves,
            cadence,
            zero_point_flux=10.0,
            seed=17,
            gamma_by_band={"g": 0.02, "r": 0.03},
            systematic_floor_mag=0.01,
        )
        torch.testing.assert_close(noisy_first.magnitude, noisy_second.magnitude)
        self.assertEqual(tuple(noisy_first.magnitude.shape), (2, 2))
        torch.testing.assert_close(noisy_first.time_days, noisy_second.time_days)

    def test_select_bands_preserves_registered_visit_metadata(self) -> None:
        cadence = mc.SurveyCadence(
            np.array([1.0, 2.0, 3.0, 4.0]),
            ("u", "g", "r", "i"),
            np.array([23.0, 24.0, 24.5, 24.2]),
            mjd=np.array([60_001.0, 60_002.0, 60_003.0, 60_004.0]),
            seeing_fwhm_arcsec=np.array([0.9, 0.8, 0.7, 0.6]),
            metadata={"field": 17},
        )
        selected = cadence.select_bands(("g", "i"))
        np.testing.assert_array_equal(selected.time_days, [2.0, 4.0])
        np.testing.assert_array_equal(selected.mjd, [60_002.0, 60_004.0])
        np.testing.assert_array_equal(selected.seeing_fwhm_arcsec, [0.8, 0.6])
        self.assertEqual(selected.band_names, ("g", "i"))
        self.assertEqual(selected.metadata["parent_visit_count"], 4)
        with self.assertRaises(ValueError):
            cadence.select_bands(("y",))

    def test_lognormal_damped_random_walk_is_positive_and_reproducible(self) -> None:
        times = torch.arange(0.0, 101.0, 1.0)
        first = mc.lognormal_damped_random_walk(times, seed=17)
        second = mc.lognormal_damped_random_walk(times, seed=17)
        torch.testing.assert_close(first.values, second.values)
        self.assertTrue(bool(torch.all(first.values > 0)))
        self.assertEqual(first.metadata()["sample_count"], 101)

    def test_paper_broken_psd_driver_is_positive_and_reproducible(self) -> None:
        times = torch.arange(0.0, 401.0, dtype=torch.float64)
        first = mc.broken_power_law_driving_signal(times, seed=21)
        second = mc.broken_power_law_driving_signal(times, seed=21)
        torch.testing.assert_close(first.values, second.values)
        self.assertTrue(bool(torch.all(first.values > 0)))
        metadata = first.metadata()
        self.assertEqual(metadata["psd"], "smooth_broken_power_law")
        self.assertEqual(metadata["fourier_sampling"], "gaussian")
        self.assertEqual(metadata["padding_factor"], 5)
        self.assertEqual(metadata["crop_start_samples"], times.numel())
        self.assertEqual(metadata["break_timescale_days"], 200.0)

    def test_broken_psd_driver_supports_even_fft_length(self) -> None:
        """Project the Nyquist bin without an overlapping-view assignment."""

        signal = mc.broken_power_law_driving_signal(
            torch.arange(0.0, 400.0, dtype=torch.float64),
            seed=22,
        )
        self.assertEqual(tuple(signal.values.shape), (400, 1))
        self.assertTrue(bool(torch.all(torch.isfinite(signal.values))))
        self.assertTrue(bool(torch.all(signal.values > 0)))

    def test_fixed_horizon_driver_matches_explicit_gaussian_interior_crop(self) -> None:
        times = torch.arange(-20., 101., dtype=torch.float64)
        fixed = mc.broken_power_law_driving_signal(
            cadence_days=1., history_days=20., max_duration_days=100.,
            seed=17, dtype=torch.float64,
        )
        explicit = mc.driving_signal_from_psd(
            times, mc.BrokenPowerLawPSD(), seed=17, padding_factor=5,
            crop_start_samples=len(times), fourier_sampling="gaussian",
            dtype=torch.float64,
        )
        actual = fixed.amplitudes(times, bands=1, dtype=torch.float64, device="cpu")
        torch.testing.assert_close(actual, explicit.values)
        # Querying only part of the same fixed horizon must not redraw it.
        subset = fixed.amplitudes(times[30:60], bands=1, dtype=torch.float64, device="cpu")
        torch.testing.assert_close(subset, actual[30:60])

    def test_arbitrary_multiband_psd_callable_is_supported(self) -> None:
        def psd(frequency: torch.Tensor) -> torch.Tensor:
            base = 1.0 / (1.0 + frequency.square())
            return torch.stack((base, 0.5 * base))

        signal = mc.driving_signal_from_psd(
            torch.arange(0.0, 64.0),
            psd,
            mean_amplitude=(1.0, 2.0),
            standard_deviation=(0.1, 0.2),
            seed=9,
        )
        self.assertEqual(tuple(signal.values.shape), (64, 2))
        self.assertTrue(bool(torch.all(signal.values > 0)))
        self.assertEqual(signal.metadata()["signal_bands"], 2)


if __name__ == "__main__":
    unittest.main()
