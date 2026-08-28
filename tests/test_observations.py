"""Survey-observation and stochastic-driver contracts."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import microcaustics as mc


class ObservationTests(unittest.TestCase):
    def test_random_wfd_cadence_reads_op_sim_without_modifying_it(self) -> None:
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

            first = mc.sample_random_rubin_wfd_cadence(
                path,
                seed=42,
                radius_deg=0.5,
                min_visits=4,
                max_visits=10,
                duration_days=4.0,
            )
            second = mc.sample_random_rubin_wfd_cadence(
                path,
                seed=42,
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
        self.assertEqual(metadata["fourier_sampling"], "random_phase")
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
