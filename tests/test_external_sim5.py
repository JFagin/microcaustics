"""Opt-in validation of frozen package/SIM5 comparison products."""

from __future__ import annotations

import json
import math
import os
import unittest
from pathlib import Path

SUMMARY = os.environ.get("MICROCAUSTICS_SIM5_SUMMARY", "")


@unittest.skipUnless(
    SUMMARY and Path(SUMMARY).is_file(),
    "set MICROCAUSTICS_SIM5_SUMMARY to paper_gr_sim5_summary.json",
)
class Sim5ExternalComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(Path(SUMMARY).read_text(encoding="utf-8"))

    def test_summary_has_matched_scope_and_multiple_physical_cases(self) -> None:
        self.assertIn(
            "matched primary Kerr transfer physics", self.payload["comparison_scope"]
        )
        cases = self.payload["cases"]
        self.assertGreaterEqual(len(cases), 3)
        self.assertEqual(len({case["case"] for case in cases}), len(cases))
        self.assertTrue(all(case["resolution"] >= 256 for case in cases))

    def test_sim5_agreement_stays_within_frozen_scientific_thresholds(self) -> None:
        maximum_nmse = float(
            os.environ.get("MICROCAUSTICS_SIM5_MAX_IMAGE_NMSE", "1e-6")
        )
        maximum_flux_error = float(
            os.environ.get("MICROCAUSTICS_SIM5_MAX_FLUX_FRACTIONAL_ERROR", "1e-4")
        )
        for case in self.payload["cases"]:
            with self.subTest(case=case["case"]):
                numeric = (
                    case["hit_iou"],
                    case["image_nmse"],
                    case["integrated_flux_ratio"],
                    case["radius_fractional_median_abs"],
                    case["gfactor_fractional_median_abs"],
                )
                self.assertTrue(all(math.isfinite(float(value)) for value in numeric))
                self.assertGreaterEqual(case["hit_iou"], 0.999)
                self.assertLess(case["image_nmse"], maximum_nmse)
                self.assertLess(
                    abs(case["integrated_flux_ratio"] - 1.0),
                    maximum_flux_error,
                )
                self.assertLess(case["radius_fractional_median_abs"], 1e-5)
                self.assertLess(case["gfactor_fractional_median_abs"], 1e-5)


if __name__ == "__main__":
    unittest.main()
