"""Opt-in scientific comparison against Weisenbach et al.'s IPM code.

The external executable and its large benchmark products are deliberately not
vendored. Set ``MICROCAUSTICS_WEISENBACH_AGGREGATE`` to an aggregate ``.npz``
created by the paper adapter to run this test.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np

import microcaustics as mc

AGGREGATE = os.environ.get("MICROCAUSTICS_WEISENBACH_AGGREGATE", "")
CCF_FRAME = os.environ.get("MICROCAUSTICS_WEISENBACH_CCF_FRAME", "")
REFERENCE_CAUSTICS = os.environ.get(
    "MICROCAUSTICS_REFERENCE_CAUSTIC_FRAME",
    "",
)


@unittest.skipUnless(
    AGGREGATE and Path(AGGREGATE).is_file(),
    "set MICROCAUSTICS_WEISENBACH_AGGREGATE to a Luke/Weisenbach aggregate NPZ",
)
class WeisenbachExternalComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = np.load(AGGREGATE, allow_pickle=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.data.close()

    def test_adapter_contract_preserves_the_fixed_aperture_problem(self) -> None:
        self.assertGreaterEqual(int(self.data["external_benchmark_schema_version"]), 6)
        self.assertEqual(
            str(self.data["luke_adapter_contract"]), "fixed_aperture_recenter_v1"
        )
        self.assertEqual(
            str(self.data["starfile_format"]), "luke_binary_fixed_aperture"
        )
        self.assertEqual(int(self.data["num_rays_y"]), 1)
        self.assertRegex(str(self.data["problem_json_sha256"]), r"^[0-9a-f]{64}$")
        self.assertRegex(str(self.data["star_field_sha256"]), r"^[0-9a-f]{64}$")

        rows = json.loads(str(self.data["per_frame_rows_json"]))
        self.assertGreater(len(rows), 0)
        notes = " ".join(str(row.get("notes", "")) for row in rows)
        self.assertIn("coordinates are rotated into Luke's y1/y2", notes)
        self.assertIn("source products transformed back", notes)

    def test_external_light_curve_matches_the_shared_reference(self) -> None:
        flux = self.data["flux"]
        reference_flux = self.data["reference_flux"]
        stored_residual_mag = self.data["residual_mag"]
        self.assertEqual(flux.shape, reference_flux.shape)
        self.assertTrue(np.all(np.isfinite(flux)))
        self.assertTrue(np.all(flux > 0.0))

        expected_residual = -2.5 * np.log10(flux / reference_flux)
        np.testing.assert_allclose(
            stored_residual_mag,
            expected_residual,
            rtol=0.0,
            atol=5.0e-12,
        )
        comparison = mc.compare_light_curves(flux, reference_flux)
        expected_rmse_mmag = 1000.0 * np.sqrt(np.mean(stored_residual_mag**2))
        self.assertAlmostEqual(comparison.rmse_mmag, expected_rmse_mmag, places=8)

        maximum_rmse = float(
            os.environ.get("MICROCAUSTICS_WEISENBACH_MAX_LC_RMSE_MMAG", "20")
        )
        self.assertLess(comparison.rmse_mmag, maximum_rmse)

    def test_external_map_metric_is_finite_and_has_valid_pixels(self) -> None:
        self.assertTrue(np.isfinite(float(self.data["fractional_map_nrmse"])))
        self.assertGreater(float(self.data["fractional_map_nrmse"]), 0.0)
        self.assertGreater(int(self.data["fractional_map_valid_pixels"]), 0)
        frame_values = np.asarray(self.data["frame_fractional_map_nrmse"])
        self.assertTrue(np.all(np.isfinite(frame_values)))
        self.assertEqual(len(frame_values), len(self.data["accuracy_frame_indices"]))


@unittest.skipUnless(
    CCF_FRAME
    and REFERENCE_CAUSTICS
    and Path(CCF_FRAME).is_file()
    and Path(REFERENCE_CAUSTICS).is_file(),
    "set MICROCAUSTICS_WEISENBACH_CCF_FRAME and MICROCAUSTICS_REFERENCE_CAUSTIC_FRAME",
)
class WeisenbachWindingMapComparisonTests(unittest.TestCase):
    """Compare complete-field topology inferred from independent CCF curves.

    Weisenbach CCF exports caustic curves, not a native winding-number map.
    Its segment orientations do not encode microcaustics' determinant-side
    convention, so raw signed integers are not the same defined observable.
    Their modulo-two winding is orientation independent and is the valid
    external topological comparison.
    """

    @staticmethod
    def _segments(data, keys) -> np.ndarray:
        for key in keys:
            if key in data:
                segments = np.asarray(data[key], dtype=np.float64)
                if segments.ndim == 3 and segments.shape[1:] == (2, 2):
                    return segments
        raise KeyError(f"none of the caustic segment keys are present: {keys}")

    @staticmethod
    def _extent(data, keys) -> np.ndarray | None:
        for key in keys:
            if key in data:
                extent = np.asarray(data[key], dtype=np.float64).reshape(-1)
                if extent.shape == (4,):
                    return extent
        return None

    def test_winding_derived_parity_map_matches_weisluke_ccf(self) -> None:
        """Compare winding-derived parity from both complete curve fields."""

        import torch

        with np.load(CCF_FRAME, allow_pickle=False) as luke_data:
            self.assertEqual(str(luke_data["ccf_status"]), "ok")
            luke_segments = self._segments(luke_data, ("segments_src",))
            extent = self._extent(luke_data, ("source_extent_uas",))
        with np.load(REFERENCE_CAUSTICS, allow_pickle=False) as reference_data:
            reference_segments = self._segments(
                reference_data,
                (
                    "reference_segments_src",
                    "caustic_segments_uas",
                    "segments_src",
                ),
            )
            reference_extent = self._extent(
                reference_data,
                ("source_extent", "source_extent_uas"),
            )

        if extent is None:
            extent = reference_extent
        self.assertIsNotNone(extent)
        if reference_extent is not None:
            np.testing.assert_allclose(
                reference_extent,
                extent,
                rtol=0.0,
                atol=1.0e-5,
            )
        xmin, xmax, ymin, ymax = (float(value) for value in extent)
        bins = int(os.environ.get("MICROCAUSTICS_WEISENBACH_WINDING_BINS", "16"))
        self.assertGreaterEqual(bins, 8)
        grid = mc.PlaneGrid(
            (bins, bins),
            (ymax - ymin, xmax - xmin),
            ((ymin + ymax) / 2.0, (xmin + xmax) / 2.0),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        luke_segments_tensor = torch.as_tensor(
            luke_segments,
            device=device,
            dtype=torch.float64,
        )
        reference_segments_tensor = torch.as_tensor(
            reference_segments,
            device=device,
            dtype=torch.float64,
        )
        dummy_lens_grid = mc.PlaneGrid((2, 2), (ymax - ymin, xmax - xmin))
        luke_field = mc.CausticField(
            luke_segments_tensor,
            luke_segments_tensor,
            dummy_lens_grid,
            metadata={"external_code": "weisluke CCF"},
        )
        reference_field = mc.CausticField(
            reference_segments_tensor,
            reference_segments_tensor,
            dummy_lens_grid,
            metadata={"method": "microcaustics complete field"},
        )

        query_kwargs = {
            "point_chunk_size": 128 if device.type == "cuda" else 16,
            "segment_chunk_size": 65_536 if device.type == "cuda" else 4_096,
        }
        luke_winding = luke_field.winding_map(grid, **query_kwargs).values
        reference_winding = reference_field.winding_map(grid, **query_kwargs).values
        luke_binary = torch.remainder(luke_winding, 2)
        reference_binary = torch.remainder(reference_winding, 2)
        binary_mismatch = float(
            (luke_binary != reference_binary).to(torch.float64).mean().cpu()
        )
        maximum_mismatch = float(
            os.environ.get(
                "MICROCAUSTICS_WEISENBACH_MAX_WINDING_MISMATCH",
                "0.05",
            )
        )
        diagnostics = (
            f"binary_mismatch={binary_mismatch:.6g}, "
            f"luke_values={torch.unique(luke_winding).cpu().tolist()}, "
            f"reference_values={torch.unique(reference_winding).cpu().tolist()}"
        )
        self.assertLessEqual(binary_mismatch, maximum_mismatch, diagnostics)


if __name__ == "__main__":
    unittest.main()
