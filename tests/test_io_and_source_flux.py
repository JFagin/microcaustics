"""Tests for compact public result I/O and source-only photometry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import microcaustics as mc


class ResultIOAndSourceFluxTests(unittest.TestCase):
    def test_source_light_curve_integrates_physical_pixels_in_batches(self):
        geometry = mc.SourceGeometry(
            (2, 3), (4.0, 5.0), (5000.0, 7000.0), ("a", "b")
        )
        image = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) + 1.0
        source = mc.StaticSource(image, geometry)
        curve = mc.source_light_curve(source, [0.0, 1.0, 2.0], batch_size=2)
        expected = image.sum(dim=(0, 1)) * 20.0
        torch.testing.assert_close(curve.flux, expected.expand(3, -1))
        torch.testing.assert_close(curve.unlensed_flux, curve.flux)
        self.assertEqual(curve.band_names, ("a", "b"))

    def test_map_and_light_curve_npz_round_trip(self):
        grid = mc.PlaneGrid((3, 4), (2.0, 3.0), (0.2, -0.1))
        product = mc.MagnificationMap(
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
            grid,
            time_days=7.5,
            method="ipm",
            metadata={"rays": 1000, "nested": {"value": 2}},
        )
        curve = mc.LightCurve(
            torch.tensor([0.0, 1.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ("g", "i"),
            unlensed_flux=torch.ones(2, 2),
            metadata={"source": "test"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            map_path = mc.save_magnification_map(product, directory / "map.npz")
            curve_path = mc.save_light_curve(curve, directory / "curve.npz")
            loaded_map = mc.load_magnification_map(map_path)
            loaded_curve = mc.load_light_curve(curve_path)

        torch.testing.assert_close(loaded_map.values, product.values)
        self.assertEqual(loaded_map.grid, product.grid)
        self.assertEqual(loaded_map.metadata, product.metadata)
        self.assertEqual(loaded_map.method, "ipm")
        torch.testing.assert_close(loaded_curve.flux, curve.flux)
        torch.testing.assert_close(loaded_curve.unlensed_flux, curve.unlensed_flux)
        self.assertEqual(loaded_curve.band_names, curve.band_names)
        self.assertEqual(loaded_curve.metadata, curve.metadata)


if __name__ == "__main__":
    unittest.main()
