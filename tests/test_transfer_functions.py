"""Standalone steady and microlensed transfer-function products."""

from __future__ import annotations

import unittest

import torch

import microcaustics as mc

try:
    import microcaustics.plotting as mcp
except ImportError:  # pragma: no cover - optional plotting extra
    mcp = None


class ResponseSource:
    geometry = mc.SourceGeometry(
        shape=(1, 1),
        pixel_scale_m=(1.0, 1.0),
        wavelengths_angstrom=(5000.0, 7000.0),
        band_names=("blue", "red"),
    )

    @staticmethod
    def metadata():
        return {"type": "test_response"}

    @staticmethod
    def transfer_function(
        edges,
        *,
        magnification=None,
        driver_amplitude=1.0,
        normalize=True,
    ):
        del edges
        scale = 1.0 if magnification is None else float(magnification.mean())
        values = torch.tensor(
            [[scale, 1.0], [3.0, scale]],
            dtype=torch.float64,
        ) * driver_amplitude
        if normalize:
            values = values / values.sum(dim=0, keepdim=True)
        return values


def simulation() -> mc.MicrolensingSimulation:
    return mc.MicrolensingSimulation.create(
        mc.MacroLens(0.0, 0.0),
        mc.PointMassField(torch.empty(0), torch.empty(0), torch.empty(0)),
        runtime=mc.RuntimeConfig(
            device="cpu",
            dtype="float64",
            backend=mc.Backend.TORCH_EAGER,
        ),
    )


class TransferFunctionTests(unittest.TestCase):
    def test_steady_product_needs_no_microlensing_simulation(self) -> None:
        product = mc.steady_transfer_function(
            ResponseSource(),
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(tuple(product.values.shape), (2, 2))
        self.assertFalse(product.metadata["microlensed"])
        torch.testing.assert_close(
            product.mean_delays_days,
            torch.tensor([1.25, 1.0], dtype=torch.float64),
        )

    @unittest.skipIf(mcp is None, "plotting extra is unavailable")
    def test_plot_density_can_close_the_physical_delay_support(self) -> None:
        product = mc.steady_transfer_function(ResponseSource(), [0.0, 1.0, 2.0])
        centers, density = mcp.transfer_response_density(
            product,
            smoothing_sigma_days=0.25,
            include_zero_boundaries=True,
        )
        self.assertEqual(tuple(centers.shape), (4,))
        self.assertEqual(tuple(density.shape), (4, 2))
        self.assertEqual(centers[0], 0.0)
        self.assertEqual(centers[-1], 2.0)
        self.assertTrue((density[[0, -1]] == 0.0).all())

    def test_one_map_product_samples_the_source_geometry(self) -> None:
        magnification_map = mc.MagnificationMap(
            torch.ones((5, 5), dtype=torch.float64),
            mc.PlaneGrid((5, 5), (2.0, 2.0)),
            method="unit_test_map",
        )
        product = mc.microlensed_transfer_function(
            ResponseSource(),
            magnification_map,
            mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
            [0.0, 1.0, 2.0],
        )
        self.assertTrue(product.metadata["microlensed"])
        self.assertEqual(product.metadata["map_method"], "unit_test_map")

    def test_dynamic_single_image_series_streams_maps(self) -> None:
        solver = simulation()
        callbacks = []
        arguments = dict(
            simulation=solver,
            lens_region=mc.PlaneRegion((2.0, 2.0)),
            source_grid=mc.PlaneGrid((5, 5), (1.0, 1.0)),
            times_days=[0.0, 1.0],
            source=ResponseSource(),
            distances=mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
            delay_edges_days=[0.0, 1.0, 2.0],
            method=mc.IPMConfig(
                rays=25,
                refinement=1,
                virtual_refinement=1,
                tiled=False,
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            ),
        )
        result = mc.streaming_microlensed_transfer_functions(
            **arguments,
            map_observer=lambda index, frame: callbacks.append((index, frame)),
        )
        through_simulation = solver.transfer_functions(
            mc.PlaneRegion((2.0, 2.0)),
            mc.PlaneGrid((5, 5), (1.0, 1.0)),
            [0.0, 1.0],
            ResponseSource(),
            mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
            [0.0, 1.0, 2.0],
            method=mc.IPMConfig(
                rays=25,
                refinement=1,
                virtual_refinement=1,
                tiled=False,
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            ),
        )
        self.assertEqual(tuple(result.values.shape), (2, 2, 2))
        self.assertEqual(len(callbacks), 2)
        self.assertEqual(tuple(result.at(0).values.shape), (2, 2))
        torch.testing.assert_close(result.values, through_simulation.values)
        torch.testing.assert_close(
            result.mean_delays_days,
            through_simulation.mean_delays_days,
        )


if __name__ == "__main__":
    unittest.main()
