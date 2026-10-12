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
        values = (
            torch.tensor(
                [[scale, 1.0], [3.0, scale]],
                dtype=torch.float64,
            )
            * driver_amplitude
        )
        if normalize:
            values = values / values.sum(dim=0, keepdim=True)
        return values


class BatchedResponseSource(ResponseSource):
    @staticmethod
    def batched_transfer_function(
        edges,
        *,
        magnification,
        driver_amplitude=1.0,
        normalize=True,
        spatial_chunk_size=262_144,
    ):
        del spatial_chunk_size
        return torch.stack(
            [
                ResponseSource.transfer_function(
                    edges,
                    magnification=frame,
                    driver_amplitude=driver_amplitude,
                    normalize=normalize,
                )
                for frame in magnification
            ]
        )


def simulation() -> mc.MicrolensingSimulation:
    return mc.MicrolensingSimulation.create(
        mc.MacroLens(0.0, 0.0),
        mc.PointMassField._from_einstein_radii(
            torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
        ),
        runtime=mc.RuntimeConfig(
            device="cpu",
            dtype="float64",
            backend=mc.Backend.TORCH_EAGER,
        ),
    )


class TransferFunctionTests(unittest.TestCase):
    def test_cdf_public_steady_and_microlensed_products(self):
        native = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        output = torch.linspace(0.0, 2.0, 9, dtype=torch.float64)
        original = mc.steady_transfer_function(ResponseSource(), native)
        actual = mc.steady_transfer_function(
            ResponseSource(), output, native_delay_edges_days=native
        )
        expected = mc.rebin_transfer_function(original, output)
        torch.testing.assert_close(actual.values, expected.values)
        torch.testing.assert_close(actual.mean_delays_days, original.mean_delays_days)
        self.assertEqual(actual.metadata["mean_delay_method"], "native_bin_centers")

        class DirectSource(ResponseSource):
            @staticmethod
            def mean_response_delays(**kwargs):
                return torch.tensor([0.91, 1.09], dtype=torch.float64)

        direct = mc.steady_transfer_function(
            DirectSource(), output, native_delay_edges_days=native
        )
        torch.testing.assert_close(
            direct.mean_delays_days, DirectSource.mean_response_delays()
        )
        magnification = mc.MagnificationMap(
            torch.ones(5, 5, dtype=torch.float64),
            mc.PlaneGrid((5, 5), (2.0, 2.0)),
            method="test",
        )
        lensed = mc.microlensed_transfer_function(
            DirectSource(),
            magnification,
            mc.LensingDistances(1e25, 2e25, 1e25),
            output,
            native_delay_edges_days=native,
        )
        torch.testing.assert_close(lensed.values, direct.values)
        torch.testing.assert_close(lensed.mean_delays_days, direct.mean_delays_days)

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

    def test_dynamic_series_batches_supported_response_sources(self) -> None:
        solver = simulation()
        common = dict(
            simulation=solver,
            lens_region=mc.PlaneRegion((2.0, 2.0)),
            source_grid=mc.PlaneGrid((5, 5), (1.0, 1.0)),
            times_days=[0.0, 1.0, 2.0],
            source=BatchedResponseSource(),
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
        batched = mc.streaming_microlensed_transfer_functions(
            **common,
            response_batch_size=2,
        )
        sequential = mc.streaming_microlensed_transfer_functions(
            **common,
            response_batch_size=1,
        )
        torch.testing.assert_close(batched.values, sequential.values)
        torch.testing.assert_close(
            batched.mean_delays_days,
            sequential.mean_delays_days,
        )
        self.assertEqual(batched.metadata["response_batch_size"], 2)
        output = torch.linspace(0.0, 2.0, 17, dtype=torch.float64)
        native = common["delay_edges_days"]
        cdf_args = {
            **common,
            "delay_edges_days": output,
            "native_delay_edges_days": native,
        }
        cdf_batch = mc.streaming_microlensed_transfer_functions(
            **cdf_args, response_batch_size=2
        )
        cdf_single = mc.streaming_microlensed_transfer_functions(
            **cdf_args, response_batch_size=1
        )
        expected = mc.rebin_transfer_function(batched, output)
        torch.testing.assert_close(cdf_batch.values, expected.values)
        torch.testing.assert_close(cdf_batch.values, cdf_single.values)
        torch.testing.assert_close(cdf_batch.mean_delays_days, batched.mean_delays_days)


if __name__ == "__main__":
    unittest.main()
