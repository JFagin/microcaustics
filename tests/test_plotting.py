"""Noninteractive tests for the optional public plotting interface."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import microcaustics as mc

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    import microcaustics.plotting as mcp
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None


@unittest.skipIf(matplotlib is None, "Matplotlib plotting extra is unavailable")
class PlottingTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    @staticmethod
    def map(values=None) -> mc.MagnificationMap:
        grid = mc.PlaneGrid((4, 5), (8.0, 10.0), (1.0, -2.0))
        if values is None:
            values = torch.linspace(0.5, 2.0, 20).reshape(4, 5)
        return mc.MagnificationMap(values, grid, method="test IPM")

    @staticmethod
    def caustics() -> mc.CausticField:
        segments = torch.tensor(
            [[[-1.0, -1.0], [1.0, 1.0]], [[-1.0, 1.0], [1.0, -1.0]]]
        )
        return mc.CausticField(
            critical_segments_uas=segments + 3.0,
            caustic_segments_uas=segments,
            lens_grid=mc.PlaneGrid((8, 8), (10.0, 10.0)),
        )

    def test_magnification_and_comparison_plots(self) -> None:
        candidate = self.map()
        reference = self.map(candidate.values * 1.02)
        figure, ax = mcp.plot_magnification_map(
            candidate, caustics=self.caustics(), log10=True
        )
        self.assertIs(ax.figure, figure)
        self.assertEqual(tuple(ax.images[0].get_extent()), candidate.grid.bounds_uas)
        figure, axes = mcp.plot_map_comparison(candidate, reference)
        self.assertEqual(len(axes), 3)
        self.assertEqual(len(figure.axes), 4)

    def test_physical_scale_bar(self) -> None:
        figure, ax = mcp.plot_magnification_map(self.map(), colorbar=False)
        artist = mcp.add_scale_bar(
            ax,
            2.0,
            label=r"2 $\mu$as",
            thickness_fraction=0.01,
        )
        self.assertIn(artist, ax.artists)
        with self.assertRaises(ValueError):
            mcp.add_scale_bar(ax, 0.0)

    def test_image_axes_keep_a_panel_frame_by_default(self) -> None:
        figure, ax = plt.subplots()
        mcp.hide_image_axes(ax)
        self.assertFalse(ax.get_xticks().size)
        self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
        mcp.hide_image_axes(ax, keep_frame=False)
        self.assertFalse(any(spine.get_visible() for spine in ax.spines.values()))

    def test_enclosed_flux_contour_levels(self) -> None:
        image = np.arange(1.0, 101.0).reshape(10, 10)
        levels = mcp.enclosed_flux_contour_levels(image, (0.68, 0.95, 0.99))
        self.assertEqual(levels.shape, (3,))
        self.assertTrue(np.all(np.diff(levels) > 0.0))
        descending = np.sort(image.ravel())[::-1]
        total = float(descending.sum())
        enclosed = np.asarray(
            [float(image[image >= level].sum()) / total for level in levels[::-1]]
        )
        self.assertTrue(np.all(enclosed >= np.asarray((0.68, 0.95, 0.99))))
        with self.assertRaises(ValueError):
            mcp.enclosed_flux_contour_levels(np.zeros((4, 4)))

    def test_streaming_source_standardization_and_band_plot(self) -> None:
        geometry = mc.SourceGeometry(
            (3, 4),
            (1.0, 1.0),
            (4000.0, 8000.0),
            ("blue", "red"),
        )

        class VariableSource:
            def __init__(self):
                self.geometry = geometry

            def brightness(self, times_days, *, dtype=None, device=None):
                times = torch.as_tensor(
                    times_days,
                    dtype=dtype,
                    device=device,
                ).reshape(-1, 1, 1, 1)
                spatial = torch.arange(
                    12,
                    dtype=dtype,
                    device=device,
                ).reshape(1, 3, 4, 1)
                bands = torch.tensor(
                    [1.0, 2.0],
                    dtype=dtype,
                    device=device,
                ).reshape(1, 1, 1, 2)
                return (1.0 + times) * (1.0 + spatial) * bands

        source = VariableSource()
        summary = mcp.standardize_source_over_time(
            source,
            [0.0, 1.0, 2.0],
            batch_size=2,
        )
        self.assertEqual(summary.representative_index, 2)
        self.assertEqual(summary.mean.shape, (3, 4, 2))
        self.assertTrue(np.all(np.isfinite(summary.representative_standardized)))
        figure, axes = mcp.plot_standardized_source_bands(
            summary,
            mc.PlaneGrid((3, 4), (3.0, 4.0)),
            geometry.band_names,
            scale_bar_uas=1.0,
        )
        self.assertEqual(len(axes), 2)
        self.assertEqual(len(figure.axes), 3)

    def test_method_diagrams_execute_the_real_tree_and_scout(self) -> None:
        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(0.05, 0.02),
            mc.PointMassField._from_einstein_radii(
                torch.tensor([-0.5, 0.6]),
                torch.tensor([0.4, -0.3]),
                einstein_radius_uas=torch.tensor([0.08, 0.06]),
            ),
            runtime=mc.RuntimeConfig(
                device="cpu",
                dtype="float32",
                backend=mc.Backend.TORCH_EAGER,
            ),
        )
        lens_region = mc.PlaneRegion((4.0, 4.0))
        tree_config = mc.FarFieldApproxConfig(
            cells_per_axis=4,
            nodes_per_cell_axis=4,
            exact_radius_cells=1.0,
            taylor_order=3,
        )
        figure, axes = mcp.plot_far_field_method(
            simulation,
            lens_region,
            tree_config,
        )
        self.assertEqual(len(axes), 3)
        self.assertTrue(hasattr(figure, "canvas"))
        ipm_config = mc.IPMConfig(
            rays=256,
            scout_ratio=1,
            refinement=2,
            virtual_refinement=2,
            tiled=True,
            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
        )
        figure, axes = mcp.plot_ipm_scout_method(
            simulation,
            lens_region,
            mc.PlaneGrid((16, 16), (2.0, 2.0)),
            ipm_config,
        )
        self.assertEqual(len(axes), 2)
        self.assertGreater(len(figure._microcaustics_metadata), 0)
        figure, axes = mcp.plot_lens_field_strategies(
            simulation,
            lens_region,
            mc.PlaneGrid((16, 16), (2.0, 2.0)),
            ipm_config,
            mc.PlaneRegion((2.0, 3.0)),
            rectangle_rotation_deg=31.0,
            scale_bar_uas=0.5,
        )
        self.assertEqual(len(axes), 3)
        self.assertTrue(hasattr(figure, "canvas"))

    def test_paper_ipm_renderer_reproduces_the_archived_q2237b_diagnostic(self) -> None:
        """Keep the tutorial schematic tied to the manuscript Figure 3 code."""

        package_root = Path(__file__).resolve().parents[1]
        fixture_root = package_root / "examples" / "data" / "q2237b_method_figures"
        with tempfile.TemporaryDirectory() as temporary:
            paths = mcp.render_paper_ipm_schematic(
                fixture_root / "paper_tile_upsampling_schematic_data.npz",
                fixture_root / "paper_far_field_schematic_data.npz",
                temporary,
                refinement=2,
                virtual_refinement=4,
                source_bins=1024,
                output_prefix="q2237b_paper_figure_3",
            )
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.is_file() for path in paths))
            rendered = plt.imread(paths[0])
            self.assertGreater(rendered.shape[1], rendered.shape[0])
            self.assertGreater(rendered.shape[0], 1000)
            with np.load(paths[2], allow_pickle=False) as diagnostic:
                self.assertEqual(int(diagnostic["displayed_refinement"]), 2)
                self.assertEqual(int(diagnostic["displayed_virtual_refinement"]), 4)
                self.assertEqual(int(diagnostic["source_bins"]), 1024)
                self.assertFalse(bool(diagnostic["displayed_pixel_grid_enlarged"]))

    def test_live_paper_ipm_renderer_builds_its_own_diagnostics(self) -> None:
        """The publication layout also accepts a newly evaluated simulation."""

        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(0.05, 0.02),
            mc.PointMassField._from_einstein_radii(
                torch.tensor([-0.5, 0.6]),
                torch.tensor([0.4, -0.3]),
                einstein_radius_uas=torch.tensor([0.08, 0.06]),
            ),
            runtime=mc.RuntimeConfig(
                device="cpu",
                dtype="float32",
                backend=mc.Backend.TORCH_EAGER,
            ),
        )
        config = mc.IPMConfig(
            rays=256,
            scout_ratio=1,
            refinement=2,
            virtual_refinement=4,
            tiled=True,
            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = mcp.render_live_paper_ipm_schematic(
                simulation,
                mc.PlaneRegion((4.0, 4.0)),
                mc.PlaneGrid((16, 16), (2.0, 2.0)),
                config,
                temporary,
                output_prefix="live_ipm",
            )
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue(
                (Path(temporary) / "live_ipm_live_diagnostics.npz").is_file()
            )

    def test_caustic_label_distance_and_anchor_plots(self) -> None:
        field = self.caustics()
        _, source_ax = mcp.plot_caustics(field)
        _, lens_ax = mcp.plot_caustics(field, plane="lens")
        self.assertIn("Caustics", source_ax.get_title())
        self.assertIn("Critical", lens_ax.get_title())
        label_map = mc.LabelMap(
            torch.arange(20, dtype=torch.int64).reshape(4, 5), self.map().grid
        )
        distance_map = mc.DistanceMap(torch.ones(4, 5), self.map().grid)
        label_figure, label_ax = mcp.plot_label_map(label_map, caustics=field)
        self.assertIsNotNone(label_ax)
        label_figure.canvas.draw()
        colorbar_ticks = label_figure.axes[-1].get_yticks()
        np.testing.assert_array_equal(colorbar_ticks, np.arange(20))
        self.assertIsNotNone(mcp.plot_distance_map(distance_map, caustics=field)[1])
        diagnostic_figure, diagnostic_axes = mcp.plot_caustic_diagnostics(
            self.map(),
            mc.LabelMap(torch.zeros(4, 5, dtype=torch.int64), self.map().grid),
            distance_map,
            mc.LabelMap(torch.ones(4, 5, dtype=torch.int64), self.map().grid),
            caustics=field,
            scale_bar_uas=2.0,
        )
        self.assertEqual(len(diagnostic_axes), 4)
        self.assertEqual(len(diagnostic_figure.axes), 8)
        labels = mc.AnchorGaugeLabels(
            raw_center_label=0,
            center_label=0,
            center_crossing=False,
            center_distance_uas=1.0,
            center_vote_count=2,
            center_valid_count=2,
            gauge_labels=torch.tensor([0, 1]),
            gauge_distances_uas=torch.tensor([1.0, 2.0]),
            gauge_vote_counts=torch.tensor([2, 2]),
            gauge_valid_counts=torch.tensor([2, 2]),
            anchor_offsets=torch.tensor([0, 1]),
            anchor_points_uas=torch.tensor([[-1.0, 0.0], [1.0, 0.0]]),
            gauge_points_uas=torch.tensor([[0.0, -1.0], [0.0, 1.0]]),
        )
        _, ax = mcp.plot_anchor_gauge(labels)
        self.assertGreaterEqual(len(ax.collections), 3)

    def test_light_curve_transfer_source_and_timing_plots(self) -> None:
        times = torch.tensor([0.0, 1.0, 2.0])
        flux = torch.tensor([[1.0, 2.0], [1.1, 1.9], [0.9, 2.1]])
        curve = mc.LightCurve(times, flux, ("g", "r"), unlensed_flux=flux * 0.9)
        _, ax = mcp.plot_light_curve(
            curve, magnitude=True, normalize=True, show_unlensed=True
        )
        self.assertEqual(len(ax.lines), 4)
        _, calibrated_ax = mcp.plot_light_curve(
            curve,
            magnitude=True,
            normalize=False,
            zero_point_flux={"g": 100.0, "r": 200.0},
        )
        self.assertEqual(calibrated_ax.get_ylabel(), "Brightness [mag]")
        with self.assertRaises(ValueError):
            mcp.plot_light_curve(
                curve,
                magnitude=True,
                normalize=True,
                zero_point_flux=100.0,
            )

        geometry = mc.SourceGeometry((4, 5), (2.0, 3.0), (4800.0, 6200.0), ("g", "r"))
        source = mc.StaticSource(torch.ones(4, 5, 2), geometry)
        _, ax = mcp.plot_source_brightness(source, band="r")
        self.assertEqual(tuple(ax.images[0].get_array().shape), (4, 5))

        transfer = mc.TransferFunction(
            delay_edges_days=torch.tensor([0.0, 1.0, 2.0, 3.0]),
            values=torch.tensor([[0.2, 0.1], [0.5, 0.6], [0.3, 0.3]]),
            mean_delays_days=torch.tensor([1.6, 1.7]),
            band_names=("g", "r"),
        )
        _, ax = mcp.plot_transfer_function(transfer)
        self.assertEqual(len(ax.lines), 2)
        centers, density = mcp.transfer_response_density(
            transfer,
            smoothing_sigma_days=0.4,
        )
        self.assertEqual(tuple(centers.shape), (3,))
        np.testing.assert_allclose(
            np.sum(
                density * np.diff(transfer.delay_edges_days.numpy())[:, None], axis=0
            ),
            transfer.values.numpy().sum(axis=0),
            rtol=1.0e-7,
            atol=1.0e-7,
        )
        causal = mc.TransferFunction(
            delay_edges_days=torch.arange(6.0),
            values=torch.tensor([[0.0], [0.0], [0.25], [0.75], [0.0]]),
            mean_delays_days=torch.tensor([3.25]),
            band_names=("g",),
        )
        _, causal_density = mcp.transfer_response_density(
            causal,
            smoothing_sigma_days=1.0,
        )
        self.assertEqual(float(causal_density[0, 0]), 0.0)
        self.assertEqual(float(causal_density[1, 0]), 0.0)
        self.assertEqual(float(causal_density[4, 0]), 0.0)
        self.assertAlmostEqual(float(causal_density[:, 0].sum()), 1.0, places=6)
        timing = mc.TimingBreakdown(
            steady_seconds=0.3,
            component_seconds={"trace": 0.1, "rasterize": 0.2},
            collected=True,
        )
        _, ax = mcp.plot_timing_breakdown(timing)
        self.assertEqual(len(ax.patches), 2)
        self.assertIn("PyTorch", mcp.runtime_description())

    def test_multi_image_observation_series_and_macro_image_plots(self) -> None:
        times = torch.tensor([0.0, 1.0, 2.0])
        curve = mc.LightCurve(
            times,
            torch.tensor([[1.0, 2.0], [1.1, 1.9], [0.9, 2.1]]),
            ("g", "r"),
        )
        curves = mc.MultiImageLightCurves(
            (
                mc.MacroImageLightCurve("A", 0.0, curve),
                mc.MacroImageLightCurve("B", 2.5, curve),
            )
        )
        _, axes = mcp.plot_multi_image_light_curves(curves, band="g")
        self.assertEqual(len(axes), 2)

        observations = mc.PhotometricObservations(
            time_days=times,
            band_names=("g", "r", "g"),
            image_names=("A", "B"),
            magnitude=torch.tensor([[22.0, 22.5], [21.8, 22.3], [22.1, 22.6]]),
            magnitude_error=torch.full((3, 2), 0.05),
            noiseless_magnitude=torch.tensor(
                [[22.0, 22.5], [21.8, 22.3], [22.1, 22.6]]
            ),
        )
        _, ax = mcp.plot_photometric_observations(observations, image="B")
        self.assertEqual(len(ax.collections), 2)

        series = mc.TransferFunctionSeries(
            times_days=torch.tensor([0.0, 2.0]),
            delay_edges_days=torch.tensor([0.0, 1.0, 2.0]),
            values=torch.tensor([[[0.4, 0.6], [0.6, 0.4]], [[0.5, 0.3], [0.5, 0.7]]]),
            mean_delays_days=torch.tensor([[1.1, 0.9], [1.0, 1.2]]),
            band_names=("g", "r"),
        )
        _, ax = mcp.plot_transfer_function(series, epoch=1)
        self.assertIn("2", ax.get_title())
        _, ax = mcp.plot_mean_delays(series)
        self.assertEqual(len(ax.lines), 2)

        grid = mc.ImagePlaneGrid((4, 5), (0.8, 1.0))
        image = mc.RenderedMacroImage(
            values=torch.ones(4, 5, 2),
            noiseless_values=torch.ones(4, 5, 2),
            grid=grid,
            band_names=("g", "r"),
            wavelengths_angstrom=(4_800.0, 6_200.0),
            units="counts",
        )
        _, ax = mcp.plot_rendered_macro_image(image, band="r", colorbar=False)
        self.assertEqual(tuple(ax.images[0].get_array().shape), (4, 5))

    def test_print_benchmark_separates_first_call_and_steady_state(self) -> None:
        benchmark = mc.CallableBenchmark(
            first_result=None,
            first_call_seconds=1.0,
            warmup_seconds=(0.4,),
            steady_seconds=(0.2, 0.3, 0.25),
        )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            mcp.print_benchmark(benchmark, label="map")
        output = stream.getvalue()
        self.assertIn("first call", output)
        self.assertIn("warmed steady state", output)

    def test_fixed_palette_gif_uses_one_global_palette(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is unavailable")
        x = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        yy, xx = np.meshgrid(x, x, indexing="ij")
        first = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
        second = first[::-1].copy()
        # Mimic a fixed colorbar at the right edge of an otherwise changing
        # scientific animation. A per-frame palette can make this strip
        # flicker even though its input RGB values never change.
        fixed_strip = np.stack((x, x, x), axis=-1)[:, None, :]
        first[:, -4:, :] = fixed_strip
        second[:, -4:, :] = fixed_strip
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixed.gif"
            mcp.save_fixed_palette_gif((first, second), path, fps=4)
            with Image.open(path) as animation:
                self.assertEqual(animation.n_frames, 2)
                self.assertEqual(animation.mode, "P")
                self.assertIsNotNone(animation.getpalette())
                animation.seek(0)
                decoded_first = np.asarray(animation.convert("RGB"))
                animation.seek(1)
                self.assertEqual(animation.size, (32, 32))
                decoded_second = np.asarray(animation.convert("RGB"))
                np.testing.assert_array_equal(
                    decoded_first[:, -4:, :], decoded_second[:, -4:, :]
                )


if __name__ == "__main__":
    unittest.main()
