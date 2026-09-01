"""High-level resolved multi-image simulation contracts."""

from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace

import torch

import microcaustics as mc
from microcaustics.multi_image import MacroImageConfig, MultiImageSimulation


def _simulation() -> mc.MicrolensingSimulation:
    return mc.MicrolensingSimulation.create(
        mc.MacroLens(convergence=0.0, shear=0.0),
        mc.PointMassField._from_einstein_radii(
            torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
        ),
        runtime=mc.RuntimeConfig(
            device="cpu",
            backend=mc.Backend.TORCH_EAGER,
            dtype="float64",
        ),
    )


def _image(name: str, delay: float) -> MacroImageConfig:
    return MacroImageConfig(
        name=name,
        simulation=_simulation(),
        lens_region=mc.PlaneRegion((2.0, 2.0)),
        source_grid=mc.PlaneGrid((5, 5), (1.0, 1.0)),
        method=mc.IPMConfig(
            rays=25,
            refinement=1,
            virtual_refinement=1,
            tiled=False,
            cell_chunk_size=16,
            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
        ),
        arrival_time_delay_days=delay,
        schedule=mc.DynamicConfig(temporal_batch_size=2),
    )


def _variable_source() -> mc.CallableSource:
    geometry = mc.SourceGeometry(
        shape=(1, 1),
        pixel_scale_m=(1.0, 1.0),
        wavelengths_angstrom=(5000.0, 7000.0),
        band_names=("blue", "red"),
    )

    def brightness(times: torch.Tensor) -> torch.Tensor:
        first = times + 10.0
        bands = torch.stack((first, 2.0 * first), dim=-1)
        return bands[:, None, None]

    return mc.CallableSource(brightness, geometry, name="clock")


class MultiImageTests(unittest.TestCase):
    def test_time_shifted_source_uses_observer_minus_arrival_delay(self) -> None:
        source = _variable_source()
        shifted = mc.TimeShiftedSource(source, 2.5)
        actual = shifted.brightness(torch.tensor([5.0], dtype=torch.float64))
        expected = source.brightness(torch.tensor([2.5], dtype=torch.float64))
        torch.testing.assert_close(actual, expected)
        self.assertEqual(shifted.geometry, source.geometry)
        self.assertEqual(
            shifted.metadata()["source_time_convention"],
            "observer_time_minus_arrival_delay",
        )

    def test_known_or_solved_delay_mapping_can_override_configurations(self) -> None:
        initial = MultiImageSimulation.create([_image("A", 0.0), _image("B", 2.0)])
        updated = initial.with_arrival_time_delays({"B": 7.5})
        self.assertEqual(initial.image("B").arrival_time_delay_days, 2.0)
        self.assertEqual(updated.image("A").arrival_time_delay_days, 0.0)
        self.assertEqual(updated.image("B").arrival_time_delay_days, 7.5)
        with self.assertRaisesRegex(ValueError, "omit"):
            initial.with_arrival_time_delays({"A": 0.0}, require_all=True)

    def test_macro_solution_converts_to_local_lens_and_delay_mapping(self) -> None:
        solution = mc.MacroImageSolution(
            name="A",
            x_arcsec=0.5,
            y_arcsec=-0.2,
            arrival_time_delay_days=3.0,
            absolute_time_delay_days=12.0,
            macro_magnification=-4.0,
            parity=-1,
            convergence=0.45,
            shear=0.52,
            shear_gamma1=0.4,
            shear_gamma2=0.33,
            shear_angle_rad=0.34,
            source_residual_arcsec=1.0e-8,
        )
        local = solution.local_macro_lens(smooth_matter_fraction=0.2)
        self.assertEqual(local.convergence, 0.45)
        self.assertEqual(local.shear, 0.52)
        self.assertEqual(mc.arrival_time_delay_mapping([solution]), {"A": 3.0})

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "requires scipy")
    def test_general_callable_macro_model_is_not_profile_specific(self) -> None:
        def raytrace(x, y):
            return x, y

        def jacobian(x, y):
            identity = torch.eye(2, dtype=x.dtype, device=x.device)
            return identity.expand(*x.shape, 2, 2)

        def time_delay(x, y):
            return 0.5 * (x.square() + y.square())

        model = mc.CallableMacroModel(raytrace, jacobian, time_delay)
        solutions = mc.solve_macroimages(
            model,
            0.0,
            0.0,
            initial_grid_size=41,
            field_of_view_arcsec=1.0,
        )
        self.assertEqual(len(solutions), 1)
        self.assertAlmostEqual(solutions[0].macro_magnification, 1.0)
        self.assertAlmostEqual(solutions[0].convergence, 0.0)
        self.assertAlmostEqual(solutions[0].shear, 0.0)

    def test_resolved_light_curves_share_source_and_apply_distinct_delays(self) -> None:
        system = MultiImageSimulation.create([_image("A", 0.0), _image("B", 2.0)])
        callbacks = {"A": [], "B": []}
        result = system.light_curves(
            [5.0, 7.0],
            _variable_source(),
            mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
            map_observers={
                name: (lambda index, frame, rows=rows: rows.append((index, frame)))
                for name, rows in callbacks.items()
            },
        )
        self.assertEqual(result.image_names, ("A", "B"))
        torch.testing.assert_close(
            result.arrival_time_delays_days,
            torch.tensor([0.0, 2.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result["A"].light_curve.unlensed_flux,
            torch.tensor([[15.0, 30.0], [17.0, 34.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result["B"].light_curve.unlensed_flux,
            torch.tensor([[13.0, 26.0], [15.0, 30.0]], dtype=torch.float64),
        )
        self.assertEqual(tuple(result.flux_tensor().shape), (2, 2, 2))
        self.assertEqual(len(callbacks["A"]), 2)
        self.assertEqual(len(callbacks["B"]), 2)
        self.assertTrue(result["B"].light_curve.metadata["multi_image_simulation"])
        self.assertEqual(
            result["B"].light_curve.metadata["arrival_time_delay_days"],
            2.0,
        )

    def test_per_image_cadences_are_supported_but_not_stackable(self) -> None:
        system = MultiImageSimulation.create([_image("A", 0.0), _image("B", 1.0)])
        result = system.light_curves(
            {"A": [1.0, 2.0], "B": [1.0, 2.0, 3.0]},
            _variable_source(),
            mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
        )
        self.assertEqual(result["A"].light_curve.times_days.numel(), 2)
        self.assertEqual(result["B"].light_curve.times_days.numel(), 3)
        with self.assertRaisesRegex(ValueError, "same time axis"):
            result.flux_tensor()

    def test_multirate_curves_keep_explicit_delays_and_generate_sparse_maps(
        self,
    ) -> None:
        system = MultiImageSimulation.create([_image("A", 0.0), _image("B", 2.0)])
        callbacks = {"A": [], "B": []}
        result = system.multirate_light_curves(
            [0.0, 2.0],
            [0.0, 1.0, 2.0],
            _variable_source(),
            mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
            map_observers={
                name: (lambda index, frame, rows=rows: rows.append((index, frame)))
                for name, rows in callbacks.items()
            },
        )
        self.assertEqual(len(callbacks["A"]), 2)
        self.assertEqual(len(callbacks["B"]), 2)
        torch.testing.assert_close(
            result["A"].light_curve.unlensed_flux,
            torch.tensor(
                [[10.0, 20.0], [11.0, 22.0], [12.0, 24.0]],
                dtype=torch.float64,
            ),
        )
        torch.testing.assert_close(
            result["B"].light_curve.unlensed_flux,
            torch.tensor(
                [[8.0, 16.0], [9.0, 18.0], [10.0, 20.0]],
                dtype=torch.float64,
            ),
        )
        self.assertFalse(
            result["A"].light_curve.metadata["interpolated_maps_materialized"]
        )
        self.assertEqual(
            result["A"].light_curve.metadata["maximum_maps_retained"],
            2,
        )

    def test_multirate_curve_requires_flux_times_inside_map_cadence(self) -> None:
        system = MultiImageSimulation.create([_image("A", 0.0)])
        with self.assertRaisesRegex(ValueError, "within"):
            system.multirate_light_curves(
                [0.0, 2.0],
                [-1.0, 1.0],
                _variable_source(),
                mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
            )

    def test_multirate_curves_include_labels_at_map_epochs(self) -> None:
        image = replace(
            _image("A", 0.0),
            lens_grid=mc.PlaneGrid((9, 9), (2.0, 2.0)),
            caustic_config=mc.CausticConfig(
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
                temporal_batch_size=1,
                jacobian_chunk_size=64,
                anchor_count=5,
                gauge_count=7,
                minimum_alignment_gauges=3,
            ),
        )
        result = MultiImageSimulation((image,)).multirate_light_curves(
            [0.0, 2.0],
            [0.0, 1.0, 2.0],
            _variable_source(),
            mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
            include_labels=True,
        )
        self.assertTrue(result.metadata["labels_included"])
        self.assertEqual(result["A"].light_curve.times_days.numel(), 3)
        self.assertEqual(result["A"].crossing_labels.shape, (2,))
        self.assertEqual(len(result["A"].caustics), 2)

    def test_dynamic_maps_are_streamed_in_image_major_order(self) -> None:
        system = MultiImageSimulation((_image("A", 0.0), _image("B", 3.0)))
        frames = list(system.dynamic_maps([0.0, 1.0]))
        self.assertEqual([frame.image_name for frame in frames], ["A", "A", "B", "B"])
        self.assertEqual([frame.time_days for frame in frames], [0.0, 1.0, 0.0, 1.0])
        self.assertTrue(
            all(
                frame.magnification_map.metadata["multi_image_simulation"]
                for frame in frames
            )
        )

    def test_configuration_rejects_ambiguous_label_requests(self) -> None:
        system = MultiImageSimulation((_image("A", 0.0),))
        with self.assertRaisesRegex(ValueError, "requires lens_grid"):
            system.light_curves(
                [0.0],
                _variable_source(),
                mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
                include_labels=True,
            )
        with self.assertRaisesRegex(ValueError, "mapping mismatch"):
            list(system.dynamic_maps({"B": [0.0]}))

    def test_stationary_multi_image_curve_can_include_center_labels(self) -> None:
        image = replace(
            _image("A", 0.0),
            lens_grid=mc.PlaneGrid((9, 9), (2.0, 2.0)),
            caustic_config=mc.CausticConfig(
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
                temporal_batch_size=1,
                jacobian_chunk_size=64,
                anchor_count=5,
                gauge_count=7,
                minimum_alignment_gauges=3,
            ),
        )
        result = MultiImageSimulation((image,)).light_curves(
            [0.0],
            _variable_source(),
            mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
            include_labels=True,
        )
        self.assertIsNotNone(result["A"].caustics)
        self.assertEqual(result["A"].crossing_labels.shape, (1,))
        self.assertEqual(result["A"].caustics[0].labels.anchor_points_uas.shape[0], 5)
        self.assertEqual(result["A"].caustics[0].labels.gauge_points_uas.shape[0], 7)

    def test_microlensing_weighted_transfer_functions_report_macro_delay(
        self,
    ) -> None:
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
            def transfer_function(edges, *, magnification, normalize=True):
                del edges, normalize
                return magnification.new_tensor([[1.0, 3.0], [3.0, 1.0]]) / 4.0

        system = MultiImageSimulation.create([_image("A", 4.0)])
        result = system.transfer_functions(
            [0.0, 2.0],
            ResponseSource(),
            mc.LensingDistances(1.0e25, 2.0e25, 1.0e25),
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(result.image_names, ("A",))
        self.assertEqual(tuple(result["A"].values.shape), (2, 2, 2))
        torch.testing.assert_close(
            result["A"].mean_delays_days[0],
            torch.tensor([1.25, 0.75], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result["A"].observer_mean_delays_days[0],
            torch.tensor([5.25, 4.75], dtype=torch.float64),
        )

    def test_macroimage_names_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            MultiImageSimulation((_image("A", 0.0), _image("A", 1.0)))


if __name__ == "__main__":
    unittest.main()
