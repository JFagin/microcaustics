"""Small end-to-end workflows using only documented public interfaces."""

from __future__ import annotations

import unittest
from dataclasses import replace

import torch

import microcaustics as mc


def _moving_simulation(*, device: str = "cpu", backend: str = "torch-eager"):
    return mc.MicrolensingSimulation.create(
        mc.MacroLens(
            0.18,
            0.09,
            shear_angle_rad=0.17,
            smooth_matter_fraction=0.2,
        ),
        mc.PointMassField._from_einstein_radii(
            torch.tensor([-0.55, 0.4, 1.1]),
            torch.tensor([0.35, -0.45, 0.65]),
            velocity_x_uas_per_day=torch.tensor([0.003, -0.002, 0.001]),
            velocity_y_uas_per_day=torch.tensor([-0.001, 0.002, -0.002]),
            einstein_radius_uas=torch.tensor([0.22, 0.17, 0.13]),
        ),
        runtime=mc.RuntimeConfig(device=device, backend=backend),
    )


def _source(total_flux=(1.0, 0.7), sigma=1.1e10):
    return mc.GaussianSource(
        mc.SourceGeometry(
            shape=(5, 7),
            pixel_scale_m=(8.0e9, 8.0e9),
            wavelengths_angstrom=(4_800.0, 7_500.0),
            band_names=("blue", "red"),
        ),
        sigma_m=sigma,
        total_flux=total_flux,
        axis_ratio=0.8,
        position_angle_rad=0.2,
    )


def _method(*, chunk=17):
    return mc.IPMConfig(
        rays=64,
        scout_ratio=2,
        refinement=2,
        virtual_refinement=4,
        tiled=False,
        cell_chunk_size=chunk,
        far_field_approx=mc.FarFieldApproxConfig(enabled=False),
    )


class PublicWorkflowTests(unittest.TestCase):
    lens_region = mc.PlaneRegion((3.0, 3.0))
    source_grid = mc.PlaneGrid((9, 11), (1.8, 2.0))
    distances = mc.LensingDistances(1.0e25, 2.0e25, 1.4e25)
    times = (0.0, 1.5, 3.0, 4.5, 6.0)

    def test_identical_population_seed_reproduces_light_curve(self) -> None:
        population_kwargs = dict(
            region=self.lens_region,
            compact_convergence_value=0.12,
            distances=self.distances,
            mass_function=mc.PowerLawMassFunction(0.2, 0.8, 2.0),
            count=4,
            velocity_dispersion_uas_per_day=0.002,
            seed=7342,
            dtype=torch.float32,
        )
        fields = tuple(
            mc.sample_uniform_point_masses(**population_kwargs) for _ in range(2)
        )
        curves = tuple(
            mc.MicrolensingSimulation.create(
                mc.MacroLens(
                    0.18,
                    0.09,
                    shear_angle_rad=0.17,
                    smooth_matter_fraction=0.2,
                ),
                field,
                runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
            ).light_curve(
                self.lens_region,
                self.source_grid,
                self.times,
                _source(),
                self.distances,
                method=_method(),
                schedule=mc.DynamicConfig(
                    temporal_batch_size=2,
                    fused_temporal_ipm=True,
                    scout_refresh_frames=1,
                ),
            )
            for field in fields
        )
        torch.testing.assert_close(
            curves[0].flux,
            curves[1].flux,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            curves[0].unlensed_flux,
            curves[1].unlensed_flux,
            rtol=0.0,
            atol=0.0,
        )

    def test_independent_static_map_batch_matches_separate_maps(self) -> None:
        macro = mc.MacroLens(
            0.18,
            0.09,
            shear_angle_rad=0.17,
            smooth_matter_fraction=0.2,
        )
        runtime = mc.RuntimeConfig(device="cpu", backend="torch-eager")
        simulations = tuple(
            mc.MicrolensingSimulation.create(
                macro,
                mc.PointMassField._from_einstein_radii(
                    torch.tensor([-0.55 + shift, 0.4, 1.1]),
                    torch.tensor([0.35, -0.45 + shift, 0.65]),
                    einstein_radius_uas=torch.tensor([0.22, 0.17, 0.13]),
                ),
                runtime=runtime,
            )
            for shift in (0.0, 0.08)
        )
        method = mc.production_ipm_config(
            rays=64,
            scout_ratio=1,
            cell_chunk_size=17,
            far_field_approx=mc.FarFieldApproxConfig(
                cells_per_axis=4,
                nodes_per_cell_axis=2,
                exact_radius_cells=1.0,
                taylor_order=2,
                center_translation_order=4,
            ),
        )
        requests = tuple(
            mc.StaticMapRequest(
                simulation,
                self.lens_region,
                self.source_grid,
                name=f"field-{index}",
            )
            for index, simulation in enumerate(simulations)
        )
        batched = mc.batched_magnification_maps(
            requests,
            method=method,
            batch_size=2,
        )
        separate = tuple(
            simulation.magnification_map(
                self.lens_region,
                self.source_grid,
                method=method,
            )
            for simulation in simulations
        )
        self.assertEqual(len(batched), len(separate))
        for index, (actual, expected) in enumerate(zip(batched, separate, strict=True)):
            torch.testing.assert_close(actual.values, expected.values)
            self.assertTrue(actual.metadata["independent_map_batch"])
            self.assertEqual(actual.metadata["independent_map_batch_index"], index)

    def test_streamed_light_curve_matches_explicit_public_maps(self) -> None:
        simulation = _moving_simulation()
        method = _method()
        schedule = mc.DynamicConfig(
            temporal_batch_size=2,
            fused_temporal_ipm=True,
            scout_refresh_frames=1,
        )
        maps = tuple(
            simulation.dynamic_maps(
                self.lens_region,
                self.source_grid,
                self.times,
                method=method,
                schedule=schedule,
            )
        )
        explicit = simulation.light_curve_from_maps(
            maps,
            _source(),
            self.times,
            self.distances,
        )
        streamed = simulation.light_curve(
            self.lens_region,
            self.source_grid,
            self.times,
            _source(),
            self.distances,
            method=method,
            schedule=schedule,
        )
        torch.testing.assert_close(streamed.flux, explicit.flux, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            streamed.unlensed_flux,
            explicit.unlensed_flux,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(streamed.band_names, ("blue", "red"))
        self.assertFalse(streamed.metadata["maps_retained"])

    def test_aligned_source_uses_exact_map_pixels(self) -> None:
        """The production geometry skips a redundant bilinear resampling."""

        simulation = _moving_simulation()
        method = _method()
        schedule = mc.DynamicConfig(
            temporal_batch_size=2,
            fused_temporal_ipm=True,
            scout_refresh_frames=1,
        )
        dy_uas, dx_uas = self.source_grid.pixel_scale_uas
        dy_m, dx_m = self.distances.uas_to_source_length(
            torch.tensor((dy_uas, dx_uas)),
            dtype=torch.float64,
        ).tolist()
        geometry = mc.SourceGeometry(
            shape=self.source_grid.shape,
            pixel_scale_m=(dy_m, dx_m),
            wavelengths_angstrom=(6_000.0,),
            band_names=("band",),
        )
        source = mc.StaticSource(
            torch.linspace(
                0.5,
                1.5,
                self.source_grid.shape[0] * self.source_grid.shape[1],
            ).reshape(*self.source_grid.shape, 1),
            geometry,
        )
        maps = tuple(
            simulation.dynamic_maps(
                self.lens_region,
                self.source_grid,
                self.times,
                method=method,
                schedule=schedule,
            )
        )
        explicit = simulation.light_curve_from_maps(
            maps,
            source,
            self.times,
            self.distances,
        )
        streamed = simulation.light_curve(
            self.lens_region,
            self.source_grid,
            self.times,
            source,
            self.distances,
            method=method,
            schedule=schedule,
        )
        torch.testing.assert_close(streamed.flux, explicit.flux)
        self.assertTrue(streamed.metadata["map_aligned_source_fast_path"])

    def test_spatial_chunk_and_temporal_batch_do_not_change_maps(self) -> None:
        simulation = _moving_simulation()
        outputs = []
        for chunk, temporal_batch in ((7, 1), (17, 2), (256, 4)):
            maps = tuple(
                simulation.dynamic_maps(
                    self.lens_region,
                    self.source_grid,
                    self.times,
                    method=_method(chunk=chunk),
                    schedule=mc.DynamicConfig(
                        temporal_batch_size=temporal_batch,
                        fused_temporal_ipm=True,
                        scout_refresh_frames=1,
                    ),
                )
            )
            outputs.append(maps)
        for candidate in outputs[1:]:
            for actual, reference in zip(candidate, outputs[0], strict=True):
                torch.testing.assert_close(
                    actual.values,
                    reference.values,
                    rtol=2.0e-7,
                    atol=5.0e-7,
                )

    def test_independent_source_batching_preserves_each_light_curve(self) -> None:
        simulation = _moving_simulation()
        requests = (
            mc.LightCurveRequest(_source(), self.distances, name="ordinary"),
            mc.LightCurveRequest(
                _source(total_flux=(0.8, 1.2), sigma=1.6e10),
                self.distances,
                trajectory=mc.LinearTrajectory(
                    initial_position_uas=(0.05, -0.03),
                    velocity_uas_per_day=(0.004, -0.002),
                ),
                name="offset",
            ),
            mc.LightCurveRequest(
                _source(total_flux=(1.4, 0.6), sigma=0.9e10),
                self.distances,
                name="compact",
            ),
        )
        common = dict(
            lens_region=self.lens_region,
            source_grid=self.source_grid,
            times_days=self.times,
            requests=requests,
            method=_method(),
        )
        independent = mc.streaming_light_curves(
            simulation,
            **common,
            schedule=mc.DynamicConfig(
                temporal_batch_size=1,
                light_curve_batch_size=1,
                scout_refresh_frames=1,
            ),
        )
        batched = simulation.light_curves(
            **common,
            schedule=mc.DynamicConfig(
                temporal_batch_size=4,
                light_curve_batch_size=3,
                scout_refresh_frames=1,
            ),
        )
        self.assertEqual(len(independent), len(requests))
        for actual, reference, request in zip(
            batched, independent, requests, strict=True
        ):
            torch.testing.assert_close(actual.flux, reference.flux, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.unlensed_flux,
                reference.unlensed_flux,
                rtol=0,
                atol=0,
            )
            self.assertEqual(actual.metadata["request_name"], request.name)
            self.assertEqual(actual.metadata["light_curve_batch_size"], 3)

    def test_public_multirate_entry_point_matches_static_map_cadence(self) -> None:
        """A one-map cadence must reproduce ordinary static map photometry."""

        simulation = _moving_simulation()
        source = _source()
        map_time = (0.0,)
        flux_times = (0.0, 1.0, 2.0)
        method = _method()
        multirate = mc.multirate_streaming_light_curve(
            simulation,
            self.lens_region,
            self.source_grid,
            map_time,
            flux_times,
            source,
            self.distances,
            method=method,
        )
        magnification_map = simulation.magnification_map(
            self.lens_region,
            self.source_grid,
            method=method,
            time_days=0.0,
        )
        explicit = simulation.light_curve_from_maps(
            (magnification_map,) * len(flux_times),
            source,
            flux_times,
            self.distances,
        )
        torch.testing.assert_close(multirate.flux, explicit.flux)
        torch.testing.assert_close(multirate.unlensed_flux, explicit.unlensed_flux)
        self.assertEqual(multirate.metadata["map_epochs"], 1)
        self.assertFalse(multirate.metadata["interpolated_maps_materialized"])

    def test_multirate_factorizes_static_coherent_modulation_exactly(self) -> None:
        """The optimized coherent driver must preserve brute-force photometry."""

        simulation = _moving_simulation()
        amplitudes = torch.tensor(
            [
                [0.80, 1.05],
                [0.95, 0.90],
                [1.10, 1.15],
                [1.25, 1.00],
                [0.90, 0.85],
            ]
        )
        source = mc.ModulatedSource(
            _source(),
            mc.TabulatedDrivingSignal(torch.as_tensor(self.times), amplitudes),
        )
        method = _method()
        optimized = mc.multirate_streaming_light_curve(
            simulation,
            self.lens_region,
            self.source_grid,
            (0.0,),
            self.times,
            source,
            self.distances,
            method=method,
        )
        magnification_map = simulation.magnification_map(
            self.lens_region,
            self.source_grid,
            method=method,
            time_days=0.0,
        )
        brute_force = simulation.light_curve_from_maps(
            (magnification_map,) * len(self.times),
            source,
            self.times,
            self.distances,
        )
        torch.testing.assert_close(optimized.flux, brute_force.flux)
        torch.testing.assert_close(
            optimized.unlensed_flux,
            brute_force.unlensed_flux,
        )
        self.assertTrue(optimized.metadata["coherent_source_factorized"])

    def test_multirate_accepts_an_arbitrary_time_evolving_source(self) -> None:
        """Fine source cadence is generic, not specific to quasar drivers."""

        geometry = _source().geometry

        def evolving_brightness(times: torch.Tensor) -> torch.Tensor:
            y = torch.linspace(-1.0, 1.0, geometry.shape[0], dtype=times.dtype)
            x = torch.linspace(-1.0, 1.0, geometry.shape[1], dtype=times.dtype)
            radius2 = y[:, None].square() + x[None, :].square()
            spatial = torch.exp(-2.0 * radius2)
            amplitude = 1.0 + 0.15 * torch.sin(times / 1.7)
            bands = torch.stack((spatial, 0.7 * spatial), dim=-1)
            return amplitude[:, None, None, None] * bands[None]

        source = mc.CallableSource(
            evolving_brightness,
            geometry,
            name="generic evolving source",
        )
        simulation = _moving_simulation()
        method = _method()
        map_times = (0.0, 6.0)
        flux_times = self.times
        actual = simulation.multirate_light_curve(
            self.lens_region,
            self.source_grid,
            map_times,
            flux_times,
            source,
            self.distances,
            method=method,
        )
        endpoints = tuple(
            simulation.magnification_map(
                self.lens_region,
                self.source_grid,
                method=method,
                time_days=time,
            )
            for time in map_times
        )
        explicit_maps = tuple(
            mc.MagnificationMap(
                endpoints[0].values
                + (float(time) / 6.0) * (endpoints[1].values - endpoints[0].values),
                endpoints[0].grid,
                time_days=float(time),
                method="explicit test interpolation",
            )
            for time in flux_times
        )
        expected = simulation.light_curve_from_maps(
            explicit_maps,
            source,
            flux_times,
            self.distances,
        )
        torch.testing.assert_close(actual.flux, expected.flux)
        torch.testing.assert_close(actual.unlensed_flux, expected.unlensed_flux)
        self.assertEqual(actual.metadata["map_epochs"], 2)
        self.assertFalse(actual.metadata["interpolated_maps_materialized"])

    def test_irs_and_ipm_static_maps_flow_through_same_photometry_api(self) -> None:
        empty = torch.empty(0)
        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(0.0, 0.0),
            mc.PointMassField._from_einstein_radii(
                empty, empty, einstein_radius_uas=empty
            ),
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        methods = (
            mc.IRSConfig(
                rays=256,
                ray_chunk_size=31,
                star_chunk_size=7,
                far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            ),
            replace(_method(), rays=64),
        )
        identity_region = mc.PlaneRegion((2.0, 2.0))
        identity_grid = mc.PlaneGrid((16, 16), (2.0, 2.0))
        for method in methods:
            with self.subTest(method=type(method).__name__):
                curve = simulation.light_curve(
                    identity_region,
                    identity_grid,
                    self.times[:2],
                    _source(),
                    self.distances,
                    method=method,
                    schedule=mc.DynamicConfig(
                        temporal_batch_size=2,
                    ),
                )
                torch.testing.assert_close(curve.flux, curve.unlensed_flux)
                self.assertEqual(tuple(curve.flux.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
