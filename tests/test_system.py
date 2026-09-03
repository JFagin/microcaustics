"""High-level physical-system construction and low-level parity tests."""

from __future__ import annotations

import math
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

import microcaustics as mc


class MicrolensingSystemTests(unittest.TestCase):
    def test_direct_physical_masses_resolve_from_system_distances(self) -> None:
        stars = mc.PointMassField(
            x_uas=[-0.5, 0.75],
            y_uas=[0.2, -0.4],
            mass_solar=[0.3, 1.2],
        )
        with self.assertRaisesRegex(RuntimeError, "have not been resolved"):
            _ = stars.einstein_radius_uas
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            stars=stars,
            source_grid=self.source_grid,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        resolved = system.realized_stars
        torch.testing.assert_close(
            resolved.einstein_radius_uas,
            self.distances.einstein_radius_uas(
                torch.tensor([0.3, 1.2]), dtype=torch.float32
            ),
        )
        torch.testing.assert_close(resolved.mass_solar, torch.tensor([0.3, 1.2]))

    def test_einstein_coordinate_helpers_round_trip(self) -> None:
        values = torch.tensor([-2.0, 0.0, 3.5])
        angular = mc.einstein_units_to_uas(
            values,
            mean_mass_solar=0.3,
            distances=self.distances,
        )
        recovered = mc.uas_to_einstein_units(
            angular,
            mean_mass_solar=0.3,
            distances=self.distances,
        )
        torch.testing.assert_close(recovered, values)
        self.assertGreater(
            float(mc.einstein_radius_uas(0.3, distances=self.distances)), 0.0
        )

    def test_lens_plane_size_accepts_auto_square_and_rectangle(self) -> None:
        stars = mc.PointMassField([0.0], [0.0], [0.3])
        common = dict(
            macro=self.macro,
            distances=self.distances,
            stars=stars,
            source_grid=self.source_grid,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        automatic = mc.MicrolensingSystem(**common).resolved_lens_region
        square = mc.MicrolensingSystem(
            **common, lens_plane_uas=7.0
        ).resolved_lens_region
        rectangle = mc.MicrolensingSystem(
            **common, lens_plane_uas=(6.0, 8.0)
        ).resolved_lens_region
        self.assertGreater(automatic.field_of_view_uas[0], 0.0)
        self.assertEqual(square.field_of_view_uas, (7.0, 7.0))
        self.assertEqual(rectangle.field_of_view_uas, (6.0, 8.0))
        with self.assertRaisesRegex(ValueError, "not both"):
            mc.MicrolensingSystem(
                **common,
                lens_plane_uas=7.0,
                lens_region=self.lens_region,
            )

    def test_source_independent_square_map_needs_no_plane_grid(self) -> None:
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            stars=self._stars(),
            lens_region=self.lens_region,
            integration_domain="full",
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        method = mc.IPMConfig(
            rays=64,
            refinement=1,
            virtual_refinement=1,
            tiled=False,
            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
        )

        generated = system.magnification_map(
            map_width_uas=2.0,
            map_pixels=8,
            method=method,
        )
        explicit = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=mc.PlaneGrid((8, 8), (2.0, 2.0)),
            stars=self._stars(),
            lens_region=self.lens_region,
            integration_domain="full",
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        ).magnification_map(method=method)

        self.assertEqual(generated.grid.shape, (8, 8))
        self.assertEqual(generated.grid.field_of_view_uas, (2.0, 2.0))
        torch.testing.assert_close(generated.values, explicit.values)
        self.assertIs(
            system._with_square_map_grid(map_width_uas=2.0, map_pixels=8),
            system._with_square_map_grid(map_width_uas=2.0, map_pixels=8),
        )
        dynamic = tuple(
            system.dynamic_maps(
                (0.0, 1.0),
                map_width_uas=2.0,
                map_pixels=8,
                method=method,
                schedule=mc.DynamicConfig(temporal_batch_size=2),
            )
        )
        self.assertEqual(len(dynamic), 2)
        self.assertEqual(dynamic[0].grid, generated.grid)

        summary = system.summary(display=False)
        metadata = system.metadata()
        self.assertEqual(summary["source_shape"], (8, 8))
        self.assertEqual(metadata["source_grid"]["shape"], [8, 8])

    def test_source_independent_inspection_selects_map_geometry(self) -> None:
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        with self.assertRaisesRegex(ValueError, "map_width_uas and map_pixels"):
            system.summary(display=False)
        selected = system.summary(
            map_width_uas=3.0,
            map_pixels=12,
            display=False,
        )
        self.assertEqual(selected["source_shape"], (12, 12))
        system._with_square_map_grid(map_width_uas=4.0, map_pixels=16)
        with self.assertRaisesRegex(ValueError, "more than one"):
            system.metadata()
        metadata = system.metadata(map_width_uas=4.0, map_pixels=16)
        self.assertEqual(metadata["source_grid"]["shape"], [16, 16])

    def test_unseeded_numerical_variants_retain_one_stellar_realization(self) -> None:
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            stellar_population=mc.StellarPopulation.salpeter(count=8),
            source_grid=mc.PlaneGrid((8, 8), (2.0, 2.0)),
            integration_domain="scout",
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        scout_stars = system.realized_stars
        full_stars = system.with_integration_domain("full").realized_stars
        torch.testing.assert_close(scout_stars.x_uas, full_stars.x_uas, rtol=0, atol=0)
        torch.testing.assert_close(scout_stars.y_uas, full_stars.y_uas, rtol=0, atol=0)
        torch.testing.assert_close(
            scout_stars.mass_solar,
            full_stars.mass_solar,
            rtol=0,
            atol=0,
        )

        map_system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            stellar_population=mc.StellarPopulation.salpeter(count=8),
            integration_domain="full",
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        coarse = map_system._with_square_map_grid(
            map_width_uas=2.0,
            map_pixels=8,
        ).realized_stars
        fine = map_system._with_square_map_grid(
            map_width_uas=2.0,
            map_pixels=16,
        ).realized_stars
        torch.testing.assert_close(coarse.x_uas, fine.x_uas, rtol=0, atol=0)
        torch.testing.assert_close(coarse.y_uas, fine.y_uas, rtol=0, atol=0)
        torch.testing.assert_close(coarse.mass_solar, fine.mass_solar, rtol=0, atol=0)

    def test_source_independent_map_geometry_validation(self) -> None:
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        with self.assertRaisesRegex(ValueError, "map_width_uas and map_pixels"):
            system.magnification_map(map_width_uas=2.0)
        with self.assertRaisesRegex(ValueError, "map_width_uas and map_pixels"):
            system.magnification_map()

    def test_plain_band_mapping_and_sampling_keywords(self) -> None:
        model = mc.ThinDiskModel(
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            bands_angstrom={"blue": 4_800.0, "red": 9_700.0},
            source_grid_shape=24,
            enclosed_flux_fraction=0.995,
            source_margin=1.05,
        )
        self.assertEqual(model.band_names, ("blue", "red"))
        self.assertEqual(model.wavelengths_angstrom, (4_800.0, 9_700.0))
        self.assertEqual(model.grid.shape, (24, 24))
        self.assertAlmostEqual(model.grid.enclosed_flux_fraction, 0.995)

    def test_component_seeds_are_stable_and_independent(self) -> None:
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stars=self._stars(),
            lens_region=self.lens_region,
            seed=123,
        )
        self.assertEqual(system.seed_for("stars"), 123)
        self.assertEqual(system.seed_for("variability"), system.seed_for("variability"))
        self.assertNotEqual(
            system.seed_for("variability"), system.seed_for("observations")
        )
        overridden = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stars=self._stars(),
            lens_region=self.lens_region,
            seed={"base": 123, "observations": 77},
        )
        self.assertEqual(overridden.seed_for("observations"), 77)

    def test_summary_does_not_realize_or_compile(self) -> None:
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        summary = system.summary(duration_days=10.0, display=False)
        self.assertFalse(summary["compilation_performed"])
        self.assertEqual(summary["estimated_star_count"], len(self._stars()))
        self.assertNotIn("_resolved", system.__dict__)

    def test_time_axis_infers_realization_duration(self) -> None:
        source = mc.GaussianSource(
            mc.SourceGeometry(
                shape=self.source_grid.shape,
                pixel_scale_m=(1.0e10, 1.0e10),
                wavelengths_angstrom=(5_000.0,),
                band_names=("optical",),
            ),
            sigma_m=1.0e10,
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=source,
            stars=self._stars(),
            lens_region=self.lens_region,
            duration_days=0.0,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        realization = system._realize_for_times((0.0, 25.0, 50.0))
        self.assertEqual(realization.system.duration_days, 50.0)
        self.assertIs(realization, system._realize_for_times((0.0, 25.0, 50.0)))

    def test_plain_solver_keywords_and_cadence_execute(self) -> None:
        source = mc.GaussianSource(
            mc.SourceGeometry(
                shape=self.source_grid.shape,
                pixel_scale_m=(1.0e10, 1.0e10),
                wavelengths_angstrom=(5_000.0,),
                band_names=("optical",),
            ),
            sigma_m=1.0e10,
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=source,
            source_grid=self.source_grid,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        magnification = system.magnification_map(
            rays=64,
            refinement=2,
            virtual_refinement=4,
            far_field=False,
        )
        self.assertEqual(magnification.grid.shape, self.source_grid.shape)
        curve = system.light_curve(
            duration_days=2.0,
            map_cadence_days=1.0,
            rays=64,
            refinement=1,
            virtual_refinement=1,
            far_field=False,
        )
        torch.testing.assert_close(
            curve.times_days,
            torch.tensor((0.0, 1.0, 2.0), dtype=curve.times_days.dtype),
        )

    distances = mc.LensingDistances(
        1.0e25,
        2.0e25,
        1.2e25,
        lens_redshift=0.25,
        source_redshift=1.2,
    )
    macro = mc.MacroLens(0.2, 0.08, smooth_matter_fraction=0.25)
    source_grid = mc.PlaneGrid((9, 11), (1.8, 2.0))
    lens_region = mc.PlaneRegion((4.0, 5.0))
    method = mc.IPMConfig(
        rays=64,
        scout_ratio=2,
        refinement=2,
        virtual_refinement=4,
        tiled=False,
        cell_chunk_size=17,
        far_field_approx=mc.FarFieldApproxConfig(enabled=False),
    )

    @staticmethod
    def _stars() -> mc.PointMassField:
        return mc.PointMassField._from_einstein_radii(
            torch.tensor([-0.4, 0.5, 1.0]),
            torch.tensor([0.3, -0.5, 0.6]),
            einstein_radius_uas=torch.tensor([0.2, 0.16, 0.12]),
        )

    def test_salpeter_constructor_matches_requested_mean_and_ratio(self) -> None:
        population = mc.StellarPopulation.salpeter(
            mean_mass_solar=0.3,
            mass_ratio=100.0,
        )
        mass_function = population.mass_function
        self.assertAlmostEqual(mass_function.mean_mass(), 0.3, places=12)
        self.assertAlmostEqual(
            mass_function.maximum_mass / mass_function.minimum_mass,
            100.0,
            places=12,
        )

    def test_degree_first_macro_lens_matches_radian_compatibility(self) -> None:
        degrees = mc.MacroLens(0.3, 0.2, shear_angle_deg=37.5)
        radians = mc.MacroLens(0.3, 0.2, shear_angle_rad=math.radians(37.5))
        self.assertAlmostEqual(degrees.shear_angle_rad, radians.shear_angle_rad)
        self.assertAlmostEqual(degrees.shear_angle_deg, 37.5)

    def test_rectangle_uses_internal_shear_aligned_coordinates(self) -> None:
        angle_deg = 37.5
        source = mc.GaussianModel(
            sigma_uas=0.12,
            bands_angstrom={"optical": 6_000.0},
            axis_ratio=0.6,
            position_angle_deg=23.0,
            source_grid_shape=24,
        )
        population = mc.StellarPopulation.salpeter(
            count=12,
            kinematics=mc.SkyProjectedKinematics(ra_deg=340.126125, dec_deg=3.358611),
        )
        common = dict(
            macro=mc.MacroLens(0.3, 0.2, shear_angle_deg=angle_deg),
            distances=self.distances,
            source=source,
            stellar_population=population,
            duration_days=100.0,
            seed=1001,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        full = mc.MicrolensingSystem(
            **common,
            integration_domain="full",
        ).realize()
        rectangle = mc.MicrolensingSystem(
            **common,
            integration_domain="rectangle",
        ).realize()

        self.assertAlmostEqual(rectangle.simulation.macro_lens.shear_angle_deg, 0.0)
        self.assertAlmostEqual(rectangle.sky_to_local_rotation_deg, angle_deg)
        self.assertEqual(rectangle.metadata()["coordinate_frame"], "shear_aligned")
        self.assertAlmostEqual(
            rectangle.source.metadata()["position_angle_rad"],
            math.radians(23.0 - angle_deg),
        )
        angle = math.radians(angle_deg)
        expected_x = (
            math.cos(angle) * full.stars.x_uas + math.sin(angle) * full.stars.y_uas
        )
        expected_y = (
            -math.sin(angle) * full.stars.x_uas + math.cos(angle) * full.stars.y_uas
        )
        torch.testing.assert_close(rectangle.stars.x_uas, expected_x)
        torch.testing.assert_close(rectangle.stars.y_uas, expected_y)
        for day in (0.0, 100.0):
            sky = full.stars.at_time(day)
            local = rectangle.stars.at_time(day)
            torch.testing.assert_close(
                local.x_uas,
                math.cos(angle) * sky.x_uas + math.sin(angle) * sky.y_uas,
            )
            torch.testing.assert_close(
                local.y_uas,
                -math.sin(angle) * sky.x_uas + math.cos(angle) * sky.y_uas,
            )

        expected_region = mc.rectangular_lens_region(
            mc.MacroLens(0.3, 0.2, shear_angle_deg=0.0),
            rectangle.source_grid.region,
            self.distances,
            population.mass_function,
        )
        self.assertEqual(
            rectangle.lens_region.field_of_view_uas,
            expected_region.field_of_view_uas,
        )

    def test_rectangle_rotates_trajectory_without_rotating_source_raster(self) -> None:
        angle_deg = 30.0
        trajectory = mc.LinearTrajectory(
            initial_position_uas=(0.3, -0.2),
            velocity_uas_per_day=(0.02, 0.01),
        )
        source = mc.GaussianModel(
            sigma_uas=0.1,
            bands_angstrom={"optical": 6_000.0},
            source_grid_shape=16,
        )
        realization = mc.MicrolensingSystem(
            macro=mc.MacroLens(0.3, 0.2, shear_angle_deg=angle_deg),
            distances=self.distances,
            source=source,
            stellar_population=mc.StellarPopulation.salpeter(count=8),
            integration_domain="rectangle",
            trajectory=trajectory,
            duration_days=10.0,
            seed=1001,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        ).realize()
        original = trajectory.position_uas((0.0, 10.0), dtype=torch.float64)
        rotated = realization.trajectory.position_uas((0.0, 10.0), dtype=torch.float64)
        angle = math.radians(angle_deg)
        expected = torch.stack(
            (
                math.cos(angle) * original[:, 0] + math.sin(angle) * original[:, 1],
                -math.sin(angle) * original[:, 0] + math.cos(angle) * original[:, 1],
            ),
            dim=-1,
        )
        torch.testing.assert_close(rotated, expected)
        self.assertEqual(realization.source.geometry.shape, (16, 16))

    def test_pixelated_rectangle_remains_in_input_frame(self) -> None:
        source = mc.GaussianSource(
            mc.SourceGeometry(
                shape=(12, 12),
                pixel_scale_m=(1.0e10, 1.0e10),
                wavelengths_angstrom=(6_000.0,),
                band_names=("optical",),
            ),
            sigma_m=2.0e10,
        )
        realization = mc.MicrolensingSystem(
            macro=mc.MacroLens(0.3, 0.2, shear_angle_deg=37.5),
            distances=self.distances,
            source=source,
            stellar_population=mc.StellarPopulation.salpeter(count=8),
            integration_domain="rectangle",
            seed=1001,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        ).realize()
        self.assertAlmostEqual(
            realization.simulation.macro_lens.shear_angle_deg,
            37.5,
        )
        self.assertEqual(realization.sky_to_local_rotation_deg, 0.0)

    def test_system_constructor_hides_distance_construction(self) -> None:
        system = mc.MicrolensingSystem(
            lens_redshift=0.25,
            source_redshift=1.2,
            H0=70.0,
            Om0=0.3,
            macro=self.macro,
            source_grid=self.source_grid,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        self.assertEqual(system.distances.lens_redshift, 0.25)
        self.assertEqual(system.distances.source_redshift, 1.2)
        self.assertGreater(system.distances.lens_to_source_m, 0.0)

    def test_redshift_system_can_replace_source_without_rebuilding_geometry(
        self,
    ) -> None:
        original = mc.GaussianSource(
            mc.SourceGeometry(
                shape=self.source_grid.shape,
                pixel_scale_m=(1.0e10, 1.0e10),
                wavelengths_angstrom=(5_000.0,),
                band_names=("optical",),
            ),
            sigma_m=1.0e10,
        )
        replacement = mc.GaussianSource(
            original.geometry,
            sigma_m=1.5e10,
        )
        system = mc.MicrolensingSystem(
            lens_redshift=0.25,
            source_redshift=1.2,
            macro=self.macro,
            source=original,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        changed = system.with_source(replacement)
        self.assertIs(changed.source, replacement)
        self.assertIs(changed.distances, system.distances)

    def test_light_curve_warmup_selects_schedule_batch_automatically(self) -> None:
        source = mc.GaussianSource(
            mc.SourceGeometry(
                shape=self.source_grid.shape,
                pixel_scale_m=(1.0e10, 1.0e10),
                wavelengths_angstrom=(5_000.0,),
                band_names=("optical",),
            ),
            sigma_m=1.0e10,
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=source,
            source_grid=self.source_grid,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        result = system.warmup_light_curve(
            method=self.method,
            schedule=mc.DynamicConfig(
                temporal_batch_size=2,
                fused_temporal_ipm=False,
                scout_refresh_frames=1,
            ),
        )
        self.assertEqual(result.times_days.numel(), 2)

    def test_common_realized_properties_are_available_on_system(self) -> None:
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        self.assertIs(system.realization, system.realize())
        self.assertIs(system.realized_stars, system.realization.stars)
        self.assertEqual(system.resolved_source_grid, self.source_grid)
        self.assertEqual(system.resolved_lens_region, self.lens_region)
        self.assertEqual(system.metadata()["source_grid"]["shape"], [9, 11])

    def test_production_dynamic_defaults_are_coherent(self) -> None:
        from microcaustics.system import _production_dynamic_settings

        method = mc.production_ipm_config()
        self.assertEqual(mc.production_dynamic_config().temporal_batch_size, 49)
        schedule, caustics = _production_dynamic_settings(method, None)
        self.assertEqual(schedule.temporal_batch_size, 30)
        self.assertIsNotNone(caustics)
        assert caustics is not None
        self.assertEqual(caustics.temporal_batch_size, 30)
        self.assertEqual(caustics.far_field_approx, method.far_field_approx)

    def test_caustic_batch_inherits_schedule_and_can_be_overridden(self) -> None:
        from microcaustics.system import _production_dynamic_settings

        method = mc.production_ipm_config()
        schedule = mc.production_dynamic_config(temporal_batch_size=7)
        _, inherited = _production_dynamic_settings(
            method, schedule, mc.CausticConfig()
        )
        _, overridden = _production_dynamic_settings(
            method,
            schedule,
            mc.CausticConfig(temporal_batch_size=3),
        )
        assert inherited is not None and overridden is not None
        self.assertEqual(inherited.temporal_batch_size, 7)
        self.assertEqual(overridden.temporal_batch_size, 3)

    def test_multi_image_system_constructor_shares_redshift_geometry(self) -> None:
        system = mc.MultiImageSystem(
            lens_redshift=0.25,
            source_redshift=1.2,
            H0=70.0,
            Om0=0.3,
            images={"A": mc.MacroLens(0.2, 0.08), "B": mc.MacroLens(0.3, 0.12)},
            source_grid=self.source_grid,
            stars={"A": self._stars(), "B": self._stars()},
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        self.assertIs(system.image("A").distances, system.image("B").distances)
        self.assertEqual(system.image("A").distances.lens_redshift, 0.25)

    def test_sky_kinematics_can_defer_redshifts(self) -> None:
        kinematics = mc.SkyProjectedKinematics(
            ra_deg=10.0,
            dec_deg=-5.0,
            seed=91,
        )
        self.assertIsInstance(kinematics, mc.SkyProjectedKinematics)
        first = kinematics.mean_velocity_uas_per_day(self.distances)
        second = kinematics.mean_velocity_uas_per_day(self.distances)
        self.assertEqual(first, second)

    def test_high_level_light_curve_retains_requested_maps(self) -> None:
        pixel_scale = self.distances.uas_to_source_length(
            self.source_grid.pixel_scale_uas, dtype=torch.float64
        )
        source = mc.GaussianSource(
            mc.SourceGeometry(
                self.source_grid.shape,
                (float(pixel_scale[0]), float(pixel_scale[1])),
                (6000.0,),
                ("optical",),
            ),
            sigma_m=float(pixel_scale.mean()),
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=source,
            source_grid=self.source_grid,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        curve = system.light_curve(
            (0.0, 1.0),
            method=self.method,
            schedule=mc.DynamicConfig(
                temporal_batch_size=1,
                fused_temporal_ipm=False,
                scout_refresh_frames=1,
            ),
            keep_maps_at_days=(0.0, 1.0),
        )
        self.assertEqual(curve.map_times_days.tolist(), [0.0, 1.0])
        self.assertEqual(curve.maps[0].time_days, 0.0)

    def test_profiling_is_opt_in(self) -> None:
        common = dict(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stars=self._stars(),
            lens_region=self.lens_region,
        )
        ordinary = mc.MicrolensingSystem(
            **common,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        ).magnification_map(method=self.method)
        profiled = mc.MicrolensingSystem(
            **common,
            runtime=mc.RuntimeConfig(
                device="cpu", backend="torch-eager", profiling="detailed"
            ),
        ).magnification_map(method=self.method)
        self.assertFalse(ordinary.timing.collected)
        self.assertTrue(profiled.timing.collected)
        torch.testing.assert_close(ordinary.values, profiled.values)

    def test_angular_gaussian_and_trajectory_grid_hide_unit_plumbing(self) -> None:
        model = mc.GaussianModel(
            sigma_uas=(0.1, 0.2),
            wavelengths_angstrom=(5000.0, 7000.0),
            band_names=("g", "i"),
            grid=mc.SourceGridConfig(
                shape=(16, 20),
                enclosed_flux_fraction=0.99,
                margin=1.0,
            ),
        )
        expected = self.distances.uas_to_source_length((0.1, 0.2), dtype=torch.float64)
        torch.testing.assert_close(
            torch.tensor(model.pixelate(self.distances).sigma_m, dtype=torch.float64),
            expected,
        )
        grid = model.recommended_grid(self.distances)
        trajectory = mc.LinearTrajectory(
            initial_position_uas=(0.2, -0.1),
            velocity_uas_per_day=(0.03, -0.02),
        )
        covered = grid.covering_trajectory(trajectory, (0.0, 10.0), margin=1.1)
        positions = trajectory.position_uas(
            torch.tensor((0.0, 10.0), dtype=torch.float64),
            dtype=torch.float64,
        )
        xmin, xmax, ymin, ymax = covered.bounds_uas
        self.assertLessEqual(
            xmin,
            float(positions[:, 0].min()) - 0.5 * grid.field_of_view_uas[1],
        )
        self.assertGreaterEqual(
            xmax,
            float(positions[:, 0].max()) + 0.5 * grid.field_of_view_uas[1],
        )
        self.assertLessEqual(
            ymin,
            float(positions[:, 1].min()) - 0.5 * grid.field_of_view_uas[0],
        )
        self.assertGreaterEqual(
            ymax,
            float(positions[:, 1].max()) + 0.5 * grid.field_of_view_uas[0],
        )

    def test_kerr_disk_model_builds_static_and_reprocessing_sources(self) -> None:
        policy = mc.SourceGridConfig(
            shape=8,
            enclosed_flux_fraction=0.9,
            margin=1.0,
            radial_samples=128,
        )
        model = mc.KerrDiskModel(
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.2,
            wavelengths_angstrom=(5000.0,),
            band_names=("g",),
            spin=0.3,
            inclination_deg=20.0,
            source_redshift=1.2,
            lamp_fraction=0.1,
            grid=policy,
            compile_solver=False,
            lamppost_nalpha=16,
            lamppost_radial_bins=16,
        )
        runtime = mc.RuntimeConfig(
            device="cpu",
            backend="torch-eager",
            dtype=torch.float64,
        )
        static = model.pixelate(self.distances, runtime=runtime)
        self.assertIsInstance(static, mc.TransferredThinDiskSource)
        self.assertEqual(static.geometry.shape, (8, 8))
        signal = mc.broken_power_law_driving_signal(
            torch.arange(-100.0, 101.0, dtype=torch.float64),
            break_timescale_days=20.0,
            seed=4,
            extrapolation="hold",
        )
        variable = model.with_driving_signal(signal).pixelate(
            self.distances,
            runtime=runtime,
        )
        self.assertIsInstance(variable, mc.ThermalReprocessingSource)
        self.assertEqual(variable.geometry, static.geometry)
        self.assertTrue(bool(torch.all(torch.isfinite(variable.brightness(0.0)))))

    def test_system_materializes_a_physical_kerr_source(self) -> None:
        model = mc.KerrDiskModel(
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.2,
            wavelengths_angstrom=(5000.0,),
            band_names=("g",),
            spin=0.3,
            inclination_deg=20.0,
            source_redshift=1.2,
            grid=mc.SourceGridConfig(
                shape=8,
                enclosed_flux_fraction=0.9,
                margin=1.0,
                radial_samples=128,
            ),
            compile_solver=False,
            lamppost_nalpha=16,
            lamppost_radial_bins=16,
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=model,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(
                device="cpu",
                backend="torch-eager",
                dtype=torch.float64,
            ),
        )
        realization = system.realize()
        self.assertIsInstance(realization.source, mc.TransferredThinDiskSource)
        self.assertEqual(realization.source_grid.shape, (8, 8))
        self.assertEqual(
            realization.source.geometry.shape,
            realization.source_grid.shape,
        )

    def test_photometric_helper_uses_ab_flux_density(self) -> None:
        flux = torch.tensor((3631.0, 7262.0), dtype=torch.float64)
        magnitude = mc.flux_to_magnitude(flux)
        self.assertAlmostEqual(float(magnitude[0]), 0.0, places=12)
        self.assertAlmostEqual(
            float(magnitude[1]),
            -2.5 * math.log10(2.0),
            places=12,
        )

    def test_opsim_discovery_searches_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "baseline_v4.3.5_10yrs.db"
            database.touch()
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            self.assertEqual(
                mc.find_rubin_opsim_database((nested,)),
                database.resolve(),
            )

    def test_isotropic_kinematics_uses_observer_time_dilation(self) -> None:
        kinematics = mc.IsotropicKinematics(
            dispersion_km_s=180.0,
            bulk_velocity_km_s=(120.0, -40.0),
        )
        actual = kinematics.component_dispersion_uas_per_day(self.distances)
        expected = (
            180_000.0
            / self.distances.lens_m
            * (180.0 / math.pi * 3600.0 * 1.0e6)
            * 86_400.0
            / 1.25
        )
        self.assertAlmostEqual(actual, expected, places=18)
        conversion = expected / 180.0
        actual_mean = kinematics.mean_velocity_uas_per_day(self.distances)
        self.assertAlmostEqual(actual_mean[0], 120.0 * conversion, places=18)
        self.assertAlmostEqual(actual_mean[1], -40.0 * conversion, places=18)

    def test_sky_projected_kinematics_combines_lens_and_source_motion(self) -> None:
        kinematics = mc.SkyProjectedKinematics(
            ra_deg=340.126125,
            dec_deg=3.358611,
            stellar_dispersion_km_s=170.0,
            lens_peculiar_velocity_km_s=(210.0, -80.0),
            source_peculiar_velocity_km_s=(-120.0, 60.0),
            include_cmb_dipole=False,
        )
        actual = kinematics.mean_velocity_uas_per_day(self.distances)
        conversion = (180.0 / math.pi * 3600.0 * 1.0e6) * 86_400.0
        expected_x = (
            210_000.0 / self.distances.lens_m / 1.25
            - (-120_000.0) / self.distances.source_m / 2.2
        ) * conversion
        expected_y = (
            -80_000.0 / self.distances.lens_m / 1.25
            - 60_000.0 / self.distances.source_m / 2.2
        ) * conversion
        self.assertAlmostEqual(actual[0], expected_x, places=18)
        self.assertAlmostEqual(actual[1], expected_y, places=18)

    def test_sky_kinematics_sampling_is_reproducible(self) -> None:
        kwargs = {
            "ra_deg": 340.126125,
            "dec_deg": 3.358611,
            "lens_redshift": 0.25,
            "source_redshift": 1.2,
            "seed": 91,
        }
        first = mc.SkyProjectedKinematics(**kwargs)
        second = mc.SkyProjectedKinematics(**kwargs)
        self.assertEqual(
            first.mean_velocity_uas_per_day(self.distances),
            second.mean_velocity_uas_per_day(self.distances),
        )
        self.assertNotEqual(first.mean_velocity_uas_per_day(self.distances), (0.0, 0.0))

        population = mc.StellarPopulation.salpeter(kinematics=first)
        metadata = population.metadata()
        self.assertEqual(metadata["kinematics"]["type"], "SkyProjectedKinematics")
        self.assertEqual(
            metadata["kinematics"]["ra_deg"],
            340.126125,
        )
        resolved = population.metadata(self.distances)["kinematics"]["resolved"]
        self.assertEqual(len(resolved["lens_peculiar_velocity_km_s"]), 2)
        self.assertEqual(len(resolved["source_peculiar_velocity_km_s"]), 2)
        self.assertEqual(len(resolved["cmb_transverse_velocity_km_s"]), 2)
        self.assertEqual(len(resolved["bulk_velocity_uas_per_day"]), 2)
        self.assertGreater(
            resolved["stellar_component_dispersion_uas_per_day"],
            0.0,
        )
        self.assertEqual(resolved["coordinate_basis"], "ICRS east/north")

    def test_system_seed_is_inherited_by_sampled_sky_kinematics(self) -> None:
        population = mc.StellarPopulation.salpeter(
            count=32,
            kinematics=mc.SkyProjectedKinematics(
                ra_deg=340.126125,
                dec_deg=3.358611,
            ),
        )

        def realize(seed):
            return mc.MicrolensingSystem(
                macro=self.macro,
                distances=self.distances,
                source_grid=self.source_grid,
                stellar_population=population,
                duration_days=10.0,
                runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
                seed=seed,
            ).realize()

        first = realize(123)
        second = realize(123)
        different = realize(124)
        torch.testing.assert_close(
            first.stars.velocity_x_uas_per_day,
            second.stars.velocity_x_uas_per_day,
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(
            torch.equal(
                first.stars.velocity_x_uas_per_day,
                different.stars.velocity_x_uas_per_day,
            )
        )
        self.assertIsNone(population.kinematics.seed)
        self.assertEqual(
            first.stellar_population.kinematics.seed,
            mc.derive_seed(123, "kinematics"),
        )
        resolved = first.metadata()["stellar_population"]["kinematics"]["resolved"]
        self.assertEqual(resolved["coordinate_basis"], "ICRS east/north")

    def test_component_seed_can_override_sampled_sky_kinematics(self) -> None:
        population = mc.StellarPopulation.salpeter(
            count=8,
            kinematics=mc.SkyProjectedKinematics(ra_deg=1.0, dec_deg=2.0),
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stellar_population=population,
            duration_days=10.0,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
            seed={"base": 123, "kinematics": 77},
        )
        self.assertEqual(system.realize().stellar_population.kinematics.seed, 77)

    def test_unseeded_sky_kinematics_draws_once_without_reseeding(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(811)
            expected_first = torch.randn(4, dtype=torch.float64)
            expected_second = torch.randn(4, dtype=torch.float64)
            expected_next = torch.rand(3)
            torch.manual_seed(811)
            first = mc.SkyProjectedKinematics(ra_deg=1.0, dec_deg=2.0)
            second = mc.SkyProjectedKinematics(ra_deg=1.0, dec_deg=2.0)
            first_velocity = first.mean_velocity_uas_per_day(self.distances)
            self.assertEqual(
                first.mean_velocity_uas_per_day(self.distances), first_velocity
            )
            second_velocity = second.mean_velocity_uas_per_day(self.distances)
            self.assertNotEqual(first_velocity, second_velocity)
            self.assertIsNone(first.seed)
            self.assertIsNone(second.seed)
            self.assertEqual(first._peculiar_standard_draws, tuple(expected_first.tolist()))
            self.assertEqual(second._peculiar_standard_draws, tuple(expected_second.tolist()))
            torch.testing.assert_close(torch.rand(3), expected_next, rtol=0, atol=0)

    def test_seeded_sky_kinematics_does_not_advance_global_stream(self) -> None:
        with torch.random.fork_rng(devices=[]):
            before = torch.get_rng_state().clone()
            kinematics = mc.SkyProjectedKinematics(ra_deg=1.0, dec_deg=2.0, seed=0)
            kinematics.mean_velocity_uas_per_day(self.distances)
            torch.testing.assert_close(torch.get_rng_state(), before)

    def test_summary_and_realization_use_the_same_kinematics(self) -> None:
        for seed in (None, 0):
            with self.subTest(seed=seed):
                system = mc.MicrolensingSystem(
                    macro=self.macro,
                    distances=self.distances,
                    source_grid=self.source_grid,
                    stellar_population=mc.StellarPopulation.salpeter(
                        count=16,
                        kinematics=mc.SkyProjectedKinematics(ra_deg=1.0, dec_deg=2.0),
                    ),
                    seed=seed,
                    runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
                )
                summary = system.summary(duration_days=100.0, display=False)
                realized = system._realize_for_times([0.0, 100.0])
                self.assertEqual(
                    summary["stellar_aperture_radius_uas"],
                    realized.stellar_aperture.radius_uas,
                )

    def test_explicit_sky_seed_overrides_system_and_component_seeds(self) -> None:
        kinematics = mc.SkyProjectedKinematics(ra_deg=1.0, dec_deg=2.0, seed=77)
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stellar_population=mc.StellarPopulation.salpeter(
                count=8, kinematics=kinematics
            ),
            seed={"base": 0, "kinematics": 9},
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        self.assertIs(system.realize().stellar_population.kinematics, kinematics)

    def test_sky_kinematics_projects_the_cmb_dipole(self) -> None:
        shared = {
            "ra_deg": 340.126125,
            "dec_deg": 3.358611,
            "stellar_dispersion_km_s": 170.0,
            "lens_peculiar_velocity_km_s": (0.0, 0.0),
            "source_peculiar_velocity_km_s": (0.0, 0.0),
        }
        with_cmb = mc.SkyProjectedKinematics(
            **shared,
            include_cmb_dipole=True,
        ).mean_velocity_uas_per_day(self.distances)
        without_cmb = mc.SkyProjectedKinematics(
            **shared,
            include_cmb_dipole=False,
        ).mean_velocity_uas_per_day(self.distances)
        self.assertEqual(without_cmb, (0.0, 0.0))
        self.assertNotEqual(with_cmb, without_cmb)

    def test_dynamic_population_warns_when_motion_components_are_missing(self) -> None:
        def system(kinematics):
            return mc.MicrolensingSystem(
                macro=self.macro,
                distances=self.distances,
                source_grid=self.source_grid,
                stellar_population=mc.StellarPopulation.salpeter(
                    count=8,
                    kinematics=kinematics,
                ),
                duration_days=10.0,
                runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
                seed=0,
            )

        with self.assertWarnsRegex(mc.IncompleteKinematicsWarning, "CMB motion"):
            system(mc.IsotropicKinematics(dispersion_km_s=170.0)).realize()
        with self.assertWarnsRegex(
            mc.IncompleteKinematicsWarning, "stellar dispersion"
        ):
            system(mc.StaticKinematics()).realize()
        with self.assertWarnsRegex(
            mc.IncompleteKinematicsWarning, "lens peculiar motion"
        ):
            system(
                mc.SkyProjectedKinematics(
                    ra_deg=1.0,
                    dec_deg=2.0,
                    stellar_dispersion_km_s=170.0,
                    peculiar_velocity_dispersion_km_s=0.0,
                )
            ).realize()

    def test_dynamic_explicit_stationary_field_warns(self) -> None:
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stars=mc.PointMassField([0.0], [0.0], [0.3]),
            duration_days=10.0,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        with self.assertWarnsRegex(
            mc.IncompleteKinematicsWarning, "contains no velocities"
        ):
            system.realize()

    def test_empty_dynamic_catalog_has_no_motion_warning_or_invalid_metadata(self) -> None:
        system = mc.MicrolensingSystem(
            macro=mc.MacroLens(0.0, 0.0),
            distances=self.distances,
            source_grid=self.source_grid,
            stars=mc.PointMassField([], [], [],
                velocity_x_uas_per_day=[], velocity_y_uas_per_day=[]),
            duration_days=10.0,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", mc.IncompleteKinematicsWarning)
            self.assertFalse(system.realize().metadata()["stellar_motion"]["has_motion"])

    def test_dynamic_explicit_motion_warns_for_missing_components(self) -> None:
        def system(velocity_x, velocity_y):
            return mc.MicrolensingSystem(
                macro=self.macro,
                distances=self.distances,
                source_grid=self.source_grid,
                stars=mc.PointMassField(
                    [0.0, 0.1],
                    [0.0, -0.1],
                    [0.3, 0.2],
                    velocity_x_uas_per_day=velocity_x,
                    velocity_y_uas_per_day=velocity_y,
                ),
                duration_days=10.0,
                runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
            )

        with self.assertWarnsRegex(
            mc.IncompleteKinematicsWarning, "stellar velocity dispersion"
        ):
            system([0.1, 0.1], [-0.2, -0.2]).realize()
        with self.assertWarnsRegex(mc.IncompleteKinematicsWarning, "bulk motion"):
            system([0.1, -0.1], [-0.2, 0.2]).realize()
        with warnings.catch_warnings():
            warnings.simplefilter("error", mc.IncompleteKinematicsWarning)
            system([0.11, 0.09], [-0.18, -0.22]).realize()

    def test_complete_sky_kinematics_emits_no_dynamic_warning(self) -> None:
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stellar_population=mc.StellarPopulation.salpeter(
                count=8,
                kinematics=mc.SkyProjectedKinematics(
                    ra_deg=340.126125,
                    dec_deg=3.358611,
                    stellar_dispersion_km_s=170.0,
                    peculiar_velocity_dispersion_km_s=235.0,
                    include_cmb_dipole=True,
                    seed=0,
                ),
            ),
            duration_days=10.0,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
            seed=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            system.realize()

    def test_circular_population_adds_bulk_motion_to_random_velocities(self) -> None:
        population = mc.StellarPopulation.salpeter(
            mean_mass_solar=0.3,
            mass_ratio=20.0,
            count=2048,
            kinematics=mc.IsotropicKinematics(
                dispersion_km_s=180.0,
                bulk_velocity_km_s=(90.0, -45.0),
            ),
        )
        field = population.realize(
            mc.StellarAperture(7.0),
            self.macro,
            self.distances,
            seed=51,
            dtype=torch.float64,
        )
        expected = population.kinematics.mean_velocity_uas_per_day(self.distances)
        self.assertAlmostEqual(
            float(field.velocity_x_uas_per_day.mean()),
            expected[0],
            delta=0.08
            * population.kinematics.component_dispersion_uas_per_day(self.distances),
        )
        self.assertAlmostEqual(
            float(field.velocity_y_uas_per_day.mean()),
            expected[1],
            delta=0.08
            * population.kinematics.component_dispersion_uas_per_day(self.distances),
        )

    def test_circular_population_is_reproducible_and_inside_aperture(self) -> None:
        population = mc.StellarPopulation.salpeter(
            mean_mass_solar=0.3,
            mass_ratio=20.0,
            count=101,
        )
        aperture = mc.StellarAperture(7.0, center_uas=(1.0, -2.0))
        first = population.realize(
            aperture,
            self.macro,
            self.distances,
            seed=81,
            dtype=torch.float64,
        )
        second = population.realize(
            aperture,
            self.macro,
            self.distances,
            seed=81,
            dtype=torch.float64,
        )
        radius = torch.sqrt((first.x_uas + 2.0).square() + (first.y_uas - 1.0).square())
        self.assertTrue(bool(torch.all(radius <= 7.0)))
        torch.testing.assert_close(first.x_uas, second.x_uas, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            first.mass_solar, second.mass_solar, rtol=0.0, atol=0.0
        )

    def test_rotated_shear_aperture_encloses_transformed_source_corners(self) -> None:
        macro = mc.MacroLens(0.31, 0.22, shear_angle_rad=0.37)
        source_region = mc.PlaneRegion((1.4, 2.2), center_uas=(0.13, -0.27))
        population = mc.StellarPopulation.salpeter(
            mean_mass_solar=0.3,
            mass_ratio=20.0,
        )
        aperture = mc.circular_stellar_aperture(
            macro,
            source_region,
            self.distances,
            population,
            light_loss=0.02,
            safety_scale=1.0,
            duration_days=0.0,
        )
        angle = 2.0 * macro.shear_angle_rad
        gamma_1 = macro.shear * math.cos(angle)
        gamma_2 = macro.shear * math.sin(angle)
        inverse = torch.linalg.inv(
            torch.tensor(
                [
                    [1.0 - macro.convergence - gamma_1, -gamma_2],
                    [-gamma_2, 1.0 - macro.convergence + gamma_1],
                ],
                dtype=torch.float64,
            )
        )
        mean_mass = population.mass_function.mean_mass()
        margin = float(
            self.distances.einstein_radius_uas(mean_mass, dtype=torch.float64)
        )
        margin *= math.sqrt(
            macro.compact_convergence
            * population.mass_function.second_moment()
            / mean_mass**2
            / 0.02
        )
        half_y = 0.5 * source_region.field_of_view_uas[0] + margin
        half_x = 0.5 * source_region.field_of_view_uas[1] + margin
        deviations = (
            torch.tensor(
                [(x, y) for x in (-half_x, half_x) for y in (-half_y, half_y)],
                dtype=torch.float64,
            )
            @ inverse.T
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(deviations, dim=1).max()),
            aperture.radius_uas + 1.0e-12,
        )

    def test_direct_raytrace_is_covariant_in_the_shear_eigenframe(self) -> None:
        angle_deg = 37.5
        angle = math.radians(angle_deg)
        macro = mc.MacroLens(0.31, 0.22, shear_angle_deg=angle_deg)
        stars = self._stars().to(dtype=torch.float64)
        local_stars = mc.PointMassField._from_einstein_radii(
            math.cos(angle) * stars.x_uas + math.sin(angle) * stars.y_uas,
            -math.sin(angle) * stars.x_uas + math.cos(angle) * stars.y_uas,
            einstein_radius_uas=stars.einstein_radius_uas,
        )
        runtime = mc.RuntimeConfig(
            device="cpu", dtype=torch.float64, backend="torch-eager"
        )
        sky = mc.MicrolensingSimulation.create(macro, stars, runtime=runtime)
        local = mc.MicrolensingSimulation.create(
            mc.MacroLens(0.31, 0.22, shear_angle_deg=0.0),
            local_stars,
            runtime=runtime,
        )
        x = torch.tensor((-0.8, -0.1, 0.6), dtype=torch.float64)
        y = torch.tensor((0.4, -0.7, 0.2), dtype=torch.float64)
        local_x = math.cos(angle) * x + math.sin(angle) * y
        local_y = -math.sin(angle) * x + math.cos(angle) * y
        source_x, source_y, _ = sky.raytrace_direct(x, y)
        actual_x, actual_y, _ = local.raytrace_direct(local_x, local_y)
        expected_x = math.cos(angle) * source_x + math.sin(angle) * source_y
        expected_y = -math.sin(angle) * source_x + math.cos(angle) * source_y
        torch.testing.assert_close(actual_x, expected_x, rtol=1.0e-13, atol=1.0e-13)
        torch.testing.assert_close(actual_y, expected_y, rtol=1.0e-13, atol=1.0e-13)

    def test_rectangular_ipm_map_and_finite_source_flux_are_frame_covariant(
        self,
    ) -> None:
        """Exercise the complete rectangle, IPM, and source contraction path."""

        angle_deg = 90.0
        macro_sky = mc.MacroLens(0.31, 0.22, shear_angle_deg=angle_deg)
        macro_local = mc.MacroLens(0.31, 0.22, shear_angle_deg=0.0)
        stars_sky = self._stars().to(dtype=torch.float64)
        stars_local = mc.PointMassField._from_einstein_radii(
            stars_sky.y_uas,
            -stars_sky.x_uas,
            einstein_radius_uas=stars_sky.einstein_radius_uas,
        )
        population = mc.StellarPopulation.salpeter(
            mean_mass_solar=0.3,
            mass_ratio=20.0,
        )
        source_grid = mc.PlaneGrid((32, 32), (1.6, 1.6))
        sky_region = mc.rectangular_lens_region(
            macro_sky,
            source_grid.region,
            self.distances,
            population.mass_function,
            light_loss=0.02,
        )
        local_region = mc.rectangular_lens_region(
            macro_local,
            source_grid.region,
            self.distances,
            population.mass_function,
            light_loss=0.02,
        )
        self.assertAlmostEqual(
            sky_region.field_of_view_uas[0],
            local_region.field_of_view_uas[1],
        )
        self.assertAlmostEqual(
            sky_region.field_of_view_uas[1],
            local_region.field_of_view_uas[0],
        )
        runtime = mc.RuntimeConfig(
            device="cpu", dtype=torch.float64, backend="torch-eager"
        )
        sky_simulation = mc.MicrolensingSimulation.create(
            macro_sky, stars_sky, runtime=runtime
        )
        local_simulation = mc.MicrolensingSimulation.create(
            macro_local, stars_local, runtime=runtime
        )
        method = mc.IPMConfig(
            # This budget gives exactly transposed 138 x 71 and 71 x 138 cell
            # lattices for the reciprocal rectangle aspect ratios.
            rays=10_038,
            scout_ratio=1,
            refinement=2,
            virtual_refinement=4,
            tiled=False,
            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
        )
        sky_map = sky_simulation.magnification_map(
            sky_region, source_grid, method=method
        )
        local_map = local_simulation.magnification_map(
            local_region, source_grid, method=method
        )
        expected_local = torch.flip(sky_map.values, dims=(1,)).T
        map_fractional_nrmse = torch.sqrt(
            torch.mean((local_map.values - expected_local).square())
            / torch.mean(expected_local.square())
        )
        # A finite IPM lattice is not exactly rotation invariant because each
        # cell uses a fixed diagonal and polynomial sampling pattern. The
        # discrepancy converges with cell density; this modest test budget is
        # sufficient to verify the correct rotated solution rather than an
        # unrelated orientation.
        self.assertLess(float(map_fractional_nrmse), 0.1)

        geometry = mc.SourceGeometry(
            shape=source_grid.shape,
            field_of_view_uas=source_grid.field_of_view_uas,
            bands_angstrom={"optical": 6_000.0},
        ).resolve(self.distances)
        sigma_m = float(self.distances.uas_to_source_length(0.25, dtype=torch.float64))
        sky_source = mc.GaussianSource(
            geometry,
            sigma_m=sigma_m,
            axis_ratio=0.6,
            position_angle_rad=math.radians(27.0),
        )
        local_source = mc.GaussianSource(
            geometry,
            sigma_m=sigma_m,
            axis_ratio=0.6,
            position_angle_rad=math.radians(27.0 - angle_deg),
        )
        sky_brightness = sky_source.brightness(0.0, dtype=torch.float64)[0, ..., 0]
        local_brightness = local_source.brightness(0.0, dtype=torch.float64)[0, ..., 0]
        torch.testing.assert_close(
            local_brightness,
            torch.flip(sky_brightness, dims=(1,)).T,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        sky_flux = torch.sum(sky_map.values * sky_brightness)
        local_flux = torch.sum(local_map.values * local_brightness)
        torch.testing.assert_close(local_flux, sky_flux, rtol=5.0e-3, atol=0.0)

    def test_physical_support_avoids_treating_zero_map_corners_as_emission(
        self,
    ) -> None:
        macro = mc.MacroLens(0.31, 0.22, shear_angle_rad=0.37)
        population = mc.StellarPopulation.salpeter(
            mean_mass_solar=0.3,
            mass_ratio=20.0,
            count=16,
        )
        source_region = mc.PlaneRegion((2.0, 2.0))
        circular = mc.circular_stellar_aperture(
            macro,
            source_region,
            self.distances,
            population,
            light_loss=0.02,
            safety_scale=1.0,
            source_support_radius_uas=1.0,
        )
        square = mc.circular_stellar_aperture(
            macro,
            source_region,
            self.distances,
            population,
            light_loss=0.02,
            safety_scale=1.0,
        )
        self.assertLess(circular.radius_uas, square.radius_uas)

    def test_direct_star_system_matches_low_level_map(self) -> None:
        stars = self._stars()
        runtime = mc.RuntimeConfig(device="cpu", backend="torch-eager")
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stars=stars,
            integration_domain="full",
            lens_region=self.lens_region,
            runtime=runtime,
        )
        actual = system.magnification_map(method=self.method)
        expected = mc.MicrolensingSimulation.create(
            self.macro,
            stars,
            runtime=runtime,
        ).magnification_map(
            self.lens_region,
            self.source_grid,
            method=self.method,
        )
        torch.testing.assert_close(actual.values, expected.values, rtol=0.0, atol=0.0)
        self.assertIs(system.realize(), system.realize())
        metadata = system.realize().metadata()
        self.assertEqual(metadata["star_count"], len(stars))
        self.assertEqual(metadata["integration_domain"], "full")
        self.assertEqual(metadata["source"], None)
        self.assertFalse(metadata["stellar_motion"]["has_motion"])
        self.assertEqual(
            metadata["stellar_motion"]["coordinate_basis"],
            "realization x/y",
        )

    def test_integration_domain_changes_region_not_stellar_population(self) -> None:
        population = mc.StellarPopulation.salpeter(
            mean_mass_solar=0.3,
            mass_ratio=20.0,
            count=31,
        )
        common = dict(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stellar_population=population,
            duration_days=100.0,
            light_loss=0.01,
            safety_scale=1.5,
            seed=910,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        full = mc.MicrolensingSystem(**common, integration_domain="full").realize()
        rectangle = mc.MicrolensingSystem(
            **common,
            integration_domain="rectangle",
        ).realize()
        torch.testing.assert_close(full.stars.x_uas, rectangle.stars.x_uas)
        torch.testing.assert_close(full.stars.y_uas, rectangle.stars.y_uas)
        torch.testing.assert_close(
            full.stars.einstein_radius_uas,
            rectangle.stars.einstein_radius_uas,
        )
        self.assertEqual(full.stellar_aperture, rectangle.stellar_aperture)
        self.assertNotEqual(full.lens_region, rectangle.lens_region)

    def test_with_integration_domain_preserves_the_physical_system(self) -> None:
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stellar_population=mc.StellarPopulation.salpeter(
                mean_mass_solar=0.3,
                mass_ratio=20.0,
                count=31,
            ),
            seed=910,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        rectangle = system.with_integration_domain("rectangle")
        self.assertEqual(system.integration_domain.value, "scout")
        self.assertEqual(rectangle.integration_domain.value, "rectangle")
        self.assertEqual(rectangle.macro, system.macro)
        self.assertEqual(rectangle.distances, system.distances)
        self.assertIs(rectangle.stellar_population, system.stellar_population)
        self.assertEqual(rectangle.seed, system.seed)

    def test_with_seed_preserves_configuration_and_changes_realization(self) -> None:
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source_grid=self.source_grid,
            stellar_population=mc.StellarPopulation.salpeter(
                mean_mass_solar=0.3,
                mass_ratio=20.0,
                count=31,
            ),
            seed=910,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )

        reseeded = system.with_seed(911)

        self.assertEqual(reseeded.seed, 911)
        self.assertEqual(reseeded.macro, system.macro)
        self.assertEqual(reseeded.source_grid, system.source_grid)
        self.assertFalse(
            torch.equal(
                reseeded.realized_stars.x_uas,
                system.realized_stars.x_uas,
            )
        )

    def test_independent_system_maps_batch_without_sharing_stars(self) -> None:
        runtime = mc.RuntimeConfig(device="cpu", backend="torch-eager")
        systems = tuple(
            mc.MicrolensingSystem(
                macro=self.macro,
                distances=self.distances,
                stars=mc.PointMassField._from_einstein_radii(
                    self._stars().x_uas + shift,
                    self._stars().y_uas,
                    einstein_radius_uas=self._stars().einstein_radius_uas,
                ),
                integration_domain="full",
                lens_region=self.lens_region,
                runtime=runtime,
            )
            for shift in (0.0, 0.07)
        )
        method = mc.IPMConfig(
            rays=64,
            scout_ratio=1,
            refinement=2,
            virtual_refinement=4,
            tiled=False,
            cell_chunk_size=17,
            far_field_approx=mc.FarFieldApproxConfig(
                cells_per_axis=4,
                nodes_per_cell_axis=2,
                exact_radius_cells=1.0,
                taylor_order=2,
                center_translation_order=4,
            ),
        )
        batched = mc.batched_system_maps(
            systems,
            map_width_uas=self.source_grid.field_of_view_uas[0],
            map_pixels=self.source_grid.shape[0],
            method=method,
            batch_size=2,
        )
        separate = tuple(
            system.magnification_map(
                map_width_uas=self.source_grid.field_of_view_uas[0],
                map_pixels=self.source_grid.shape[0],
                method=method,
            )
            for system in systems
        )
        self.assertEqual(len(batched), 2)
        for actual, expected in zip(batched, separate, strict=True):
            torch.testing.assert_close(actual.values, expected.values)

    def test_source_geometry_automatically_defines_angular_grid(self) -> None:
        source = mc.GaussianSource(
            mc.SourceGeometry(
                shape=(5, 7),
                pixel_scale_m=(8.0e9, 6.0e9),
                wavelengths_angstrom=(5_000.0,),
                band_names=("optical",),
            ),
            sigma_m=1.0e10,
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=source,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        grid = system.realize().source_grid
        self.assertEqual(grid.shape, source.geometry.shape)
        expected = self.distances.source_length_to_uas(
            (5 * 8.0e9, 7 * 6.0e9),
            dtype=torch.float64,
        )
        self.assertAlmostEqual(grid.field_of_view_uas[0], float(expected[0]))
        self.assertAlmostEqual(grid.field_of_view_uas[1], float(expected[1]))

    def test_source_grid_automatically_covers_the_system_trajectory(self) -> None:
        source = mc.GaussianSource(
            mc.SourceGeometry(
                shape=(17, 19),
                pixel_scale_m=(8.0e9, 6.0e9),
                wavelengths_angstrom=(5_000.0,),
                band_names=("optical",),
            ),
            sigma_m=1.0e10,
        )
        trajectory = mc.LinearTrajectory(
            initial_position_uas=(0.2, -0.1),
            velocity_uas_per_day=(0.03, -0.02),
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=source,
            stars=self._stars(),
            lens_region=self.lens_region,
            duration_days=10.0,
            trajectory=trajectory,
            trajectory_grid_margin=1.1,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        native = mc.PlaneGrid(
            source.geometry.shape,
            tuple(
                float(value)
                for value in self.distances.source_length_to_uas(
                    (
                        source.geometry.shape[0] * source.geometry.pixel_scale_m[0],
                        source.geometry.shape[1] * source.geometry.pixel_scale_m[1],
                    ),
                    dtype=torch.float64,
                )
            ),
        )
        expected = native.covering_trajectory(
            trajectory,
            (0.0, 10.0),
            margin=1.1,
        )
        self.assertEqual(system.resolved_source_grid, expected)
        self.assertEqual(system.realization.source.geometry, source.geometry)

    def test_map_grid_must_enclose_pixelated_source_but_may_be_larger(self) -> None:
        source = mc.GaussianSource(
            mc.SourceGeometry(
                shape=(5, 7),
                pixel_scale_m=(8.0e9, 6.0e9),
                wavelengths_angstrom=(5_000.0,),
                band_names=("optical",),
            ),
            sigma_m=1.0e10,
        )
        with self.assertRaisesRegex(ValueError, "must enclose"):
            mc.MicrolensingSystem(
                macro=self.macro,
                distances=self.distances,
                source=source,
                source_grid=mc.PlaneGrid((5, 7), (1.0e-6, 1.0e-6)),
                stars=self._stars(),
                lens_region=self.lens_region,
            )
        enlarged = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=source,
            source_grid=mc.PlaneGrid((17, 19), (99.0, 99.0)),
            stars=self._stars(),
            lens_region=self.lens_region,
        )
        self.assertEqual(enlarged.realize().source_grid.shape, (17, 19))

    def test_multi_image_system_preserves_independent_fields_and_delays(self) -> None:
        pixel_scale_m = self.distances.uas_to_source_length(
            self.source_grid.pixel_scale_uas,
            dtype=torch.float64,
        )
        source = mc.GaussianSource(
            mc.SourceGeometry(
                shape=self.source_grid.shape,
                pixel_scale_m=(float(pixel_scale_m[0]), float(pixel_scale_m[1])),
                wavelengths_angstrom=(6_000.0,),
                band_names=("optical",),
            ),
            sigma_m=2.0 * float(pixel_scale_m.mean()),
        )
        systems = {
            name: mc.MicrolensingSystem(
                macro=self.macro,
                distances=self.distances,
                source=source,
                source_grid=self.source_grid,
                stars=mc.PointMassField._from_einstein_radii(
                    self._stars().x_uas + shift,
                    self._stars().y_uas,
                    einstein_radius_uas=self._stars().einstein_radius_uas,
                ),
                integration_domain="full",
                lens_region=self.lens_region,
                runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
            )
            for name, shift in (("A", 0.0), ("B", 0.08))
        }
        multi = mc.MultiImageSystem(
            systems,
            arrival_time_delays_days={"A": 0.0, "B": 3.5},
            methods=self.method,
            schedules=mc.DynamicConfig(
                temporal_batch_size=2,
                fused_temporal_ipm=False,
                scout_refresh_frames=1,
            ),
        )
        result = multi.light_curves((0.0, 1.0))
        self.assertEqual(result.image_names, ("A", "B"))
        self.assertEqual(result["B"].arrival_time_delay_days, 3.5)
        self.assertFalse(
            torch.equal(
                systems["A"].realize().stars.x_uas,
                systems["B"].realize().stars.x_uas,
            )
        )

    def test_multi_image_system_builds_images_from_shared_physics(self) -> None:
        pixel_scale_m = self.distances.uas_to_source_length(
            self.source_grid.pixel_scale_uas,
            dtype=torch.float64,
        )
        source = mc.GaussianSource(
            mc.SourceGeometry(
                shape=self.source_grid.shape,
                pixel_scale_m=(float(pixel_scale_m[0]), float(pixel_scale_m[1])),
                wavelengths_angstrom=(6_000.0,),
                band_names=("optical",),
            ),
            sigma_m=2.0 * float(pixel_scale_m.mean()),
        )
        multi = mc.MultiImageSystem(
            images={
                "A": mc.MacroLens(0.2, 0.08),
                "B": mc.MacroLens(0.3, 0.12),
            },
            distances=self.distances,
            source=source,
            stars={"A": self._stars(), "B": self._stars()},
            integration_domain={"A": "full", "B": "rectangle"},
            lens_region=self.lens_region,
            arrival_time_delays_days={"B": 3.5},
            seed=1001,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
            caustic_grid_shape=32,
        )
        self.assertEqual(multi.image_names, ("A", "B"))
        self.assertIsInstance(multi.image("A"), mc.MicrolensingSystem)
        self.assertEqual(multi.image("A").seed, mc.derive_seed(1001, "image:A"))
        self.assertEqual(multi.image("B").seed, mc.derive_seed(1001, "image:B"))
        self.assertEqual(multi.image("A").integration_domain.value, "full")
        self.assertEqual(multi.image("B").integration_domain.value, "rectangle")
        self.assertEqual(multi.arrival_time_delays_days, {"B": 3.5})

        maps = multi.magnification_maps(
            time_days={"A": 0.0, "B": 1.0},
            methods=self.method,
        )
        self.assertEqual(tuple(maps), ("A", "B"))
        self.assertEqual(tuple(maps["A"].values.shape), self.source_grid.shape)
        with self.assertRaisesRegex(ValueError, "omit macroimage"):
            multi.magnification_maps(
                time_days={"A": 0.0},
                methods=self.method,
            )

        updated = multi.with_arrival_time_delays(
            {"A": 0.0, "B": 4.25},
            require_all=True,
        )
        self.assertEqual(updated.arrival_time_delays_days["B"], 4.25)
        self.assertEqual(multi.arrival_time_delays_days["B"], 3.5)

    def test_multi_image_system_shares_physical_bulk_motion_seed(self) -> None:
        population = mc.StellarPopulation.salpeter(
            count=16,
            kinematics=mc.SkyProjectedKinematics(
                ra_deg=340.126125,
                dec_deg=3.358611,
            ),
        )
        multi = mc.MultiImageSystem(
            images={
                "A": mc.MacroLens(0.2, 0.08),
                "B": mc.MacroLens(0.3, 0.12),
            },
            distances=self.distances,
            source_grid=self.source_grid,
            stellar_population=population,
            duration_days=10.0,
            seed=123,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
            caustic_grid_shape=32,
        )
        first = multi.image("A").realize()
        second = multi.image("B").realize()
        first_kinematics = first.stellar_population.kinematics
        second_kinematics = second.stellar_population.kinematics
        self.assertEqual(first_kinematics.seed, second_kinematics.seed)
        self.assertEqual(
            first_kinematics._peculiar_velocities(self.distances),
            second_kinematics._peculiar_velocities(self.distances),
        )
        self.assertFalse(torch.equal(first.stars.x_uas, second.stars.x_uas))

    def test_unseeded_multi_image_population_keeps_one_shared_bulk_draw(self) -> None:
        population = mc.StellarPopulation.salpeter(
            count=16,
            kinematics=mc.SkyProjectedKinematics(ra_deg=1.0, dec_deg=2.0),
        )
        multi = mc.MultiImageSystem(
            images={"A": self.macro, "B": self.macro},
            distances=self.distances,
            source_grid=self.source_grid,
            stellar_population=population,
            duration_days=10.0,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        first = multi.image("A").realize()
        second = multi.image("B").realize()
        self.assertIsNone(first.stellar_population.kinematics.seed)
        self.assertIsNone(second.stellar_population.kinematics.seed)
        self.assertEqual(
            first.stellar_population.kinematics.mean_velocity_uas_per_day(self.distances),
            second.stellar_population.kinematics.mean_velocity_uas_per_day(self.distances),
        )
        self.assertFalse(torch.equal(first.stars.x_uas, second.stars.x_uas))

    def test_multi_image_system_builds_directly_from_macro_solutions(self) -> None:
        solutions = (
            mc.MacroImageSolution(
                "A",
                0.5,
                0.2,
                0.0,
                12.0,
                4.0,
                1,
                0.35,
                0.20,
                0.18,
                0.09,
                0.23,
                1.0e-9,
            ),
            mc.MacroImageSolution(
                "B",
                -0.4,
                0.3,
                7.5,
                19.5,
                -3.0,
                -1,
                0.55,
                0.45,
                0.40,
                0.21,
                0.24,
                2.0e-9,
            ),
        )
        multi = mc.MultiImageSystem.from_macroimage_solutions(
            solutions,
            smooth_matter_fraction={"A": 0.1, "B": 0.2},
            distances=self.distances,
            source_grid=self.source_grid,
            stars={"A": self._stars(), "B": self._stars()},
            integration_domain="full",
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
            caustic_grid_shape=32,
        )
        self.assertEqual(multi.arrival_time_delays_days, {"A": 0.0, "B": 7.5})
        self.assertAlmostEqual(multi.image("A").macro.convergence, 0.35)
        self.assertAlmostEqual(multi.image("B").macro.smooth_matter_fraction, 0.2)

    def test_physical_thin_disk_selects_and_materializes_its_grid(self) -> None:
        model = mc.ThinDiskModel(
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            wavelengths_angstrom=(4_800.0, 9_700.0),
            band_names=("blue", "red"),
            spin=0.3,
            inclination_deg=45.0,
            grid=mc.SourceGridConfig(
                shape=(24, 32),
                enclosed_flux_fraction=0.995,
                margin=1.05,
                radial_samples=1024,
            ),
        )
        grid = model.recommended_grid(self.distances)
        self.assertEqual(grid.shape, (24, 32))
        self.assertAlmostEqual(
            grid.field_of_view_uas[0],
            grid.field_of_view_uas[1],
        )
        source = model.pixelate(self.distances, grid=grid)
        self.assertEqual(source.geometry.shape, grid.shape)
        brightness = source.brightness(0.0, dtype=torch.float64)
        self.assertEqual(tuple(brightness.shape), (1, 24, 32, 2))
        self.assertTrue(bool(torch.isfinite(brightness).all()))
        self.assertGreater(float(brightness.sum()), 0.0)

    def test_physical_gaussian_selects_grid_and_preserves_requested_flux(self) -> None:
        model = mc.GaussianModel(
            sigma_m=(1.0e10, 2.0e10),
            wavelengths_angstrom=(4_800.0, 9_700.0),
            band_names=("blue", "red"),
            total_flux=(2.0, 3.0),
            axis_ratio=0.6,
            position_angle_deg=math.degrees(0.4),
            center_m=(2.0e9, -1.0e9),
            grid=mc.SourceGridConfig(
                shape=(31, 37),
                enclosed_flux_fraction=0.999,
                margin=1.05,
            ),
        )
        grid = model.recommended_grid(self.distances)
        source = model.pixelate(self.distances, grid=grid)
        self.assertEqual(grid.shape, (31, 37))
        self.assertNotAlmostEqual(
            grid.field_of_view_uas[0],
            grid.field_of_view_uas[1],
        )
        brightness = source.brightness(0.0, dtype=torch.float64)[0]
        pixel_area = math.prod(source.geometry.pixel_scale_m)
        integrated = brightness.sum(dim=(0, 1)) * pixel_area
        torch.testing.assert_close(
            integrated,
            torch.tensor((2.0, 3.0), dtype=torch.float64),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_system_resolves_physical_source_only_once(self) -> None:
        model = mc.ThinDiskModel(
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            wavelengths_angstrom=(7_500.0,),
            band_names=("optical",),
            grid=mc.SourceGridConfig(
                shape=16,
                enclosed_flux_fraction=0.99,
                radial_samples=512,
            ),
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=model,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        realization = system.realize()
        self.assertIs(realization, system.realize())
        self.assertIsInstance(realization.source, mc.ThinDiskSource)
        self.assertEqual(realization.source_grid.shape, (16, 16))
        self.assertAlmostEqual(
            2.0 * realization.source_support_radius_uas,
            realization.source_grid.field_of_view_uas[0],
        )

    def test_system_rejects_grid_that_truncates_physical_support(self) -> None:
        model = mc.GaussianModel(
            sigma_m=2.0e13,
            wavelengths_angstrom=(7_500.0,),
            band_names=("optical",),
            grid=mc.SourceGridConfig(shape=16, enclosed_flux_fraction=0.99),
        )
        recommended = model.recommended_grid(self.distances)
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=model,
            source_grid=mc.PlaneGrid(
                recommended.shape,
                tuple(0.8 * value for value in recommended.field_of_view_uas),
            ),
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        with self.assertRaisesRegex(ValueError, "does not enclose"):
            system.realize()

    def test_expanding_photosphere_geometry_defines_system_grid(self) -> None:
        source = mc.paper_type_ia_supernova_source(
            redshift=self.distances.source_redshift,
            wavelengths_angstrom=(4_800.0, 7_500.0),
            maximum_observer_time_days=30.0,
            band_names=("blue", "red"),
            source_grid_shape=24,
            luminosity_distance_m=4.0e25,
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=source,
            stars=self._stars(),
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        realization = system.realize()
        self.assertIsNot(realization.source, source)
        self.assertEqual(realization.source.geometry, source.geometry)
        expected_support = float(
            self.distances.source_length_to_uas(
                source.maximum_photosphere_radius_m,
                dtype=torch.float64,
            )
        )
        self.assertAlmostEqual(
            realization.source_support_radius_uas,
            expected_support,
        )
        self.assertAlmostEqual(
            realization.source_grid.field_of_view_uas[0]
            / (2.0 * realization.source_support_radius_uas),
            source.source_margin,
        )
        self.assertEqual(realization.source_grid.shape, (24, 24))
        expected_m = 2.0 * source.maximum_photosphere_radius_m * 1.05
        expected_uas = float(
            self.distances.source_length_to_uas(
                expected_m,
                dtype=torch.float64,
            )
        )
        self.assertAlmostEqual(
            realization.source_grid.field_of_view_uas[0],
            expected_uas,
        )

    def test_high_level_labeled_light_curve_matches_low_level_pipeline(self) -> None:
        pixel_scale_m = self.distances.uas_to_source_length(
            self.source_grid.pixel_scale_uas,
            dtype=torch.float64,
        )
        source = mc.GaussianSource(
            mc.SourceGeometry(
                shape=(9, 11),
                pixel_scale_m=(float(pixel_scale_m[0]), float(pixel_scale_m[1])),
                wavelengths_angstrom=(6_000.0,),
                band_names=("optical",),
            ),
            sigma_m=0.2 * float(pixel_scale_m.mean()),
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=source,
            source_grid=self.source_grid,
            stars=self._stars(),
            integration_domain="full",
            lens_region=self.lens_region,
            caustic_grid_shape=16,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        caustic_config = mc.CausticConfig(
            far_field_approx=mc.FarFieldApproxConfig(enabled=False),
            minimum_determinant_sign_pixels=1,
            anchor_count=3,
            gauge_count=3,
            minimum_alignment_gauges=1,
        )
        actual = system.light_curve(
            (0.0,),
            include_labels=True,
            method=self.method,
            schedule=mc.DynamicConfig(
                temporal_batch_size=1,
                fused_temporal_ipm=False,
                scout_refresh_frames=1,
            ),
            caustics=caustic_config,
            keep_maps_at_days=(0.0,),
        )
        self.assertEqual(actual.map_times_days.tolist(), [0.0])
        self.assertIsInstance(actual.maps[0], mc.MagnificationMap)
        realization = system.realize()
        expected = realization.simulation.light_curve_with_labels(
            realization.lens_region,
            realization.source_grid,
            realization.lens_grid,
            (0.0,),
            source,
            self.distances,
            method=self.method,
            map_schedule=mc.DynamicConfig(
                temporal_batch_size=1,
                fused_temporal_ipm=False,
                scout_refresh_frames=1,
            ),
            caustic_config=caustic_config,
        )
        torch.testing.assert_close(
            actual.flux,
            expected.light_curve.flux,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(actual.labels.crossing_labels, expected.crossing_labels)

    def test_high_level_multirate_light_curve_matches_low_level_pipeline(self) -> None:
        pixel_scale_m = self.distances.uas_to_source_length(
            self.source_grid.pixel_scale_uas,
            dtype=torch.float64,
        )
        source = mc.GaussianSource(
            mc.SourceGeometry(
                shape=self.source_grid.shape,
                pixel_scale_m=(float(pixel_scale_m[0]), float(pixel_scale_m[1])),
                wavelengths_angstrom=(6_000.0,),
                band_names=("optical",),
            ),
            sigma_m=0.3 * float(pixel_scale_m.mean()),
        )
        system = mc.MicrolensingSystem(
            macro=self.macro,
            distances=self.distances,
            source=source,
            source_grid=self.source_grid,
            stars=self._stars(),
            integration_domain="full",
            lens_region=self.lens_region,
            runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
        )
        schedule = mc.DynamicConfig(
            temporal_batch_size=1,
            fused_temporal_ipm=False,
            scout_refresh_frames=1,
        )
        actual = system.light_curve(
            (0.0, 1.0),
            flux_times_days=(0.0, 0.5, 1.0),
            method=self.method,
            schedule=schedule,
        )
        realization = system.realize()
        expected = realization.simulation.multirate_light_curve(
            realization.lens_region,
            realization.source_grid,
            (0.0, 1.0),
            (0.0, 0.5, 1.0),
            source,
            self.distances,
            method=self.method,
            schedule=schedule,
        )
        torch.testing.assert_close(actual.flux, expected.flux, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
