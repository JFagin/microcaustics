"""High-level physical-system construction and low-level parity tests."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

import microcaustics as mc


class MicrolensingSystemTests(unittest.TestCase):
    def test_plain_band_mapping_and_sampling_keywords(self) -> None:
        model = mc.ThinDiskModel(
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            bands={"blue": 4_800.0, "red": 9_700.0},
            resolution=24,
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
        return mc.PointMassField(
            torch.tensor([-0.4, 0.5, 1.0]),
            torch.tensor([0.3, -0.5, 0.6]),
            torch.tensor([0.2, 0.16, 0.12]),
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
        source = mc.GaussianModel.from_angular(
            self.distances,
            sigma_uas=0.12,
            bands={"optical": 6_000.0},
            axis_ratio=0.6,
            position_angle_rad=math.radians(23.0),
            resolution=24,
        )
        population = mc.StellarPopulation.salpeter(count=12)
        common = dict(
            macro=mc.MacroLens(0.3, 0.2, shear_angle_deg=angle_deg),
            distances=self.distances,
            source=source,
            stellar_population=population,
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
        expected_x = math.cos(angle) * full.stars.x_uas + math.sin(angle) * full.stars.y_uas
        expected_y = -math.sin(angle) * full.stars.x_uas + math.cos(angle) * full.stars.y_uas
        torch.testing.assert_close(rectangle.stars.x_uas, expected_x)
        torch.testing.assert_close(rectangle.stars.y_uas, expected_y)

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
        source = mc.GaussianModel.from_angular(
            self.distances,
            sigma_uas=0.1,
            bands={"optical": 6_000.0},
            resolution=16,
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
        rotated = realization.trajectory.position_uas(
            (0.0, 10.0), dtype=torch.float64
        )
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

    def test_system_from_redshifts_hides_distance_construction(self) -> None:
        system = mc.MicrolensingSystem.from_redshifts(
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
        schedule, caustics = _production_dynamic_settings(method, None)
        self.assertEqual(schedule.temporal_batch_size, 40)
        self.assertIsNotNone(caustics)
        assert caustics is not None
        self.assertEqual(caustics.temporal_batch_size, 40)
        self.assertEqual(caustics.far_field_approx, method.far_field_approx)

    def test_multi_image_system_from_redshifts_shares_geometry(self) -> None:
        system = mc.MultiImageSystem.from_redshifts(
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

    def test_sampled_sky_kinematics_can_defer_redshifts(self) -> None:
        kinematics = mc.SkyProjectedKinematics.sampled(
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
        self.assertEqual(tuple(curve.maps), (0.0, 1.0))
        self.assertEqual(curve.maps[0.0].time_days, 0.0)

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
        model = mc.GaussianModel.from_angular(
            self.distances,
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
            torch.tensor(model.sigma_m, dtype=torch.float64),
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

    def test_sampled_sky_kinematics_is_reproducible(self) -> None:
        kwargs = {
            "ra_deg": 340.126125,
            "dec_deg": 3.358611,
            "lens_redshift": 0.25,
            "source_redshift": 1.2,
            "seed": 91,
        }
        first = mc.SkyProjectedKinematics.sampled(**kwargs)
        second = mc.SkyProjectedKinematics.sampled(**kwargs)
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
        local_stars = mc.PointMassField(
            math.cos(angle) * stars.x_uas + math.sin(angle) * stars.y_uas,
            -math.sin(angle) * stars.x_uas + math.cos(angle) * stars.y_uas,
            stars.einstein_radius_uas,
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
        stars_local = mc.PointMassField(
            stars_sky.y_uas,
            -stars_sky.x_uas,
            stars_sky.einstein_radius_uas,
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

        geometry = mc.SourceGeometry.from_angular(
            self.distances,
            shape=source_grid.shape,
            field_of_view_uas=source_grid.field_of_view_uas,
            bands={"optical": 6_000.0},
        )
        sigma_m = float(
            self.distances.uas_to_source_length(0.25, dtype=torch.float64)
        )
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
                source_grid=self.source_grid,
                stars=mc.PointMassField(
                    self._stars().x_uas + shift,
                    self._stars().y_uas,
                    self._stars().einstein_radius_uas,
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
        batched = mc.batched_system_maps(systems, method=method, batch_size=2)
        separate = tuple(system.magnification_map(method=method) for system in systems)
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
                stars=mc.PointMassField(
                    self._stars().x_uas + shift,
                    self._stars().y_uas,
                    self._stars().einstein_radius_uas,
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
            position_angle_rad=0.4,
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
            redshift=0.8,
            wavelengths_angstrom=(4_800.0, 7_500.0),
            maximum_observer_time_days=30.0,
            band_names=("blue", "red"),
            resolution=24,
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
        self.assertIs(realization.source, source)
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
            source.source_fov_margin,
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
            minimum_sign_component_pixels=1,
            anchor_count=3,
            gauge_count=3,
            minimum_safe_gauges=1,
        )
        actual = system.light_curve_with_labels(
            (0.0,),
            method=self.method,
            schedule=mc.DynamicConfig(
                temporal_batch_size=1,
                fused_temporal_ipm=False,
                scout_refresh_frames=1,
            ),
            caustics=caustic_config,
            keep_maps_at_days=(0.0,),
        )
        self.assertEqual(tuple(actual.maps), (0.0,))
        self.assertIsInstance(actual.maps[0.0], mc.MagnificationMap)
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
            actual.light_curve.flux,
            expected.light_curve.flux,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(actual.crossing_labels, expected.crossing_labels)

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
        actual = system.multirate_light_curve(
            (0.0, 1.0),
            (0.0, 0.5, 1.0),
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
