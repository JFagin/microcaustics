"""Small CPU tests for the initial public data-model contracts."""

from __future__ import annotations

import unittest
import warnings
from unittest import mock

import torch

from microcaustics import (
    Backend,
    CallableDrivingSignal,
    CausticField,
    DelayedModulatedSource,
    DynamicConfig,
    FarFieldApproxConfig,
    IPMConfig,
    IRSConfig,
    LensingDistances,
    LightCurveRequest,
    MacroLens,
    MicrolensingSimulation,
    ModulatedSource,
    ObserverScreen,
    ObserverTransfer,
    PlaneGrid,
    PlaneRegion,
    PointMassField,
    RuntimeCapabilities,
    RuntimeConfig,
    StaticSource,
    TabulatedDrivingSignal,
    ThinDiskSource,
    TransferredThinDiskSource,
    benchmark_callable,
    compact_convergence,
    gravitational_radius_m,
    kerr_isco_radius,
    lamppost_irradiation_efficiency,
    lamppost_source_height_rg,
    novikov_thorne_radiative_efficiency,
    production_dynamic_config,
    production_ipm_config,
    resolve_runtime,
    thin_disk_flux_radius_rg,
)
from microcaustics.config import _production_static_ipm_config
from microcaustics.lens import kroupa_mass_function, salpeter_mass_function
from microcaustics.runtime import warn_backend_fallback
from microcaustics.sources import GaussianSource, SourceGeometry


class FoundationTests(unittest.TestCase):
    def test_production_runtime_memory_cap(self) -> None:
        self.assertEqual(RuntimeConfig().memory_fraction, 0.95)
        self.assertTrue(RuntimeConfig().warn_on_compile)

    def test_gravitational_radius_uses_public_units(self) -> None:
        self.assertAlmostEqual(float(gravitational_radius_m(1.0)), 1476.625, places=3)

    def test_production_preset_supports_explicit_complete_scout(self) -> None:
        dynamic = production_ipm_config()
        static = _production_static_ipm_config()
        conservative = production_ipm_config(scout_trace_centers=True)
        self.assertEqual(static.scout_ratio, 1)
        self.assertFalse(static.dual_scout_scalar_correction)
        self.assertTrue(static.scout_trace_centers)
        self.assertTrue(conservative.scout_trace_centers)
        self.assertEqual(dynamic.scout_ratio, 2)
        self.assertTrue(dynamic.dual_scout_scalar_correction)
        self.assertFalse(dynamic.scout_trace_centers)
        self.assertTrue(dynamic.compact_sparse_nodes)
        self.assertFalse(IPMConfig().compact_sparse_nodes)
        self.assertEqual(static.scout_halo_pixels, 0.0)
        self.assertEqual(dynamic.scout_halo_pixels, 0.0)
        self.assertEqual(static.scout_dilation_cells, 1)
        self.assertEqual(dynamic.scout_dilation_cells, 1)
        self.assertEqual(dynamic.far_field_approx.cells_per_axis, 16)
        self.assertEqual(dynamic.far_field_approx.nodes_per_cell_axis, 8)
        schedule = production_dynamic_config()
        self.assertEqual(schedule.temporal_batch_size, 49)
        self.assertEqual(schedule.scout_refresh_frames, 10)
        self.assertTrue(schedule.endpoint_union)

    def test_flux_enclosing_q2237_disk_radius_matches_production(self) -> None:
        radius = thin_disk_flux_radius_rg(
            black_hole_mass_solar=10.0**9.08,
            eddington_ratio=0.34,
            spin=0.74,
            observed_wavelength_angstrom=9712.0,
            source_redshift=1.695,
            lamp_fraction=0.1,
            corona_height_above_isco_rg=20.0,
        )
        self.assertAlmostEqual(radius, 690.2878, places=2)

    def test_far_field_approx_config_is_public(self) -> None:
        config = FarFieldApproxConfig(cells_per_axis=16, nodes_per_cell_axis=8)
        self.assertEqual(config.cells_per_axis, 16)
        self.assertEqual(config.nodes_per_cell_axis, 8)

    def test_callable_benchmark_separates_first_and_steady_calls(self) -> None:
        calls = []
        synchronizations = []

        def calculation():
            calls.append(len(calls))
            return calls[-1]

        measurement = benchmark_callable(
            calculation,
            warmup=2,
            repeats=3,
            synchronize=lambda: synchronizations.append(None),
        )
        self.assertEqual(measurement.first_result, 0)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(synchronizations), 12)
        self.assertEqual(len(measurement.warmup_seconds), 2)
        self.assertEqual(len(measurement.steady_seconds), 3)
        self.assertGreaterEqual(measurement.median_steady_seconds, 0.0)

    def test_grid_cell_centers_and_rectangular_shape(self) -> None:
        grid = PlaneGrid((2, 4), (2.0, 8.0))
        y, x = grid.axes(dtype=torch.float64)
        self.assertTrue(torch.equal(y, torch.tensor([-0.5, 0.5], dtype=torch.float64)))
        self.assertTrue(
            torch.equal(x, torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float64))
        )

    def test_moving_point_mass_field(self) -> None:
        field = PointMassField._from_einstein_radii(
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            velocity_x_uas_per_day=torch.tensor([0.1]),
            velocity_y_uas_per_day=torch.tensor([-0.2]),
            einstein_radius_uas=torch.tensor([0.5]),
        )
        moved = field.at_time(10.0)
        self.assertAlmostEqual(float(moved.x_uas[0]), 2.0)
        self.assertAlmostEqual(float(moved.y_uas[0]), 0.0)
        self.assertIs(moved.einstein_radius_uas, field.einstein_radius_uas)
        self.assertIs(
            moved.velocity_x_uas_per_day,
            field.velocity_x_uas_per_day,
        )
        self.assertIs(
            moved.velocity_y_uas_per_day,
            field.velocity_y_uas_per_day,
        )

    def test_rectangular_static_source_batch(self) -> None:
        geometry = SourceGeometry(
            shape=(2, 3),
            pixel_scale_m=(1.0, 2.0),
            wavelengths_angstrom=(5000.0, 7000.0),
            band_names=("blue", "red"),
        )
        source = StaticSource(torch.ones(2, 3, 2), geometry)
        self.assertEqual(source.brightness([0.0, 1.0]).shape, (2, 2, 3, 2))

    def test_source_geometry_from_angular_field(self) -> None:
        distances = LensingDistances(8.0e24, 1.6e25, 9.0e24)
        geometry = SourceGeometry(
            shape=(20, 40),
            field_of_view_uas=(2.0, 8.0),
            bands_angstrom={"g": 4800.0, "i": 7600.0},
        ).resolve(distances)
        field_m = distances.uas_to_source_length(
            torch.tensor((2.0, 8.0)), dtype=torch.float32
        )
        self.assertEqual(geometry.shape, (20, 40))
        self.assertAlmostEqual(geometry.pixel_scale_m[0], float(field_m[0]) / 20)
        self.assertAlmostEqual(geometry.pixel_scale_m[1], float(field_m[1]) / 40)
        self.assertEqual(geometry.band_names, ("g", "i"))
        self.assertEqual(geometry.wavelengths_angstrom, (4800.0, 7600.0))

    def test_source_geometry_from_angular_requires_named_wavelengths(self) -> None:
        """Reject empty wavelength mappings with the public argument name."""
        with self.assertRaisesRegex(ValueError, "bands_angstrom"):
            SourceGeometry(
                shape=4, field_of_view_uas=2.0, bands_angstrom={}
            )

    def test_backend_falls_back_without_triton(self) -> None:
        capabilities = RuntimeCapabilities(
            platform="test",
            torch_version="test",
            cuda_available=False,
            cuda_version=None,
            mps_available=False,
            torch_compile_available=False,
            triton_importable=False,
            gpu_name=None,
            total_device_memory_bytes=None,
        )
        runtime = resolve_runtime(
            RuntimeConfig(device="cpu", backend=Backend.TRITON),
            capabilities=capabilities,
        )
        self.assertEqual(runtime.backend, Backend.TORCH_EAGER)
        self.assertIsNotNone(runtime.fallback_reason)

    def test_backend_operation_fallback_warns_once(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            error = RuntimeError("synthetic kernel failure")
            warn_backend_fallback("foundation-test kernel", error)
            warn_backend_fallback("foundation-test kernel", error)
        self.assertEqual(len(caught), 1)
        self.assertIn("portable Torch", str(caught[0].message))

    def test_simulation_moves_lens_arrays_to_runtime(self) -> None:
        field = PointMassField._from_einstein_radii(
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            einstein_radius_uas=torch.tensor([1.0], dtype=torch.float64),
        )
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.4, shear=0.2),
            field,
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float32",
            ),
        )
        self.assertEqual(simulation.point_masses.x_uas.dtype, torch.float32)
        self.assertEqual(simulation.runtime.device.type, "cpu")

    def test_direct_raytrace_matches_manual_lens_equation(self) -> None:
        field = PointMassField._from_einstein_radii(
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            einstein_radius_uas=torch.tensor([0.5], dtype=torch.float64),
        )
        macro = MacroLens(
            convergence=0.2,
            shear=0.1,
            shear_angle_deg=0.0,
            smooth_matter_fraction=0.5,
        )
        simulation = MicrolensingSimulation.create(
            macro,
            field,
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float64",
            ),
        )
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        y = torch.tensor([0.5, -1.0], dtype=torch.float64)
        source_x, source_y, diagnostics = simulation.raytrace_direct(
            x,
            y,
            star_chunk_size=1,
            ray_chunk_size=1,
        )
        radius2 = x.square() + y.square()
        point_alpha_x = 0.25 * x / radius2
        point_alpha_y = 0.25 * y / radius2
        expected_x = x - point_alpha_x - 0.1 * x - 0.1 * x
        expected_y = y - point_alpha_y - 0.1 * y + 0.1 * y
        torch.testing.assert_close(source_x, expected_x, rtol=0.0, atol=1e-14)
        torch.testing.assert_close(source_y, expected_y, rtol=0.0, atol=1e-14)
        self.assertEqual(diagnostics.ray_chunks, 2)

    def test_mass_functions_are_reproducible_and_bounded(self) -> None:
        generator_a = torch.Generator().manual_seed(7)
        generator_b = torch.Generator().manual_seed(7)
        mass_function = salpeter_mass_function(0.1, 10.0)
        first = mass_function.sample(1000, generator=generator_a, dtype=torch.float64)
        second = mass_function.sample(1000, generator=generator_b, dtype=torch.float64)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertGreaterEqual(float(first.min()), 0.1)
        self.assertLessEqual(float(first.max()), 10.0)
        self.assertGreater(kroupa_mass_function().mean_mass(), 0.0)

    def test_uniform_point_population_is_reproducible(self) -> None:
        distances = LensingDistances(1.0e25, 2.0e25, 1.0e25)
        macro = MacroLens(
            convergence=0.4,
            shear=0.2,
            smooth_matter_fraction=0.25,
        )
        region = PlaneRegion((20.0, 30.0), center_uas=(2.0, -1.0))
        mass_function = salpeter_mass_function(0.1, 1.0)
        first = PointMassField.sample_uniform(
            region,
            macro,
            distances,
            mass_function,
            seed=17,
            velocity_dispersion_uas_per_day=(0.001, 0.002),
            dtype=torch.float64,
        )
        second = PointMassField.sample_uniform(
            region,
            macro,
            distances,
            mass_function,
            seed=17,
            velocity_dispersion_uas_per_day=(0.001, 0.002),
            dtype=torch.float64,
        )
        self.assertGreater(len(first), 0)
        torch.testing.assert_close(first.x_uas, second.x_uas)
        torch.testing.assert_close(first.y_uas, second.y_uas)
        torch.testing.assert_close(first.mass_solar, second.mass_solar)
        self.assertTrue(first.has_motion)
        xmin, xmax, ymin, ymax = region.bounds_uas
        self.assertTrue(bool(torch.all((first.x_uas >= xmin) & (first.x_uas < xmax))))
        self.assertTrue(bool(torch.all((first.y_uas >= ymin) & (first.y_uas < ymax))))
        self.assertGreater(compact_convergence(first, region), 0.0)

    def test_gaussian_source_flux_and_hole(self) -> None:
        geometry = SourceGeometry(
            shape=(65, 65),
            pixel_scale_m=(1.0, 1.0),
            wavelengths_angstrom=(5000.0, 7000.0),
            band_names=("blue", "red"),
        )
        source = GaussianSource(
            geometry,
            sigma_m=(5.0, 8.0),
            total_flux=(2.0, 3.0),
            hole_radius_m=2.0,
        )
        frame = source.brightness(0.0, dtype=torch.float64)[0]
        torch.testing.assert_close(
            frame.sum(dim=(0, 1)),
            torch.tensor([2.0, 3.0], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(float(frame[32, 32].max()), 0.0)

    def test_kerr_isco_and_novikov_thorne_are_finite_through_zero_spin(self) -> None:
        from microcaustics.relativity import (
            kerr_isco_radius,
            novikov_thorne_flux_factor,
        )

        spins = torch.tensor([-1.0e-5, 0.0, 1.0e-5], dtype=torch.float64)
        spins.requires_grad_(True)
        isco = kerr_isco_radius(spins)
        self.assertGreater(float(isco[0].detach()), 6.0)
        self.assertEqual(float(isco[1].detach()), 6.0)
        self.assertLess(float(isco[2].detach()), 6.0)
        radius = torch.tensor([1.0, 6.0, 8.0, 20.0, 100.0], dtype=torch.float64)
        factor = novikov_thorne_flux_factor(radius[None], spins[:, None])
        self.assertTrue(bool(torch.all(torch.isfinite(factor))))
        self.assertTrue(bool(torch.all(factor[:, 0] == 0)))
        self.assertEqual(float(factor[1, 1].detach()), 0.0)
        factor.sum().backward()
        self.assertTrue(bool(torch.all(torch.isfinite(spins.grad))))

    def test_elliptic_primitives_match_identities_and_are_differentiable(
        self,
    ) -> None:
        from microcaustics.relativity import (
            carlson_rc,
            carlson_rd,
            carlson_rf,
            carlson_rj,
            elliptic_e,
            elliptic_f,
            elliptic_k,
            elliptic_pi_principal,
            jacobi_sn_cn,
        )

        one = torch.ones(3, dtype=torch.float64, requires_grad=True)
        zero = torch.zeros(3, dtype=torch.float64)
        torch.testing.assert_close(carlson_rf(one, one, one), one)
        torch.testing.assert_close(carlson_rd(one, one, one), one)
        torch.testing.assert_close(carlson_rj(one, one, one, one), one)
        torch.testing.assert_close(
            carlson_rc(zero, one),
            torch.full_like(one, torch.pi / 2.0),
        )
        amplitude = torch.tensor([-0.7, 0.2, 1.1], dtype=torch.float64)
        torch.testing.assert_close(
            elliptic_k(zero),
            torch.full_like(zero, torch.pi / 2.0),
        )
        torch.testing.assert_close(elliptic_f(amplitude, zero), amplitude)
        torch.testing.assert_close(elliptic_e(amplitude, zero), amplitude)
        torch.testing.assert_close(
            elliptic_pi_principal(zero, amplitude, zero),
            amplitude,
        )
        sn, cn = jacobi_sn_cn(amplitude, zero)
        torch.testing.assert_close(sn, torch.sin(amplitude))
        torch.testing.assert_close(cn, torch.cos(amplitude))
        (
            carlson_rf(one, one + 0.2, one + 0.4).sum()
            + carlson_rd(one, one + 0.2, one + 0.4).sum()
        ).backward()
        self.assertTrue(bool(torch.all(torch.isfinite(one.grad))))

    def test_kerr_radial_roots_satisfy_null_potential_quartic(self) -> None:
        from microcaustics.relativity import (
            kerr_radial_root_parts,
            radial_root_real_count,
        )

        spin = torch.tensor([-0.8, 0.0, 0.6, 0.9], dtype=torch.float64)
        eta = torch.tensor([2.0, 18.0, 45.0, 6.0], dtype=torch.float64)
        lam = torch.tensor([-3.0, 1.0, 7.0, -5.0], dtype=torch.float64)
        real, imag = kerr_radial_root_parts(spin, eta, lam)
        roots = torch.complex(real, imag)
        coefficient_a = spin.square() - eta - lam.square()
        coefficient_b = 2.0 * (eta + (lam - spin).square())
        coefficient_c = -spin.square() * eta
        residual = (
            roots.pow(4)
            + coefficient_a[:, None] * roots.square()
            + coefficient_b[:, None] * roots
            + coefficient_c[:, None]
        )
        scale = (
            roots.abs().pow(4)
            + coefficient_a[:, None].abs() * roots.abs().square()
            + coefficient_b[:, None].abs() * roots.abs()
            + coefficient_c[:, None].abs()
        ).clamp_min(1.0)
        self.assertLess(float((residual.abs() / scale).max()), 2.0e-14)
        counts = radial_root_real_count(real, imag)
        self.assertTrue(bool(torch.all((counts == 0) | (counts == 2) | (counts == 4))))

    def test_primary_kerr_trace_matches_frozen_paper_fixture(self) -> None:
        from microcaustics.relativity.primary import trace_primary_equatorial

        screen = ObserverScreen.uniform(
            (16, 20),
            (24.0, 30.0),
            gravitational_radius_m=1.0,
            observer_distance_m=1.0e6,
            dtype=torch.float64,
        )
        trace = trace_primary_equatorial(
            screen,
            spin=0.7,
            inclination_deg=53.0,
            disk_outer_rg=35.0,
        )
        hit = trace.transfer.hit
        self.assertEqual(int(hit.sum()), 262)
        self.assertEqual(trace.repaired_pixels, 3)
        self.assertIsNotNone(trace.interpolated_mask)
        self.assertEqual(int(trace.interpolated_mask.sum()), trace.repaired_pixels)
        self.assertAlmostEqual(
            float(trace.transfer.radius_rg[hit].sum()),
            5772.234197356813,
            places=10,
        )
        self.assertAlmostEqual(
            float(trace.transfer.gfactor[hit].sum()),
            243.5259394524515,
            places=11,
        )
        self.assertEqual(
            torch.bincount(trace.radial_root_count.reshape(-1), minlength=5).tolist(),
            [0, 0, 8, 0, 312],
        )
        self.assertEqual(
            trace.transfer.metadata["backend"],
            "analytic_separated_kerr",
        )

    def test_primary_kerr_compile_cache_is_reused_by_static_shape(self) -> None:
        from microcaustics.relativity.primary import (
            clear_compiled_primary_cache,
            trace_primary_equatorial,
        )

        screen = ObserverScreen.uniform(
            (4, 5),
            10.0,
            gravitational_radius_m=1.0,
            observer_distance_m=1.0e6,
            dtype=torch.float64,
        )
        clear_compiled_primary_cache()
        with (
            mock.patch(
                "microcaustics.relativity.primary._torch_compile_supported",
                return_value=True,
            ),
            mock.patch.object(
                torch, "compile", side_effect=lambda function, **_: function
            ) as compile_mock,
        ):
            first = trace_primary_equatorial(
                screen,
                spin=0.2,
                inclination_deg=30.0,
                compile_solver=True,
                warn_on_compile=False,
            )
            second = trace_primary_equatorial(
                screen,
                spin=0.5,
                inclination_deg=45.0,
                compile_solver=True,
                warn_on_compile=False,
            )
        self.assertEqual(compile_mock.call_count, 1)
        self.assertIn("cache=miss", first.transfer.metadata["execution"])
        self.assertIn("cache=hit", second.transfer.metadata["execution"])
        clear_compiled_primary_cache()

    def test_primary_kerr_iterative_pinhole_repair_is_monotone(self) -> None:
        from microcaustics.relativity.primary import trace_primary_equatorial

        screen = ObserverScreen.uniform(
            (32, 40),
            (48.0, 60.0),
            gravitational_radius_m=1.0,
            observer_distance_m=1.0e6,
            dtype=torch.float64,
        )
        one_pass = trace_primary_equatorial(
            screen,
            spin=0.7,
            inclination_deg=53.0,
            disk_outer_rg=35.0,
            repair_max_passes=1,
        )
        iterative = trace_primary_equatorial(
            screen,
            spin=0.7,
            inclination_deg=53.0,
            disk_outer_rg=35.0,
            repair_max_passes=8,
        )
        self.assertGreaterEqual(
            int(iterative.transfer.hit.sum()), int(one_pass.transfer.hit.sum())
        )
        self.assertGreaterEqual(iterative.repaired_pixels, one_pass.repaired_pixels)
        self.assertEqual(iterative.transfer.metadata["pinhole_repair_max_passes"], 8)

    def test_observer_coordinates_match_frozen_paper_fixture(self) -> None:
        from microcaustics.relativity import add_observer_coordinates
        from microcaustics.relativity.primary import trace_primary_equatorial

        screen = ObserverScreen.uniform(
            (16, 20),
            (24.0, 30.0),
            gravitational_radius_m=1.0,
            observer_distance_m=1.0e6,
            dtype=torch.float64,
        )
        primary = trace_primary_equatorial(
            screen,
            spin=0.7,
            inclination_deg=53.0,
            disk_outer_rg=35.0,
        )
        coordinates = add_observer_coordinates(
            primary,
            screen,
            black_hole_mass_solar=1.0e9,
            spin=0.7,
            inclination_deg=53.0,
            quadrature_order=24,
            coordinate_dtype=torch.float64,
        )
        hit = coordinates.transfer.hit
        self.assertEqual(coordinates.failed_pixels, 0)
        self.assertAlmostEqual(
            float(coordinates.transfer.emission_azimuth_rad[hit].sum()),
            0.8424633053864028,
            places=12,
        )
        self.assertAlmostEqual(
            float(coordinates.transfer.relative_delay_days[hit].sum()),
            428.7570236620868,
            delta=1.0e-9,
        )
        self.assertAlmostEqual(
            float(coordinates.transfer.relative_delay_days[hit].max()),
            3.0446956619552896,
            places=12,
        )
        self.assertLess(
            float(coordinates.polar_consistency_error[hit].max()),
            2.0e-15,
        )
        self.assertEqual(
            coordinates.transfer.metadata["primary_pinhole_coordinate_repairs"],
            primary.repaired_pixels,
        )
        redshifted = add_observer_coordinates(
            primary,
            screen,
            black_hole_mass_solar=1.0e9,
            spin=0.7,
            inclination_deg=53.0,
            source_redshift=1.0,
            quadrature_order=24,
            coordinate_dtype=torch.float64,
        )
        self.assertTrue(
            torch.allclose(
                redshifted.transfer.relative_delay_days[hit],
                2.0 * coordinates.transfer.relative_delay_days[hit],
                rtol=2.0e-15,
                atol=2.0e-15,
            )
        )

    def test_observer_coordinates_repair_rare_float32_rays(self) -> None:
        from microcaustics.relativity import add_observer_coordinates
        from microcaustics.relativity.primary import trace_primary_equatorial

        screen = ObserverScreen.uniform(
            (32, 32),
            (60.0, 60.0),
            gravitational_radius_m=1.0,
            observer_distance_m=1.0e6,
            dtype=torch.float32,
        )
        primary = trace_primary_equatorial(
            screen,
            spin=0.7,
            inclination_deg=53.0,
            disk_outer_rg=50.0,
        )
        coordinates = add_observer_coordinates(
            primary,
            screen,
            black_hole_mass_solar=1.0e9,
            spin=0.7,
            inclination_deg=53.0,
        )
        self.assertEqual(coordinates.failed_pixels, 0)
        self.assertGreater(coordinates.repaired_pixels, 0)
        self.assertEqual(
            coordinates.transfer.metadata["coordinate_repaired_pixels"],
            coordinates.repaired_pixels,
        )
        self.assertTrue(
            torch.all(
                torch.isfinite(
                    coordinates.transfer.relative_delay_days[coordinates.transfer.hit]
                )
            )
        )
        self.assertLess(
            float(
                coordinates.transfer.relative_delay_days[coordinates.transfer.hit].max()
            ),
            20.0,
        )

    def test_thermal_reprocessing_response_matches_finite_difference(self) -> None:
        from microcaustics.relativity import (
            add_observer_coordinates,
            axis_lamppost_profile,
        )
        from microcaustics.relativity.primary import trace_primary_equatorial
        from microcaustics.sources import (
            CallableDrivingSignal,
            SourceGeometry,
            ThermalReprocessingSource,
            thin_disk_temperature4,
        )

        screen = ObserverScreen.uniform(
            (12, 14),
            (24.0, 28.0),
            gravitational_radius_m=1.0e12,
            observer_distance_m=1.0e25,
            dtype=torch.float64,
        )
        primary = trace_primary_equatorial(
            screen,
            spin=0.5,
            inclination_deg=45.0,
            disk_outer_rg=30.0,
        )
        coordinates = add_observer_coordinates(
            primary,
            screen,
            black_hole_mass_solar=1.0e8,
            spin=0.5,
            inclination_deg=45.0,
            source_redshift=1.0,
            coordinate_dtype=torch.float64,
        )
        geometry = SourceGeometry(
            screen.shape,
            (1.0e12, 1.0e12),
            (4000.0, 7000.0),
            ("blue", "red"),
        )
        viscous4, _ = thin_disk_temperature4(
            coordinates.transfer.radius_rg,
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            spin=0.5,
        )

        def make_source(amplitude: float):
            signal = CallableDrivingSignal(
                lambda times: torch.full_like(times, amplitude),
                name=f"constant_{amplitude}",
            )
            return ThermalReprocessingSource(
                geometry,
                coordinates.transfer,
                signal,
                0.2 * viscous4,
                coordinates.transfer.relative_delay_days,
                black_hole_mass_solar=1.0e8,
                eddington_ratio=0.1,
                spin=0.5,
                source_redshift=1.0,
            )

        epsilon = 1.0e-4
        source = make_source(1.0)
        perturbed = make_source(1.0 + epsilon)
        numerical = (
            perturbed.brightness(0.0, dtype=torch.float64)[0]
            - source.brightness(0.0, dtype=torch.float64)[0]
        ) / epsilon
        analytic = source.linear_response_weights(dtype=torch.float64)
        hit = coordinates.transfer.hit[..., None].expand_as(analytic)
        relative = (numerical[hit] - analytic[hit]).abs() / analytic[hit].clamp_min(
            1.0e-30
        )
        self.assertLess(float(relative.max()), 2.0e-4)
        transfer_function = source.transfer_function(
            torch.linspace(0.0, 5.0, 17, dtype=torch.float64)
        )
        self.assertEqual(tuple(transfer_function.shape), (16, 2))
        self.assertTrue(
            torch.allclose(
                transfer_function.sum(0),
                torch.ones(2, dtype=torch.float64),
            )
        )
        lamp = axis_lamppost_profile(
            spin=0.5,
            source_height_rg=10.0,
            disk_outer_rg=30.0,
            nalpha=32,
            radial_bins=24,
            dtype=torch.float64,
        )
        lamp_source = ThermalReprocessingSource.from_axis_lamppost(
            geometry,
            coordinates.transfer,
            make_source(1.0).signal,
            lamp,
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            spin=0.5,
            source_redshift=1.0,
            irradiation_efficiency=0.1,
        )
        self.assertEqual(
            lamp_source.metadata()["heating"]["model"],
            "axis_kerr_lamppost",
        )
        self.assertTrue(
            torch.all(torch.isfinite(lamp_source.brightness(0.0, dtype=torch.float64)))
        )

    def test_axis_lamppost_matches_frozen_transfer_fixture(self) -> None:
        from microcaustics.relativity import axis_lamppost_profile

        profile = axis_lamppost_profile(
            spin=0.7,
            source_height_rg=10.0,
            disk_outer_rg=50.0,
            nalpha=128,
            radial_bins=64,
            dtype=torch.float64,
            quadrature_order=24,
        )
        rays = profile.rays
        hit = rays.hit
        self.assertEqual(int(hit.sum()), 91)
        self.assertAlmostEqual(
            float(rays.radius_rg[hit].sum()),
            1266.8307827755639,
            places=10,
        )
        self.assertAlmostEqual(
            float(rays.travel_time_rg[hit].sum()),
            592.5028160000445,
            places=9,
        )
        self.assertAlmostEqual(
            float(profile.illumination.sum()),
            0.13778851271845494,
            places=13,
        )
        self.assertAlmostEqual(
            float(profile.hit_fraction),
            0.45300421317498346,
            places=14,
        )

    def test_axis_lamppost_compile_cache_uses_static_launch_shape(self) -> None:
        from microcaustics.relativity.lamppost import (
            clear_compiled_lamppost_cache,
            trace_axis_lamppost,
        )

        clear_compiled_lamppost_cache()
        with (
            mock.patch(
                "microcaustics.relativity.lamppost._torch_compile_supported",
                return_value=True,
            ),
            mock.patch.object(
                torch, "compile", side_effect=lambda function, **_: function
            ) as compile_mock,
        ):
            first = trace_axis_lamppost(
                spin=0.2,
                source_height_rg=10.0,
                disk_outer_rg=30.0,
                nalpha=32,
                dtype=torch.float64,
                compile_solver=True,
                warn_on_compile=False,
            )
            second = trace_axis_lamppost(
                spin=0.7,
                source_height_rg=12.0,
                disk_outer_rg=35.0,
                nalpha=32,
                dtype=torch.float64,
                compile_solver=True,
                warn_on_compile=False,
            )
        self.assertEqual(compile_mock.call_count, 1)
        self.assertIn("cache=miss", first.execution)
        self.assertIn("cache=hit", second.execution)
        clear_compiled_lamppost_cache()

    def test_thin_disk_source_has_physical_multiband_brightness(self) -> None:
        geometry = SourceGeometry(
            shape=(33, 33),
            pixel_scale_m=(1.0e12, 1.0e12),
            wavelengths_angstrom=(4000.0, 7000.0),
            band_names=("blue", "red"),
        )
        source = ThinDiskSource(
            geometry,
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            spin=0.5,
            inclination_deg=35.0,
            source_redshift=1.5,
            luminosity_distance_m=1.0e26,
            relativity="none",
        )
        brightness = source.brightness([0.0, 10.0], dtype=torch.float64)
        self.assertEqual(tuple(brightness.shape), (2, 33, 33, 2))
        self.assertTrue(bool(torch.all(torch.isfinite(brightness))))
        self.assertTrue(bool(torch.all(brightness >= 0)))
        self.assertGreater(float(brightness.sum()), 0.0)
        torch.testing.assert_close(brightness[0], brightness[1])
        self.assertEqual(float(brightness[0, 16, 16].max()), 0.0)
        brightness32 = source.brightness([0.0], dtype=torch.float32)
        self.assertTrue(bool(torch.all(torch.isfinite(brightness32))))
        self.assertGreater(float(brightness32.sum()), 0.0)
        torch.testing.assert_close(
            brightness32[0].double().sum(dim=(0, 1)),
            brightness[0].sum(dim=(0, 1)),
            rtol=2.0e-5,
            atol=0.0,
        )

        shifted = ThinDiskSource(
            geometry,
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            spin=0.5,
            inclination_deg=35.0,
            source_redshift=1.5,
            luminosity_distance_m=1.0e26,
            relativity="approximate",
        ).brightness(0.0, dtype=torch.float64)
        self.assertGreater(float((shifted - brightness[:1]).abs().sum()), 0.0)
        inferred = ThinDiskSource.from_lensing_distances(
            geometry,
            1.0e8,
            0.1,
            LensingDistances(1.0e25, 2.0e25, 1.0e25),
            source_redshift=1.5,
        )
        self.assertAlmostEqual(inferred.luminosity_distance_m / 1.25e26, 1.0)

    def test_observer_transfer_reproduces_face_on_disk(self) -> None:
        from microcaustics.relativity import kerr_isco_radius

        geometry = SourceGeometry(
            shape=(33, 33),
            pixel_scale_m=(1.0e12, 1.0e12),
            wavelengths_angstrom=(4000.0, 7000.0),
            band_names=("blue", "red"),
        )
        mass = 1.0e8
        luminosity_distance = 1.0e26
        redshift = 1.5
        angular_diameter_distance = luminosity_distance / (1.0 + redshift) ** 2
        spin = 0.5
        ordinary = ThinDiskSource(
            geometry,
            mass,
            0.1,
            spin=spin,
            inclination_deg=0.0,
            source_redshift=redshift,
            luminosity_distance_m=luminosity_distance,
            relativity="none",
        )
        ny, nx = geometry.shape
        dy, dx = geometry.pixel_scale_m
        x = (torch.arange(nx, dtype=torch.float64) + 0.5 - nx / 2) * dx
        y = (torch.arange(ny, dtype=torch.float64) + 0.5 - ny / 2) * dy
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        gravitational_radius = (
            6.67430e-11 * 1.988409870698051e30 / (299_792_458.0**2) * mass
        )
        radius_rg = torch.sqrt(xx.square() + yy.square()) / gravitational_radius
        outer_radius = 0.5 * min(ny * dy, nx * dx) / gravitational_radius
        hit = (radius_rg > kerr_isco_radius(torch.tensor(spin))) & (
            radius_rg <= outer_radius
        )
        transfer = ObserverTransfer(
            radius_rg,
            torch.ones_like(radius_rg),
            torch.full_like(
                radius_rg,
                dx * dy / angular_diameter_distance**2,
            ),
            hit,
            relative_delay_days=torch.zeros_like(radius_rg),
            emission_azimuth_rad=torch.atan2(yy, xx),
            metadata={"backend": "flat_test"},
        )
        transferred = TransferredThinDiskSource(
            geometry,
            transfer,
            mass,
            0.1,
            spin,
            redshift,
        )
        expected = ordinary.brightness(0.0, dtype=torch.float64)
        actual = transferred.brightness(0.0, dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=0.0)
        self.assertEqual(transfer.shape, geometry.shape)

    def test_uniform_observer_screen_has_centered_geometry_and_solid_angle(
        self,
    ) -> None:
        screen = ObserverScreen.uniform(
            (2, 4),
            (1.0, 4.0),
            gravitational_radius_m=2.0,
            observer_distance_m=10.0,
            dtype=torch.float64,
        )
        self.assertEqual(screen.shape, (2, 4))
        torch.testing.assert_close(
            screen.y_rg[:, 0],
            torch.tensor([-0.5, 0.5], dtype=torch.float64),
        )
        torch.testing.assert_close(
            screen.x_rg[0],
            torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            screen.solid_angle_sr,
            torch.full((2, 4), 0.08, dtype=torch.float64),
        )
        transfer = ObserverTransfer.from_screen(
            screen,
            torch.ones(screen.shape, dtype=torch.float64),
            torch.ones(screen.shape, dtype=torch.float64),
            torch.ones(screen.shape, dtype=torch.bool),
            metadata={"backend": "test"},
        )
        self.assertEqual(transfer.metadata["sampling"], "pixel_centers")
        self.assertEqual(transfer.metadata["backend"], "test")

    def test_observer_screen_rotation_is_core_and_preserves_solid_angle(self) -> None:
        screen = ObserverScreen.uniform(
            (2, 2),
            1.0,
            gravitational_radius_m=2.0,
            observer_distance_m=10.0,
            dtype=torch.float64,
        )
        rotated = screen.rotated(90.0)
        torch.testing.assert_close(rotated.x_rg, -screen.y_rg, atol=1.0e-15, rtol=0)
        torch.testing.assert_close(rotated.y_rg, screen.x_rg, atol=1.0e-15, rtol=0)
        torch.testing.assert_close(rotated.solid_angle_sr, screen.solid_angle_sr)
        self.assertAlmostEqual(rotated.metadata["position_angle_deg"], 90.0)

    def test_lamppost_physical_helpers_replace_manual_notebook_math(self) -> None:
        spin = torch.tensor(0.74, dtype=torch.float64)
        isco = kerr_isco_radius(spin)
        expected_efficiency = 1.0 - torch.sqrt(1.0 - 2.0 / (3.0 * isco))
        torch.testing.assert_close(
            novikov_thorne_radiative_efficiency(spin), expected_efficiency
        )
        torch.testing.assert_close(
            lamppost_source_height_rg(spin, 20.0),
            isco + 20.0,
        )
        torch.testing.assert_close(
            lamppost_irradiation_efficiency(0.1, 0.34, spin),
            0.1 * expected_efficiency / 0.34,
        )

    def test_temperature_tilt_preserves_coordinate_area_luminosity(self) -> None:
        from microcaustics.sources import thin_disk_temperature4

        radius = torch.logspace(-4.0, 6.0, 20_000, dtype=torch.float64)
        widths = torch.empty_like(radius)
        widths[1:-1] = 0.5 * (radius[2:] - radius[:-2])
        widths[0] = radius[1] - radius[0]
        widths[-1] = radius[-1] - radius[-2]
        area_weight = radius * widths
        baseline, _ = thin_disk_temperature4(
            radius,
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            spin=0.6,
            temperature_slope_beta=0.75,
        )
        tilted, _ = thin_disk_temperature4(
            radius,
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            spin=0.6,
            temperature_slope_beta=0.6,
        )
        torch.testing.assert_close(
            (tilted * area_weight).sum(),
            (baseline * area_weight).sum(),
            rtol=2.0e-12,
            atol=0.0,
        )

    def test_tabulated_signal_modulates_any_source_by_band(self) -> None:
        geometry = SourceGeometry(
            shape=(2, 3),
            pixel_scale_m=(1.0, 1.0),
            wavelengths_angstrom=(4000.0, 7000.0),
            band_names=("blue", "red"),
        )
        base = StaticSource(torch.ones(2, 3, 2), geometry)
        signal = TabulatedDrivingSignal(
            torch.tensor([0.0, 10.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )
        source = ModulatedSource(base, signal)
        brightness = source.brightness([0.0, 5.0, 10.0])
        expected = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])[
            :, None, None, :
        ].expand(-1, 2, 3, -1)
        torch.testing.assert_close(brightness, expected)
        self.assertFalse(source.is_time_static)

    def test_delayed_modulation_evaluates_pixelwise_retarded_times(self) -> None:
        geometry = SourceGeometry(
            shape=(2, 2),
            pixel_scale_m=(1.0, 1.0),
            wavelengths_angstrom=(5000.0,),
            band_names=("band",),
        )
        base = StaticSource(torch.ones(2, 2, 1), geometry)
        signal = CallableDrivingSignal(lambda time: time + 10.0)
        delayed = DelayedModulatedSource(
            base,
            signal,
            torch.tensor([[0.0, 1.0], [2.0, float("nan")]]),
        )
        value = delayed.brightness([5.0])
        torch.testing.assert_close(
            value[0, :, :, 0],
            torch.tensor([[15.0, 14.0], [13.0, 0.0]]),
        )
        self.assertEqual(delayed.metadata()["valid_delay_pixels"], 3)

    def test_tabulated_signal_requires_explicit_extrapolation(self) -> None:
        signal = TabulatedDrivingSignal(
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 2.0]),
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            signal.amplitudes(
                [-1.0],
                bands=1,
                dtype=torch.float32,
                device="cpu",
            )

    def test_uniform_irs_returns_absolute_unit_magnification(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float64",
            ),
        )
        result = simulation.magnification_map(
            PlaneRegion((2.0, 2.0)),
            PlaneGrid((16, 16), (2.0, 2.0)),
            method=IRSConfig(
                rays=4096,
                ray_chunk_size=257,
                far_field_approx=FarFieldApproxConfig(enabled=False),
            ),
        )
        torch.testing.assert_close(
            result.values,
            torch.ones_like(result.values),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(result.metadata["actual_rays"], 4096)
        self.assertEqual(result.metadata["sampling"], "cartesian")
        self.assertIsNone(result.metadata["seed"])

    def test_random_irs_is_reproducible_and_chunk_invariant(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float64",
            ),
        )
        region = PlaneRegion((2.0, 2.0))
        grid = PlaneGrid((16, 16), (2.0, 2.0))
        first = simulation.magnification_map(
            region,
            grid,
            method=IRSConfig(rays=4097, sampling="random", seed=12, ray_chunk_size=257),
        )
        rechunked = simulation.magnification_map(
            region,
            grid,
            method=IRSConfig(rays=4097, sampling="random", seed=12, ray_chunk_size=1024),
        )
        different = simulation.magnification_map(
            region,
            grid,
            method=IRSConfig(rays=4097, sampling="random", seed=13, ray_chunk_size=257),
        )
        torch.testing.assert_close(first.values, rechunked.values, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(first.values, different.values))
        self.assertEqual(first.metadata["sampling"], "random")
        self.assertEqual(first.metadata["seed"], 12)
        self.assertEqual(first.metadata["actual_rays"], 4097)
        self.assertIsNone(first.metadata["ray_grid_shape"])
        pixel_area = grid.pixel_scale_uas[0] * grid.pixel_scale_uas[1]
        self.assertAlmostEqual(
            float(first.values.sum()) * pixel_area,
            4.0,
            places=12,
        )

    def test_random_irs_reuses_ray_coordinates_across_frames(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        maps = list(
            simulation.dynamic_maps(
                PlaneRegion((2.0, 2.0)),
                PlaneGrid((12, 12), (2.0, 2.0)),
                [0.0, 1.0, 2.0],
                method=IRSConfig(
                    rays=2048,
                    sampling="random",
                    seed=7,
                    ray_chunk_size=300,
                ),
            )
        )
        torch.testing.assert_close(maps[0].values, maps[1].values, rtol=0.0, atol=0.0)
        torch.testing.assert_close(maps[0].values, maps[2].values, rtol=0.0, atol=0.0)

    def test_random_irs_is_temporal_schedule_invariant(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.tensor([0.1]),
                torch.tensor([-0.2]),
                einstein_radius_uas=torch.tensor([0.08]),
                velocity_x_uas_per_day=torch.tensor([0.01]),
                velocity_y_uas_per_day=torch.tensor([-0.005]),
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        region = PlaneRegion((2.0, 2.5))
        grid = PlaneGrid((12, 15), (1.5, 2.0))
        method = IRSConfig(
            rays=4097,
            sampling="random",
            seed=8,
            ray_chunk_size=257,
        )

        def calculate(batch: int):
            return tuple(
                simulation.dynamic_maps(
                    region,
                    grid,
                    [0.0, 1.0, 2.0],
                    method=method,
                    schedule=DynamicConfig(temporal_batch_size=batch),
                )
            )

        scalar = calculate(1)
        batched = calculate(3)
        for first, second in zip(scalar, batched, strict=True):
            torch.testing.assert_close(first.values, second.values, rtol=0.0, atol=0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_random_irs_matches_cpu_and_cuda_and_rechunking(self) -> None:
        region = PlaneRegion((2.0, 2.0))
        grid = PlaneGrid((16, 16), (2.0, 2.0))

        def calculate(device: str, chunk: int):
            simulation = MicrolensingSimulation.create(
                MacroLens(convergence=0.0, shear=0.0),
                PointMassField._from_einstein_radii(
                    torch.empty(0, device=device),
                    torch.empty(0, device=device),
                    einstein_radius_uas=torch.empty(0, device=device),
                ),
                runtime=RuntimeConfig(
                    device=device,
                    backend=Backend.TORCH_EAGER,
                    dtype="float64",
                ),
            )
            return simulation.magnification_map(
                region,
                grid,
                method=IRSConfig(
                    rays=4097,
                    sampling="random",
                    seed=12,
                    ray_chunk_size=chunk,
                ),
            ).values.cpu()

        cpu = calculate("cpu", 257)
        cuda = calculate("cuda", 257)
        cuda_rechunked = calculate("cuda", 1024)
        torch.testing.assert_close(cuda, cpu, rtol=0.0, atol=0.0)
        torch.testing.assert_close(cuda_rechunked, cuda, rtol=0.0, atol=0.0)

    def test_irs_sampling_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "cartesian.*random"):
            IRSConfig(sampling="hexagonal")
        with self.assertRaisesRegex(TypeError, "seed"):
            IRSConfig(seed=True)

    def test_uniform_irs_accepts_eager_taylor_far_field(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        for sampling in ("cartesian", "random"):
            with self.subTest(sampling=sampling):
                result = simulation.magnification_map(
                    PlaneRegion((2.0, 2.0)),
                    PlaneGrid((8, 8), (2.0, 2.0)),
                    method=IRSConfig(
                        rays=1024,
                        sampling=sampling,
                        seed=5,
                        ray_chunk_size=128,
                        far_field_approx=FarFieldApproxConfig(
                            cells_per_axis=4,
                            nodes_per_cell_axis=2,
                        ),
                    ),
                )
                if sampling == "cartesian":
                    torch.testing.assert_close(
                        result.values, torch.ones_like(result.values)
                    )
                else:
                    self.assertAlmostEqual(float(result.values.mean()), 1.0, places=6)
                self.assertTrue(result.metadata["far_field"]["enabled"])

    def test_full_field_ipm_preserves_identity_magnification(self) -> None:
        from microcaustics import IPMConfig

        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float64",
            ),
        )
        result = simulation.magnification_map(
            PlaneRegion((2.0, 2.0)),
            PlaneGrid((4, 4), (2.0, 2.0)),
            method=IPMConfig(
                rays=16,
                refinement=2,
                virtual_refinement=4,
                tiled=False,
                cell_chunk_size=5,
                far_field_approx=FarFieldApproxConfig(enabled=False),
            ),
        )
        torch.testing.assert_close(
            result.values,
            torch.ones_like(result.values),
            rtol=0.0,
            atol=2.0e-14,
        )
        self.assertEqual(result.metadata["actual_base_cells"], 16)
        self.assertTrue(
            all(value >= 0.0 for value in result.timing.component_seconds.values())
        )

    def test_ipm_production_lattice_is_square_for_square_field(self) -> None:
        """The 10M, k=2 paper setting must retain its 3162-square lattice."""

        from microcaustics.solvers.ipm import _ipm_cell_shape

        self.assertEqual(
            _ipm_cell_shape(
                10_000_000,
                PlaneRegion((100.0, 100.0)),
                scout_ratio=2,
            ),
            (3162, 3162),
        )

    def test_interpolated_nodes_recover_quadratic_surface(self) -> None:
        from microcaustics.solvers import interpolated_nodes

        coarse = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
        yy, xx = torch.meshgrid(coarse, coarse, indexing="ij")
        node_x = (2.0 * xx.square() + 0.5 * yy + xx * yy)[None]
        node_y = (-yy.square() + 0.25 * xx)[None]
        fine_x, fine_y = interpolated_nodes(
            node_x,
            node_y,
            virtual_refinement=8,
        )
        fine = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
        fine_yy, fine_xx = torch.meshgrid(fine, fine, indexing="ij")
        torch.testing.assert_close(
            fine_x[0],
            2.0 * fine_xx.square() + 0.5 * fine_yy + fine_xx * fine_yy,
            rtol=0.0,
            atol=3.0e-15,
        )
        torch.testing.assert_close(
            fine_y[0],
            -fine_yy.square() + 0.25 * fine_xx,
            rtol=0.0,
            atol=3.0e-15,
        )

    def test_generic_ipm_interpolation_recovers_cubic_surface(self) -> None:
        from microcaustics.solvers import interpolated_nodes

        traced = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
        yy, xx = torch.meshgrid(traced, traced, indexing="ij")
        node_x = (xx**3 - 0.4 * yy**2 + xx * yy)[None]
        node_y = (0.5 * yy**3 + xx**2 * yy)[None]
        fine_x, fine_y = interpolated_nodes(
            node_x,
            node_y,
            virtual_refinement=7,
        )
        fine = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
        fine_yy, fine_xx = torch.meshgrid(fine, fine, indexing="ij")
        torch.testing.assert_close(
            fine_x[0],
            fine_xx**3 - 0.4 * fine_yy**2 + fine_xx * fine_yy,
            rtol=0.0,
            atol=4.0e-15,
        )
        torch.testing.assert_close(
            fine_y[0],
            0.5 * fine_yy**3 + fine_xx**2 * fine_yy,
            rtol=0.0,
            atol=4.0e-15,
        )

    def test_full_field_ipm_accepts_general_r_and_v(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        for refinement, virtual in ((1, 1), (3, 5)):
            with self.subTest(refinement=refinement, virtual=virtual):
                result = simulation.magnification_map(
                    PlaneRegion((2.0, 2.0)),
                    PlaneGrid((3, 5), (2.0, 2.0)),
                    method=IPMConfig(
                        rays=6,
                        refinement=refinement,
                        virtual_refinement=virtual,
                        tiled=False,
                        cell_chunk_size=3,
                        far_field_approx=FarFieldApproxConfig(enabled=False),
                    ),
                )
                torch.testing.assert_close(
                    result.values,
                    torch.ones_like(result.values),
                    rtol=0.0,
                    atol=2.0e-6,
                )

    def test_tiled_ipm_accepts_general_scout_ratio(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        for scout_ratio in (1, 2, 3):
            with self.subTest(scout_ratio=scout_ratio):
                result = simulation.magnification_map(
                    PlaneRegion((2.0, 3.0)),
                    PlaneGrid((4, 6), (2.0, 3.0)),
                    method=IPMConfig(
                        rays=24,
                        scout_ratio=scout_ratio,
                        refinement=1,
                        virtual_refinement=1,
                        tiled=True,
                        scout_halo_pixels=0.0,
                        scout_dilation_cells=0,
                        cell_chunk_size=7,
                        far_field_approx=FarFieldApproxConfig(enabled=False),
                    ),
                )
                torch.testing.assert_close(
                    result.values,
                    torch.ones_like(result.values),
                    rtol=0.0,
                    atol=2.0e-6,
                )
                self.assertEqual(result.metadata["scout_ratio"], scout_ratio)

    def test_compact_sparse_ipm_matches_cell_local_nodes(self) -> None:
        """Deduplicating shared nodes must preserve the portable IPM map."""

        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.17, shear=0.06),
            PointMassField._from_einstein_radii(
                torch.tensor([-0.45, 0.35], dtype=torch.float64),
                torch.tensor([0.25, -0.15], dtype=torch.float64),
                einstein_radius_uas=torch.tensor([0.18, 0.12], dtype=torch.float64),
            ),
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float64",
            ),
        )
        common = dict(
            rays=64,
            scout_ratio=2,
            refinement=3,
            virtual_refinement=5,
            tiled=True,
            scout_trace_centers=False,
            scout_dilation_cells=1,
            cell_chunk_size=7,
            far_field_approx=FarFieldApproxConfig(enabled=False),
        )
        lens_region = PlaneRegion((3.0, 3.0))
        source_grid = PlaneGrid((7, 8), (1.4, 1.6))
        dense = simulation.magnification_map(
            lens_region,
            source_grid,
            method=IPMConfig(compact_sparse_nodes=False, **common),
        )
        compact = simulation.magnification_map(
            lens_region,
            source_grid,
            method=IPMConfig(compact_sparse_nodes=True, **common),
        )
        torch.testing.assert_close(compact.values, dense.values, rtol=0.0, atol=0.0)
        self.assertTrue(compact.metadata["compact_sparse_nodes"])
        self.assertLess(
            compact.metadata["compact_unique_nodes"],
            compact.metadata["selected_fine_cells"] * 16,
        )

    def test_compact_sparse_temporal_ipm_matches_cell_local_nodes(self) -> None:
        """The fused temporal scheduler must preserve compact-node parity."""

        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.12, shear=0.04),
            PointMassField._from_einstein_radii(
                torch.tensor([-0.4, 0.3]),
                torch.tensor([0.2, -0.1]),
                velocity_x_uas_per_day=torch.tensor([0.002, -0.001]),
                velocity_y_uas_per_day=torch.tensor([-0.001, 0.0015]),
                einstein_radius_uas=torch.tensor([0.16, 0.11]),
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        common = dict(
            rays=36,
            scout_ratio=2,
            refinement=2,
            virtual_refinement=4,
            tiled=True,
            scout_trace_centers=False,
            cell_chunk_size=11,
            far_field_approx=FarFieldApproxConfig(enabled=False),
        )
        schedule = DynamicConfig(
            temporal_batch_size=3,
            scout_refresh_frames=3,
            fused_temporal_ipm=True,
        )
        arguments = (
            PlaneRegion((3.0, 3.0)),
            PlaneGrid((5, 6), (1.2, 1.4)),
            [0.0, 1.0, 2.0],
        )
        dense = list(
            simulation.dynamic_maps(
                *arguments,
                method=IPMConfig(compact_sparse_nodes=False, **common),
                schedule=schedule,
            )
        )
        compact = list(
            simulation.dynamic_maps(
                *arguments,
                method=IPMConfig(compact_sparse_nodes=True, **common),
                schedule=schedule,
            )
        )
        for actual, expected in zip(compact, dense, strict=True):
            torch.testing.assert_close(
                actual.values,
                expected.values,
                rtol=2.0e-7,
                atol=2.0e-7,
            )
            self.assertTrue(actual.metadata["compact_sparse_nodes"])

    def test_tiled_scout_matches_full_field_for_linear_mapping(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.2, shear=0.1, shear_angle_deg=17.188733853924695),
            PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        lens_region = PlaneRegion((6.0, 6.0))
        source_grid = PlaneGrid((8, 10), (2.0, 2.5))
        common = dict(
            rays=144,
            refinement=1,
            virtual_refinement=1,
            scout_halo_pixels=1.0,
            scout_dilation_cells=1,
            cell_chunk_size=48,
            far_field_approx=FarFieldApproxConfig(enabled=False),
        )
        full = simulation.magnification_map(
            lens_region,
            source_grid,
            method=IPMConfig(tiled=False, **common),
        )
        for scout_ratio in (1, 2, 3):
            with self.subTest(scout_ratio=scout_ratio):
                tiled = simulation.magnification_map(
                    lens_region,
                    source_grid,
                    method=IPMConfig(
                        tiled=True,
                        scout_ratio=scout_ratio,
                        **common,
                    ),
                )
                torch.testing.assert_close(
                    tiled.values,
                    full.values,
                    rtol=0.0,
                    atol=3.0e-6,
                )

    def test_einstein_radius_scales_as_square_root_of_mass(self) -> None:
        distances = LensingDistances(
            lens_m=1.0e25,
            source_m=2.0e25,
            lens_to_source_m=1.2e25,
        )
        radii = distances.einstein_radius_uas(
            torch.tensor([1.0, 4.0], dtype=torch.float64),
            dtype=torch.float64,
        )
        self.assertAlmostEqual(float(radii[1] / radii[0]), 2.0, places=14)
        field = (
            PointMassField(
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 4.0],
            )
            .to(dtype=torch.float64)
            .resolve(distances)
        )
        torch.testing.assert_close(field.einstein_radius_uas, radii)
        self.assertIsNotNone(field.mass_solar)

    def test_direct_jacobian_matches_finite_difference(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(
                convergence=0.3,
                shear=0.12,
                shear_angle_deg=13.178029288008934,
                smooth_matter_fraction=0.4,
            ),
            PointMassField._from_einstein_radii(
                torch.tensor([-0.4, 0.7], dtype=torch.float64),
                torch.tensor([0.2, -0.5], dtype=torch.float64),
                einstein_radius_uas=torch.tensor([0.15, 0.2], dtype=torch.float64),
            ),
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float64",
            ),
        )
        x = torch.tensor(1.3, dtype=torch.float64)
        y = torch.tensor(0.9, dtype=torch.float64)
        determinant, _ = simulation.jacobian_determinant_direct(x, y)
        step = 1.0e-5
        bx_plus, by_plus, _ = simulation.raytrace_direct(x + step, y)
        bx_minus, by_minus, _ = simulation.raytrace_direct(x - step, y)
        bx_up, by_up, _ = simulation.raytrace_direct(x, y + step)
        bx_down, by_down, _ = simulation.raytrace_direct(x, y - step)
        d_bx_dx = (bx_plus - bx_minus) / (2.0 * step)
        d_by_dx = (by_plus - by_minus) / (2.0 * step)
        d_bx_dy = (bx_up - bx_down) / (2.0 * step)
        d_by_dy = (by_up - by_down) / (2.0 * step)
        finite_difference = d_bx_dx * d_by_dy - d_bx_dy * d_by_dx
        torch.testing.assert_close(
            determinant,
            finite_difference,
            rtol=1.0e-9,
            atol=1.0e-10,
        )

    def test_single_point_lens_critical_curve_and_caustic(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.tensor([0.0], dtype=torch.float64),
                torch.tensor([0.0], dtype=torch.float64),
                einstein_radius_uas=torch.tensor([1.0], dtype=torch.float64),
            ),
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float64",
            ),
        )
        result = simulation.caustics(
            PlaneGrid((129, 129), (4.0, 4.0)),
            ray_chunk_size=4096,
        )
        self.assertGreater(result.segment_count, 100)
        critical_radius = torch.linalg.vector_norm(
            result.critical_segments_uas.reshape(-1, 2),
            dim=1,
        )
        self.assertLess(abs(float(critical_radius.mean()) - 1.0), 2.0e-3)
        source_radius = torch.linalg.vector_norm(
            result.caustic_segments_uas.reshape(-1, 2),
            dim=1,
        )
        self.assertLess(float(source_radius.mean()), 2.0e-3)
        accelerated = simulation.caustics(
            PlaneGrid((129, 129), (4.0, 4.0)),
            far_field_approx=FarFieldApproxConfig(
                cells_per_axis=4,
                nodes_per_cell_axis=4,
                exact_radius_cells=1.0,
            ),
        )
        self.assertEqual(accelerated.segment_count, result.segment_count)
        torch.testing.assert_close(
            accelerated.caustic_segments_uas,
            result.caustic_segments_uas,
            rtol=0.0,
            atol=2.0e-6,
        )

    def test_single_point_lens_ipm_map_matches_analytic_magnification(self) -> None:
        """Recover the exact point-lens magnification away from its pole."""

        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                einstein_radius_uas=torch.tensor([1.0]),
            ),
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float32",
            ),
        )
        source_grid = PlaneGrid((24, 24), (4.0, 4.0))
        numerical = simulation.magnification_map(
            PlaneRegion((6.0, 6.0)),
            source_grid,
            method=IPMConfig(
                rays=4096,
                scout_ratio=1,
                refinement=2,
                virtual_refinement=4,
                tiled=False,
                cell_chunk_size=4096,
                far_field_approx=FarFieldApproxConfig(enabled=False),
            ),
        ).values
        x, y = source_grid.mesh(dtype=numerical.dtype)
        separation = torch.sqrt(x.square() + y.square())
        analytic = (separation.square() + 2.0) / (
            separation * torch.sqrt(separation.square() + 4.0)
        )
        # The point-source solution diverges at the origin, while a pixelized
        # map stores a cell average. The outer cut also avoids finite-aperture
        # boundary cells. Neither cut is tuned to the numerical result.
        audit = (separation > 0.3) & (separation < 1.8)
        relative = (numerical[audit] - analytic[audit]).abs() / analytic[audit]
        self.assertLess(float(relative.median()), 1.0e-3)
        self.assertLess(float(torch.quantile(relative, 0.95)), 6.0e-3)

    def test_half_open_caustic_crossing_labels(self) -> None:
        from microcaustics.results import CausticField

        square = torch.tensor(
            [
                [[-1.0, -1.0], [1.0, -1.0]],
                [[1.0, -1.0], [1.0, 1.0]],
                [[1.0, 1.0], [-1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, -1.0]],
            ],
            dtype=torch.float64,
        )
        field = CausticField(
            critical_segments_uas=square,
            caustic_segments_uas=square,
            lens_grid=PlaneGrid((2, 2), (2.0, 2.0)),
        )
        labels = field.crossing_parity(
            torch.tensor(
                [[0.0, 0.0], [2.0, 0.0], [0.0, 1.5], [0.0, -1.0]],
                dtype=torch.float64,
            ),
            point_chunk_size=2,
            segment_chunk_size=1,
        )
        torch.testing.assert_close(
            labels,
            torch.tensor([1, 0, 0, 1], dtype=torch.int8),
            rtol=0,
            atol=0,
        )
        label_map = field.label_map(PlaneGrid((3, 3), (3.0, 3.0)))
        self.assertEqual(int(label_map.values[1, 1]), 1)
        distances = field.distance(
            torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64),
            point_chunk_size=1,
            segment_chunk_size=2,
        )
        torch.testing.assert_close(
            distances,
            torch.ones(2, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        distance_map = field.distance_map(PlaneGrid((3, 3), (3.0, 3.0)))
        self.assertEqual(float(distance_map.values_uas[1, 1]), 1.0)

    def test_caustic_discovery_controls_validate(self) -> None:
        from microcaustics import CausticConfig

        config = CausticConfig(discovery_downsample_ratio=8)
        self.assertEqual(config.discovery_downsample_ratio, 8)
        with self.assertRaisesRegex(ValueError, "discovery_downsample_ratio"):
            CausticConfig(discovery_downsample_ratio=0)
        with self.assertRaisesRegex(ValueError, "discovery_near_zero_quantile"):
            CausticConfig(discovery_near_zero_quantile=1.1)
        with self.assertRaisesRegex(ValueError, "discovery_dilation_cells"):
            CausticConfig(discovery_dilation_cells=-1)

    def test_production_anchor_gauge_labels_support_shifted_source_region(self) -> None:
        from microcaustics import CausticConfig
        from microcaustics.caustics import label_caustic_fields
        from microcaustics.results import CausticField

        square = torch.tensor(
            [
                [[-1.0, -1.0], [1.0, -1.0]],
                [[1.0, -1.0], [1.0, 1.0]],
                [[1.0, 1.0], [-1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, -1.0]],
            ],
            dtype=torch.float64,
        )
        shift = torch.tensor([2.0, 3.0], dtype=torch.float64)
        field = CausticField(
            square + shift,
            square + shift,
            PlaneGrid((4, 4), (6.0, 6.0), (3.0, 2.0)),
        )
        outputs, gauges, distances, center_label, center_distance = (
            label_caustic_fields(
                (field,),
                PlaneRegion((6.0, 6.0), (3.0, 2.0)),
                CausticConfig(),
            )
        )
        labels = outputs[0].labels
        self.assertEqual(labels.center_label, 1)
        self.assertEqual(labels.center_vote_count, 9)
        self.assertTrue(torch.equal(gauges, torch.zeros(9, dtype=torch.int8)))
        self.assertEqual(tuple(distances.shape), (9,))
        self.assertEqual(center_label, 1)
        self.assertAlmostEqual(center_distance, 1.0, places=12)

    def test_missing_in_field_caustic_distance_is_capped_at_source_radius(self) -> None:
        from microcaustics import CausticConfig
        from microcaustics.caustics import label_caustic_fields
        from microcaustics.results import CausticField

        empty = torch.empty((0, 2, 2), dtype=torch.float64)
        field = CausticField(
            empty,
            empty,
            PlaneGrid((4, 4), (8.0, 6.0)),
        )
        outputs, _, _, _, center_distance = label_caustic_fields(
            (field,),
            PlaneRegion((8.0, 6.0)),
            CausticConfig(),
        )
        labels = outputs[0].labels
        self.assertEqual(labels.center_distance_uas, 3.0)
        self.assertTrue(labels.center_distance_censored)
        self.assertEqual(center_distance, 3.0)

    def test_anchor_and_gauge_counts_are_independently_configurable(self) -> None:
        from microcaustics import CausticConfig
        from microcaustics.caustics import label_caustic_fields
        from microcaustics.results import CausticField

        square = torch.tensor(
            [
                [[-1.0, -1.0], [1.0, -1.0]],
                [[1.0, -1.0], [1.0, 1.0]],
                [[1.0, 1.0], [-1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, -1.0]],
            ],
            dtype=torch.float64,
        )
        field = CausticField(
            square,
            square,
            PlaneGrid((4, 4), (6.0, 6.0)),
        )
        for anchor_count, gauge_count in ((5, 7), (12, 16)):
            outputs, *_ = label_caustic_fields(
                (field,),
                PlaneRegion((6.0, 6.0)),
                CausticConfig(
                    anchor_count=anchor_count,
                    gauge_count=gauge_count,
                    minimum_alignment_gauges=min(3, gauge_count),
                ),
            )
            labels = outputs[0].labels
            self.assertEqual(labels.center_label, 1)
            self.assertEqual(tuple(labels.anchor_points_uas.shape), (anchor_count, 2))
            self.assertEqual(tuple(labels.gauge_points_uas.shape), (gauge_count, 2))
            self.assertEqual(tuple(labels.gauge_labels.shape), (gauge_count,))
            self.assertEqual(labels.center_valid_count, anchor_count)

    def test_production_gauges_remove_temporal_global_xor(self) -> None:
        from microcaustics import CausticConfig
        from microcaustics.caustics import label_caustic_fields
        from microcaustics.results import CausticField

        empty = torch.empty((0, 2, 2), dtype=torch.float64)
        radius = 2.8
        surrounding = torch.tensor(
            [
                [[-radius, -radius], [radius, -radius]],
                [[radius, -radius], [radius, radius]],
                [[radius, radius], [-radius, radius]],
                [[-radius, radius], [-radius, -radius]],
            ],
            dtype=torch.float64,
        )
        grid = PlaneGrid((4, 4), (6.0, 6.0))
        fields = (
            CausticField(empty, empty, grid),
            CausticField(surrounding, surrounding, grid),
        )
        outputs, *_ = label_caustic_fields(
            fields,
            PlaneRegion((6.0, 6.0)),
            CausticConfig(),
        )
        self.assertEqual(outputs[1].labels.raw_center_label, 1)
        self.assertEqual(outputs[1].labels.frame_xor, 1)
        self.assertEqual(outputs[1].labels.center_label, 0)
        self.assertFalse(outputs[1].labels.center_crossing)

    def test_open_component_crossings_invalidate_only_affected_votes(self) -> None:
        from microcaustics import CausticConfig
        from microcaustics.caustics import label_caustic_fields
        from microcaustics.results import CausticField

        square = torch.tensor(
            [
                [[-1.0, -1.0], [1.0, -1.0]],
                [[1.0, -1.0], [1.0, 1.0]],
                [[1.0, 1.0], [-1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, -1.0]],
            ],
            dtype=torch.float64,
        )
        field = CausticField(
            square,
            square,
            PlaneGrid((4, 4), (6.0, 6.0)),
            invalid_segment_mask=torch.tensor([True, False, False, False]),
        )
        outputs, *_ = label_caustic_fields(
            (field,),
            PlaneRegion((6.0, 6.0)),
            CausticConfig(),
        )
        labels = outputs[0].labels
        self.assertEqual(labels.center_label, 1)
        self.assertGreater(labels.center_valid_count, 0)
        self.assertLess(labels.center_valid_count, 9)
        self.assertEqual(labels.center_vote_count, labels.center_valid_count)

    def test_boundary_connected_critical_components_are_invalid_for_labels(
        self,
    ) -> None:
        from microcaustics.caustics.production import _boundary_component_mask

        critical = torch.tensor(
            [
                [[-1.6, 0.0], [-1.0, 0.0]],
                [[-1.0, 0.0], [-0.5, 0.5]],
                [[0.2, 0.2], [0.7, 0.2]],
                [[0.7, 0.2], [0.7, 0.7]],
                [[0.7, 0.7], [0.2, 0.7]],
                [[0.2, 0.7], [0.2, 0.2]],
            ],
            dtype=torch.float64,
        )
        invalid = _boundary_component_mask(
            critical,
            PlaneGrid((5, 5), (4.0, 4.0)),
        )
        torch.testing.assert_close(
            invalid,
            torch.tensor([True, True, False, False, False, False]),
        )

    def test_local_determinant_cleanup_removes_only_unresolved_speckle(self) -> None:
        from microcaustics.caustics.production import _clean_small_sign_islands

        determinant = torch.ones((9, 9), dtype=torch.float64)
        determinant[2, 2] = -1.0
        determinant[5:7, 5:7] = -1.0
        cleaned = _clean_small_sign_islands(determinant, 4)
        self.assertEqual(float(cleaned[2, 2]), 1.0)
        torch.testing.assert_close(
            cleaned[5:7, 5:7],
            -torch.ones((2, 2), dtype=torch.float64),
        )

    def test_production_caustic_pipeline_returns_aligned_frames(self) -> None:
        from microcaustics import CausticConfig

        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.tensor([0.0], dtype=torch.float64),
                torch.tensor([0.0], dtype=torch.float64),
                einstein_radius_uas=torch.tensor([1.0], dtype=torch.float64),
            ),
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float64",
            ),
        )
        results = simulation.dynamic_labeled_caustics(
            PlaneGrid((33, 33), (4.0, 4.0)),
            PlaneRegion((1.0, 1.0)),
            [0.0, 1.0],
            config=CausticConfig(
                far_field_approx=FarFieldApproxConfig(
                    cells_per_axis=4,
                    nodes_per_cell_axis=4,
                    exact_radius_cells=1.0,
                ),
                temporal_batch_size=2,
                jacobian_chunk_size=257,
            ),
        )
        self.assertEqual(len(results), 2)
        self.assertGreater(results[0].caustics.segment_count, 20)
        self.assertEqual(results[0].labels.center_valid_count, 9)
        self.assertEqual(results[1].labels.center_label, results[0].labels.center_label)
        self.assertFalse(results[1].labels.center_crossing)

    def test_production_labeled_light_curve_reuses_temporal_far_fields(self) -> None:
        from microcaustics import CausticConfig

        far_field = FarFieldApproxConfig(
            cells_per_axis=4,
            nodes_per_cell_axis=4,
            exact_radius_cells=1.0,
        )
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                velocity_x_uas_per_day=torch.tensor([1.0e-3]),
                velocity_y_uas_per_day=torch.tensor([-2.0e-3]),
                einstein_radius_uas=torch.tensor([0.4]),
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        source = StaticSource(
            torch.ones((3, 3, 1)),
            SourceGeometry((3, 3), (1.0e10, 1.0e10), (5000.0,), ("x",)),
        )
        result = simulation.light_curve_with_labels(
            PlaneRegion((4.0, 4.0)),
            PlaneGrid((12, 12), (2.0, 2.0)),
            PlaneGrid((33, 33), (4.0, 4.0)),
            [0.0, 1.0],
            source,
            LensingDistances(1.0e25, 2.0e25, 1.0e25),
            method=IPMConfig(
                rays=64,
                scout_ratio=2,
                refinement=1,
                virtual_refinement=1,
                tiled=True,
                far_field_approx=far_field,
            ),
            map_schedule=DynamicConfig(
                temporal_batch_size=2,
                scout_refresh_frames=2,
            ),
            caustic_config=CausticConfig(
                far_field_approx=far_field,
                temporal_batch_size=2,
                jacobian_chunk_size=257,
            ),
        )
        self.assertEqual(tuple(result.light_curve.flux.shape), (2, 1))
        self.assertEqual(tuple(result.crossing_labels.shape), (2,))
        self.assertTrue(
            all(
                frame.caustics.metadata["shared_far_fields"]
                for frame in result.caustics
            )
        )
        self.assertTrue(
            all(
                frame.caustics.metadata["scout_sparse_determinant"]
                for frame in result.caustics
            )
        )
        self.assertTrue(
            all(
                0.0 < frame.caustics.metadata["determinant_grid_fraction"] <= 1.0
                and frame.caustics.metadata["sparse_unique_vertices"] is not None
                for frame in result.caustics
            )
        )
        self.assertTrue(
            all(
                frame.caustics.metadata[
                    "critical_discovery_downsample_ratio"
                ]
                == 16
                and frame.caustics.metadata[
                    "critical_discovery_source_active_after"
                ]
                <= frame.caustics.metadata[
                    "critical_discovery_source_active_before"
                ]
                for frame in result.caustics
            )
        )

        observed = []
        multirate = simulation.multirate_light_curve_with_labels(
            PlaneRegion((4.0, 4.0)),
            PlaneGrid((12, 12), (2.0, 2.0)),
            PlaneGrid((33, 33), (4.0, 4.0)),
            [0.0, 1.0],
            [0.0, 0.5, 1.0],
            source,
            LensingDistances(1.0e25, 2.0e25, 1.0e25),
            method=IPMConfig(
                rays=64,
                scout_ratio=2,
                refinement=1,
                virtual_refinement=1,
                tiled=True,
                far_field_approx=far_field,
            ),
            map_schedule=DynamicConfig(
                temporal_batch_size=2,
                scout_refresh_frames=2,
            ),
            caustic_config=CausticConfig(
                far_field_approx=far_field,
                temporal_batch_size=2,
                jacobian_chunk_size=257,
                discovery_downsample_ratio=8,
            ),
            map_observer=lambda index, frame: observed.append((index, frame.time_days)),
        )
        self.assertEqual(tuple(multirate.light_curve.flux.shape), (3, 1))
        self.assertEqual(multirate.map_times_days.tolist(), [0.0, 1.0])
        self.assertEqual(observed, [(0, 0.0), (1, 1.0)])
        self.assertTrue(
            all(
                frame.caustics.metadata["shared_far_fields"]
                for frame in multirate.caustics
            )
        )
        self.assertTrue(
            all(
                frame.caustics.metadata[
                    "critical_discovery_downsample_ratio"
                ]
                == 8
                for frame in multirate.caustics
            )
        )

    def test_signed_winding_number_respects_orientation(self) -> None:
        from microcaustics.caustics import winding_number

        vertices = torch.tensor(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ],
            dtype=torch.float64,
        )
        counterclockwise = torch.stack((vertices[:-1], vertices[1:]), dim=1)
        points = torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [0.0, -1.0]],
            dtype=torch.float64,
        )
        positive = winding_number(counterclockwise, points)
        negative = winding_number(counterclockwise.flip(dims=(1,)), points)
        self.assertEqual(positive.tolist(), [1, 0, 1])
        self.assertEqual(negative.tolist(), [-1, 0, -1])

    def test_regular_grid_winding_and_parity_match_point_queries(self) -> None:
        """The scanline diagnostic maps preserve the pointwise predicates."""

        from microcaustics.results import CausticField

        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                generator = torch.Generator().manual_seed(1729)
                segments = 4.0 * torch.rand(
                    (257, 2, 2), generator=generator, dtype=dtype
                ) - 2.0
                grid = PlaneGrid((31, 29), (4.5, 4.25), (0.13, -0.17))
                field = CausticField(segments, segments, grid)
                x, y = grid.mesh(dtype=dtype)
                points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
                expected_winding = field.winding_number(
                    points,
                    point_chunk_size=37,
                    segment_chunk_size=41,
                ).reshape(grid.shape)
                expected_parity = field.crossing_parity(
                    points,
                    point_chunk_size=37,
                    segment_chunk_size=41,
                ).reshape(grid.shape)
                actual_winding = field.winding_map(
                    grid,
                    segment_chunk_size=43,
                ).values
                actual_parity = field.label_map(
                    grid,
                    segment_chunk_size=43,
                ).values
                torch.testing.assert_close(actual_winding, expected_winding)
                torch.testing.assert_close(actual_parity, expected_parity)

    def test_unordered_closed_segments_are_registered_in_core(self) -> None:
        from microcaustics.caustics import orient_mapped_closed_segments

        vertices = torch.tensor(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]],
            dtype=torch.float64,
        )
        segments = torch.stack((vertices[:-1], vertices[1:]), dim=1)
        shuffled = segments[torch.tensor([2, 0, 3, 1])].flip(1)
        critical, caustic = orient_mapped_closed_segments(shuffled, 2.0 * shuffled)
        field = CausticField(
            critical,
            caustic,
            PlaneGrid((8, 8), (4.0, 4.0)),
        )
        self.assertEqual(field.center_winding_number(), 1)
        positive_map = field.winding_map(
            PlaneGrid((8, 8), (4.0, 4.0)), orientation="positive"
        )
        self.assertGreaterEqual(int(positive_map.values.min()), 0)

    def test_marching_segments_form_an_oriented_closed_contour(self) -> None:
        """Marching output must be usable directly for signed winding maps."""

        from microcaustics.caustics import (
            crossing_parity,
            marching_squares_zero,
            winding_number,
        )

        axis = torch.linspace(-2.0, 2.0, 129, dtype=torch.float64)
        y_grid, x_grid = torch.meshgrid(axis, axis, indexing="ij")
        field = x_grid.square() + y_grid.square() - 1.0
        segments = marching_squares_zero(field, x_grid, y_grid)
        points = torch.tensor(
            [[0.0, 0.0], [0.75, 0.0], [1.25, 0.0], [1.75, 1.75]],
            dtype=torch.float64,
        )
        signed = winding_number(segments, points)
        parity = crossing_parity(segments, points)
        self.assertEqual(signed.tolist(), [1, 1, 0, 0])
        self.assertEqual(parity.tolist(), [1, 1, 0, 0])
        torch.testing.assert_close(
            torch.remainder(signed, 2).to(parity.dtype), parity, rtol=0, atol=0
        )

    def test_dynamic_irs_streams_moving_lens_frames(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                velocity_x_uas_per_day=torch.tensor([0.1]),
                velocity_y_uas_per_day=torch.tensor([0.0]),
                einstein_radius_uas=torch.tensor([0.2]),
            ),
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float32",
            ),
        )
        maps = list(
            simulation.dynamic_maps(
                PlaneRegion((2.0, 2.0)),
                PlaneGrid((8, 8), (1.0, 1.0)),
                [0.0, 10.0],
                method=IRSConfig(
                    rays=4096,
                    ray_chunk_size=1024,
                    far_field_approx=FarFieldApproxConfig(enabled=False),
                ),
            )
        )
        self.assertEqual([item.time_days for item in maps], [0.0, 10.0])
        self.assertFalse(torch.equal(maps[0].values, maps[1].values))

    def test_dynamic_static_lens_reuses_one_exact_map(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        maps = list(
            simulation.dynamic_maps(
                PlaneRegion((2.0, 2.0)),
                PlaneGrid((4, 4), (2.0, 2.0)),
                [0.0, 2.0, 5.0],
                method=IPMConfig(
                    rays=16,
                    scout_ratio=2,
                    refinement=1,
                    virtual_refinement=1,
                    tiled=True,
                    far_field_approx=FarFieldApproxConfig(enabled=False),
                ),
            )
        )
        self.assertEqual(len(maps), 3)
        self.assertTrue(all(item.metadata["dynamic_static_map_reuse"] for item in maps))
        self.assertEqual(len({item.values.data_ptr() for item in maps}), 1)

    def test_dynamic_tiled_ipm_records_endpoint_union(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                velocity_x_uas_per_day=torch.tensor([0.01]),
                velocity_y_uas_per_day=torch.tensor([0.0]),
                einstein_radius_uas=torch.tensor([0.2]),
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        maps = list(
            simulation.dynamic_maps(
                PlaneRegion((3.0, 3.0)),
                PlaneGrid((4, 4), (1.0, 1.0)),
                [0.0, 1.0, 2.0],
                method=IPMConfig(
                    rays=36,
                    scout_ratio=2,
                    refinement=1,
                    virtual_refinement=1,
                    tiled=True,
                    cell_chunk_size=32,
                    far_field_approx=FarFieldApproxConfig(enabled=False),
                ),
                schedule=DynamicConfig(
                    temporal_batch_size=2,
                    scout_refresh_frames=3,
                ),
            )
        )
        self.assertEqual(len(maps), 3)
        self.assertTrue(
            all(item.metadata["dynamic_scout_reuse_approximate"] for item in maps)
        )
        self.assertTrue(
            all(item.metadata["dynamic_scout_endpoint_union"] for item in maps)
        )
        self.assertEqual(maps[0].metadata["dynamic_scout_anchor_last"], 2)

    def test_fused_temporal_ipm_matches_scalar_scheduler(self) -> None:
        """The temporal contract must not change portable IPM map values."""

        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.15, shear=0.08),
            PointMassField._from_einstein_radii(
                torch.tensor([-0.5, 0.4]),
                torch.tensor([0.3, -0.2]),
                velocity_x_uas_per_day=torch.tensor([0.002, -0.001]),
                velocity_y_uas_per_day=torch.tensor([-0.001, 0.0015]),
                einstein_radius_uas=torch.tensor([0.25, 0.18]),
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        lens_region = PlaneRegion((3.0, 3.0))
        source_grid = PlaneGrid((5, 6), (1.2, 1.4))
        method = IPMConfig(
            rays=36,
            scout_ratio=2,
            refinement=2,
            virtual_refinement=4,
            tiled=True,
            scout_trace_centers=True,
            cell_chunk_size=17,
            far_field_approx=FarFieldApproxConfig(enabled=False),
        )
        common = dict(
            temporal_batch_size=2,
            scout_refresh_frames=3,
        )
        fused = list(
            simulation.dynamic_maps(
                lens_region,
                source_grid,
                [0.0, 1.0, 2.0],
                method=method,
                schedule=DynamicConfig(fused_temporal_ipm=True, **common),
            )
        )
        scalar = list(
            simulation.dynamic_maps(
                lens_region,
                source_grid,
                [0.0, 1.0, 2.0],
                method=method,
                schedule=DynamicConfig(fused_temporal_ipm=False, **common),
            )
        )
        for actual, expected in zip(fused, scalar, strict=True):
            # The two schedules reduce otherwise identical float32
            # contributions in a different order.  CPU vector libraries may
            # therefore differ by one ULP even though the numerical contract
            # is unchanged.
            tolerance = 2.0 * torch.finfo(actual.values.dtype).eps
            torch.testing.assert_close(
                actual.values,
                expected.values,
                rtol=tolerance,
                atol=tolerance,
            )
            self.assertEqual(
                actual.metadata["temporal_batch_real_frames"],
                2 if actual.time_days < 2 else 1,
            )
            self.assertFalse(actual.metadata["dynamic_temporal_solver_fused"])

    def test_fused_scout_intervals_remain_global_across_batch_boundaries(self) -> None:
        """Batch boundaries must not redefine endpoint-reuse intervals."""

        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                velocity_x_uas_per_day=torch.tensor([0.01]),
                velocity_y_uas_per_day=torch.tensor([0.0]),
                einstein_radius_uas=torch.tensor([0.2]),
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        maps = list(
            simulation.dynamic_maps(
                PlaneRegion((3.0, 3.0)),
                PlaneGrid((4, 4), (1.0, 1.0)),
                list(range(7)),
                method=IPMConfig(
                    rays=36,
                    scout_ratio=2,
                    refinement=1,
                    virtual_refinement=1,
                    tiled=True,
                    cell_chunk_size=32,
                    far_field_approx=FarFieldApproxConfig(enabled=False),
                ),
                schedule=DynamicConfig(
                    temporal_batch_size=4,
                    scout_refresh_frames=3,
                ),
            )
        )
        self.assertEqual(
            maps[0].metadata["dynamic_scout_interval_pairs"], [(0, 2), (3, 5)]
        )
        self.assertEqual(
            maps[4].metadata["dynamic_scout_interval_pairs"], [(3, 5), (6, 6)]
        )
        self.assertEqual(maps[4].metadata["dynamic_scout_anchor_frames"], [3, 5, 6])

    def test_fused_tiled_refresh_one_uses_exact_per_frame_scouts(self) -> None:
        """A refresh of one remains fused but introduces no scout reuse."""

        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                velocity_x_uas_per_day=torch.tensor([0.01]),
                velocity_y_uas_per_day=torch.tensor([0.0]),
                einstein_radius_uas=torch.tensor([0.2]),
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        maps = list(
            simulation.dynamic_maps(
                PlaneRegion((3.0, 3.0)),
                PlaneGrid((4, 4), (1.0, 1.0)),
                [0.0, 1.0, 2.0],
                method=IPMConfig(
                    rays=36,
                    scout_ratio=2,
                    refinement=1,
                    virtual_refinement=1,
                    tiled=True,
                    cell_chunk_size=32,
                    far_field_approx=FarFieldApproxConfig(enabled=False),
                ),
                schedule=DynamicConfig(
                    temporal_batch_size=3,
                    scout_refresh_frames=1,
                ),
            )
        )
        self.assertTrue(
            all(not item.metadata["dynamic_scout_reuse_approximate"] for item in maps)
        )
        self.assertEqual(maps[0].metadata["dynamic_scout_anchor_frames"], [0, 1, 2])

    def test_fused_full_field_temporal_ipm_matches_scalar_maps(self) -> None:
        """Full-field IPM uses the same temporal dispatcher without scouting."""

        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.tensor([0.2]),
                torch.tensor([-0.1]),
                velocity_x_uas_per_day=torch.tensor([0.01]),
                velocity_y_uas_per_day=torch.tensor([0.0]),
                einstein_radius_uas=torch.tensor([0.12]),
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        lens_region = PlaneRegion((2.0, 2.0))
        source_grid = PlaneGrid((4, 5), (1.5, 1.5))
        method = IPMConfig(
            rays=25,
            refinement=1,
            virtual_refinement=2,
            tiled=False,
            cell_chunk_size=11,
            far_field_approx=FarFieldApproxConfig(enabled=False),
        )
        times = [0.0, 1.5, 3.0]
        fused = list(
            simulation.dynamic_maps(
                lens_region,
                source_grid,
                times,
                method=method,
                schedule=DynamicConfig(
                    temporal_batch_size=2,
                    fused_temporal_ipm=True,
                ),
            )
        )
        expected = [
            simulation.magnification_map(
                lens_region,
                source_grid,
                method=method,
                time_days=time_days,
            )
            for time_days in times
        ]
        for actual, reference in zip(fused, expected, strict=True):
            torch.testing.assert_close(actual.values, reference.values, rtol=0, atol=0)

    def test_dual_scout_correction_is_a_constant_map_offset(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.35, shear=0.25),
            PointMassField._from_einstein_radii(
                torch.tensor([-0.7, 0.2, 0.9]),
                torch.tensor([0.3, -0.2, 0.6]),
                einstein_radius_uas=torch.tensor([0.45, 0.32, 0.38]),
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        common = dict(
            rays=64,
            scout_ratio=2,
            refinement=1,
            virtual_refinement=1,
            tiled=True,
            scout_halo_pixels=0.0,
            scout_dilation_cells=0,
            scout_trace_centers=False,
            cell_chunk_size=64,
            far_field_approx=FarFieldApproxConfig(enabled=False),
        )
        lens_region = PlaneRegion((5.0, 5.0))
        source_grid = PlaneGrid((6, 6), (1.0, 1.0))
        ordinary = simulation.magnification_map(
            lens_region,
            source_grid,
            method=IPMConfig(dual_scout_scalar_correction=False, **common),
        )
        corrected = simulation.magnification_map(
            lens_region,
            source_grid,
            method=IPMConfig(dual_scout_scalar_correction=True, **common),
        )
        correction = corrected.metadata["dual_scout_scalar_correction"]
        self.assertNotEqual(correction, 0.0)
        torch.testing.assert_close(
            corrected.values - ordinary.values,
            torch.full_like(ordinary.values, correction),
            rtol=0.0,
            atol=2.0e-6,
        )

    def test_unit_map_preserves_multiband_source_flux(self) -> None:
        from microcaustics import (
            LensingDistances,
            LinearTrajectory,
            MagnificationMap,
            SourceGeometry,
            StaticSource,
            light_curve_from_maps,
        )

        geometry = SourceGeometry(
            shape=(3, 5),
            pixel_scale_m=(1.0e10, 2.0e10),
            wavelengths_angstrom=(4000.0, 7000.0),
            band_names=("blue", "red"),
        )
        image = torch.arange(30, dtype=torch.float64).reshape(3, 5, 2) + 1.0
        source = StaticSource(image, geometry)
        grid = PlaneGrid((9, 11), (10.0, 12.0))
        maps = [
            MagnificationMap(
                torch.ones(grid.shape, dtype=torch.float64),
                grid,
                time_days=t,
            )
            for t in (0.0, 2.0)
        ]
        result = light_curve_from_maps(
            maps,
            source,
            [0.0, 2.0],
            LensingDistances(1.0e25, 2.0e25, 1.0e25),
            trajectory=LinearTrajectory(velocity_uas_per_day=(0.1, 0.0)),
        )
        torch.testing.assert_close(result.flux, result.unlensed_flux)
        expected = image.sum(dim=(0, 1)) * 2.0e20
        torch.testing.assert_close(result.flux[0], expected)

    def test_streaming_light_curve_matches_precomputed_maps(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        distances = LensingDistances(1.0e25, 2.0e25, 1.0e25)
        source = StaticSource(
            torch.arange(18, dtype=torch.float32).reshape(3, 3, 2) + 1.0,
            SourceGeometry(
                (3, 3),
                (1.0e10, 1.0e10),
                (4500.0, 7000.0),
                ("blue", "red"),
            ),
        )
        times = [0.0, 2.0, 4.0]
        lens_region = PlaneRegion((2.0, 2.0))
        source_grid = PlaneGrid((6, 6), (2.0, 2.0))
        method = IRSConfig(
            rays=36,
            ray_chunk_size=18,
            far_field_approx=FarFieldApproxConfig(enabled=False),
        )
        maps = list(
            simulation.dynamic_maps(
                lens_region,
                source_grid,
                times,
                method=method,
            )
        )
        expected = simulation.light_curve_from_maps(
            maps,
            source,
            times,
            distances,
        )
        observed: list[int] = []
        actual = simulation.light_curve(
            lens_region,
            source_grid,
            times,
            source,
            distances,
            method=method,
            schedule=DynamicConfig(temporal_batch_size=2),
            map_observer=lambda index, _: observed.append(index),
        )
        torch.testing.assert_close(actual.flux, expected.flux)
        torch.testing.assert_close(actual.unlensed_flux, expected.unlensed_flux)
        self.assertEqual(observed, [0, 1, 2])
        self.assertFalse(actual.metadata["maps_retained"])

    def test_streaming_light_curve_batch_matches_independent_requests(self) -> None:
        simulation = MicrolensingSimulation.create(
            MacroLens(convergence=0.0, shear=0.0),
            PointMassField._from_einstein_radii(
                torch.empty(0), torch.empty(0), einstein_radius_uas=torch.empty(0)
            ),
            runtime=RuntimeConfig(device="cpu", backend=Backend.TORCH_EAGER),
        )
        distances = LensingDistances(1.0e25, 2.0e25, 1.0e25)
        geometry = SourceGeometry(
            (3, 3),
            (1.0e10, 1.0e10),
            (4500.0,),
            ("blue",),
        )
        first = StaticSource(torch.arange(9).reshape(3, 3, 1).float() + 1, geometry)
        second = StaticSource(torch.flip(first.image, dims=(0,)), geometry)
        times = [0.0, 2.0, 4.0]
        lens_region = PlaneRegion((2.0, 2.0))
        source_grid = PlaneGrid((6, 6), (2.0, 2.0))
        method = IRSConfig(
            rays=36,
            ray_chunk_size=18,
            far_field_approx=FarFieldApproxConfig(enabled=False),
        )
        requests = (
            LightCurveRequest(first, distances, name="first"),
            LightCurveRequest(second, distances, name="second"),
        )
        batched = simulation.light_curves(
            lens_region,
            source_grid,
            times,
            requests,
            method=method,
            schedule=DynamicConfig(
                temporal_batch_size=2,
                light_curve_batch_size=2,
            ),
        )
        independent = tuple(
            simulation.light_curve(
                lens_region,
                source_grid,
                times,
                request.source,
                request.distances,
                method=method,
                schedule=DynamicConfig(temporal_batch_size=2),
            )
            for request in requests
        )
        self.assertEqual(len(batched), 2)
        for shared, separate in zip(batched, independent, strict=True):
            torch.testing.assert_close(shared.flux, separate.flux)
            torch.testing.assert_close(shared.unlensed_flux, separate.unlensed_flux)
            self.assertEqual(shared.metadata["shared_map_request_count"], 2)
            self.assertEqual(shared.metadata["light_curve_batch_size"], 2)

    def test_light_curve_rejects_incomplete_map_coverage(self) -> None:
        from microcaustics import (
            LensingDistances,
            MagnificationMap,
            SourceGeometry,
            StaticSource,
            light_curve_from_maps,
        )

        source = StaticSource(
            torch.ones((3, 3, 1), dtype=torch.float32),
            SourceGeometry((3, 3), (1.0e20, 1.0e20), (5000.0,), ("x",)),
        )
        grid = PlaneGrid((4, 4), (1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            light_curve_from_maps(
                [MagnificationMap(torch.ones(grid.shape), grid)],
                source,
                [0.0],
                LensingDistances(1.0e25, 1.0e25, 1.0e25),
            )

    def test_complex_taylor_far_field_matches_exact_deflection(self) -> None:
        from microcaustics.solvers import (
            complex_taylor_coefficients,
            evaluate_complex_taylor,
        )

        center_x = torch.tensor([0.0, 0.5], dtype=torch.float64)
        center_y = torch.tensor([0.0, -0.25], dtype=torch.float64)
        star_x = torch.tensor([8.0, -10.0, 11.0], dtype=torch.float64)
        star_y = torch.tensor([9.0, 12.0, -7.0], dtype=torch.float64)
        mass = torch.tensor([0.7, 1.1, 0.3], dtype=torch.float64)
        real, imag = complex_taylor_coefficients(
            center_x,
            center_y,
            star_x,
            star_y,
            mass,
            order=8,
            star_chunk_size=2,
        )
        delta_x = torch.tensor([0.15, -0.12], dtype=torch.float64)
        delta_y = torch.tensor([-0.11, 0.08], dtype=torch.float64)
        approximate_x, approximate_y = evaluate_complex_taylor(
            real,
            imag,
            delta_x,
            delta_y,
        )
        query_x = center_x + delta_x
        query_y = center_y + delta_y
        dx = query_x[:, None] - star_x[None]
        dy = query_y[:, None] - star_y[None]
        weight = mass[None] / (dx.square() + dy.square())
        exact_x = (dx * weight).sum(dim=1)
        exact_y = (dy * weight).sum(dim=1)
        torch.testing.assert_close(approximate_x, exact_x, rtol=0.0, atol=2.0e-13)
        torch.testing.assert_close(approximate_y, exact_y, rtol=0.0, atol=2.0e-13)

    def test_center_translation_preserves_taylor_evaluation(self) -> None:
        from microcaustics.solvers import (
            complex_taylor_coefficients,
            evaluate_complex_taylor,
            translate_complex_taylor,
        )

        center_real, center_imag = complex_taylor_coefficients(
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([6.0, -7.0], dtype=torch.float64),
            torch.tensor([8.0, 9.0], dtype=torch.float64),
            torch.tensor([1.0, 0.5], dtype=torch.float64),
            order=10,
        )
        offset_x = torch.tensor([0.2, -0.15], dtype=torch.float64)
        offset_y = torch.tensor([-0.1, 0.12], dtype=torch.float64)
        node_real, node_imag = translate_complex_taylor(
            center_real,
            center_imag,
            offset_x,
            offset_y,
            output_order=4,
        )
        query_offset_x = torch.tensor([0.01, -0.015], dtype=torch.float64)
        query_offset_y = torch.tensor([-0.02, 0.01], dtype=torch.float64)
        translated_x, translated_y = evaluate_complex_taylor(
            node_real[0],
            node_imag[0],
            query_offset_x,
            query_offset_y,
        )
        direct_x, direct_y = evaluate_complex_taylor(
            center_real.expand(2, -1),
            center_imag.expand(2, -1),
            offset_x + query_offset_x,
            offset_y + query_offset_y,
        )
        torch.testing.assert_close(translated_x, direct_x, rtol=0.0, atol=2.0e-10)
        torch.testing.assert_close(translated_y, direct_y, rtol=0.0, atol=2.0e-10)

    def test_map_metrics_retain_constant_bias(self) -> None:
        from microcaustics import compare_magnification_maps

        reference = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
        comparison = compare_magnification_maps(reference * 2.0, reference)
        self.assertAlmostEqual(comparison.fractional_nrmse, 1.0, places=12)
        expected = abs(-2500.0 * torch.log10(torch.tensor(2.0)).item())
        self.assertAlmostEqual(comparison.rmse_mmag, expected, places=3)
        self.assertEqual(comparison.valid_pixels, 4)

    def test_light_curve_shape_only_metric_is_explicit(self) -> None:
        from microcaustics import compare_light_curves

        reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        raw = compare_light_curves(reference * 1.5, reference)
        adjusted = compare_light_curves(
            reference * 1.5,
            reference,
            remove_constant_offset=True,
        )
        self.assertGreater(raw.rmse_mmag, 400.0)
        self.assertLess(adjusted.rmse_mmag, 1.0e-9)
        self.assertFalse(raw.constant_offset_removed)
        self.assertTrue(adjusted.constant_offset_removed)

    def test_temporal_far_field_builds_every_frame_exactly(self) -> None:
        """Temporal batching preserves independently constructed far-field approximations."""

        from microcaustics.solvers import (
            TaylorFarFieldApproximation,
            temporal_taylor_far_fields,
        )

        simulation = MicrolensingSimulation.create(
            MacroLens(0.2, 0.1),
            PointMassField._from_einstein_radii(
                torch.tensor([-1.1, -0.2, 0.8, 1.3], dtype=torch.float64),
                torch.tensor([0.6, -0.9, 0.4, -0.3], dtype=torch.float64),
                velocity_x_uas_per_day=torch.tensor(
                    [0.004, -0.003, 0.002, -0.001], dtype=torch.float64
                ),
                velocity_y_uas_per_day=torch.tensor(
                    [-0.002, 0.001, -0.003, 0.002], dtype=torch.float64
                ),
                einstein_radius_uas=torch.tensor(
                    [0.12, 0.09, 0.11, 0.08], dtype=torch.float64
                ),
            ),
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float64",
            ),
        )
        config = FarFieldApproxConfig(
            cells_per_axis=3,
            nodes_per_cell_axis=4,
            exact_radius_cells=0.8,
            taylor_order=4,
            center_translation_order=8,
        )
        region = PlaneRegion((4.0, 4.0))
        sequence, metadata = temporal_taylor_far_fields(
            simulation,
            region,
            config,
            [0.0, 1.0, 2.0],
        )
        self.assertTrue(metadata["far_field_exact_each_frame"])
        self.assertEqual(metadata["far_field_frame_count"], 3)
        for index, time_days in enumerate((0.0, 1.0, 2.0)):
            independent = TaylorFarFieldApproximation(
                simulation,
                region,
                config,
                time_days=time_days,
            )
            torch.testing.assert_close(sequence[index].local_x, independent.local_x)
            torch.testing.assert_close(sequence[index].local_y, independent.local_y)
            torch.testing.assert_close(
                sequence[index].local_mass, independent.local_mass
            )
            torch.testing.assert_close(
                sequence[index].coefficient_real,
                independent.coefficient_real,
            )

    def test_eager_taylor_far_field_matches_direct_raytrace(self) -> None:
        from microcaustics import FarFieldApproxConfig
        from microcaustics.solvers import TaylorFarFieldApproximation

        generator = torch.Generator().manual_seed(42)
        stars_x = torch.rand(30, generator=generator, dtype=torch.float64) * 8.0 - 4.0
        stars_y = torch.rand(30, generator=generator, dtype=torch.float64) * 8.0 - 4.0
        simulation = MicrolensingSimulation.create(
            MacroLens(
                convergence=0.2,
                shear=0.12,
                shear_angle_deg=17.188733853924695,
                smooth_matter_fraction=0.5,
            ),
            PointMassField._from_einstein_radii(
                stars_x,
                stars_y,
                einstein_radius_uas=torch.full((30,), 0.08, dtype=torch.float64),
            ),
            runtime=RuntimeConfig(
                device="cpu",
                backend=Backend.TORCH_EAGER,
                dtype="float64",
            ),
        )
        approximation = TaylorFarFieldApproximation(
            simulation,
            PlaneRegion((4.0, 4.0)),
            FarFieldApproxConfig(
                cells_per_axis=4,
                nodes_per_cell_axis=8,
                exact_radius_cells=1.0,
                taylor_order=4,
                center_translation_order=10,
            ),
            star_chunk_size=11,
        )
        query_x = torch.rand(64, generator=generator, dtype=torch.float64) * 3.8 - 1.9
        query_y = torch.rand(64, generator=generator, dtype=torch.float64) * 3.8 - 1.9
        approximate_x, approximate_y = approximation.raytrace(query_x, query_y)
        exact_x, exact_y, _ = simulation.raytrace_direct(
            query_x,
            query_y,
            star_chunk_size=13,
        )
        torch.testing.assert_close(approximate_x, exact_x, rtol=0.0, atol=2.0e-7)
        torch.testing.assert_close(approximate_y, exact_y, rtol=0.0, atol=2.0e-7)
        approximate_det = approximation.jacobian_determinant(query_x, query_y)
        exact_det, _ = simulation.jacobian_determinant_direct(
            query_x,
            query_y,
            star_chunk_size=13,
        )
        torch.testing.assert_close(approximate_det, exact_det, rtol=0.0, atol=8.0e-7)


if __name__ == "__main__":
    unittest.main()
