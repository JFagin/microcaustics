"""Independent numerical oracles for package mathematics.

These tests deliberately import optional libraries only from ``tests``.  The
installed package has no runtime dependency on Astropy, SciPy, or
lenstronomy.  Lenstronomy is additionally opt-in because its first import can
compile dependencies and take appreciable time on a clean machine.
"""

from __future__ import annotations

import importlib.util
import math
import os
import unittest

import numpy as np
import torch

import microcaustics as mc
from microcaustics.relativity.elliptic import (
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

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
ASTROPY_AVAILABLE = importlib.util.find_spec("astropy") is not None
LENSTRONOMY_AVAILABLE = importlib.util.find_spec("lenstronomy") is not None
RUN_LENSTRONOMY = os.environ.get("MICROCAUSTICS_RUN_LENSTRONOMY", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _empty_field(dtype=torch.float64) -> mc.PointMassField:
    empty = torch.empty(0, dtype=dtype)
    return mc.PointMassField(empty, empty, empty)


class NumPyOracleTests(unittest.TestCase):
    def test_plane_grid_matches_numpy_cell_centers_and_mesh_convention(self) -> None:
        grid = mc.PlaneGrid(
            shape=(5, 7),
            field_of_view_uas=(4.0, 9.8),
            center_uas=(0.6, -1.2),
        )
        y_axis, x_axis = grid.axes(dtype=torch.float64)
        expected_y = np.linspace(-1.4, 2.6, 5, endpoint=False) + 0.4
        expected_x = np.linspace(-6.1, 3.7, 7, endpoint=False) + 0.7
        np.testing.assert_allclose(y_axis.numpy(), expected_y, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(x_axis.numpy(), expected_x, rtol=0.0, atol=1e-14)
        mesh_x, mesh_y = grid.mesh(dtype=torch.float64)
        expected_mesh_y, expected_mesh_x = np.meshgrid(
            expected_y,
            expected_x,
            indexing="ij",
        )
        np.testing.assert_allclose(mesh_x.numpy(), expected_mesh_x)
        np.testing.assert_allclose(mesh_y.numpy(), expected_mesh_y)

    def test_direct_lens_equation_matches_independent_numpy_sum(self) -> None:
        dtype = torch.float64
        macro = mc.MacroLens(
            convergence=0.43,
            shear=0.21,
            shear_angle_rad=0.37,
            smooth_matter_fraction=0.35,
        )
        stars = mc.PointMassField(
            torch.tensor([-0.8, 0.35, 1.1], dtype=dtype),
            torch.tensor([0.25, -0.55, 0.9], dtype=dtype),
            torch.tensor([0.13, 0.22, 0.17], dtype=dtype),
        )
        simulation = mc.MicrolensingSimulation.create(
            macro,
            stars,
            runtime=mc.RuntimeConfig(
                device="cpu",
                backend="torch-eager",
                dtype=dtype,
            ),
        )
        x = torch.tensor([-1.3, -0.2, 0.7, 1.4], dtype=dtype)
        y = torch.tensor([0.8, -1.1, 0.15, -0.4], dtype=dtype)
        package_x, package_y, _ = simulation.raytrace_direct(x, y)

        xn, yn = x.numpy(), y.numpy()
        sx, sy = stars.x_uas.numpy(), stars.y_uas.numpy()
        radius2 = stars.einstein_radius_uas.numpy() ** 2
        dx = xn[:, None] - sx[None, :]
        dy = yn[:, None] - sy[None, :]
        denominator = dx * dx + dy * dy
        alpha_x = np.sum(dx * radius2[None, :] / denominator, axis=1)
        alpha_y = np.sum(dy * radius2[None, :] / denominator, axis=1)
        gamma1 = macro.shear * math.cos(2.0 * macro.shear_angle_rad)
        gamma2 = macro.shear * math.sin(2.0 * macro.shear_angle_rad)
        sheet = macro.smooth_convergence
        alpha_x += (sheet + gamma1) * xn + gamma2 * yn
        alpha_y += gamma2 * xn + (sheet - gamma1) * yn
        np.testing.assert_allclose(package_x.numpy(), xn - alpha_x, rtol=2e-14)
        np.testing.assert_allclose(package_y.numpy(), yn - alpha_y, rtol=2e-14)

    def test_validation_metrics_match_numpy_definitions(self) -> None:
        reference = np.asarray([[1.0, 2.0], [4.0, 8.0]])
        candidate = reference * np.asarray([[1.02, 0.97], [1.01, 1.04]])
        comparison = mc.compare_magnification_maps(candidate, reference)
        residual = candidate - reference
        magnitude = -2500.0 * np.log10(candidate / reference)
        self.assertAlmostEqual(
            comparison.linear_nrmse,
            float(np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(reference**2))),
            places=14,
        )
        self.assertAlmostEqual(
            comparison.fractional_nrmse,
            float(np.sqrt(np.mean((residual / reference) ** 2))),
            places=14,
        )
        self.assertAlmostEqual(
            comparison.rmse_mmag,
            float(np.sqrt(np.mean(magnitude**2))),
            places=11,
        )


@unittest.skipUnless(SCIPY_AVAILABLE, "SciPy validation dependency is unavailable")
class SciPyOracleTests(unittest.TestCase):
    def test_carlson_integrals_match_scipy_special(self) -> None:
        from scipy import special

        x = torch.tensor([0.2, 1.0, 3.0], dtype=torch.float64)
        y = torch.tensor([0.4, 2.0, 4.0], dtype=torch.float64)
        z = torch.tensor([0.7, 5.0, 6.0], dtype=torch.float64)
        p = torch.tensor([1.1, 3.0, 7.0], dtype=torch.float64)
        xn, yn, zn, pn = x.numpy(), y.numpy(), z.numpy(), p.numpy()
        np.testing.assert_allclose(
            carlson_rf(x, y, z), special.elliprf(xn, yn, zn), rtol=2e-14
        )
        np.testing.assert_allclose(carlson_rc(x, y), special.elliprc(xn, yn), rtol=2e-14)
        np.testing.assert_allclose(
            carlson_rd(x, y, z), special.elliprd(xn, yn, zn), rtol=2e-14
        )
        np.testing.assert_allclose(
            carlson_rj(x, y, z, p),
            special.elliprj(xn, yn, zn, pn),
            rtol=2e-14,
        )

    def test_legendre_and_jacobi_functions_match_scipy(self) -> None:
        from scipy import special

        parameter = torch.tensor([0.0, 0.2, 0.8], dtype=torch.float64)
        amplitude = torch.tensor([-0.7, 0.4, 1.2], dtype=torch.float64)
        characteristic = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
        np.testing.assert_allclose(
            elliptic_k(parameter),
            special.ellipk(parameter.numpy()),
            rtol=2e-14,
        )
        np.testing.assert_allclose(
            elliptic_f(amplitude, parameter),
            special.ellipkinc(amplitude.numpy(), parameter.numpy()),
            rtol=2e-14,
        )
        np.testing.assert_allclose(
            elliptic_e(amplitude, parameter),
            special.ellipeinc(amplitude.numpy(), parameter.numpy()),
            rtol=2e-14,
        )
        sine = np.sin(amplitude.numpy())
        sine2 = sine * sine
        expected_pi = sine * special.elliprf(
            1.0 - sine2,
            1.0 - parameter.numpy() * sine2,
            1.0,
        ) + characteristic.numpy() * sine**3 / 3.0 * special.elliprj(
            1.0 - sine2,
            1.0 - parameter.numpy() * sine2,
            1.0,
            1.0 - characteristic.numpy() * sine2,
        )
        np.testing.assert_allclose(
            elliptic_pi_principal(characteristic, amplitude, parameter),
            expected_pi,
            rtol=3e-14,
        )
        argument = torch.tensor([-1.4, -0.2, 0.8], dtype=torch.float64)
        sn, cn = jacobi_sn_cn(argument, parameter)
        expected_sn, expected_cn, _, _ = special.ellipj(
            argument.numpy(),
            parameter.numpy(),
        )
        np.testing.assert_allclose(sn, expected_sn, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(cn, expected_cn, rtol=2e-12, atol=2e-12)

    def test_gaussian_psf_kernel_matches_scipy_window(self) -> None:
        from scipy.signal.windows import gaussian

        size = 17
        fwhm_arcsec = 0.14
        pixel_scale_arcsec = 0.04
        oversample = 2
        package = mc.gaussian_psf_kernel(
            fwhm_arcsec,
            pixel_scale_arcsec,
            size=size,
            oversample_factor=oversample,
            dtype=torch.float64,
        ).numpy()
        sigma_pixels = (
            fwhm_arcsec / (pixel_scale_arcsec / oversample) / 2.354820045
        )
        axis = gaussian(size, sigma_pixels)
        expected = np.outer(axis, axis)
        expected /= expected.sum()
        np.testing.assert_allclose(package, expected, rtol=2.0e-14, atol=1.0e-16)


@unittest.skipUnless(ASTROPY_AVAILABLE, "Astropy validation dependency is unavailable")
class AstropyOracleTests(unittest.TestCase):
    def test_ab_magnitude_matches_astropy_photometric_units(self) -> None:
        import astropy.units as units

        flux_jy = np.asarray([3631.0, 1.0, 1.0e-3, 1.0e-6])
        package = mc.flux_to_magnitude(
            torch.as_tensor(flux_jy, dtype=torch.float64)
        ).numpy()
        expected = (flux_jy * units.Jy).to_value(units.ABmag)
        # microcaustics uses the conventional rounded 3631 Jy AB zero point.
        np.testing.assert_allclose(package, expected, rtol=0.0, atol=7.0e-5)

    def test_torch_float32_flat_cosmology_matches_astropy(self) -> None:
        import astropy.units as units
        from astropy.cosmology import FlatLambdaCDM

        cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)
        for lens_redshift, source_redshift in ((0.0395, 1.695), (0.45, 1.8)):
            distances = mc.LensingDistances.from_redshifts(
                lens_redshift,
                source_redshift,
                H0=70.0,
                Om0=0.3,
                dtype=torch.float32,
            )
            expected = (
                cosmology.angular_diameter_distance(lens_redshift).to_value(units.m),
                cosmology.angular_diameter_distance(source_redshift).to_value(units.m),
                cosmology.angular_diameter_distance(
                    lens_redshift, source_redshift
                ).to_value(units.m),
            )
            np.testing.assert_allclose(
                (distances.lens_m, distances.source_m, distances.lens_to_source_m),
                expected,
                rtol=2.0e-6,
                atol=0.0,
            )

    def test_redshift_distances_match_astropy_cosmology(self) -> None:
        import astropy.units as units
        from astropy.cosmology import FlatLambdaCDM

        cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)
        lens_redshift, source_redshift = 0.45, 1.8
        distances = mc.LensingDistances.from_redshifts(
            lens_redshift,
            source_redshift,
            cosmology=cosmology,
        )
        self.assertAlmostEqual(
            distances.lens_m,
            cosmology.angular_diameter_distance(lens_redshift).to_value(units.m),
            delta=1e-13 * distances.lens_m,
        )
        self.assertAlmostEqual(
            distances.source_m,
            cosmology.angular_diameter_distance(source_redshift).to_value(units.m),
            delta=1e-13 * distances.source_m,
        )
        self.assertAlmostEqual(
            distances.lens_to_source_m,
            cosmology.angular_diameter_distance(
                lens_redshift,
                source_redshift,
            ).to_value(units.m),
            delta=1e-13 * distances.lens_to_source_m,
        )

    def test_einstein_radius_matches_astropy_constants_and_units(self) -> None:
        import astropy.constants as constants
        import astropy.units as units

        distances = mc.LensingDistances(1.2e25, 2.7e25, 1.9e25)
        masses = np.asarray([0.08, 0.5, 3.0])
        package = distances.einstein_radius_uas(masses, dtype=torch.float64).numpy()
        expected = np.sqrt(
            4.0
            * constants.G
            * (masses * constants.M_sun)
            / constants.c**2
            * (distances.lens_to_source_m * units.m)
            / ((distances.lens_m * units.m) * (distances.source_m * units.m))
        ).to_value(units.uas, equivalencies=units.dimensionless_angles())
        np.testing.assert_allclose(package, expected, rtol=2e-8, atol=0.0)


@unittest.skipUnless(
    LENSTRONOMY_AVAILABLE and RUN_LENSTRONOMY,
    "set MICROCAUSTICS_RUN_LENSTRONOMY=1 for the lenstronomy oracle",
)
class LenstronomyOracleTests(unittest.TestCase):
    def test_macro_deflection_jacobian_and_magnification(self) -> None:
        from lenstronomy.LensModel.lens_model import LensModel

        convergence = 0.27
        shear = 0.14
        angle = 0.31
        gamma1 = shear * math.cos(2.0 * angle)
        gamma2 = shear * math.sin(2.0 * angle)
        x = np.asarray([-0.7, 0.2, 1.1])
        y = np.asarray([0.4, -0.5, 0.8])
        model = LensModel(lens_model_list=["CONVERGENCE", "SHEAR"])
        kwargs = [
            {"kappa": convergence, "ra_0": 0.0, "dec_0": 0.0},
            {"gamma1": gamma1, "gamma2": gamma2, "ra_0": 0.0, "dec_0": 0.0},
        ]
        alpha_x, alpha_y = model.alpha(x, y, kwargs)
        f_xx, f_xy, f_yx, f_yy = model.hessian(x, y, kwargs)

        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(
                convergence,
                shear,
                shear_angle_rad=angle,
                smooth_matter_fraction=1.0,
            ),
            _empty_field(),
            runtime=mc.RuntimeConfig(
                device="cpu",
                backend="torch-eager",
                dtype=torch.float64,
            ),
        )
        source_x, source_y, _ = simulation.raytrace_direct(
            torch.as_tensor(x),
            torch.as_tensor(y),
        )
        np.testing.assert_allclose(source_x, x - alpha_x, rtol=0.0, atol=2e-14)
        np.testing.assert_allclose(source_y, y - alpha_y, rtol=0.0, atol=2e-14)
        determinant, _ = simulation.jacobian_determinant_direct(
            torch.as_tensor(x),
            torch.as_tensor(y),
        )
        expected_determinant = (
            (1.0 - f_xx) * (1.0 - f_yy) - (-f_xy) * (-f_yx)
        )
        np.testing.assert_allclose(
            determinant,
            expected_determinant,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            torch.reciprocal(determinant),
            model.magnification(x, y, kwargs),
            rtol=2e-14,
        )

    def test_multiple_point_masses_match_lenstronomy(self) -> None:
        """Compare the complete local lens equation, not only its macro term."""

        from lenstronomy.LensModel.lens_model import LensModel

        convergence = 0.19
        shear = 0.11
        angle = -0.28
        gamma1 = shear * math.cos(2.0 * angle)
        gamma2 = shear * math.sin(2.0 * angle)
        star_x = np.asarray([-0.8, 0.45, 1.25])
        star_y = np.asarray([0.35, -0.7, 0.6])
        theta_e = np.asarray([0.12, 0.08, 0.15])
        x = np.asarray([-1.4, -0.15, 0.65, 1.6])
        y = np.asarray([0.9, -1.2, 0.1, -0.45])

        model = LensModel(
            lens_model_list=["CONVERGENCE", "SHEAR"]
            + ["POINT_MASS"] * len(star_x)
        )
        kwargs = [
            {"kappa": convergence, "ra_0": 0.0, "dec_0": 0.0},
            {"gamma1": gamma1, "gamma2": gamma2, "ra_0": 0.0, "dec_0": 0.0},
            *[
                {
                    "theta_E": float(radius),
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                }
                for center_x, center_y, radius in zip(
                    star_x, star_y, theta_e, strict=True
                )
            ],
        ]
        alpha_x, alpha_y = model.alpha(x, y, kwargs)
        f_xx, f_xy, f_yx, f_yy = model.hessian(x, y, kwargs)

        simulation = mc.MicrolensingSimulation.create(
            mc.MacroLens(
                convergence,
                shear,
                shear_angle_rad=angle,
                smooth_matter_fraction=1.0,
            ),
            mc.PointMassField(
                torch.as_tensor(star_x),
                torch.as_tensor(star_y),
                torch.as_tensor(theta_e),
            ),
            runtime=mc.RuntimeConfig(
                device="cpu",
                backend="torch-eager",
                dtype=torch.float64,
            ),
        )
        source_x, source_y, _ = simulation.raytrace_direct(x, y)
        determinant, _ = simulation.jacobian_determinant_direct(x, y)
        expected_determinant = (
            (1.0 - f_xx) * (1.0 - f_yy) - (-f_xy) * (-f_yx)
        )
        np.testing.assert_allclose(source_x, x - alpha_x, rtol=0.0, atol=3e-14)
        np.testing.assert_allclose(source_y, y - alpha_y, rtol=0.0, atol=3e-14)
        np.testing.assert_allclose(
            determinant,
            expected_determinant,
            rtol=2e-14,
            atol=3e-14,
        )

    def test_public_macroimage_solver_matches_point_mass_oracle(self) -> None:
        """Validate image positions, magnifications, and relative delays."""

        from lenstronomy.LensModel.lens_model import LensModel

        theta_e = 0.9
        beta_x, beta_y = 0.18, -0.07

        def raytrace(x, y):
            radius2 = (x.square() + y.square()).clamp_min(1.0e-30)
            scale = theta_e**2 / radius2
            return x * (1.0 - scale), y * (1.0 - scale)

        def jacobian(x, y):
            radius2 = (x.square() + y.square()).clamp_min(1.0e-30)
            radius4 = radius2.square()
            potential_xx = theta_e**2 * (y.square() - x.square()) / radius4
            potential_xy = -2.0 * theta_e**2 * x * y / radius4
            result = torch.empty((*x.shape, 2, 2), dtype=x.dtype, device=x.device)
            result[..., 0, 0] = 1.0 - potential_xx
            result[..., 0, 1] = -potential_xy
            result[..., 1, 0] = -potential_xy
            result[..., 1, 1] = 1.0 + potential_xx
            return result

        def delay(x, y):
            radius = torch.hypot(x, y).clamp_min(1.0e-30)
            return (
                0.5 * ((x - beta_x).square() + (y - beta_y).square())
                - theta_e**2 * torch.log(radius)
            )

        model = mc.CallableMacroModel(raytrace, jacobian, delay, name="point_mass")
        solutions = mc.solve_macroimages(
            model,
            beta_x,
            beta_y,
            initial_grid_size=100,
            field_of_view_arcsec=2.2,
            source_tolerance_arcsec=0.045,
            cluster_tolerance_arcsec=0.12,
            deduplication_tolerance_arcsec=0.05,
        )
        self.assertEqual(len(solutions), 2)

        beta_radius = math.hypot(beta_x, beta_y)
        direction = np.asarray([beta_x, beta_y]) / beta_radius
        root_scale = math.sqrt(beta_radius**2 + 4.0 * theta_e**2)
        radial_roots = (
            0.5 * (beta_radius + root_scale),
            0.5 * (beta_radius - root_scale),
        )
        expected_positions = np.stack([value * direction for value in radial_roots])
        actual_positions = np.asarray(
            [[solution.x_arcsec, solution.y_arcsec] for solution in solutions]
        )
        for expected in expected_positions:
            self.assertLess(np.min(np.linalg.norm(actual_positions - expected, axis=1)), 2e-9)

        lens = LensModel(lens_model_list=["POINT_MASS"])
        kwargs = [{"theta_E": theta_e, "center_x": 0.0, "center_y": 0.0}]
        x, y = actual_positions[:, 0], actual_positions[:, 1]
        source_x, source_y = lens.ray_shooting(x, y, kwargs)
        np.testing.assert_allclose(source_x, beta_x, rtol=0.0, atol=2e-9)
        np.testing.assert_allclose(source_y, beta_y, rtol=0.0, atol=2e-9)
        expected_magnification = lens.magnification(x, y, kwargs)
        np.testing.assert_allclose(
            [solution.macro_magnification for solution in solutions],
            expected_magnification,
            rtol=2e-9,
            atol=2e-9,
        )
        expected_delays = np.asarray(delay(torch.as_tensor(x), torch.as_tensor(y)))
        expected_delays -= expected_delays.min()
        np.testing.assert_allclose(
            [solution.arrival_time_delay_days for solution in solutions],
            expected_delays,
            rtol=0.0,
            atol=2e-9,
        )


if __name__ == "__main__":
    unittest.main()
