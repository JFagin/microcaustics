"""Apple MPS parity with CPU, skipped automatically where MPS is unavailable."""

from __future__ import annotations

import pytest
import torch

import microcaustics as mc
from microcaustics.solvers.ipm import rasterize_triangles_exact_eager
from microcaustics.sources.transferred_disk import _transferred_flux_kernel

pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS is unavailable",
)


def _runtime(device: str) -> mc.RuntimeConfig:
    return mc.RuntimeConfig(
        device=device, backend="torch-eager", strict_backend=True, warn_on_compile=False
    )


def _thin_disk_system(device: str) -> mc.MultiImageSystem:
    """A physical quasar disk whose faint pixels have subnormal Jy m^-2 brightness."""

    model = mc.ThinDiskModel(
        black_hole_mass_solar=1.5e8,
        eddington_ratio=0.14,
        inclination_deg=56.0,
        bands_angstrom={"u": 3600.0, "i": 7500.0},
        source_grid_shape=32,
        driving_signal=mc.broken_power_law_driving_signal(
            seed=3, max_duration_days=10, history_days=200
        ),
    )
    return mc.MultiImageSystem(
        images={"A": mc.MacroLens(0.5, 0.4, shear_angle_deg=30.0, smooth_matter_fraction=0.8)},
        arrival_time_delays_days={"A": 0.0},
        lens_redshift=0.86,
        source_redshift=1.95,
        source=model,
        stellar_population=mc.StellarPopulation.salpeter(mean_mass_solar=0.3),
        methods=mc.production_ipm_config(rays=10_000),
        integration_domain="scout",
        duration_days=2.0,
        seed=11,
        runtime=_runtime(device),
    )


def test_rasterizer_reads_mps_triangles():
    """One MPS -> CPU float64 ``.to`` returns zeros, which emptied every IPM map."""

    triangles = torch.tensor(
        [[[0.1, 0.2], [1.7, 0.4], [0.6, 1.9]], [[-1.2, -0.8], [0.3, -1.1], [-0.4, 0.9]]]
    )
    grid = mc.PlaneGrid((8, 8), (4.0, 4.0))
    expected = rasterize_triangles_exact_eager(
        triangles, grid, lens_area_per_triangle_uas2=0.5
    )
    actual = rasterize_triangles_exact_eager(
        triangles.to("mps"), grid, lens_area_per_triangle_uas2=0.5
    )
    assert actual.device.type == "mps"
    assert float(expected.sum()) > 0.0
    torch.testing.assert_close(actual.cpu(), expected)


def _faint_disk_flux(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Pixel solid angle, inverse pixel area and dimming of a z ~ 2 quasar disk;
    # 3000-5000 K pixels at 300 nm have subnormal Jy m^-2 brightness.
    temperature = torch.linspace(3000.0, 5000.0, 32 * 32).reshape(1, 32, 32)
    shape = temperature.shape[1:]
    arguments = (
        temperature.pow(4).expand(2, -1, -1),
        torch.ones(shape),
        torch.full(shape, 1.339e-28),
        torch.ones(shape, dtype=torch.bool),
        torch.tensor([2.0e-7, 3.0e-7]),
        torch.tensor(1.0),
        torch.tensor(0.0388),
        torch.tensor(2.619e-24),
    )
    magnification = torch.full((2, 1, 1), 3.0)
    arguments = (magnification, magnification, torch.zeros(2), *arguments)
    return _transferred_flux_kernel(*(value.to(device) for value in arguments))


def test_faint_disk_flux_matches_cpu():
    """MPS flushes subnormals, so faint pixels must stay normal in the flux kernel."""

    expected = _faint_disk_flux("cpu")
    actual = _faint_disk_flux("mps")
    assert bool((expected[1] > 0).all())
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value.cpu(), reference, rtol=1e-5, atol=0.0)


def test_intrinsic_disk_flux_matches_cpu():
    times = torch.arange(0.0, 3.0)
    expected = _thin_disk_system("cpu").image("A").source_light_curve(times).flux
    actual = _thin_disk_system("mps").image("A").source_light_curve(times).flux
    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-5, atol=0.0)


def _light_curve(device: str) -> mc.LightCurve:
    curves = _thin_disk_system(device).light_curves(
        duration_days=2.0, map_cadence_days=1.0, source_cadence_days=1.0
    )
    return curves["A"].light_curve


def test_microlensed_light_curve_matches_cpu():
    expected, actual = _light_curve("cpu"), _light_curve("mps")
    assert bool((expected.flux > 0).all())
    # MPS maps are rasterized on the GPU; the CPU keeps the exact reference.
    assert "torch_direct_cell" in str(actual.metadata)
    assert "exact_sutherland_hodgman" in str(expected.metadata)
    torch.testing.assert_close(actual.flux.cpu(), expected.flux, rtol=2e-4, atol=0.0)
    torch.testing.assert_close(
        actual.unlensed_flux.cpu(), expected.unlensed_flux, rtol=5e-5, atol=0.0
    )
