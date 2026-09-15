"""Concise source constructors preserve physical geometry and photometry."""

import math
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import microcaustics as mc


def uniform_brightness(times, *, geometry):
    """A two-band custom source with constant integrated flux in Jy."""
    flux = times.new_tensor([2e-5, 3e-5])
    return flux.expand(len(times), *geometry.shape, 2) / (
        math.prod(geometry.shape) * geometry.pixel_area_m2
    )


def small_system(source, redshift=1.5, dtype=torch.float32):
    """Use a negligible lens for fast high-level geometry and batching checks."""
    return mc.MicrolensingSystem(
        lens_redshift=0.3,
        source_redshift=redshift,
        macro=mc.MacroLens(0.0, 0.0),
        stars=mc.PointMassField([0.0], [0.0], [1e-10]),
        source=source,
        seed=0,
        caustic_grid_shape=16,
        runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager", dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_angular_geometry_is_resolved_without_mutating_input(dtype):
    geometry = mc.SourceGeometry(
        (16, 32),
        field_of_view_uas=(2.0, 8.0),
        bands_angstrom={"red": 7500.0, "blue": 4800.0},
    )
    with pytest.raises(RuntimeError, match="not resolved"):
        _ = geometry.pixel_area_m2
    distances = mc.LensingDistances.from_redshifts(0.3, 1.5)
    resolved = geometry.resolve(distances, dtype=dtype)
    expected = distances.uas_to_source_length((2.0, 8.0), dtype=dtype)
    assert resolved.pixel_scale_m == (float(expected[0]) / 16, float(expected[1]) / 32)
    assert resolved.band_names == ("red", "blue")
    assert geometry.pixel_scale_m is None
    assert resolved.resolve(distances) is resolved

    rebound = resolved.with_bands({"u": 3671.0, "y": 9712.0})
    assert rebound.shape == resolved.shape
    assert rebound.pixel_scale_m == resolved.pixel_scale_m
    assert rebound.band_names == ("u", "y")
    assert rebound.wavelengths_angstrom == (3671.0, 9712.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": 0},
        {"shape": 4.5},
        {"shape": (4, 2.5)},
        {"field_of_view_uas": 0},
        {"field_of_view_uas": float("nan")},
        {"pixel_scale_m": (1.0, 1.0)},
        {"bands_angstrom": {}},
    ],
)
def test_invalid_or_conflicting_custom_geometry(kwargs):
    inputs = dict(shape=8, field_of_view_uas=2.0, bands_angstrom={"g": 4800})
    inputs.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        mc.SourceGeometry(**inputs)


def test_static_image_infers_shape_without_resampling():
    image = torch.ones((16, 32, 2))
    source = mc.StaticSource(
        image, field_of_view_uas=(1, 2), bands_angstrom={"g": 4800, "i": 7500}
    )
    distances = mc.LensingDistances.from_redshifts(0.3, 1.5)
    resolved = source.pixelate(distances, grid=mc.PlaneGrid((64, 64), (3, 3)))
    assert resolved.geometry.shape == (16, 32)
    assert resolved.image.data_ptr() == image.data_ptr()
    torch.testing.assert_close(resolved.brightness([0, 1])[0], image)
    with pytest.raises(ValueError, match="enclose"):
        source.pixelate(distances, grid=mc.PlaneGrid((64, 64), (0.5, 0.5)))
    with pytest.raises(ValueError, match="not both"):
        mc.StaticSource(image, source.geometry, field_of_view_uas=2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_callable_geometry_keyword_and_system_distance_resolution(dtype):
    source = mc.CallableSource(
        uniform_brightness,
        source_grid_shape=16,
        field_of_view_uas=2.0,
        bands_angstrom={"g": 4800, "i": 7500},
    )
    with pytest.raises(RuntimeError, match="not resolved"):
        source.brightness([0, 1])
    first = small_system(source, 1.0, dtype).realize().source
    second = small_system(source, 2.0, dtype).realize().source
    assert first.geometry.pixel_area_m2 != second.geometry.pixel_area_m2
    assert source.geometry.pixel_scale_m is None
    # Detect optional callback arguments once, never in the evaluation loop.
    with patch(
        "inspect.signature", side_effect=AssertionError("repeated introspection")
    ):
        for resolved in (first, second):
            values = resolved.brightness([0, 1], dtype=dtype)
            assert values.dtype == dtype
            flux = values.sum((1, 2)) * resolved.geometry.pixel_area_m2
            torch.testing.assert_close(
                flux, torch.tensor([[2e-5, 3e-5]] * 2, dtype=dtype)
            )


def test_times_only_callable_still_works_with_physical_geometry():
    geometry = mc.SourceGeometry(8, (2.0, 3.0), (4800.0,), ("g",))
    source = mc.CallableSource(
        lambda times: times.new_ones((len(times), 8, 8, 1)), geometry
    )
    assert not source._accepts_geometry
    assert source.pixelate(mc.LensingDistances(1e25, 2e25, 1e25)) is source
    torch.testing.assert_close(source.brightness([0, 1]), torch.ones((2, 8, 8, 1)))
    with pytest.raises(ValueError, match="not both"):
        mc.CallableSource(uniform_brightness, geometry, source_grid_shape=8)


def test_angular_gaussian_matches_physical_source_with_center_and_hole():
    distances = mc.LensingDistances.from_redshifts(0.3, 1.5)
    common = dict(
        bands_angstrom={"g": 4800, "i": 7500},
        source_grid_shape=32,
        position_angle_deg=37.0,
        axis_ratio=0.6,
        total_flux=(2e-5, 3e-5),
    )
    angular = mc.GaussianModel(
        sigma_uas=(0.1, 0.2),
        center_uas=(0.03, -0.01),
        hole_radius_uas=0.02,
        **common,
    )

    def length(value):
        return distances.uas_to_source_length(value, dtype=torch.float64)

    physical = mc.GaussianModel(
        sigma_m=tuple(length((0.1, 0.2))),
        center_m=tuple(length((0.03, -0.01))),
        hole_radius_m=float(length(0.02)),
        **common,
    )
    assert angular.sigma_m is None
    assert angular.recommended_grid(distances) == physical.recommended_grid(distances)
    first, second = angular.pixelate(distances), physical.pixelate(distances)
    torch.testing.assert_close(
        first.brightness([0], dtype=torch.float64),
        second.brightness([0], dtype=torch.float64),
    )
    for kwargs in (
        {"sigma_m": 1e12},
        {"center_m": (0, 0)},
        {"hole_radius_m": 0.1},
    ):
        with pytest.raises(ValueError, match="exactly one|not both"):
            mc.GaussianModel(
                sigma_uas=0.1,
                center_uas=(0, 0),
                hole_radius_uas=0.1,
                **kwargs,
                **common,
            )


def test_modulated_physical_disk_matches_explicit_pixelization():
    distances = mc.LensingDistances.from_redshifts(0.3, 1.5)
    model = mc.ThinDiskModel(
        black_hole_mass_solar=1e8,
        eddington_ratio=0.2,
        bands_angstrom={"g": 4800},
        relativity="approximate",
        source_redshift=1.5,
        source_grid_shape=16,
    )
    signal = mc.TabulatedDrivingSignal([0.0, 1.0, 2.0], [1.0, 1.5, 0.5])
    deferred = mc.ModulatedSource(model, signal)
    direct = mc.ModulatedSource(model.pixelate(distances), signal)
    resolved = deferred.pixelate(distances)
    assert resolved.signal is signal
    assert deferred.source is model
    torch.testing.assert_close(
        resolved.brightness([0, 1, 2]), direct.brightness([0, 1, 2])
    )
    common = dict(
        macro=mc.MacroLens(0.3, 0.2),
        distances=distances,
        stellar_population=mc.StellarPopulation.salpeter(count=8),
        seed=0,
        runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
    )
    first = mc.MicrolensingSystem(source=deferred, **common).realize()
    second = mc.MicrolensingSystem(source=direct, **common).realize()
    assert first.stellar_aperture.radius_uas == pytest.approx(
        second.stellar_aperture.radius_uas
    )
    torch.testing.assert_close(first.stars.x_uas, second.stars.x_uas)
    torch.testing.assert_close(first.stars.y_uas, second.stars.y_uas)


def test_custom_source_works_in_independent_and_multi_image_light_curves(tmp_path):
    source = mc.CallableSource(
        uniform_brightness,
        source_grid_shape=8,
        field_of_view_uas=2,
        bands_angstrom={"g": 4800, "i": 7500},
    )
    systems = [small_system(source, z) for z in (1.0, 2.0)]
    settings = dict(duration_days=2, map_cadence_days=1, rays=64, temporal_batch_size=2)
    direct = [system.light_curve(**settings) for system in systems]
    batch = mc.batched_system_light_curves(systems, curves_per_batch=2, **settings)
    for curve, expected in zip(batch.light_curves, direct, strict=True):
        torch.testing.assert_close(curve.flux, expected.flux)
    multi = mc.MultiImageSystem(images={"A": systems[0], "B": systems[0].with_seed(1)})
    result = multi.light_curves(**settings)
    torch.testing.assert_close(result["A"].flux, direct[0].flux)
    mixed = mc.batched_system_light_curves(
        [systems[1], multi], curves_per_batch=2, **settings
    )
    assert mixed.completed_systems == 2
    assert mixed.completed_light_curves == 3
    assert mixed.executed_batch_sizes == (2, 1)
    assert mixed.light_curves[1].image_names == ("A", "B")
    torch.testing.assert_close(mixed.light_curves[0].flux, direct[1].flux)
    torch.testing.assert_close(mixed.light_curves[1]["A"].flux, result["A"].flux)
    torch.testing.assert_close(mixed.light_curves[1]["B"].flux, result["B"].flux)
    stored = mc.batched_system_light_curves(
        [systems[1], multi],
        curves_per_batch=2,
        output_path=tmp_path / "mixed.npz",
        **settings,
    )
    assert stored.light_curves == ()
    assert stored.stored_systems[1].image_names == ("A", "B")
    loaded = stored.load_system(1)
    assert loaded.image_names == ("A", "B")
    torch.testing.assert_close(loaded["A"].flux, result["A"].flux)
    torch.testing.assert_close(loaded["B"].flux, result["B"].flux)
    requests = [mc.LightCurveRequest(source, systems[0].distances)]
    shared = systems[0].light_curves(
        [0, 1, 2],
        requests,
        rays=64,
        schedule=mc.production_dynamic_config(temporal_batch_size=2),
    )
    torch.testing.assert_close(shared[0].flux, direct[0].flux)


@pytest.mark.cuda
@pytest.mark.skipif(
    not (
        mc.RuntimeCapabilities.detect().cuda_available
        and mc.RuntimeCapabilities.detect().triton_importable
    ),
    reason="requires CUDA and Triton",
)
def test_deferred_custom_source_torch_and_triton_agree():
    source = mc.CallableSource(
        uniform_brightness,
        source_grid_shape=16,
        field_of_view_uas=2,
        bands_angstrom={"g": 4800, "i": 7500},
    )
    base = small_system(source)
    curves = []
    for backend in ("torch-eager", "triton"):
        system = replace(
            base,
            runtime=mc.RuntimeConfig(
                device="cuda", backend=backend, strict_backend=True
            ),
        )
        curves.append(
            system.light_curve(
                duration_days=2,
                map_cadence_days=1,
                rays=4096,
                temporal_batch_size=2,
            )
        )
    torch.testing.assert_close(curves[0].flux, curves[1].flux, rtol=2e-5, atol=1e-10)
