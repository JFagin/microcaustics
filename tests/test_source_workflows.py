"""High-level source geometry and batching agree with explicit calculations."""

from dataclasses import replace

import pytest
import torch

import microcaustics as mc
from microcaustics.photometry import streaming_light_curves


def photosphere(**kwargs):
    return mc.ExpandingPhotosphereSource(
        evolution=mc.PowerLawExponentialPhotosphere(
            peak_time_rest_days=15,
            peak_luminosity_watts=1e36,
            rise_power=2,
            decline_time_rest_days=25,
            photosphere_velocity_km_s=10000,
            initial_radius_m=1e9,
            temperature_floor_k=3000,
            temperature_ceiling_k=15000,
        ),
        bands_angstrom={"g": 4800, "r": 6200},
        maximum_observer_time_days=100,
        source_grid_shape=16,
        **kwargs,
    )


def system(source, *, dtype=torch.float32, redshift=1.5):
    return mc.MicrolensingSystem(
        lens_redshift=0.3,
        source_redshift=redshift,
        H0=70,
        Om0=0.3,
        macro=mc.MacroLens(convergence=0, shear=0),
        stars=mc.PointMassField([0.0], [0.0], [1e-10]),
        source=source,
        runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager", dtype=dtype),
        seed=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_supernova_inherits_system_geometry_without_mutating_model(dtype):
    model = photosphere(
        source_margin=1.1, appearance=mc.PhotosphereAppearance(position_angle_deg=25)
    )
    first = system(model, dtype=dtype)
    resolved = first.realize().source
    assert model.geometry is None and model.source_redshift is None
    assert resolved.redshift == first.distances.source_redshift
    assert (
        resolved.luminosity_distance_m
        == first.distances.source_m * (1 + resolved.redshift) ** 2
    )
    reference = photosphere(
        source_redshift=1.5,
        source_margin=1.1,
        appearance=mc.PhotosphereAppearance(position_angle_deg=25),
        luminosity_distance_m=resolved.luminosity_distance_m,
    )
    torch.testing.assert_close(
        resolved.brightness([10, 30, 70], dtype=dtype),
        reference.brightness([10, 30, 70], dtype=dtype),
    )
    assert resolved.geometry == reference.geometry
    other = system(model, dtype=dtype, redshift=2).realize().source
    assert other.geometry != resolved.geometry
    assert other.luminosity_distance_m != resolved.luminosity_distance_m
    assert first._realize_for_times([0, 25]).source.geometry == resolved.geometry
    with pytest.raises(ValueError, match="maximum_observer_time_days"):
        resolved.brightness([101])
    with pytest.raises(ValueError, match="resolve"):
        model.brightness([10])


def test_supernova_custom_distance_and_conflict_checks():
    model = photosphere(source_redshift=1.5, luminosity_distance_m=4e25)
    assert system(model).realize().source.luminosity_distance_m == 4e25
    with pytest.raises(ValueError, match="redshift differs"):
        system(model, redshift=2).realize()
    with pytest.raises(ValueError, match="not both"):
        model.pixelate(system(model).distances, H0=70)
    with pytest.raises(ValueError, match="enclose"):
        model.pixelate(
            system(model).distances, grid=mc.PlaneGrid((16, 16), (1e-8, 1e-8))
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["thin", "kerr", "gaussian"])
def test_standalone_sources_match_system_distances(kind, dtype):
    runtime = mc.RuntimeConfig(device="cpu", backend="torch-eager", dtype=dtype)
    settings = dict(bands_angstrom={"g": 4800}, source_grid_shape=16)
    if kind == "gaussian":
        model = mc.GaussianModel(sigma_uas=0.3, **settings)
    else:
        cls = mc.ThinDiskModel if kind == "thin" else mc.KerrDiskModel
        model = cls(
            black_hole_mass_solar=1e8,
            eddington_ratio=0.1,
            **settings,
            **({"compile_solver": False} if kind == "kerr" else {}),
        )
    distances = mc.LensingDistances.from_redshifts(
        0.3, 1.5, H0=70, Om0=0.3, device="cpu", dtype=dtype
    )
    expected = model.pixelate(distances, runtime=runtime)
    actual = model.pixelate(source_redshift=1.5, H0=70, Om0=0.3, runtime=runtime)
    torch.testing.assert_close(
        actual.brightness([0], dtype=dtype),
        expected.brightness([0], dtype=dtype),
        rtol=1e-5,
        atol=0,
    )
    assert actual.geometry.pixel_scale_m == pytest.approx(
        expected.geometry.pixel_scale_m, rel=1e-7
    )
    if kind == "kerr":
        torch.testing.assert_close(
            actual.transfer.gfactor, expected.transfer.gfactor, equal_nan=True
        )
    with pytest.raises(ValueError, match="source_redshift"):
        model.pixelate(runtime=runtime)


def test_standalone_reverberation_transfer_requires_no_lens():
    model = mc.KerrDiskModel(
        black_hole_mass_solar=1e8,
        eddington_ratio=0.1,
        source_grid_shape=16,
        bands_angstrom={"g": 4800},
        compile_solver=False,
        lamppost_nalpha=128,
        lamppost_radial_bins=64,
        driving_signal=mc.TabulatedDrivingSignal([-100.0, 100.0], [1.0, 1.0]),
    )
    source = model.pixelate(
        source_redshift=1.5,
        H0=70,
        Om0=0.3,
        runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
    )
    edges = torch.linspace(
        0, float(source.delay_days[torch.isfinite(source.delay_days)].max()) + 1, 64
    )
    result = mc.steady_transfer_function(source, edges)
    torch.testing.assert_close(result.values, source.transfer_function(edges))
    assert torch.isfinite(result.values).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_multirate_batch_reuses_maps_and_matches_individual_curves(dtype):
    base = mc.GaussianModel(
        sigma_uas=0.25, bands_angstrom={"g": 4800}, source_grid_shape=16
    )
    source = mc.ModulatedSource(
        base,
        mc.TabulatedDrivingSignal([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 0.5, 1.5, 1.0]),
    )
    owner = system(source, dtype=dtype)
    trajectory = mc.LinearTrajectory(velocity_uas_per_day=(0.01, 0))
    requests = [
        mc.LightCurveRequest(source),
        mc.LightCurveRequest(source, trajectory=trajectory),
    ]
    settings = dict(
        duration_days=4,
        map_cadence_days=2,
        source_cadence_days=0.5,
        rays=64,
        temporal_batch_size=2,
    )
    expected = [
        owner.light_curve(trajectory=r.trajectory, **settings) for r in requests
    ]
    epochs = []
    result = owner.light_curves(
        requests=requests,
        light_curve_batch_size=2,
        map_observer=lambda i, m: epochs.append(m.time_days),
        **settings,
    )
    assert epochs == [0, 2, 4]
    assert all(r.distances is None for r in requests)
    for actual, reference in zip(result, expected, strict=True):
        torch.testing.assert_close(actual.flux, reference.flux, rtol=2e-5, atol=1e-12)
        torch.testing.assert_close(actual.unlensed_flux, reference.unlensed_flux)
        assert actual.metadata["shared_map_request_count"] == 2
        assert actual.metadata["map_epochs"] == 3
        assert actual.metadata["flux_epochs"] == 9


def test_shared_batch_irregular_times_and_static_field():
    source = mc.GaussianModel(
        sigma_uas=0.25, bands_angstrom={"g": 4800}, source_grid_shape=8
    )
    owner = system(source)
    for maps, flux in [([0, 2, 4], [0.1, 0.9, 3.7]), ([0], None)]:
        calls = []
        result = owner.light_curves(
            maps,
            [mc.LightCurveRequest(source)],
            flux_times_days=flux,
            rays=64,
            map_observer=lambda i, m, calls=calls: calls.append(i),
        )
        expected = owner.light_curve(maps, flux_times_days=flux, rays=64)
        torch.testing.assert_close(result[0].flux, expected.flux)
        assert len(calls) == len(maps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_multirate_interpolation_matches_known_time_varying_maps(dtype):
    model = mc.ModulatedSource(
        mc.GaussianModel(
            sigma_uas=0.2, bands_angstrom={"g": 4800}, source_grid_shape=8
        ),
        mc.TabulatedDrivingSignal([0.0, 4.0], [1.0, 2.0]),
    )
    owner = system(model, dtype=dtype)
    realization = owner.realize()
    epochs = torch.tensor([0, 0.5, 2, 3.5, 4], dtype=dtype)
    generated = []

    def maps():
        for time, value in zip([0, 2, 4], [2, 6, 3], strict=True):
            generated.append(time)
            yield mc.MagnificationMap(
                torch.full(realization.source_grid.shape, value, dtype=dtype),
                realization.source_grid,
                time_days=time,
            )

    result = streaming_light_curves(
        realization.simulation,
        realization.lens_region,
        realization.source_grid,
        [0, 2, 4],
        [mc.LightCurveRequest(realization.source, owner.distances)],
        method=mc.production_ipm_config(rays=64),
        flux_times_days=epochs,
        _map_iterator=maps(),
    )[0]
    brightness = realization.source.brightness(epochs, dtype=dtype)
    area = (
        realization.source.geometry.pixel_scale_m[0]
        * realization.source.geometry.pixel_scale_m[1]
    )
    unlensed = brightness.sum(dim=(1, 2)) * area
    expected = unlensed * torch.tensor([2, 3, 6, 3.75, 3], dtype=dtype)[:, None]
    torch.testing.assert_close(result.flux, expected)
    torch.testing.assert_close(result.unlensed_flux, unlensed)
    assert generated == [0, 2, 4]


def test_static_batch_plain_rays_and_advanced_method_agree():
    owner = system(
        mc.GaussianModel(sigma_uas=0.2, bands_angstrom={"g": 4800}, source_grid_shape=8)
    )
    systems = [owner, owner.with_seed(1)]
    actual = mc.batched_system_maps(systems, rays=64, batch_size=2)
    expected = mc.batched_system_maps(
        systems, method=mc.production_ipm_config(rays=64, scout_ratio=1), batch_size=2
    )
    for first, second in zip(actual, expected, strict=True):
        torch.testing.assert_close(first.values, second.values)
    with pytest.raises(TypeError, match="unsupported"):
        mc.batched_system_maps(systems, bad_option=1)


def test_tuner_accepts_same_plain_controls_as_independent_batches():
    owner = system(
        mc.GaussianModel(sigma_uas=0.2, bands_angstrom={"g": 4800}, source_grid_shape=8)
    )
    result = mc.tune_system_light_curve_batch(
        [owner, owner.with_seed(1)],
        duration_days=2,
        map_cadence_days=1,
        source_cadence_days=0.5,
        rays=64,
        temporal_batch_size=2,
        candidates=(1, 2),
    )
    assert all(trial.accepted for trial in result.trials)


@pytest.mark.cuda
def test_cuda_shared_multirate_batch_matches_torch():
    capabilities = mc.RuntimeCapabilities.detect()
    if not (capabilities.cuda_available and capabilities.triton_importable):
        pytest.skip("requires CUDA and Triton")
    model = mc.GaussianModel(
        sigma_uas=0.2, bands_angstrom={"g": 4800}, source_grid_shape=16
    )
    curves = []
    for backend in ("torch-eager", "triton"):
        owner = replace(
            system(model), runtime=mc.RuntimeConfig(device="cuda", backend=backend)
        )
        curves.append(
            owner.light_curves(
                requests=[mc.LightCurveRequest(model)],
                duration_days=2,
                map_cadence_days=1,
                source_cadence_days=0.5,
                rays=4096,
                temporal_batch_size=2,
            )[0]
        )
    torch.testing.assert_close(curves[0].flux, curves[1].flux, rtol=3e-5, atol=1e-10)
