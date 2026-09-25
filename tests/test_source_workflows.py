"""High-level source geometry and batching agree with explicit calculations."""

from dataclasses import replace
from unittest.mock import patch

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


def test_reprocessing_brightness_reuses_static_evaluation_tensors(monkeypatch):
    model = mc.ThinDiskModel(
        black_hole_mass_solar=1e8,
        eddington_ratio=0.1,
        source_grid_shape=16,
        bands_angstrom={"g": 4800, "i": 7500},
        driving_signal=mc.TabulatedDrivingSignal(
            [-1000.0, 1000.0],
            [1.0, 1.0],
        ),
    )
    source = model.pixelate(
        source_redshift=1.5,
        H0=70,
        Om0=0.3,
        runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
    )
    calls = 0
    original = type(source)._static_temperature4

    def counted(instance, *, device, dtype):
        nonlocal calls
        calls += 1
        return original(instance, device=device, dtype=dtype)

    monkeypatch.setattr(type(source), "_static_temperature4", counted)
    with torch.no_grad():
        source.brightness([-1.0, 0.0])
        source.brightness([1.0, 2.0])
    assert calls == 1
    assert len(source._evaluation_cache) == 1


def test_newtonian_disk_exposes_the_full_reverberation_interface():
    driver = mc.TabulatedDrivingSignal(
        [-100.0, 0.0, 100.0],
        [0.8, 1.2, 0.8],
    )
    model = mc.ThinDiskModel(
        black_hole_mass_solar=1e8,
        eddington_ratio=0.1,
        bands_angstrom={"g": 4800, "i": 7500},
        source_grid_shape=32,
        spin=0.4,
        inclination_deg=35.0,
        position_angle_deg=20.0,
        lamp_fraction=0.1,
        corona_height_above_isco_rg=15.0,
        driving_signal=driver,
    )
    source = model.pixelate(
        source_redshift=1.5,
        H0=70,
        Om0=0.3,
        runtime=mc.RuntimeConfig(device="cpu", backend="torch-eager"),
    )
    assert isinstance(source, mc.ThermalReprocessingSource)
    assert source.heating_metadata["model"] == "axis_newtonian_lamppost"
    brightness = source.brightness([-1.0, 0.0, 1.0])
    assert brightness.shape == (3, 32, 32, 2)
    assert torch.isfinite(brightness).all()
    steady = mc.steady_mean_response_delays(source)
    magnification = torch.stack(
        (torch.ones((32, 32)), torch.linspace(0.5, 1.5, 32).expand(32, -1))
    )
    microlensed = source.batched_mean_response_delays(magnification=magnification)
    assert steady.shape == (2,)
    assert microlensed.shape == (2, 2)
    assert torch.isfinite(microlensed).all()
    torch.testing.assert_close(microlensed[0], steady)
    maximum_delay = source.delay_days[torch.isfinite(source.delay_days)].max()
    edges = torch.linspace(0.0, float(maximum_delay) + 1.0, 65)
    transfer = source.batched_transfer_function(edges, magnification=magnification)
    assert transfer.shape == (2, 64, 2)
    torch.testing.assert_close(transfer.sum(dim=1), torch.ones((2, 2)))


def test_straight_screen_disks_approach_the_kerr_weak_field_image():
    runtime = mc.RuntimeConfig(
        device="cpu",
        backend="torch-eager",
        dtype=torch.float64,
    )
    distances = mc.LensingDistances.from_redshifts(
        0.3, 1.5, H0=70, Om0=0.3, device="cpu", dtype=torch.float64
    )

    def normalized_image(source):
        image = source.brightness([0.0], dtype=torch.float64)[0, ..., 0]
        return image / image.sum()

    for inclination_deg in (1.0e-2, 45.0):
        settings = dict(
            black_hole_mass_solar=1.0e8,
            eddington_ratio=0.1,
            bands_angstrom={"long": 20_000.0},
            source_redshift=1.5,
            spin=0.0,
            inclination_deg=inclination_deg,
            position_angle_deg=0.0,
            source_grid_shape=96,
            enclosed_flux_fraction=0.99,
            source_margin=1.05,
        )
        kerr = mc.KerrDiskModel(**settings, compile_solver=False).pixelate(
            distances, runtime=runtime
        )
        grid = mc.PlaneGrid(
            kerr.geometry.shape,
            tuple(kerr.transfer.metadata["source_field_of_view_uas"]),
        )
        straight = {
            relativity: mc.ThinDiskModel(
                **settings,
                relativity=relativity,
            ).pixelate(distances, grid=grid, runtime=runtime)
            for relativity in ("none", "approximate")
        }
        reference = normalized_image(kerr)
        errors = {
            name: float((normalized_image(source) - reference).abs().sum())
            for name, source in straight.items()
        }
        if inclination_deg < 1.0:
            assert errors["none"] < 0.01
            assert errors["approximate"] < 0.01
        else:
            assert errors["none"] < 0.10
            assert errors["approximate"] < 0.02
            assert errors["approximate"] < 0.25 * errors["none"]


def test_kerr_and_straight_screen_reprocessing_agree_in_controlled_limit():
    runtime = mc.RuntimeConfig(
        device="cpu",
        backend="torch-eager",
        dtype=torch.float64,
    )
    distances = mc.LensingDistances.from_redshifts(
        0.3, 1.5, H0=70, Om0=0.3, device="cpu", dtype=torch.float64
    )
    driver = mc.TabulatedDrivingSignal([-10_000.0, 10_000.0], [1.0, 1.0])
    settings = dict(
        black_hole_mass_solar=1.0e8,
        eddington_ratio=0.1,
        bands_angstrom={"near": 3000.0, "middle": 5000.0, "far": 10_000.0},
        source_redshift=1.5,
        spin=0.0,
        inclination_deg=1.0e-2,
        position_angle_deg=0.0,
        source_grid_shape=96,
        enclosed_flux_fraction=0.95,
        source_margin=1.05,
        lamp_fraction=0.1,
        corona_height_above_isco_rg=100.0,
        driving_signal=driver,
    )
    kerr = mc.KerrDiskModel(
        **settings,
        compile_solver=False,
        lamppost_nalpha=1024,
        lamppost_radial_bins=512,
    ).pixelate(distances, runtime=runtime)
    grid = mc.PlaneGrid(
        kerr.geometry.shape,
        tuple(kerr.transfer.metadata["source_field_of_view_uas"]),
    )
    straight_sources = tuple(
        mc.ThinDiskModel(**settings, relativity=relativity).pixelate(
            distances,
            grid=grid,
            runtime=runtime,
        )
        for relativity in ("none", "approximate")
    )
    sources = (kerr, *straight_sources)
    maximum_delay = max(
        float(source.delay_days[torch.isfinite(source.delay_days)].max())
        for source in sources
    )
    edges = torch.linspace(0.0, maximum_delay + 1.0, 257, dtype=torch.float64)
    kerr_lags = kerr.mean_response_delays()
    kerr_differential = kerr_lags - kerr_lags[0]
    kerr_transfer = kerr.transfer_function(edges)
    torch.testing.assert_close(
        kerr_transfer.sum(dim=0),
        torch.ones(3, dtype=torch.float64),
    )
    for source in straight_sources:
        lags = source.mean_response_delays()
        differential = lags - lags[0]
        assert bool(torch.all(differential[1:] > 0.0))
        relative_error = (
            (differential[1:] - kerr_differential[1:]).abs()
            / kerr_differential[1:]
        )
        assert float(relative_error.max()) < 0.25
        transfer = source.transfer_function(edges)
        torch.testing.assert_close(
            transfer.sum(dim=0),
            torch.ones(3, dtype=torch.float64),
        )
        cdf_error = (transfer.cumsum(0) - kerr_transfer.cumsum(0)).abs()
        assert float(cdf_error.max()) < 0.15


def test_batched_kerr_source_setup_matches_serial_sources():
    runtime = mc.RuntimeConfig(device="cpu", backend="torch-eager", dtype=torch.float32)
    distances = mc.LensingDistances.from_redshifts(
        0.3, 1.5, device="cpu", dtype=torch.float32
    )
    driver = mc.TabulatedDrivingSignal([-100.0, 100.0], [1.0, 1.0])
    base = mc.KerrDiskModel(
        black_hole_mass_solar=1e8,
        eddington_ratio=0.1,
        bands_angstrom={"g": 4800},
        source_grid_shape=8,
        compile_solver=False,
        lamppost_nalpha=32,
        lamppost_radial_bins=32,
        driving_signal=driver,
    )
    models = (base, replace(base, spin=0.4, inclination_deg=30.0))
    expected = tuple(model.pixelate(distances, runtime=runtime) for model in models)
    actual = mc.batched_pixelate_sources(
        models, distances, batch_size=2, runtime=runtime
    )
    assert len(actual) == len(expected)
    for source, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(source.transfer.hit, reference.transfer.hit)
        torch.testing.assert_close(
            source.delay_days, reference.delay_days, equal_nan=True
        )
        torch.testing.assert_close(source.brightness([0]), reference.brightness([0]))
    with pytest.raises(ValueError, match="batch_size"):
        mc.batched_pixelate_sources(models, distances, batch_size=0, runtime=runtime)


def test_wide_kerr_disk_places_coordinate_observer_outside_emission():
    runtime = mc.RuntimeConfig(device="cpu", backend="torch-eager", dtype=torch.float32)
    distances = mc.LensingDistances.from_redshifts(
        0.3, 1.301, device="cpu", dtype=torch.float32
    )
    model = mc.KerrDiskModel(
        black_hole_mass_solar=4.8e7,
        eddington_ratio=0.34,
        bands_angstrom={"red": 10990.0},
        spin=-0.277,
        inclination_deg=33.28,
        source_grid_shape=64,
        compile_solver=False,
        lamppost_nalpha=32,
        lamppost_radial_bins=32,
        driving_signal=mc.TabulatedDrivingSignal([-100.0, 100.0], [1.0, 1.0]),
    )
    serial = model.pixelate(distances, runtime=runtime)
    batched = mc.batched_pixelate_sources(
        (model, replace(model, spin=-0.2)), distances, batch_size=2, runtime=runtime
    )
    for source in (serial, *batched):
        metadata = source.transfer.metadata
        assert metadata["disk_outer_rg"] > 3000.0
        assert metadata["observer_radius_rg"] > metadata["disk_outer_rg"]
        assert bool(source.transfer.hit.any())
        assert bool(torch.isfinite(source.delay_days[source.transfer.hit]).all())
    torch.testing.assert_close(serial.delay_days, batched[0].delay_days, equal_nan=True)


def test_system_source_setup_batch_counts_one_shared_multi_image_disk():
    from microcaustics.batching import _batch_pixelate_system_sources

    model = mc.KerrDiskModel(
        black_hole_mass_solar=1e8,
        eddington_ratio=0.1,
        bands_angstrom={"g": 4800},
        source_grid_shape=8,
        compile_solver=False,
    )
    first = system(model)
    multi = mc.MultiImageSystem(
        images={"A": first, "B": first.with_seed(1)}, source=model
    )
    geometry = mc.SourceGeometry((4, 4), (1.0, 1.0), (4800.0,), ("g",))
    resolved = mc.StaticSource(torch.ones((4, 4, 1)), geometry)
    with patch(
        "microcaustics.sources.batched_pixelate_sources",
        return_value=(resolved, resolved),
    ) as pixelate:
        updated = _batch_pixelate_system_sources((first, multi), 3)
    models = pixelate.call_args.args[0]
    assert len(models) == 2
    assert updated[0].source is resolved
    assert updated[1].source is resolved
    assert all(image.source is resolved for image in updated[1].images.values())


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
        keep_maps_at_days=[0.0, 4.0],
        **settings,
    )
    assert epochs == [0, 2, 4]
    assert result[0].map_times_days.tolist() == [0.0, 4.0]
    assert result[1].map_times_days.tolist() == [0.0, 4.0]
    assert result[0].maps[0] is result[1].maps[0]
    assert result[0].maps[1] is result[1].maps[1]
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
