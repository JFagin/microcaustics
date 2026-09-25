"""Optimized spectral contractions preserve dense photometry and driver semantics."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import microcaustics as mc
from microcaustics.bandpasses import BandpassGrid
from microcaustics.compile import run_tensor_kernel
from microcaustics.photometry import (
    _flexible_streaming_light_curves,
    _source_band_chunks,
)
from microcaustics.sources.linear_response import (
    LinearResponsePlan,
    _project_response_chunk,
)
from microcaustics.sources.transferred_disk import (
    _transferred_arguments,
    _transferred_brightness_kernel,
    _transferred_flux_kernel,
)
from microcaustics.sources.triton_thermal_flux import triton_thermal_flux


def test_thermal_flux_block_pixels_are_validated_and_resolved():
    assert mc.RuntimeConfig().thermal_flux_block_pixels is None
    for pixels in (128, 256, 512, 1024):
        config = mc.RuntimeConfig(
            device="cpu", backend="torch-eager", thermal_flux_block_pixels=pixels
        )
        assert mc.resolve_runtime(config).thermal_flux_block_pixels == pixels
    for invalid in (0, 64, 2048, True, 256.0):
        with pytest.raises(ValueError, match="thermal_flux_block_pixels"):
            mc.RuntimeConfig(thermal_flux_block_pixels=invalid)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "bands,block_pixels,shape",
    [
        (8, 512, (128, 160)),
        (16, 128, (128, 160)),
        (16, 256, (128, 160)),
        (16, 512, (128, 160)),
        (16, 1024, (128, 160)),
        (16, 1024, (17, 19)),
    ],
)
def test_thermal_flux_block_override_preserves_flux(bands, block_pixels, shape):
    temperature = torch.full((2, *shape), 1e16, device="cuda")
    gfactor = torch.full(shape, 0.9, device="cuda")
    solid_angle = torch.full(shape, 1e-25, device="cuda")
    hit = torch.zeros(shape, dtype=torch.bool, device="cuda")
    if shape == (17, 19):
        hit[2:15, 3:17] = True
    else:
        hit[16:96, 32:128] = True
    wavelength = torch.linspace(3e-7, 1.1e-6, bands, device="cuda")
    scalars = tuple(torch.tensor(value, device="cuda") for value in (1.2, 0.08, 1e-26))
    arguments = (temperature, gfactor, solid_angle, hit, wavelength, *scalars)
    left = torch.full((1, *shape), 1.3, device="cuda")
    right = torch.full((1, *shape), 1.7, device="cuda")
    fraction = torch.tensor([0.0, 1.0], device="cuda")
    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(
            device="cuda",
            backend="triton",
            dtype=torch.float32,
            thermal_flux_block_pixels=block_pixels,
        )
    )
    with torch.inference_mode():
        actual = triton_thermal_flux(left, right, fraction, arguments, runtime)
        assert actual is not None
        expected = _transferred_flux_kernel(left, right, fraction, *arguments)
        for observed, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(observed, reference, rtol=2e-5, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_active_tile_cache_separates_user_selected_block_sizes():
    from microcaustics.sources.triton_thermal_flux import _active_source_tiles

    hit = torch.zeros((128, 160), dtype=torch.bool, device="cuda")
    hit[16:96, 32:128] = True
    small = _active_source_tiles(hit, 256)
    large = _active_source_tiles(hit, 512)
    assert small is not None and large is not None
    assert small[0].numel() == large[0].numel() == hit.numel()
    assert small[1].numel() > large[1].numel()
    assert _active_source_tiles(hit, 256) is small
    assert _active_source_tiles(hit, 512) is large


@pytest.mark.parametrize("shape", [(32, 32), (17, 19)])
@pytest.mark.parametrize("map_kind", ["shared", "evolving", "unity"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is unavailable"
                ),
            ],
        ),
    ],
)
def test_tiled_thermal_reduction_matches_dense_values_and_gradients(
    shape, map_kind, device
):
    """Independent dense sums cover tiled/irregular grids and broadcast maps."""
    generator = torch.Generator(device=device).manual_seed(92)

    def rand(*sizes):
        return torch.rand(
            *sizes, device=device, dtype=torch.float64, generator=generator
        )

    times = 3
    temperature = (1e16 * (0.5 + rand(times, *shape))).requires_grad_()
    gfactor = (0.8 + rand(*shape)).requires_grad_()
    solid_angle = 1e-25 * (0.5 + rand(*shape))
    hit = rand(*shape) > 0.15
    wavelength = torch.tensor(
        [2e-7, 4e-7, 8e-7], device=device, dtype=torch.float64, requires_grad=True
    )
    args = (
        temperature,
        gfactor,
        solid_angle,
        hit,
        wavelength,
        torch.tensor(1.2, device=device, dtype=torch.float64),
        torch.tensor(0.08, device=device, dtype=torch.float64),
        torch.tensor(1e-26, device=device, dtype=torch.float64),
    )
    map_shape = (
        (1, 1, 1)
        if map_kind == "unity"
        else ((times if map_kind == "evolving" else 1), *shape)
    )
    left = (0.5 + rand(*map_shape)).requires_grad_()
    right = 0.5 + rand(*map_shape)
    fraction = torch.tensor([0.0, 0.4, 1.0], device=device, dtype=torch.float64)
    if device == "cuda":
        from microcaustics.compile import run_tensor_kernel

        runtime = mc.resolve_runtime(
            mc.RuntimeConfig(
                device=device,
                backend="torch-compile",
                dtype=torch.float64,
                strict_backend=True,
            )
        )
        actual, compiled = run_tensor_kernel(
            runtime,
            "thermal reduction test",
            _transferred_flux_kernel,
            left,
            right,
            fraction,
            *args,
        )
        assert compiled
    else:
        actual = _transferred_flux_kernel(left, right, fraction, *args)
    brightness = _transferred_brightness_kernel(*args)
    magnification = left + fraction[:, None, None] * (right - left)
    expected = (
        (brightness * magnification[..., None]).sum((1, 2)),
        brightness.sum((1, 2)),
    )
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-12, atol=0)
    inputs = (temperature, gfactor, wavelength, left)
    gradients = torch.autograd.grad(sum(value.sum() for value in actual), inputs)
    reference_gradients = torch.autograd.grad(
        sum(value.sum() for value in expected), inputs
    )
    for a, b in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-12, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "shape,bands",
    [((17, 19), 6), ((32, 32), 8), ((17, 19), 16), ((32, 32), 16)],
)
@pytest.mark.parametrize("map_kind", ["shared", "evolving", "unity"])
def test_triton_thermal_flux_matches_dense_source_and_map_contract(
    shape, bands, map_kind
):
    generator = torch.Generator(device="cuda").manual_seed(113)

    def rand(*sizes):
        return torch.rand(
            *sizes, device="cuda", dtype=torch.float32, generator=generator
        )

    epochs = 3
    temperature = 1e16 * (0.5 + rand(epochs, *shape))
    gfactor = 0.8 + rand(*shape)
    solid_angle = 1e-25 * (0.5 + rand(*shape))
    hit = rand(*shape) > 0.15
    wavelength = torch.linspace(2e-7, 8e-7, bands, device="cuda")
    scalars = tuple(torch.tensor(value, device="cuda") for value in (1.2, 0.08, 1e-26))
    arguments = (temperature, gfactor, solid_angle, hit, wavelength, *scalars)
    map_shape = (
        (1, 1, 1)
        if map_kind == "unity"
        else ((epochs if map_kind == "evolving" else 1), *shape)
    )
    left, right = 0.5 + rand(*map_shape), 0.5 + rand(*map_shape)
    fraction = torch.tensor([0.0, 0.4, 1.0], device="cuda")
    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(device="cuda", backend="triton", dtype=torch.float32)
    )
    with torch.inference_mode():
        actual = triton_thermal_flux(left, right, fraction, arguments, runtime)
        assert actual is not None
        expected = _transferred_flux_kernel(left, right, fraction, *arguments)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=0)
    # Autograd must retain the differentiable compiled/eager PyTorch path.
    assert triton_thermal_flux(left, right, fraction, arguments, runtime) is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("map_kind", ["shared", "evolving", "unity"])
@pytest.mark.parametrize("source_times", [1, 3])
def test_exact_compact_thermal_tiles_follow_hit_updates(map_kind, source_times):
    from microcaustics.sources.triton_thermal_flux import _active_source_tiles

    generator = torch.Generator(device="cuda").manual_seed(198)
    shape = (128, 160)
    epochs = 3
    temperature = (
        1e16
        * (0.5 + torch.rand(source_times, *shape, device="cuda", generator=generator))
    )
    gfactor = torch.full(shape, 0.9, device="cuda")
    solid_angle = torch.full(shape, 1e-25, device="cuda")
    hit = torch.zeros(shape, device="cuda", dtype=torch.bool)
    hit[16:96, 32:128] = True
    wavelength = torch.linspace(3e-7, 1.1e-6, 16, device="cuda")
    scalars = tuple(torch.tensor(value, device="cuda") for value in (1.2, 0.08, 1e-26))
    map_shape = (
        (1, 1, 1)
        if map_kind == "unity"
        else ((epochs if map_kind == "evolving" else 1), *shape)
    )
    left = 0.5 + torch.rand(*map_shape, device="cuda", generator=generator)
    right = 0.5 + torch.rand(*map_shape, device="cuda", generator=generator)
    fraction = torch.tensor([0.0, 0.4, 1.0], device="cuda")
    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(device="cuda", backend="triton", dtype=torch.float32)
    )
    with torch.inference_mode():
        for step in range(2):
            if step:
                hit[:16, :32] = True
            tiles = _active_source_tiles(hit)
            assert tiles is not None
            assert tiles[1].numel() == 15 + step
            arguments = (
                temperature,
                gfactor,
                solid_angle,
                hit,
                wavelength,
                *scalars,
            )
            actual = triton_thermal_flux(left, right, fraction, arguments, runtime)
            expected = _transferred_flux_kernel(left, right, fraction, *arguments)
            assert actual is not None
            for observed, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(
                    observed, reference.expand_as(observed), rtol=2e-5, atol=0
                )
        alternate = torch.cuda.Stream()
        alternate.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(alternate):
            reused = triton_thermal_flux(left, right, fraction, arguments, runtime)
        alternate.synchronize()
        for observed, reference in zip(reused, expected, strict=True):
            torch.testing.assert_close(
                observed, reference.expand_as(observed), rtol=2e-5, atol=0
            )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compact_tiles_handle_empty_and_full_observer_masks():
    from microcaustics.sources.triton_thermal_flux import _active_source_tiles

    shape = (128, 128)
    temperature = torch.full((1, *shape), 1e16, device="cuda")
    gfactor = torch.full(shape, 0.9, device="cuda")
    solid_angle = torch.full(shape, 1e-25, device="cuda")
    hit = torch.zeros(shape, device="cuda", dtype=torch.bool)
    wavelength = torch.linspace(3e-7, 1.1e-6, 16, device="cuda")
    scalars = tuple(torch.tensor(value, device="cuda") for value in (1.2, 0.08, 1e-26))
    magnification = torch.ones((1, 1, 1), device="cuda")
    fraction = torch.tensor([0.0, 0.5, 1.0], device="cuda")
    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(device="cuda", backend="triton", dtype=torch.float32)
    )
    with torch.inference_mode():
        arguments = (
            temperature,
            gfactor,
            solid_angle,
            hit,
            wavelength,
            *scalars,
        )
        tiles = _active_source_tiles(hit)
        assert tiles is not None and tiles[1].numel() == 0
        result = triton_thermal_flux(
            magnification, magnification, fraction, arguments, runtime
        )
        assert all(torch.count_nonzero(value) == 0 for value in result)

        hit.fill_(True)
        assert _active_source_tiles(hit) is None
        result = triton_thermal_flux(
            magnification, magnification, fraction, arguments, runtime
        )
        expected = _transferred_flux_kernel(
            magnification, magnification, fraction, *arguments
        )
        for observed, reference in zip(result, expected, strict=True):
            torch.testing.assert_close(
                observed, reference.expand_as(observed), rtol=2e-5, atol=0
            )


def test_cached_spectral_constants_follow_source_and_tensor_parameters():
    source, _, _, _ = thermal_fixture()
    with torch.no_grad():
        original = source.brightness([0.0])
        assert len(source._spectral_cache) == 1
        cached = next(iter(source._spectral_cache.values()))[0]
        source.brightness([1.0])
        assert next(iter(source._spectral_cache.values()))[0] is cached

        shifted = source.with_bands({"shifted": 9500.0})
        assert not shifted._spectral_cache
        assert not torch.equal(original[..., :1], shifted.brightness([0.0]))

        redshift = torch.tensor(1.5, dtype=torch.float64)
        tensor_parameter_source = replace(source, source_redshift=redshift)
        before = tensor_parameter_source.brightness([0.0])
        redshift.add_(0.1)
        after = tensor_parameter_source.brightness([0.0])
        assert not torch.equal(before, after)
        assert not tensor_parameter_source._spectral_cache


def test_prepared_transfer_is_reused_without_revalidating_observer_pixels():
    source, _, _, _ = thermal_fixture()
    with torch.no_grad():
        state = source._brightness_state([0.0], dtype=torch.float64)
        temperature, transfer, _ = state
        with patch.object(
            mc.ObserverTransfer, "to", side_effect=AssertionError("transfer rebuilt")
        ):
            arguments, _ = _transferred_arguments(
                temperature,
                geometry=source.geometry,
                transfer=transfer,
                source_redshift=source.source_redshift,
                color_correction=source.color_correction,
            )
        assert arguments[1] is transfer.gfactor
        # A genuinely new dtype still gets a converted transfer.
        converted, _ = _transferred_arguments(
            temperature.float(),
            geometry=source.geometry,
            transfer=transfer,
            source_redshift=source.source_redshift,
            color_correction=source.color_correction,
        )
        assert converted[1].dtype == torch.float32


def thermal_fixture():
    distances = mc.LensingDistances.from_redshifts(0.5, 1.5)
    grid = mc.PlaneGrid((4, 4), (2.0, 2.0))
    geometry = mc.SourceGeometry(
        4,
        field_of_view_uas=2.0,
        bands_angstrom={f"b{i}": 4000.0 + 600.0 * i for i in range(5)},
    ).resolve(distances)
    image = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    transfer = mc.ObserverTransfer(
        20.0 + image,
        torch.ones_like(image),
        torch.ones_like(image),
        torch.ones_like(image, dtype=torch.bool),
    )
    signal = mc.TabulatedDrivingSignal(
        torch.tensor([-2.0, 0.0, 1.0, 2.0], dtype=torch.float64),
        torch.tensor([0.7, 1.0, 1.4, 0.9], dtype=torch.float64),
    )
    source = mc.ThermalReprocessingSource(
        geometry,
        transfer,
        signal,
        torch.full_like(image, 1e20),
        image / 50.0,
        black_hole_mass_solar=1e8,
        eddington_ratio=0.1,
        spin=0.0,
        source_redshift=1.5,
    )
    maps = tuple(
        mc.MagnificationMap(
            1.0 + image / (10.0 + i) + 0.3 * i, grid, time_days=float(i)
        )
        for i in range(3)
    )
    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(device="cpu", backend="torch-eager", dtype=torch.float64)
    )
    return source, distances, maps, runtime


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("moving", [False, True])
def test_batched_spectral_flux_matches_dense_reference(static, moving):
    source, distances, maps, runtime = thermal_fixture()
    if static:
        source = source.at_driver_mean()
    trajectory = (
        mc.LinearTrajectory(velocity_uas_per_day=(0.03, 0.02)) if moving else None
    )
    times = torch.tensor([0.0, 0.25, 0.5, 1.0, 1.75, 2.0], dtype=torch.float64)
    sampled_maps = []
    for t in times:
        right = max(1, min(2, int(torch.ceil(t))))
        weight = t - (right - 1)
        values = maps[right - 1].values + weight * (
            maps[right].values - maps[right - 1].values
        )
        sampled_maps.append(
            mc.MagnificationMap(values, maps[0].grid, time_days=float(t))
        )
    # Dense reference explicitly evaluates every wavelength and flux epoch.
    expected = mc.light_curve_from_maps(
        sampled_maps,
        source,
        times,
        distances,
        trajectory=trajectory,
        strict_coverage=False,
        batch_size=2,
    )
    original = mc.ThermalReprocessingSource._brightness_state
    with patch.object(
        mc.ThermalReprocessingSource,
        "_brightness_state",
        autospec=True,
        side_effect=original,
    ) as prepare:
        (actual,) = _flexible_streaming_light_curves(
            SimpleNamespace(runtime=runtime),
            None,
            maps[0].grid,
            [0.0, 1.0, 2.0],
            (
                mc.LightCurveRequest(
                    source, distances, trajectory, strict_coverage=False
                ),
            ),
            (times,),
            method=None,
            schedule=mc.DynamicConfig(temporal_batch_size=3),
            map_observer=None,
            band_batch_size=2,
            map_iterator=iter(maps),
        )
    # The five wavelengths (including the padded last chunk) share one heating
    # state per temporal batch. A static disk prepares it just once in total.
    assert prepare.call_count == (1 if static else 3)
    torch.testing.assert_close(actual.flux, expected.flux, rtol=1e-12, atol=0)
    torch.testing.assert_close(
        actual.unlensed_flux, expected.unlensed_flux, rtol=1e-12, atol=0
    )
    assert actual.metadata["map_aligned_source_fast_path"] is (not moving)


@pytest.mark.parametrize("varying", [False, True])
def test_linear_response_preserves_daily_driver_and_map_interpolation(varying):
    source, distances, maps, runtime = thermal_fixture()
    values = [0.99, 1.0, 1.01, 0.995] if varying else [1.0] * 4
    source = replace(
        source,
        signal=mc.TabulatedDrivingSignal(
            torch.tensor([-2.0, 0.0, 1.0, 2.0], dtype=torch.float64),
            torch.tensor(values, dtype=torch.float64),
        ),
    )
    times = torch.tensor([0.0, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0], dtype=torch.float64)

    def evaluate(mode):
        (curve,) = _flexible_streaming_light_curves(
            SimpleNamespace(runtime=runtime),
            None,
            maps[0].grid,
            [0.0, 1.0, 2.0],
            (
                mc.LightCurveRequest(
                    source,
                    distances,
                    source_evolution=mode,
                    response_delay_bin_days=0.025,
                ),
            ),
            (times,),
            method=None,
            schedule=mc.DynamicConfig(temporal_batch_size=3),
            map_observer=None,
            band_batch_size=2,
            map_iterator=iter(maps),
        )
        return curve

    exact = evaluate("exact")
    approximate = evaluate("linear_response")
    tolerance = 1e-12 if not varying else 2e-3
    torch.testing.assert_close(
        approximate.flux, exact.flux, rtol=tolerance, atol=0
    )
    torch.testing.assert_close(
        approximate.unlensed_flux, exact.unlensed_flux,
        rtol=tolerance, atol=0,
    )
    assert approximate.metadata["source_evolution"] == "linear_response"
    assert approximate.times_days.numel() == times.numel()


def test_linear_response_rejects_moving_source_and_invalid_mode():
    with pytest.raises(ValueError, match="source_evolution"):
        mc.LightCurveRequest(source_evolution="other")
    source, distances, maps, runtime = thermal_fixture()
    with pytest.raises(ValueError, match="stationary source trajectory"):
        _flexible_streaming_light_curves(
            SimpleNamespace(runtime=runtime), None, maps[0].grid, [0.0, 1.0],
            (
                mc.LightCurveRequest(
                    source, distances,
                    trajectory=mc.LinearTrajectory(velocity_uas_per_day=(0.01, 0.0)),
                    source_evolution="linear_response",
                ),
            ),
            ([0.0, 0.5, 1.0],), method=None, schedule=None,
            map_observer=None, band_batch_size=2, map_iterator=iter(maps[:2]),
        )


def test_linear_response_respects_macroimage_arrival_shift():
    source, distances, maps, runtime = thermal_fixture()
    source = replace(
        source,
        signal=mc.TabulatedDrivingSignal(
            torch.tensor([-3.0, -2.0, 0.0, 1.0, 2.0], dtype=torch.float64),
            torch.tensor([1.0, 0.998, 1.0, 1.004, 0.999], dtype=torch.float64),
        ),
    )
    shifted = mc.TimeShiftedSource(source, 0.5)
    times = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64)

    def evaluate(mode):
        (curve,) = _flexible_streaming_light_curves(
            SimpleNamespace(runtime=runtime), None, maps[0].grid,
            [0.0, 1.0, 2.0],
            (mc.LightCurveRequest(shifted, distances, source_evolution=mode),),
            (times,), method=None, schedule=None, map_observer=None,
            band_batch_size=2, map_iterator=iter(maps),
        )
        return curve

    torch.testing.assert_close(
        evaluate("linear_response").flux,
        evaluate("exact").flux,
        rtol=2e-3, atol=0,
    )


def test_linear_response_delay_deposition_preserves_mean_lag():
    source, _, maps, _ = thermal_fixture()
    times = torch.tensor([0.0, 1.0], dtype=torch.float64)
    with torch.no_grad():
        plan = LinearResponsePlan(
            source, ((source, len(source.geometry.band_names)),),
            times, delay_bin_days=0.11,
        )
        _, kernel, _ = plan.project_map(maps[1].values)
        weights = source.linear_response_weights(dtype=torch.float64)
        weighted = weights * maps[1].values[..., None]
        true_mass = weighted.sum(dim=(0, 1))
        true_first_moment = (
            weighted * source.delay_days[..., None]
        ).sum(dim=(0, 1))
    torch.testing.assert_close(kernel.sum(dim=0), true_mass, rtol=1e-12, atol=0)
    torch.testing.assert_close(
        (kernel * plan.delay_centers_days[:, None]).sum(dim=0),
        true_first_moment, rtol=1e-12, atol=0,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_response_projection_matches_eager():
    generator = torch.Generator(device="cuda").manual_seed(173)
    pixels, bands, bins = 8192, 6, 24
    sampled = torch.rand(pixels, device="cuda", generator=generator)
    brightness = torch.rand(pixels, bands, device="cuda", generator=generator)
    response = torch.rand(pixels, bands, device="cuda", generator=generator)
    left = torch.randint(0, bins - 1, (pixels,), device="cuda", generator=generator)
    right = left + 1
    fraction = torch.rand(pixels, device="cuda", generator=generator)
    arguments = (sampled, brightness, response, left, right, fraction, bins)
    runtime = mc.resolve_runtime(mc.RuntimeConfig(device="cuda", backend="triton"))
    with torch.no_grad():
        expected = _project_response_chunk(*arguments)
        actual, compiled = run_tensor_kernel(
            runtime, "test delay-response projection", _project_response_chunk,
            *arguments,
        )
    assert compiled
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("static_shapes", [False, True])
def test_response_projection_shape_setting_preserves_flux(static_shapes):
    source, _, maps, _ = thermal_fixture()
    times = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(
            device="cpu", backend="torch-eager", dtype=torch.float64,
            static_response_projection=static_shapes,
        )
    )
    plan = LinearResponsePlan(
        source, ((source, len(source.geometry.band_names)),), times,
        delay_bin_days=0.2, response_order=2, runtime=runtime,
    )
    with patch(
        "microcaustics.sources.linear_response.run_tensor_kernel",
        wraps=run_tensor_kernel,
    ) as kernel:
        actual = plan.project_map(maps[1].values)
    assert kernel.call_args.kwargs["dynamic"] is not static_shapes
    reference = _project_response_chunk(
        maps[1].values.reshape(-1)[plan.positions],
        plan.parts[0][0], plan.parts[0][1], plan.left_index,
        plan.right_index, plan.right_fraction, plan.bin_count,
    )
    torch.testing.assert_close(actual[0], reference[0])
    torch.testing.assert_close(actual[1], reference[1][:, :len(source.geometry.band_names)])
    torch.testing.assert_close(actual[2], reference[1][:, len(source.geometry.band_names):])


def test_fixed_response_projection_is_default_and_can_be_disabled():
    assert mc.RuntimeConfig().static_response_projection is True
    assert mc.resolve_runtime(mc.RuntimeConfig(device="cpu")).static_response_projection
    assert not mc.resolve_runtime(
        mc.RuntimeConfig(device="cpu", static_response_projection=False)
    ).static_response_projection
    with pytest.raises(ValueError, match="static_response_projection"):
        mc.RuntimeConfig(static_response_projection="yes")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("order", [1, 2])
def test_fused_response_chunks_preserve_kernel_and_memory_fallback(order):
    source, _, maps, _ = thermal_fixture()
    chunks = _source_band_chunks(source, 2)
    times = torch.tensor([0.0, 0.5, 1.0], device="cuda")
    sampled = maps[1].values.to(device="cuda", dtype=times.dtype)
    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(device="cuda", backend="triton", dtype=times.dtype)
    )
    with torch.no_grad():
        fused = LinearResponsePlan(
            source, chunks, times, delay_bin_days=0.2,
            response_order=order, runtime=runtime,
        )
        chunked = LinearResponsePlan(
            source, chunks, times, delay_bin_days=0.2,
            response_order=order, runtime=replace(runtime, memory_fraction=1e-10),
        )
        assert len(fused.parts) == 1
        assert len(chunked.parts) == 3
        actual = fused.project_map(sampled)
        expected = chunked.project_map(sampled)
    for result, reference in zip(actual, expected, strict=True):
        if result is not None:
            torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(
        fused.flux(actual, 0, 3), chunked.flux(expected, 0, 3),
        rtol=1e-4, atol=1e-3,
    )


def test_quadratic_thermal_response_matches_brightness_finite_difference():
    source, _, _, _ = thermal_fixture()
    epsilon = 0.005

    def brightness(amplitude):
        frozen = replace(
            source,
            signal=mc.CallableDrivingSignal(
                lambda times: torch.full_like(times, amplitude)
            ),
        )
        return frozen.brightness([0.0], dtype=torch.float64)[0]

    with torch.no_grad():
        center = brightness(1.0)
        plus = brightness(1.0 + epsilon)
        minus = brightness(1.0 - epsilon)
        numerical = (plus - 2.0 * center + minus) / epsilon**2
        analytic = source.quadratic_response_weights(dtype=torch.float64)
    torch.testing.assert_close(analytic, numerical, rtol=1e-3, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_quadratic_weights_match_independent_linear_weights(dtype):
    source, _, _, _ = thermal_fixture()
    for amplitude in (0.0, 0.7, 1.4):
        expected_first = source.linear_response_weights(
            driver_amplitude=amplitude, dtype=dtype
        )
        with torch.no_grad():
            first, second = source._quadratic_response_weight_pair(
                driver_amplitude=amplitude, dtype=dtype
            )
            torch.testing.assert_close(
                first,
                expected_first,
                rtol=2e-6,
                atol=0,
            )
            torch.testing.assert_close(
                second,
                source.quadratic_response_weights(
                    driver_amplitude=amplitude, dtype=dtype
                ),
                rtol=0,
                atol=0,
            )
            assert first.data_ptr() == source.linear_response_weights(
                driver_amplitude=amplitude, dtype=dtype
            ).data_ptr()


def test_fused_quadratic_weights_preserve_response_gradient():
    source, _, _, _ = thermal_fixture()
    response = source.response_temperature4.clone().requires_grad_()
    source = replace(source, response_temperature4=response)
    first, second = source._quadratic_response_weight_pair(dtype=torch.float64)
    first.sum().backward(retain_graph=True)
    first_gradient = response.grad.clone()
    response.grad = None
    second.sum().backward()
    assert torch.isfinite(first_gradient).all()
    assert torch.isfinite(response.grad).all()
    assert bool(first_gradient.abs().sum() > 0)
    assert bool(response.grad.abs().sum() > 0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_quadratic_weights_match_eager_and_reuse_cache():
    source, _, _, _ = thermal_fixture()
    device = torch.empty((), device="cuda").device
    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(device="cuda", backend="triton", dtype=torch.float32)
    )
    with torch.no_grad():
        reference = source._quadratic_response_weight_pair(
            device=device, dtype=torch.float32
        )
        source._linear_response_cache.clear()
        with patch(
            "microcaustics.sources.reprocessing.run_tensor_kernel",
            wraps=run_tensor_kernel,
        ) as compiled_kernel:
            compiled = source._quadratic_response_weight_pair(
                device=device, dtype=torch.float32, runtime=runtime
            )
        compiled_kernel.assert_called_once()
        for actual, expected in zip(compiled, reference, strict=True):
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=0)
        cached = source._quadratic_response_weight_pair(
            device=device, dtype=torch.float32, runtime=runtime
        )
        assert all(a.data_ptr() == b.data_ptr() for a, b in zip(compiled, cached, strict=True))


def test_quadratic_response_reduces_nonlinear_driver_error():
    source, distances, maps, runtime = thermal_fixture()
    times = torch.tensor([0.0, 0.25, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64)

    def evaluate(mode):
        (curve,) = _flexible_streaming_light_curves(
            SimpleNamespace(runtime=runtime), None, maps[0].grid,
            [0.0, 1.0, 2.0],
            (mc.LightCurveRequest(source, distances, source_evolution=mode),),
            (times,), method=None, schedule=None, map_observer=None,
            band_batch_size=2, map_iterator=iter(maps),
        )
        return curve.flux

    exact = evaluate("exact")
    linear_error = ((evaluate("linear_response") - exact) / exact).abs().max()
    quadratic_error = ((evaluate("quadratic_response") - exact) / exact).abs().max()
    assert quadratic_error < linear_error


def test_sparse_variable_source_does_not_pad_to_map_temporal_batch():
    source, distances, maps, runtime = thermal_fixture()
    times = torch.tensor([0.0, 2.0], dtype=torch.float64)
    dense_times = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
    expected = mc.light_curve_from_maps(
        (maps[0], maps[2]), source, times, distances, batch_size=2
    )
    original = mc.ThermalReprocessingSource._brightness_state
    with patch.object(
        mc.ThermalReprocessingSource,
        "_brightness_state",
        autospec=True,
        side_effect=original,
    ) as prepare:
        actual, dense = _flexible_streaming_light_curves(
            SimpleNamespace(runtime=runtime),
            None,
            maps[0].grid,
            [0.0, 1.0, 2.0],
            (
                mc.LightCurveRequest(source, distances),
                mc.LightCurveRequest(source, distances),
            ),
            (times, dense_times),
            method=None,
            schedule=mc.DynamicConfig(temporal_batch_size=30),
            map_observer=None,
            band_batch_size=2,
            map_iterator=iter(maps),
        )
    assert actual.metadata["source_batch_size"] == 1
    assert dense.metadata["source_batch_size"] == 3
    assert [call.args[1].numel() for call in prepare.call_args_list] == [1, 3, 1, 3]
    torch.testing.assert_close(actual.flux, expected.flux, rtol=1e-12, atol=0)


def test_aligned_wavelength_batches_do_not_resample_maps():
    source, distances, maps, runtime = thermal_fixture()
    with patch(
        "microcaustics.photometry._sample_map_batch",
        side_effect=AssertionError("unexpected resampling"),
    ):
        _flexible_streaming_light_curves(
            SimpleNamespace(runtime=runtime),
            None,
            maps[0].grid,
            [0.0, 1.0, 2.0],
            (mc.LightCurveRequest(source, distances),),
            ([0.0, 0.5, 2.0],),
            method=None,
            schedule=mc.DynamicConfig(temporal_batch_size=3),
            map_observer=None,
            band_batch_size=2,
            map_iterator=iter(maps),
        )


@pytest.mark.parametrize("kind", ["tabulated", "fixed", "callable"])
def test_intrinsic_spectral_batches_match_explicit_brightness(kind):
    source, _, _, _ = thermal_fixture()
    if kind == "fixed":
        source = replace(
            source,
            signal=mc.broken_power_law_driving_signal(
                cadence_days=0.25, max_duration_days=2.0, history_days=2.0, seed=41
            ),
        )
    elif kind == "callable":
        source = replace(
            source, signal=mc.CallableDrivingSignal(lambda t: 1.0 + 0.1 * torch.sin(t))
        )
    times = torch.tensor([0.0, 0.25, 1.0, 2.0], dtype=torch.float64)
    expected = source.brightness(times, dtype=torch.float64).sum(dim=(1, 2))
    expected *= source.geometry.pixel_scale_m[0] * source.geometry.pixel_scale_m[1]
    actual = mc.source_light_curve(source, times, batch_size=3, band_batch_size=2)
    torch.testing.assert_close(actual.flux, expected)
    changed = replace(
        source, signal=mc.CallableDrivingSignal(lambda t: torch.full_like(t, 1.8))
    )
    assert not torch.equal(
        mc.source_light_curve(changed, times, band_batch_size=2).flux, actual.flux
    )


def test_retarded_driver_bounds_and_hold_are_preserved():
    source, _, _, _ = thermal_fixture()
    with pytest.raises(ValueError, match="outside"):
        source.brightness([2.1])
    with pytest.raises(ValueError, match="finite"):
        source.brightness([float("nan")])
    held = replace(source, signal=replace(source.signal, extrapolation="hold"))
    assert torch.isfinite(held.brightness([2.1])).all()
    fixed = replace(
        source,
        signal=mc.broken_power_law_driving_signal(
            cadence_days=0.25, max_duration_days=2.0, history_days=2.0, seed=41
        ),
    )
    with pytest.raises(ValueError, match="Driving-signal queries"):
        fixed.brightness([2.1])


def test_valid_retarded_batches_read_driver_table_without_endpoint_interpolation():
    source, _, _, _ = thermal_fixture()
    with patch.object(
        mc.TabulatedDrivingSignal,
        "amplitudes",
        side_effect=AssertionError("redundant driver interpolation"),
    ):
        assert torch.isfinite(source.brightness([0.0, 0.5])).all()


def test_bandpass_device_cache_respects_weight_updates_and_gradients():
    weights = torch.tensor([[0.5], [0.5]])
    grid = BandpassGrid((4000.0, 5000.0), ("b",), weights)
    flux = torch.tensor([[1.0, 3.0]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(
        grid.integrate(flux), torch.tensor([[2.0]], dtype=torch.float64)
    )
    weights[0, 0] = 1.5
    result = grid.integrate(flux)
    torch.testing.assert_close(result, torch.tensor([[3.0]], dtype=torch.float64))
    result.sum().backward()
    torch.testing.assert_close(
        flux.grad, torch.tensor([[1.5, 0.5]], dtype=torch.float64)
    )


def test_delayed_intrinsic_source_keeps_fused_heating_and_arrival_times():
    source, _, _, _ = thermal_fixture()
    delayed = mc.TimeShiftedSource(source, delay_days=0.25)
    times = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float64)
    expected = mc.source_light_curve(source, times - 0.25, batch_size=2)
    actual = mc.source_light_curve(delayed, times, batch_size=3, band_batch_size=2)
    torch.testing.assert_close(actual.flux, expected.flux)
    torch.testing.assert_close(actual.times_days, times)


def test_static_cache_budget_falls_back_without_changing_flux():
    source, distances, maps, runtime = thermal_fixture()
    source = source.at_driver_mean()

    def evaluate(active_runtime):
        return _flexible_streaming_light_curves(
            SimpleNamespace(runtime=active_runtime),
            None,
            maps[0].grid,
            [0.0, 1.0, 2.0],
            (mc.LightCurveRequest(source, distances),),
            ([0.0, 0.5, 1.5, 2.0],),
            method=None,
            schedule=mc.DynamicConfig(temporal_batch_size=3),
            map_observer=None,
            band_batch_size=2,
            map_iterator=iter(maps),
        )[0]

    expected = evaluate(runtime)
    # Advertise a deliberately tiny device budget without allocating anything.
    limited = replace(
        runtime,
        capabilities=replace(runtime.capabilities, total_device_memory_bytes=32),
    )
    actual = evaluate(limited)
    torch.testing.assert_close(actual.flux, expected.flux)
    torch.testing.assert_close(actual.unlensed_flux, expected.unlensed_flux)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_intrinsic_batches_reuse_graphs_for_changed_driver_and_disk():
    from torch._dynamo.utils import counters

    from microcaustics.compile import _DISABLED_KERNELS

    source, _, _, _ = thermal_fixture()
    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(
            device="cuda",
            backend="torch-compile",
            dtype=torch.float64,
            strict_backend=True,
        )
    )
    source = replace(source, _runtime=runtime)
    times = torch.tensor(
        [0.0, 0.25, 0.5, 1.0, 1.5, 2.0], device="cuda", dtype=torch.float64
    )

    def evaluate(disk, batch=3):
        return mc.source_light_curve(disk, times, batch_size=batch, band_batch_size=2)

    with torch.inference_mode():
        reference = source.brightness(times, device="cuda", dtype=torch.float64).sum(
            dim=(1, 2)
        )
        reference *= source.geometry.pixel_scale_m[0] * source.geometry.pixel_scale_m[1]
        actual = evaluate(source)
        torch.testing.assert_close(actual.flux, reference)
        graphs = counters["stats"]["unique_graphs"]
        changed = replace(
            source,
            black_hole_mass_solar=2e8,
            signal=replace(source.signal, values=source.signal.values * 1.1),
        )
        again = evaluate(source)
        different = evaluate(changed)
        torch.cuda.synchronize()
        assert counters["stats"]["unique_graphs"] == graphs
        assert not _DISABLED_KERNELS
        torch.testing.assert_close(again.flux, actual.flux)
        assert not torch.equal(actual.flux, different.flux)
