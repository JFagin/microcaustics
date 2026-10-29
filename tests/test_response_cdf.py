"""Shared CDF export/convolution regression tests, with no production checks."""

import numpy as np
import pytest
import torch

import microcaustics as mc
from microcaustics._response_cdf import (
    cdf_mean_delays,
    central_response_widths,
    choose_cdf_grid,
    driver_integrals,
    pchip_coefficients,
    project_histograms,
    rebin_coefficients,
)
from microcaustics.compile import run_tensor_kernel


@pytest.mark.parametrize("scale", [1.0, 1e-30, 1e10])
def test_grid_selection_uses_narrowest_band_not_flux_or_bh_mass(scale):
    x = torch.arange(20000, dtype=torch.float32) * 0.005
    widths = torch.tensor([3.0, 1.0, 0.2])
    mass = x[:, None] * torch.exp(-x[:, None] / widths)
    measured = central_response_widths(mass * scale, 0.005)
    choice = choose_cdf_grid(measured, 1.0, 100.0)
    assert choice.step_days <= float(measured.min()) / 16
    assert choice.step_days <= 0.25
    assert choice.step_days == 0.025
    assert choose_cdf_grid(measured.flip(0), 1.0, 100.0) == choice


def test_selection_cadence_memory_and_unresolved_fallbacks():
    assert choose_cdf_grid([10.0, 20.0], 0.4, 100.0).step_days == 0.1
    assert choose_cdf_grid([10.0, 20.0], 4.0, 100.0).step_days == 0.5
    with pytest.raises(ValueError, match="No CDF bucket"):
        choose_cdf_grid([0.01], 1.0, 10.0)
    with pytest.raises(ValueError, match="memory budget"):
        choose_cdf_grid([0.2], 1.0, 1000.0, max_native_bins=100)
    for widths in ([0.0], [float("nan")], [-1.0], []):
        with pytest.raises(ValueError, match="response widths"):
            choose_cdf_grid(widths, 1.0, 10.0)


def test_pixel_scale_selection_remains_bounded_by_response_and_driver():
    assert choose_cdf_grid([10.0], 1.0, 100.0, pixel_delay_days=0.249).step_days == 0.25
    assert (
        choose_cdf_grid([10.0], 1.0, 100.0, pixel_delay_days=0.0623).step_days == 0.05
    )
    assert choose_cdf_grid([0.8], 1.0, 100.0, pixel_delay_days=0.249).step_days == 0.05
    assert (
        choose_cdf_grid([10.0], 0.025, 100.0, pixel_delay_days=0.249).step_days == 0.025
    )


def test_quantile_width_is_conservative_for_binned_response():
    # Uniform density q84-q16 = .68. A binned lower bound must never overstate it.
    mass = torch.ones(100, 2)
    mass[:, 1] *= 1e-20
    width = central_response_widths(mass, 0.01)
    assert bool(((width >= 0.65) & (width <= 0.68)).all())
    assert float(central_response_widths(torch.zeros(100, 1), 0.01)) == 0
    spike = torch.zeros(100, 1)
    spike[40] = 1
    assert float(central_response_widths(spike, 0.01)) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_constant_driver_mean_mass_positive_causal_response(dtype):
    edges = torch.arange(41, dtype=dtype) * 0.1
    centers = (edges[:-1] + edges[1:]) / 2
    mass = (centers.square() * torch.exp(-centers))[:, None]
    mass[:3] = 0
    coeff = pchip_coefficients(mass)
    knots = torch.arange(-10, 11, dtype=dtype)
    times = torch.arange(6, dtype=dtype) + 0.137
    d1, d2 = driver_integrals(
        times, edges, knots, torch.full_like(knots, 1.7), 1.7, True
    )
    assert torch.count_nonzero(d1) == torch.count_nonzero(d2) == 0
    output = torch.linspace(-1, 5, 601, dtype=dtype)
    rebinner = mc.CDFResponseRebinner(edges, output)
    rendered = rebin_coefficients(coeff, rebinner)
    assert float(rendered.min()) >= -1e-6
    assert torch.count_nonzero(rendered[:100]) == 0
    assert torch.count_nonzero(rendered[500:]) == 0
    torch.testing.assert_close(rendered.sum(0), mass.sum(0), rtol=2e-6, atol=1e-6)
    mean = cdf_mean_delays(coeff, edges)
    numerical = (rendered * ((output[:-1] + output[1:]) / 2)[:, None]).sum(
        0
    ) / rendered.sum(0)
    torch.testing.assert_close(mean, numerical, rtol=1e-5, atol=1e-5)


def test_irregular_tabulated_driver_knots_and_signed_zero_total():
    # The analytic integrator must split nonuniform driver segments correctly.
    knots = torch.tensor(
        [-4.0, -3.7, -2.9, -2.0, -0.8, -0.2, 0.4, 1.5, 2.1, 3.0], dtype=torch.float64
    )
    values = 1 + 0.2 * torch.sin(knots * 3)
    edges = torch.arange(21, dtype=torch.float64) * 0.1
    times = torch.tensor([0.137, 0.9], dtype=torch.float64)
    mass = torch.stack((torch.linspace(0, 1, 20), torch.linspace(1, 0, 20)), 1).double()
    coeff = pchip_coefficients(mass)
    signed = coeff[:, :1] - coeff[:, 1:]
    fine = torch.linspace(0, 2, 20001, dtype=torch.float64)
    displayed = rebin_coefficients(signed, mc.CDFResponseRebinner(edges, fine))
    assert abs(float(displayed.sum())) < 1e-12
    d1, d2 = driver_integrals(times, edges, knots, values, 1.0, True)
    query = times[:, None] - ((fine[:-1] + fine[1:]) / 2)[None]
    delta = (
        torch.from_numpy(np.interp(query.numpy(), knots.numpy(), values.numpy())) - 1
    )
    torch.testing.assert_close(d1 @ signed, delta @ displayed, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(
        d2 @ signed, delta.square() @ displayed, atol=1e-7, rtol=1e-7
    )


@pytest.mark.parametrize("phase", [0.0, 0.137, 0.5])
@pytest.mark.parametrize("step", [0.1, 0.25, 0.5, 1.0])
def test_analytic_driver_integrals_against_split_gauss(phase, step):
    knots = np.arange(-5, 11, dtype=float)
    values = 1 + 0.2 * np.sin(knots * 2.3)
    times = np.arange(3, dtype=float) + phase
    edges = np.arange(round(3 / step) + 1) * step
    first, second = driver_integrals(
        *[torch.from_numpy(x) for x in (times, edges, knots, values)], 1.0, True
    )
    expected = np.zeros((2, len(times), 3, len(edges) - 1))
    nodes, weights = np.polynomial.legendre.leggauss(3)
    for ti, time in enumerate(times):
        for i, edge in enumerate(edges[:-1]):
            cuts = (time - edge - knots) / step
            cuts = np.r_[0.0, np.sort(cuts[(cuts > 0) & (cuts < 1)]), 1.0]
            for low, high in zip(cuts[:-1], cuts[1:], strict=True):
                t = (low + high) / 2 + (high - low) / 2 * nodes
                w = weights * (high - low) / 2
                driver = np.interp(time - edge - step * t, knots, values) - 1
                basis = np.array(
                    [6 * (t - t * t), 1 - 4 * t + 3 * t * t, 3 * t * t - 2 * t]
                )
                for order in range(2):
                    expected[order, ti, :, i] += (
                        basis * w * driver ** (order + 1)
                    ).sum(1)
    np.testing.assert_allclose(first, expected[0].reshape(len(times), -1), atol=2e-14)
    np.testing.assert_allclose(second, expected[1].reshape(len(times), -1), atol=2e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_coefficients_reproduce_public_cdf_and_first_moment(dtype):
    generator = torch.Generator().manual_seed(141)
    mass = torch.rand(12, 4, generator=generator, dtype=dtype)
    mass[3, 0] = 0
    edges = torch.arange(13, dtype=dtype) * 0.25
    output = torch.linspace(0, 3, 1201, dtype=dtype)
    public = mc.CDFResponseRebinner(edges, output)(mass)
    coefficients = pchip_coefficients(mass)
    m, left, right = coefficients.chunk(3)
    q = torch.linspace(0, 1, 101, dtype=dtype)
    cdf = (
        m[:, None] * (3 * q.square() - 2 * q**3)[None, :, None]
        + left[:, None] * (q**3 - 2 * q.square() + q)[None, :, None]
        + right[:, None] * (q**3 - q.square())[None, :, None]
    )
    actual = cdf.diff(dim=1).reshape(-1, 4)
    torch.testing.assert_close(
        actual,
        public,
        rtol=2e-3 if dtype == torch.float32 else 1e-10,
        atol=2e-6 if dtype == torch.float32 else 1e-14,
    )
    moment = (edges[:-1, None] * m + 0.25 * (0.5 * m - left / 12 + right / 12)).sum(0)
    numerical = (public * ((output[:-1] + output[1:]) / 2)[:, None]).sum(0)
    torch.testing.assert_close(moment, numerical, rtol=1e-5, atol=1e-5)


def test_signed_projection_splits_before_binning():
    sample = torch.tensor([1.0, 3.0, 2.0])
    brightness = torch.ones(3, 1)
    response = torch.tensor([[1.0, -2.0], [2.0, 1.0], [3.0, -1.0]])
    mean, mass = project_histograms(
        sample, brightness, response, torch.tensor([0, 0, 1]), 2, 1, True
    )
    torch.testing.assert_close(mean, torch.tensor([6.0]))
    torch.testing.assert_close(mass, torch.tensor([[7.0, 3.0, 2.0], [6.0, 0.0, 2.0]]))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_default_transfer_and_light_curve_use_same_cdf_without_reference_work(order):
    from unittest.mock import patch

    from test_spectral_batch_optimization import thermal_fixture

    from microcaustics.sources.linear_response import LinearResponsePlan

    source, _, maps, runtime = thermal_fixture()
    times = torch.tensor([0.0, 0.37, 1.1, 2.0], dtype=torch.float64)
    with patch(
        "microcaustics.sources.linear_response._project_response_chunk",
        side_effect=AssertionError("ordinary response must not run"),
    ):
        plan = LinearResponsePlan(
            source,
            ((source, 5),),
            times,
            delay_bin_days=None,
            response_order=order,
            runtime=runtime,
        )
        projection = plan.project_map(maps[1].values)
    edges = torch.linspace(0, float(plan.edges[-1]), 501, dtype=times.dtype)
    mass = plan.response_bin_masses(projection, edges)
    expected = source.transfer_function(edges, magnification=maps[1].values)
    torch.testing.assert_close(mass / mass.sum(0), expected, rtol=1e-9, atol=1e-12)
    batch = source.batched_transfer_function(
        edges,
        magnification=torch.stack((torch.ones_like(maps[1].values), maps[1].values)),
    )
    torch.testing.assert_close(batch[1], expected)
    center = (edges[1:] + edges[:-1]) / 2
    reconstructed_mean = (expected * center[:, None]).sum(0) / expected.sum(0)
    direct = source.mean_response_delays(magnification=maps[1].values)
    assert (
        (reconstructed_mean - direct).abs()
        <= torch.maximum(0.01 * direct.abs(), direct.new_tensor(0.1))
    ).all()
    torch.testing.assert_close(
        source.transfer_function(edges[:251], magnification=maps[1].values),
        expected[:250],
        rtol=1e-10,
        atol=1e-12,
    )
    assert plan.metadata()["response_reconstruction"] == "cdf_pchip"


def test_ordinary_override_and_exact_mode_default():
    from dataclasses import replace

    from test_spectral_batch_optimization import thermal_fixture

    from microcaustics.sources.linear_response import (
        LinearResponsePlan,
        OrdinaryResponsePlan,
    )

    source, _, maps, runtime = thermal_fixture()
    assert mc.RuntimeConfig().response_reconstruction == "cdf"
    assert mc.LightCurveRequest().source_evolution == "exact"
    runtime = replace(runtime, response_reconstruction="ordinary")
    args = (source, ((source, 5),), torch.tensor([0.0, 1.0], dtype=torch.float64))
    plan = LinearResponsePlan(*args, delay_bin_days=0.1, runtime=runtime)
    control = OrdinaryResponsePlan(*args, delay_bin_days=0.1, runtime=runtime)
    torch.testing.assert_close(
        plan.flux(plan.project_map(maps[1].values), 0, 2),
        control.flux(control.project_map(maps[1].values), 0, 2),
    )
    with pytest.raises(ValueError, match="response_reconstruction"):
        mc.RuntimeConfig(response_reconstruction="unknown")


@pytest.mark.parametrize("order", [1, 2])
def test_response_chunks_share_heating_and_match_public_source_weights(order):
    from dataclasses import replace
    from unittest.mock import patch

    from test_spectral_batch_optimization import thermal_fixture

    from microcaustics.photometry import _source_band_chunks
    from microcaustics.sources.linear_response import LinearResponsePlan
    from microcaustics.sources.reprocessing import ThermalReprocessingSource

    source, _, _, runtime = thermal_fixture()
    amplitude = 1.3
    source = replace(
        source,
        signal=mc.CallableDrivingSignal(
            lambda t: torch.full_like(t, amplitude),
            user_metadata={"mean_amplitude": amplitude},
        ),
    )
    times = torch.tensor([0.0, 1.0], dtype=torch.float64)
    mean = source.at_driver_mean().brightness(times[:1], dtype=times.dtype)[0]
    first = source.linear_response_weights(driver_amplitude=amplitude)
    second = source.quadratic_response_weights(driver_amplitude=amplitude)
    chunks = _source_band_chunks(source, 2)  # includes a padded final band
    source._evaluation_cache.clear()
    original = ThermalReprocessingSource._static_temperature4
    with (
        patch.object(
            ThermalReprocessingSource,
            "_static_temperature4",
            autospec=True,
            side_effect=original,
        ) as temperature,
        patch.object(
            ThermalReprocessingSource,
            "at_driver_mean",
            side_effect=AssertionError(
                "response setup must not clone the source per band"
            ),
        ),
    ):
        plan = LinearResponsePlan(
            source,
            chunks,
            times,
            delay_bin_days=0.25,
            response_order=order,
            runtime=runtime,
        )
    assert temperature.call_count == 1
    column = 0
    for (brightness, response, bands), scale in zip(
        plan.parts, plan.scales, strict=True
    ):
        sl = slice(column, column + bands)

        def select(x, sl=sl, bands=bands):
            return x[..., sl].reshape(-1, bands)[plan.positions]

        torch.testing.assert_close(brightness, select(mean))
        torch.testing.assert_close(response[:, :bands] * scale[:bands], select(first))
        if order == 2:
            torch.testing.assert_close(
                response[:, bands:] * scale[bands:], select(second)
            )
        column += bands


def test_cdf_transfer_spatial_chunks_preserve_values_and_bound_temporaries():
    from unittest.mock import patch

    from test_spectral_batch_optimization import thermal_fixture

    from microcaustics import _response_cdf

    source, _, maps, _ = thermal_fixture()
    edges = torch.linspace(0, 2, 401, dtype=torch.float64)
    magnification = torch.stack([item.values for item in maps])
    expected = source.batched_transfer_function(edges, magnification=magnification)
    with patch.object(
        _response_cdf, "deposit_response", wraps=_response_cdf.deposit_response
    ) as histogram:
        actual = source.batched_transfer_function(
            edges, magnification=magnification, spatial_chunk_size=7
        )
    assert histogram.call_count > len(maps)
    assert all(len(call.args[0]) <= 7 for call in histogram.call_args_list)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "support,step", [(1.79, 0.25), (0.0, 0.25), (0.15, 0.1), (2.7, 0.02)]
)
def test_shifted_cdf_matches_independent_scipy_and_preserves_mass(dtype, support, step):
    from scipy.interpolate import PchipInterpolator

    from microcaustics._response_cdf import NativeCDFGrid, phase_cdf_coefficients

    delay = torch.tensor([0.0, support * 0.03, support * 0.5, support], dtype=dtype)
    grid = NativeCDFGrid(delay, [torch.ones(4, 2, dtype=dtype)], step=step)
    generator = torch.Generator().manual_seed(295)
    fine = torch.rand(grid.deposition_count, 2, generator=generator, dtype=dtype)
    fine[30:36, 0] = 0
    fine[-4:] = 0
    fine[:, 1] *= 1e-20
    coeff = phase_cdf_coefficients(fine, grid.count)
    edges = torch.linspace(0, float(grid.edges[-1]), 1001, dtype=dtype)
    actual = grid.rebin(coeff, edges)
    fine64 = fine.double().numpy()
    cdf = np.vstack((np.zeros((1, 2)), fine64.cumsum(0)))
    expected = np.zeros((1000, 2))
    for phase in range(4):
        idx = np.r_[0, np.arange(phase or 4, grid.count, 4), grid.count] * 4
        x = idx * grid.deposition_step
        expected += (
            np.diff(
                PchipInterpolator(x, cdf[idx], axis=0)(edges.double().numpy()), axis=0
            )
            / 4
        )
    torch.testing.assert_close(
        actual.double(),
        torch.from_numpy(expected),
        rtol=3e-4 if dtype == torch.float32 else 1e-10,
        atol=0,
    )
    torch.testing.assert_close(
        actual.sum(0),
        fine.sum(0),
        rtol=3e-6 if dtype == torch.float32 else 1e-12,
        atol=0,
    )
    assert bool((actual >= -torch.finfo(dtype).eps).all())


def test_fine_deposition_does_not_introduce_uniform_time_shift():
    from microcaustics._response_cdf import NativeCDFGrid, deposit_response

    delay = torch.tensor([0.0, 0.107, 0.233, 0.999, 1.25], dtype=torch.float64)
    weight = torch.tensor([[0.0], [1.0], [2.0], [1.0], [3.0]], dtype=delay.dtype)
    grid = NativeCDFGrid(delay, [weight], step=0.25)
    mass = deposit_response(
        torch.ones_like(delay), weight, grid.index, grid.fraction, grid.deposition_count
    )
    centers = (
        torch.arange(grid.deposition_count, dtype=delay.dtype) + 0.5
    ) * grid.deposition_step
    torch.testing.assert_close(
        (mass * centers[:, None]).sum(0),
        (weight * delay[:, None]).sum(0),
        rtol=1e-14,
        atol=1e-14,
    )
    coeff = grid.project_response(torch.ones_like(delay), weight, 2)
    assert (
        abs(
            float(
                cdf_mean_delays(coeff, grid.edges)
                - (weight * delay[:, None]).sum(0) / weight.sum(0)
            )
        )
        < 0.01
    )


def test_exact_evolution_is_bitwise_unchanged_by_response_reconstruction():
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import patch

    from test_spectral_batch_optimization import thermal_fixture

    from microcaustics.photometry import _flexible_streaming_light_curves

    source, distances, maps, runtime = thermal_fixture()
    times = torch.tensor([0.0, 0.25, 0.5, 1.0, 1.75, 2.0], dtype=torch.float64)
    curves = []
    with patch(
        "microcaustics.sources.linear_response.LinearResponsePlan.__init__",
        side_effect=AssertionError("exact evolution must not construct a response"),
    ):
        for mode in ("ordinary", "cdf"):
            (curve,) = _flexible_streaming_light_curves(
                SimpleNamespace(runtime=replace(runtime, response_reconstruction=mode)),
                None,
                maps[0].grid,
                [0.0, 1.0, 2.0],
                (mc.LightCurveRequest(source, distances, strict_coverage=False),),
                (times,),
                method=None,
                schedule=mc.DynamicConfig(temporal_batch_size=3),
                map_observer=None,
                band_batch_size=2,
                map_iterator=iter(maps),
            )
            curves.append(curve)
    for field in ("flux", "unlensed_flux", "magnitude"):
        torch.testing.assert_close(
            getattr(curves[0], field), getattr(curves[1], field), rtol=0, atol=0
        )


def test_display_and_convolution_share_signed_polynomials():
    torch.manual_seed(193)
    mass = torch.rand(12, 3, dtype=torch.float64)
    coeff = pchip_coefficients(mass)
    signed = coeff[:, 1:2] - coeff[:, 2:3]
    edges = torch.arange(13, dtype=torch.float64) * 0.25
    fine_edges = torch.linspace(0, 3, 12001, dtype=torch.float64)
    plan = mc.CDFResponseRebinner(edges, fine_edges)
    displayed = rebin_coefficients(coeff, plan)
    torch.testing.assert_close(displayed, plan(mass), rtol=1e-9, atol=1e-14)
    signed_display = rebin_coefficients(signed, plan)
    torch.testing.assert_close(
        signed_display, displayed[:, 1:2] - displayed[:, 2:3], rtol=1e-9, atol=1e-14
    )
    times = torch.tensor([0.0, 0.137, 1.0, 2.3], dtype=torch.float64)
    sample_times = torch.arange(-5, 6, dtype=torch.float64)
    values = 1 + 0.2 * torch.sin(sample_times * 1.7)
    first, second = driver_integrals(times, edges, sample_times, values, 1.0, True)
    query = times[:, None] - (fine_edges[:-1] + fine_edges[1:])[None] / 2
    delta = np.interp(query.numpy(), sample_times.numpy(), values.numpy()) - 1
    sampled = torch.from_numpy(delta)
    torch.testing.assert_close(
        first @ coeff[:, :1], sampled @ displayed[:, :1], atol=2e-8, rtol=1e-7
    )
    torch.testing.assert_close(
        second @ signed, sampled.square() @ signed_display, atol=2e-8, rtol=1e-7
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_float32_integrals_projection_and_coefficients():
    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(
            device="cuda",
            dtype=torch.float32,
            backend="triton",
            strict_backend=True,
            warn_on_compile=False,
        )
    )
    knots = torch.arange(-10, 3660, device="cuda", dtype=torch.float32)
    values = 1 + 0.2 * torch.sin(knots * 2.3)
    times = torch.arange(3630, 3651, device="cuda", dtype=torch.float32) + 0.137
    edges = torch.arange(41, device="cuda", dtype=torch.float32) * 0.25
    args = (times, edges, knots, values, 1.0, True)
    actual, compiled = run_tensor_kernel(
        runtime,
        "experimental analytic CDF driver integration",
        driver_integrals,
        *args,
        dynamic=False,
    )
    assert compiled
    expected = driver_integrals(
        *[x.double() if isinstance(x, torch.Tensor) else x for x in args]
    )
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a.double(), b, atol=2e-7, rtol=2e-4)
    torch.manual_seed(125)
    sampled = torch.rand(4096, device="cuda")
    brightness = torch.rand(4096, 3, device="cuda")
    response = torch.randn(4096, 6, device="cuda")
    response[:, :3].abs_()
    index = torch.randint(40, (4096,), device="cuda")
    arguments = (sampled, brightness, response, index, 40, 3, True)
    (mean, mass), compiled = run_tensor_kernel(
        runtime,
        "experimental fused CDF map projection",
        project_histograms,
        *arguments,
        dynamic=False,
    )
    assert compiled
    expected_mean, expected_mass = project_histograms(*arguments)
    torch.testing.assert_close(mean, expected_mean, rtol=2e-5, atol=1e-5)
    torch.testing.assert_close(mass, expected_mass, rtol=2e-5, atol=1e-5)
    coeff, compiled = run_tensor_kernel(
        runtime,
        "experimental native CDF coefficients",
        pchip_coefficients,
        mass,
        dynamic=False,
    )
    assert compiled
    torch.testing.assert_close(
        coeff, pchip_coefficients(expected_mass), rtol=2e-5, atol=1e-5
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_four_phase_projection_matches_eager_without_recompilation():
    from torch._dynamo.utils import counters

    from microcaustics._response_cdf import NativeCDFGrid

    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(
            device="cuda",
            dtype=torch.float32,
            backend="triton",
            strict_backend=True,
            warn_on_compile=False,
        )
    )
    torch.manual_seed(713)
    delay = torch.rand(4096, device="cuda") * 10
    sampled = torch.rand_like(delay)
    brightness = torch.rand(4096, 3, device="cuda")
    response = torch.randn(4096, 6, device="cuda")
    response[:, :3].abs_()
    grid = NativeCDFGrid(delay, [response[:, :3]], step=0.25, runtime=runtime)
    reference = NativeCDFGrid(delay.double(), [response[:, :3].double()], step=0.25)
    expected = reference.project(
        sampled.double(), brightness.double(), response.double(), 3, True
    )
    grid.project(sampled, brightness, response, 3, True)
    graphs = counters["stats"]["unique_graphs"]
    for factor in (1.0, 1.5):
        actual = grid.project(sampled * factor, brightness, response, 3, True)
        for value, target in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                value.double(), target * factor, rtol=3e-5, atol=2e-5
            )
        assert counters["stats"]["unique_graphs"] == graphs
