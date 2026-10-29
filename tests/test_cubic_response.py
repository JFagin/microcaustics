"""Experimental third-order response, independent derivatives and integration."""

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import microcaustics as mc
from microcaustics._response_cdf import (
    callable_driver_integrals,
    driver_integrals,
    project_deposited_response,
)
from microcaustics.compile import run_tensor_kernel
from microcaustics.photometry import (
    _flexible_streaming_light_curves,
    _source_band_chunks,
)
from microcaustics.sources.linear_response import LinearResponsePlan
from microcaustics.sources.reprocessing import _quadratic_response_weight_kernel


@pytest.mark.parametrize("amplitude", [0.2, 1.0, 1.7])
def test_cubic_thermal_derivative_against_autograd(amplitude):
    # One wavelength, independent amplitudes per pixel. Cover the Rayleigh-Jeans
    # and Wien regimes without differentiating through the numerical exp cap.
    static = torch.logspace(12, 24, 17, dtype=torch.float64)[:, None]
    heating = 0.8 * static
    x = torch.full_like(static, amplitude, requires_grad=True)
    wave = torch.tensor([4e-7], dtype=torch.float64)
    g = torch.full_like(static, 0.9)
    solid = torch.full_like(static, 1e-25)
    hit = torch.ones_like(static, dtype=torch.bool)
    color, z, area = 1.2, 1.5, 1e20
    # Independently differentiate the physical Planck law, not the derivative kernel.
    temperature = (static + x * heating).pow(0.25)
    exponent = 6.62607015e-34 * 299792458 / (wave * g * 1.380649e-23 * color * temperature)
    flux = 2 * 6.62607015e-34 * 299792458 / wave**3 / torch.expm1(exponent)
    flux = flux / color**4 * solid / (1 + z)**3 * 1e26 / area
    expected = [flux[..., None]]
    derivative = flux
    for _ in range(3):
        derivative = torch.autograd.grad(derivative.sum(), x, create_graph=True)[0]
        expected.append(derivative[..., None])
    actual = _quadratic_response_weight_kernel(
        static, heating, g, solid, hit, wave, torch.tensor(z), torch.tensor(color, dtype=static.dtype),
        area, amplitude, True, True,
    )
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference, rtol=3e-11, atol=1e-50)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cubic_kernel_matches_existing_mean_and_derivatives(dtype):
    from test_spectral_batch_optimization import thermal_fixture

    source, _, _, _ = thermal_fixture()
    transfer, _, static, response, _ = source._evaluation_tensors(device="cpu", dtype=dtype)
    args = (
        static, response, transfer.gfactor, transfer.solid_angle_sr, transfer.hit,
        torch.tensor(source.geometry.wavelengths_angstrom, dtype=dtype) * 1e-10 / 2.5,
        torch.tensor(1.5, dtype=dtype), torch.tensor(1., dtype=dtype),
        source.geometry.pixel_area_m2, 1., True,
    )
    mean, first, second, third = _quadratic_response_weight_kernel(*args, True)
    torch.testing.assert_close(mean, source.at_driver_mean().brightness([0.], dtype=dtype)[0], rtol=3e-6, atol=0)
    old_first, old_second = _quadratic_response_weight_kernel(*args)
    torch.testing.assert_close(first, old_first, rtol=0, atol=0)
    torch.testing.assert_close(second, old_second, rtol=0, atol=0)
    assert torch.isfinite(third).all()


@pytest.mark.parametrize("phase", [0., 0.137, 0.5])
@pytest.mark.parametrize("step", [0.1, 0.25, 0.5])
def test_cubic_driver_integrals_against_independent_split_gauss(phase, step):
    knots = np.arange(-5, 6, dtype=float)
    values = 1 + 0.5 * np.sin(knots * 1.9)
    times = np.arange(3, dtype=float) + phase
    edges = np.arange(round(3 / step) + 1) * step
    actual = driver_integrals(*[torch.from_numpy(x) for x in (times, edges, knots, values)], 1., True, True)
    expected = np.zeros((3, len(times), 3, len(edges) - 1))
    nodes, weights = np.polynomial.legendre.leggauss(4)
    for ti, time in enumerate(times):
        for i, edge in enumerate(edges[:-1]):
            cuts = (time - edge - knots) / step
            cuts = np.r_[0., np.sort(cuts[(cuts > 0) & (cuts < 1)]), 1.]
            for low, high in zip(cuts[:-1], cuts[1:], strict=True):
                t = (low + high) / 2 + (high - low) / 2 * nodes
                w = weights * (high - low) / 2
                driver = np.interp(time - edge - step * t, knots, values) - 1
                basis = np.array([6*(t-t*t), 1-4*t+3*t*t, 3*t*t-2*t])
                for order in range(3):
                    expected[order, ti, :, i] += (basis * w * driver**(order+1)).sum(1)
    for order, result in enumerate(actual):
        np.testing.assert_allclose(result, expected[order].reshape(len(times), -1), rtol=1e-10, atol=3e-14)


def test_cubic_callable_integrals_for_linear_driver():
    times = torch.tensor([0.137, 1.2], dtype=torch.float64)
    edges = torch.linspace(0, 2, 21, dtype=times.dtype)
    knots = torch.arange(-5, 6, dtype=times.dtype)
    signal = mc.CallableDrivingSignal(lambda t: 1 + 0.1 * t)
    actual = callable_driver_integrals(signal, times, edges, 1., True, True)
    expected = driver_integrals(times, edges, knots, 1 + 0.1 * knots, 1., True, True)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-10)


def test_cubic_signed_deposition_preserves_each_order_mass():
    response = torch.tensor([[1., -2., 4.], [2., 3., -1.], [1., -1., -3.]])
    mu = torch.tensor([2., 1., 1.])
    _, mass = project_deposited_response(mu, torch.ones(3, 1), response,
        torch.tensor([0, 1, 2]), torch.tensor([0.2, 0.3, 0.4]), 4, 1, True, True)
    assert (mass >= 0).all()
    total = mass.sum(0)
    signed = torch.stack((total[0], total[1]-total[2], total[3]-total[4]))
    torch.testing.assert_close(signed, mu @ response)


@pytest.mark.parametrize("reconstruction", ["ordinary", "cdf"])
def test_cubic_packed_contraction_equals_three_separate_orders(reconstruction):
    from test_spectral_batch_optimization import thermal_fixture

    source, _, maps, runtime = thermal_fixture()
    times = torch.tensor([0., 0.37, 1.2, 2.], dtype=torch.float64)
    plan = LinearResponsePlan(source, _source_band_chunks(source, 2), times,
        delay_bin_days=0.2, response_order=3,
        runtime=replace(runtime, response_reconstruction=reconstruction))
    projection = plan.project_map(maps[2].values)
    powers = plan.driver_powers.chunk(3, dim=1)
    coefficients = projection[1].chunk(3, dim=0)
    expected = projection[0][None] + sum(d @ c for d, c in zip(powers, coefficients, strict=True))
    torch.testing.assert_close(plan.flux(projection, 0, len(times)), expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("reconstruction", ["ordinary", "cdf"])
def test_cubic_public_mode_and_fused_flux_improve_small_amplitude_error(reconstruction):
    from test_spectral_batch_optimization import thermal_fixture

    source, distances, maps, runtime = thermal_fixture()
    runtime = replace(runtime, response_reconstruction=reconstruction)
    # Uniform delays remove CDF shape error, isolating Taylor order.
    source = replace(source, delay_days=torch.zeros_like(source.delay_days),
                     signal=mc.CallableDrivingSignal(lambda t: 1 + 0.2 * torch.cos(t)))
    times = torch.tensor([0., 0.25, 0.5, 1., 1.5, 2.], dtype=torch.float64)

    def evaluate(mode):
        (curve,) = _flexible_streaming_light_curves(
            SimpleNamespace(runtime=runtime), None, maps[0].grid, [0., 1., 2.],
            (mc.LightCurveRequest(source, distances, source_evolution=mode),), (times,),
            method=None, schedule=None, map_observer=None, band_batch_size=2,
            map_iterator=iter(maps))
        return curve.flux

    exact = evaluate("exact")
    quadratic = ((evaluate("quadratic_response") - exact) / exact).abs().max()
    cubic = ((evaluate("cubic_response") - exact) / exact).abs().max()
    assert cubic < quadratic / 2


@pytest.mark.parametrize("reconstruction", ["ordinary", "cdf"])
def test_cubic_mean_driver_and_single_matmul(reconstruction):
    from test_spectral_batch_optimization import thermal_fixture

    source, _, maps, runtime = thermal_fixture()
    runtime = replace(runtime, response_reconstruction=reconstruction)
    source = replace(source, signal=mc.CallableDrivingSignal(
        lambda t: torch.full_like(t, 1.3), user_metadata={"mean_amplitude": 1.3}))
    times = torch.tensor([0., 0.5, 1.], dtype=torch.float64)
    plan = LinearResponsePlan(source, _source_band_chunks(source, 2), times,
                             delay_bin_days=0.25, response_order=3, runtime=runtime)
    projection = plan.project_map(maps[1].values)
    with patch("torch.addmm", wraps=torch.addmm) as gemm:
        flux = plan.flux(projection, 0, 3)
        assert gemm.call_count == 1
    torch.testing.assert_close(flux, projection[0][None].expand(3, -1))
    assert torch.count_nonzero(plan.driver_powers) == 0
    if reconstruction == "cdf":
        for order in (1, 2, 3):
            mass = plan.response_bin_masses(projection, plan.edges, order=order)
            coeff = projection[1].chunk(3)[order-1] * math.factorial(order)
            torch.testing.assert_close(mass.sum(0), coeff.chunk(3)[0].sum(0))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cubic_compiled_fused_and_chunked_paths_match(isolated_compiler_cache):
    from test_spectral_batch_optimization import thermal_fixture

    source, _, maps, _ = thermal_fixture()
    chunks = _source_band_chunks(source, 2)
    times = torch.tensor([0., 0.5, 1.], device="cuda")
    mu = maps[1].values.to(times)
    runtime = mc.resolve_runtime(mc.RuntimeConfig(
        device="cuda", backend="triton", strict_backend=True))
    with torch.no_grad():
        fused = LinearResponsePlan(source, chunks, times, response_order=3,
                                   delay_bin_days=0.2, runtime=runtime)
        chunked = LinearResponsePlan(source, chunks, times, response_order=3,
            delay_bin_days=0.2, runtime=replace(runtime, memory_fraction=1e-10))
        eager = LinearResponsePlan(source, chunks, times, response_order=3,
            delay_bin_days=0.2, runtime=mc.resolve_runtime(mc.RuntimeConfig(device="cuda", backend="torch-eager")))
        assert len(fused.parts) == 1 and len(chunked.parts) == 3
        expected = eager.flux(eager.project_map(mu), 0, 3)
        for plan in (fused, chunked):
            torch.testing.assert_close(plan.flux(plan.project_map(mu), 0, 3), expected, rtol=2e-5, atol=0)
        edges = torch.linspace(0, 2, 21, device="cuda")
        knots = torch.arange(-5, 6, device="cuda")
        values = 1 + 0.3 * torch.sin(knots)
        args = (times, edges, knots, values, 1., True, True)
        compiled, _ = run_tensor_kernel(runtime, "test fused cubic integrals", driver_integrals, *args)
        for a, b in zip(compiled, driver_integrals(*args), strict=True):
            torch.testing.assert_close(a, b, atol=5e-7, rtol=2e-5)


@pytest.fixture
def isolated_compiler_cache():
    """Keep this deliberately varied shape test out of other compile budgets."""
    from microcaustics.compile import clear_compiled_kernel_cache

    torch._dynamo.reset()
    clear_compiled_kernel_cache()
    try:
        yield
    finally:
        torch._dynamo.reset()
        clear_compiled_kernel_cache()
