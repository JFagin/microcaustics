"""Numerical contracts for conservative CDF/PCHIP transfer reconstruction."""

import numpy as np
import pytest
import torch

import microcaustics as mc


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched", [False, True])
def test_scipy_reference_nonuniform_coarse_fine_and_cropped(dtype, batched):
    scipy = pytest.importorskip("scipy.interpolate")
    gen = torch.Generator().manual_seed(21)
    edges = torch.cat(
        (torch.zeros(1), torch.rand(80, generator=gen).add(0.1).cumsum(0))
    ).to(dtype)
    shape = (3, 80, 6) if batched else (80, 6)
    mass = torch.rand(shape, generator=gen, dtype=dtype)
    mass[..., 15:25, :] = 0
    for output in (
        torch.linspace(-2, float(edges[-1]) + 2, 170, dtype=dtype),
        torch.linspace(3, 20, 7, dtype=dtype),
        edges,
    ):
        actual = mc.CDFResponseRebinner(edges, output)(mass)
        cumulative = np.concatenate(
            (np.zeros((*shape[:-2], 1, 6)), mass.double().numpy().cumsum(axis=-2)),
            axis=-2,
        )
        reference = scipy.PchipInterpolator(edges.double().numpy(), cumulative, axis=-2)
        expected = np.diff(
            reference(output.clamp(edges[0], edges[-1]).double().numpy()), axis=-2
        )
        np.testing.assert_allclose(
            actual.numpy(),
            expected,
            rtol=1e-4 if dtype == torch.float32 else 1e-11,
            atol=2e-6 if dtype == torch.float32 else 1e-12,
        )
        assert (actual >= 0).all()


def test_identity_conservation_empty_causal_and_weak_tail():
    edges = torch.arange(6.0)
    mass = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1e-12, 0.0], [0.0, 0.0]])
    torch.testing.assert_close(mc.CDFResponseRebinner(edges, edges)(mass), mass)
    output = torch.linspace(-1, 6, 701)
    values = mc.CDFResponseRebinner(edges, output)(mass)
    torch.testing.assert_close(values.sum(0), mass.sum(0), rtol=2e-6, atol=0)
    assert (values[output[1:] <= 1] == 0).all()
    assert (values[output[:-1] >= 4] == 0).all()
    assert values[:, 1].count_nonzero() == 0
    tail = values[(output[:-1] >= 3) & (output[1:] <= 4), 0].sum()
    torch.testing.assert_close(tail, mass[3, 0], rtol=2e-5, atol=0)


def test_truncated_single_interval_is_not_renormalized():
    mass = torch.tensor([[10.0, 20.0]])
    actual = mc.CDFResponseRebinner([0.0, 10.0], [2.0, 3.0, 4.0])(mass)
    torch.testing.assert_close(actual, torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    assert mc.CDFResponseRebinner([0.0, 10.0], [11.0, 12.0])(mass).count_nonzero() == 0


@pytest.mark.parametrize("edges", [[0.0, 0.0], [0.0, float("nan")], [1.0, 0.0], [0.0]])
def test_invalid_edges(edges):
    with pytest.raises(ValueError):
        mc.CDFResponseRebinner(edges, [0.0, 1.0])
    with pytest.raises(ValueError):
        mc.CDFResponseRebinner([0.0, 1.0], edges)


def test_invalid_mass():
    plan = mc.CDFResponseRebinner([0.0, 1.0], [0.0, 0.5, 1.0])
    for mass in (
        torch.tensor([[-1.0]]),
        torch.tensor([[float("inf")]]),
        torch.ones(2, 1),
        torch.ones(1),
    ):
        with pytest.raises(ValueError):
            plan(mass)


def test_signed_quadratic_experiment_retains_negative_amplitude():
    plan = mc.CDFResponseRebinner(torch.arange(7.0), torch.linspace(0, 6, 241))
    mass = torch.tensor(
        [[1.0, -2.0], [-1.0, 2.0], [3.0, 0.0], [-3.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
    )
    actual = plan.experimental_signed(mass)
    assert (actual < 0).any() and (actual > 0).any()
    torch.testing.assert_close(actual.sum(0), mass.sum(0), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        actual, plan(mass.clamp_min(0)) - plan((-mass).clamp_min(0))
    )


def test_product_preserves_original_lag_and_cropped_mass():
    product = mc.TransferFunction(
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([[0.4], [0.6]]),
        torch.tensor([0.91]),
        ("g",),
    )
    result = mc.rebin_transfer_function(product, [0.0, 0.5, 1.0])
    torch.testing.assert_close(result.mean_delays_days, product.mean_delays_days)
    torch.testing.assert_close(result.values.sum(), torch.tensor(0.4))
    assert result.metadata["response_reconstruction"] == "cdf_pchip"


def test_signed_kernel_convolution_converges_without_phase_shift():
    # A signed second-order kernel. A constant squared driver must recover
    # its signed integral, not a normalized/clipped probability distribution.
    output = torch.linspace(0.0, 8.0, 3201, dtype=torch.float64)
    center = (output[1:] + output[:-1]) / 2
    times = torch.linspace(9.0, 20.0, 40, dtype=torch.float64)
    exact_mass = output[1:] * (-output[1:]).exp() - output[:-1] * (-output[:-1]).exp()
    driver_squared = torch.sin((times[:, None] - center[None]) * 0.7).square()
    exact_flux = driver_squared @ exact_mass
    errors = []
    for n in (20, 40, 80):
        native = torch.linspace(0.0, 8.0, n + 1, dtype=torch.float64)
        mass = (native[1:] * (-native[1:]).exp() - native[:-1] * (-native[:-1]).exp())[
            :, None
        ]
        actual = mc.CDFResponseRebinner(native, output).experimental_signed(mass)[:, 0]
        torch.testing.assert_close(actual.sum(), mass.sum(), atol=2e-15, rtol=1e-12)
        errors.append(float((driver_squared @ actual - exact_flux).abs().max()))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 3e-4


def test_analytic_exponential_response_and_native_grid_convergence():
    output = torch.linspace(0, 10, 2001, dtype=torch.float64)
    exact = (-output[:-1]).exp() - (-output[1:]).exp()
    errors = []
    for n in (25, 50, 100):
        edges = torch.linspace(0, 10, n + 1, dtype=torch.float64)
        mass = ((-edges[:-1]).exp() - (-edges[1:]).exp())[:, None]
        values = mc.CDFResponseRebinner(edges, output)(mass)[:, 0]
        errors.append((values - exact).abs().sum().item())
        if n == 100:
            center = (output[:-1] + output[1:]) / 2
            assert (
                abs(
                    float((values * center).sum() / values.sum())
                    - (1 - 11 * np.exp(-10)) / (1 - np.exp(-10))
                )
                < 2e-5
            )
    assert errors[2] < errors[1] < errors[0]


@pytest.mark.parametrize("inclination", [0.0, 60.0])
@pytest.mark.parametrize(
    "mass_solar,native_days",
    [
        (10**9.08, 0.25),
        (10**9.08, 0.5),
        (10**9.08, 1.0),
        (1e8, 0.1),
        (1e8, 0.25),
        (1e7, 0.1),
        (1e7, 0.25),
    ],
)
def test_reconstructed_physical_centroid_matches_direct_pixel_lag(
    inclination, mass_solar, native_days
):
    """Check reconstructed values, not the preserved mean-delay metadata.

    Permit 1% or 0.1 day, whichever is larger. The output covers the entire
    response. Cropped kernels are intentionally not required to keep the lag.
    A moving idealized fold weights different parts of the inclined disk.
    """
    distances = mc.LensingDistances.from_redshifts(0.0395, 1.695)
    runtime = mc.RuntimeConfig(device="cpu", dtype=torch.float32, backend="torch-eager")
    source = mc.ThinDiskModel(
        black_hole_mass_solar=mass_solar,
        eddington_ratio=0.34,
        spin=0.74,
        inclination_deg=inclination,
        bands_angstrom=dict(
            zip("ugrizy", (3671, 4827, 6223, 7546, 8691, 9712), strict=True)
        ),
        driving_signal=mc.CallableDrivingSignal(lambda t: torch.ones_like(t)),
        lamp_fraction=0.1,
        source_grid_shape=128,
    ).pixelate(distances, runtime=runtime)
    maximum = float(source.delay_days[source.transfer.hit].max())
    bins = int(np.ceil(maximum / native_days)) + 1
    native = torch.arange(bins + 1, dtype=torch.float32) * native_days
    output = torch.linspace(0, float(native[-1]), bins * round(native_days / 0.05) + 1)
    centers = (output[:-1] + output[1:]) / 2
    x = torch.linspace(-1, 1, 128)[None].expand(128, -1)
    maps = [torch.ones_like(x)]
    for offset in (-0.15, 0.0, 0.15):
        d = x - offset
        maps.append(1 + torch.where(d > 0, 0.4 / d.clamp_min(1e-3).sqrt(), 0))
    plan = mc.CDFResponseRebinner(native, output)
    for magnification in maps:
        mass = source.transfer_function(native, magnification=magnification, reconstruction="ordinary")
        reconstructed = plan(mass)
        actual = (reconstructed * centers[:, None]).sum(0) / reconstructed.sum(0)
        expected = source.mean_response_delays(magnification=magnification)
        tolerance = torch.maximum(expected.abs() * 0.01, expected.new_tensor(0.1))
        assert torch.isfinite(actual).all()
        assert ((actual - expected).abs() <= tolerance).all(), (actual, expected)
        torch.testing.assert_close(reconstructed.sum(0), mass.sum(0), rtol=3e-6, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_float32_compiled_matches_eager_and_reuses_specialization():
    from torch._dynamo.utils import counters

    from microcaustics.compile import _PREPARED_SPECIALIZATIONS
    from microcaustics.response_rebin import _pchip_rebin
    from microcaustics.runtime import resolve_runtime

    runtime = resolve_runtime(
        mc.RuntimeConfig(
            device="cuda",
            backend=mc.Backend.TORCH_COMPILE,
            strict_backend=True,
            warn_on_compile=False,
        )
    )
    native = torch.linspace(0, 60, 129, device="cuda")
    output = torch.linspace(0, 60, 1201, device="cuda")
    mass = torch.rand(4, 128, 6, device="cuda")
    plan = mc.CDFResponseRebinner(native, output, runtime=runtime)
    actual = plan(mass, validate=False)
    expected = mc.CDFResponseRebinner(native.cpu(), output.cpu())(mass.cpu())
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-5)
    signatures = {x for x in _PREPARED_SPECIALIZATIONS if x[0] is _pchip_rebin}
    graphs = counters["stats"]["unique_graphs"]
    for scale in (2.0, 0.5):
        torch.testing.assert_close(
            plan(mass * scale, validate=False), actual * scale, rtol=2e-5, atol=2e-6
        )
    assert signatures == {x for x in _PREPARED_SPECIALIZATIONS if x[0] is _pchip_rebin}
    assert counters["stats"]["unique_graphs"] == graphs


def test_plotting_rejects_removed_gaussian_option():
    import microcaustics.plotting as mcp

    with pytest.raises(TypeError, match="smoothing_sigma_days"):
        mcp.transfer_response_density(None, smoothing_sigma_days=0.5)
