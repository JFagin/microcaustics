"""Grouped deposition must preserve every pixel and its fractional delay weight."""

import pytest
import torch

import microcaustics as mc
from microcaustics._response_cdf import NativeCDFGrid, project_deposited_response

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("concentrated", [False, True])
def test_grouped_deposition_matches_independent_float64_sum(order, concentrated):
    pytest.importorskip("triton")
    from microcaustics.sources.triton_response import grouped_deposit, prepare_groups

    device = "cuda"
    torch.manual_seed(510)
    pixels, bands, bins = 8193, 7, 27
    index = (
        torch.zeros(pixels, device=device, dtype=torch.long)
        if concentrated
        else torch.randint(bins - 1, (pixels,), device=device)
    )
    fraction = torch.rand(pixels, device=device)
    fraction[:3] = torch.tensor([0.0, 0.5, 1.0], device=device)
    mu = torch.rand(pixels, device=device) * 20
    response = torch.randn(pixels, bands * order, device=device)
    response[:, :bands].abs_()
    brightness = torch.rand(pixels, bands, device=device)
    runtime = mc.resolve_runtime(mc.RuntimeConfig(device="cuda", backend="triton"))
    groups = prepare_groups(index, bins)
    actual = grouped_deposit(
        mu, response, fraction, groups, bins, bands, order, runtime
    )
    _, expected = project_deposited_response(
        mu.double(),
        brightness.double(),
        response.double(),
        index,
        fraction.double(),
        bins,
        bands,
        order >= 2,
        order == 3,
    )
    torch.testing.assert_close(actual.double(), expected, atol=3e-3, rtol=3e-6)
    torch.testing.assert_close(
        actual.sum(0).double(), expected.sum(0), atol=3e-3, rtol=3e-6
    )


def test_grouped_cdf_matches_eager_and_reuses_memberships():
    pytest.importorskip("triton")
    from unittest.mock import patch

    from microcaustics.sources.triton_response import prepare_groups

    torch.manual_seed(22)
    delay = torch.rand(8193, device="cuda") * 9
    response = torch.randn(8193, 15, device="cuda")
    response[:, :5].abs_()
    brightness = torch.rand(8193, 5, device="cuda")
    mu = torch.rand(8193, device="cuda")
    runtime = mc.resolve_runtime(
        mc.RuntimeConfig(device="cuda", backend="triton", strict_backend=True)
    )
    grid = NativeCDFGrid(delay, [response[:, :5]], step=0.25, runtime=runtime)
    eager = NativeCDFGrid(delay, [response[:, :5]], step=0.25)
    with patch(
        "microcaustics.sources.triton_response.prepare_groups", wraps=prepare_groups
    ) as prepare:
        for factor in (1.0, 1.3):
            actual = grid.project(mu * factor, brightness, response, 5, True, True)
            expected = eager.project(mu * factor, brightness, response, 5, True, True)
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-5)
        assert prepare.call_count == 1
