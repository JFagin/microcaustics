"""Independent lamp-ray tests, including branches hidden by profile continuation."""

import math

import numpy as np
import pytest
import torch

from microcaustics.relativity import axis_lamppost_profile
from microcaustics.relativity.geodesics import polar_mino_time
from microcaustics.relativity.kerr import kerr_isco_radius
from microcaustics.relativity.lamppost import _trace_chunk


def _integrated_rays(spin, height, angles, outer):
    """Integrate the radial potential directly, without roots or inversions."""
    integrate = pytest.importorskip("scipy.integrate")
    special = pytest.importorskip("scipy.special")
    sigma = height**2 + spin**2
    horizon = 1 + math.sqrt(1 - spin**2)
    inner = float(kerr_isco_radius(torch.tensor(spin, dtype=torch.float64)))
    expected = []
    for alpha in angles:
        eta = (
            sigma**2 / (height**2 - 2 * height + spin**2) * math.sin(alpha) ** 2
            - spin**2
        )
        if eta <= 0:
            expected.append((False, math.nan))
            continue
        polar_time = special.ellipk(-(spin**2) / eta) / math.sqrt(eta)

        def rhs(_, state, eta=eta):
            r, momentum = state
            # r'' = R'(r)/2 crosses finite turning points without a sign switch.
            return momentum, 2 * r**3 + (spin**2 - eta) * r + eta + spin**2

        def capture(_, state):
            return state[0] - (horizon + 1e-6)

        def escape(_, state):
            return 2 * outer - state[0]

        capture.terminal = escape.terminal = True
        result = integrate.solve_ivp(
            rhs,
            (0, polar_time),
            (height, -sigma * math.cos(alpha)),
            method="DOP853",
            rtol=1e-11,
            atol=1e-12,
            events=(capture, escape),
        )
        radius = result.y[0, -1]
        expected.append((result.status == 0 and inner <= radius <= outer, radius))
    return expected


@pytest.mark.parametrize("dtype,rtol", [(torch.float32, 4e-5), (torch.float64, 2e-8)])
@pytest.mark.parametrize(
    "spin,height",
    [
        (0.0, 23.0),
        (0.74, 23.20620346069336),
        (-0.74, 23.0),
        (0.99, 10.0),
        (0.74, 50.0),
        (0.99, 2.0),
    ],
)
def test_lamp_hits_match_independently_integrated_radial_potential(
    dtype, rtol, spin, height
):
    angles = torch.deg2rad(
        torch.tensor(
            [
                45.0,
                60.0,
                75.0,
                85.0,
                89.0,
                89.999,
                90.001,
                91.0,
                92.0,
                93.0,
                94.0,
                120.0,
                140.0,
            ],
            dtype=dtype,
        )
    )
    signs = torch.where(angles <= math.pi / 2, -1.0, 1.0)

    def scalar(x):
        return torch.tensor(x, dtype=dtype)

    inner = kerr_isco_radius(scalar(spin))
    result = _trace_chunk(
        angles,
        signs,
        scalar(spin),
        scalar(height),
        scalar(0),
        inner,
        scalar(690.2876),
        quadrature_order=24,
    )
    expected = _integrated_rays(
        float(scalar(spin)),
        float(scalar(height)),
        angles.double().tolist(),
        float(scalar(690.2876)),
    )
    hit = torch.tensor([row[0] for row in expected])
    torch.testing.assert_close(result[-1], hit)
    reference = torch.tensor([row[1] for row in expected], dtype=dtype)
    torch.testing.assert_close(result[0][hit], reference[hit], rtol=rtol, atol=2e-5)
    assert result[0].dtype == dtype
    assert bool(torch.isfinite(result[1][hit]).all())


@pytest.mark.parametrize("dtype,rtol", [(torch.float32, 3e-7), (torch.float64, 3e-14)])
def test_axial_polar_time_uses_complete_integral(dtype, rtol):
    special = pytest.importorskip("scipy.special")
    spin = torch.tensor([0.0, -0.74, 0.74, 0.99], dtype=dtype)
    eta = torch.tensor([1.0, 20.0, 600.0, 10000.0], dtype=dtype)
    actual, valid = polar_mino_time(
        spin, eta, torch.zeros_like(eta), -torch.ones_like(eta), torch.zeros_like(eta)
    )
    reference = special.ellipk(
        -(spin.double().numpy() ** 2) / eta.double().numpy()
    ) / np.sqrt(eta.double().numpy())
    np.testing.assert_allclose(actual.numpy(), reference, rtol=rtol, atol=0)
    assert bool(valid.all())


@pytest.mark.parametrize(
    "dtype,order,rtol",
    [
        (torch.float32, 24, 4e-5),
        (torch.float64, 24, 2e-5),
        (torch.float64, 48, 2e-8),
    ],
)
def test_outward_coordinate_time_matches_independent_quadrature(dtype, order, rtol):
    integrate = pytest.importorskip("scipy.integrate")
    spin, height = 0.74, 23.20620346069336

    def scalar(x):
        return torch.tensor(x, dtype=dtype)

    a, h = float(scalar(spin)), float(scalar(height))
    angles = torch.deg2rad(torch.tensor([91.0, 92.0, 93.0], dtype=dtype))
    expected = _integrated_rays(a, h, angles.double().tolist(), 690.2876)
    result = _trace_chunk(
        angles,
        torch.ones_like(angles),
        scalar(spin),
        scalar(height),
        scalar(0),
        kerr_isco_radius(scalar(spin)),
        scalar(690.2876),
        quadrature_order=order,
    )
    assert bool(result[-1].all())
    times = []
    for alpha, (_, radius) in zip(angles.double().tolist(), expected, strict=True):
        eta = (h * h + a * a) ** 2 / (h * h - 2 * h + a * a) * math.sin(
            alpha
        ) ** 2 - a * a

        def radial_integrand(r, eta=eta):
            delta = r * r - 2 * r + a * a
            potential = (r * r + a * a) ** 2 - delta * (eta + a * a)
            return (r * r + 2 * r * (r * r + a * a) / delta) / math.sqrt(potential)

        radial = integrate.quad(radial_integrand, h, radius, epsabs=1e-9, epsrel=1e-11)[
            0
        ]
        polar = integrate.quad(
            lambda theta, eta=eta: (
                a
                * a
                * math.cos(theta) ** 2
                / math.sqrt(eta + a * a * math.cos(theta) ** 2)
            ),
            0,
            math.pi / 2,
            epsabs=1e-12,
        )[0]
        times.append(radial + polar)
    # Fixed 24-node quadrature has a small truncation error toward outer
    # escape; doubling the order must recover the independent integral.
    torch.testing.assert_close(
        result[1], torch.tensor(times, dtype=dtype), rtol=rtol, atol=1e-5
    )


@pytest.mark.parametrize("nalpha", [1024, 8192, 16384])
def test_dense_float32_lamp_profiles_do_not_collapse_near_tangent(nalpha):
    kwargs = dict(
        spin=0.74,
        source_height_rg=23.20620346069336,
        disk_outer_rg=690.2876238256858,
        nalpha=nalpha,
        radial_bins=512,
    )
    reference = axis_lamppost_profile(**kwargs, dtype=torch.float64)
    actual = axis_lamppost_profile(**kwargs, dtype=torch.float32)
    hit = actual.rays.hit
    assert int((hit & (actual.rays.initial_radial_sign > 0)).sum()) > 0
    assert float(actual.rays.radius_rg[hit].max()) > 680.0
    radii, order = torch.sort(actual.rays.radius_rg[hit])
    angles = actual.rays.launch_angle_rad[hit][order]
    assert bool((angles.diff() >= 0).all())
    assert bool((radii.diff() > 0).all())
    # Check the annular profile, not just globally normalized total power.
    torch.testing.assert_close(
        actual.illumination.double(), reference.illumination, rtol=0.02, atol=1e-9
    )
    torch.testing.assert_close(
        actual.lamp_delay_rg.double(), reference.lamp_delay_rg, rtol=2e-4, atol=0.015
    )


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_dense_lamp_profile_preserves_float32_and_reuses_cache():
    from microcaustics.relativity.lamppost import clear_compiled_lamppost_cache

    kwargs = dict(
        spin=0.74,
        source_height_rg=23.20620346069336,
        disk_outer_rg=690.2876238256858,
        nalpha=8192,
        radial_bins=512,
    )
    reference = axis_lamppost_profile(**kwargs, dtype=torch.float64)
    clear_compiled_lamppost_cache()
    actual = axis_lamppost_profile(
        **kwargs,
        device="cuda",
        dtype=torch.float32,
        compile_solver=True,
        fallback_to_eager=False,
        warn_on_compile=False,
    )
    repeat = axis_lamppost_profile(
        **kwargs,
        device="cuda",
        dtype=torch.float32,
        compile_solver=True,
        fallback_to_eager=False,
        warn_on_compile=False,
    )
    assert actual.illumination.dtype == torch.float32
    assert "cache=hit" in repeat.rays.execution
    assert repeat.rays.compile_warmup_s == 0.0
    torch.testing.assert_close(
        actual.illumination.double().cpu(), reference.illumination, rtol=0.02, atol=1e-9
    )
    torch.testing.assert_close(
        actual.lamp_delay_rg.double().cpu(),
        reference.lamp_delay_rg,
        rtol=2e-4,
        atol=0.02,
    )
    torch.testing.assert_close(actual.illumination, repeat.illumination, rtol=0, atol=0)
