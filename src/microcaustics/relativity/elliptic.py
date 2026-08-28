"""Differentiable real Carlson and Legendre elliptic integrals.

The fixed-count duplication algorithms are friendly to ``torch.compile`` and
CUDA graphs. They cover the non-negative real domains required by the package's
separated Kerr geodesics. Singular analytic continuations are intentionally
outside this private numerical layer.
"""

from __future__ import annotations

import math

import torch


def _validate_real_floating(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    values = torch.broadcast_tensors(*values)
    if any(value.is_complex() for value in values):
        raise TypeError("elliptic primitives accept real tensors only")
    if values[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("elliptic primitives require float32 or float64")
    return values


def carlson_rf(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    iterations: int | None = None,
) -> torch.Tensor:
    """Return Carlson's symmetric integral :math:`R_F(x,y,z)`."""

    x, y, z = _validate_real_floating(x, y, z)
    if iterations is None:
        iterations = 5 if x.dtype == torch.float32 else 12
    tiny = float(torch.finfo(x.dtype).tiny) ** 0.25
    xn, yn, zn = x.clamp_min(0.0), y.clamp_min(0.0), z.clamp_min(0.0)
    invalid_pair = torch.minimum(
        torch.minimum(xn + yn, xn + zn),
        yn + zn,
    ) < tiny
    xn = torch.where(invalid_pair, xn + tiny, xn)
    yn = torch.where(invalid_pair, yn + tiny, yn)
    zn = torch.where(invalid_pair, zn + tiny, zn)
    for _ in range(int(iterations)):
        sx, sy, sz = torch.sqrt(xn), torch.sqrt(yn), torch.sqrt(zn)
        lam = sx * (sy + sz) + sy * sz
        xn, yn, zn = (
            0.25 * (xn + lam),
            0.25 * (yn + lam),
            0.25 * (zn + lam),
        )
    mean = (xn + yn + zn) / 3.0
    xdev = (mean - xn) / mean
    ydev = (mean - yn) / mean
    zdev = (mean - zn) / mean
    e2 = xdev * ydev - zdev.square()
    e3 = xdev * ydev * zdev
    series = 1.0 + e2 * (-0.1 + e2 / 24.0 - 3.0 * e3 / 44.0) + e3 / 14.0
    return series / torch.sqrt(mean)


def carlson_rc(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    iterations: int | None = None,
) -> torch.Tensor:
    """Return Carlson's degenerate integral :math:`R_C(x,y)`."""

    x, y = _validate_real_floating(x, y)
    if iterations is None:
        iterations = 5 if x.dtype == torch.float32 else 12
    tiny = float(torch.finfo(x.dtype).tiny) ** 0.25
    xn, yn = x.clamp_min(0.0), y.clamp_min(tiny)
    for _ in range(int(iterations)):
        sx, sy = torch.sqrt(xn), torch.sqrt(yn)
        lam = 2.0 * sx * sy + yn
        xn, yn = 0.25 * (xn + lam), 0.25 * (yn + lam)
    mean = (xn + 2.0 * yn) / 3.0
    sn = (yn - mean) / mean
    series = 1.0 + sn.square() * (
        0.3 + sn * (1.0 / 7.0 + sn * (0.375 + sn * (9.0 / 22.0)))
    )
    return series / torch.sqrt(mean)


def carlson_rd(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    iterations: int | None = None,
) -> torch.Tensor:
    """Return Carlson's symmetric integral :math:`R_D(x,y,z)`."""

    x, y, z = _validate_real_floating(x, y, z)
    if iterations is None:
        iterations = 6 if x.dtype == torch.float32 else 12
    tiny = float(torch.finfo(x.dtype).tiny) ** 0.25
    xn, yn, zn = x.clamp_min(0.0), y.clamp_min(0.0), z.clamp_min(tiny)
    invalid_pair = torch.minimum(
        xn + yn,
        torch.minimum(xn + zn, yn + zn),
    ) < tiny
    xn = torch.where(invalid_pair, xn + tiny, xn)
    yn = torch.where(invalid_pair, yn + tiny, yn)
    sigma = torch.zeros_like(xn)
    power4 = torch.ones_like(xn)
    for _ in range(int(iterations)):
        sx, sy, sz = torch.sqrt(xn), torch.sqrt(yn), torch.sqrt(zn)
        lam = sx * (sy + sz) + sy * sz
        sigma = sigma + power4 / (sz * (zn + lam))
        power4 = 0.25 * power4
        xn, yn, zn = (
            0.25 * (xn + lam),
            0.25 * (yn + lam),
            0.25 * (zn + lam),
        )
    mean = (xn + yn + 3.0 * zn) / 5.0
    xdev = (mean - xn) / mean
    ydev = (mean - yn) / mean
    zdev = (mean - zn) / mean
    ea = xdev * ydev
    eb = zdev.square()
    ec = ea - eb
    ed = ea - 6.0 * eb
    ef = ed + 2.0 * ec
    s1 = ed * (
        -3.0 / 14.0
        + 0.25 * (9.0 / 22.0) * ed
        - 1.5 * (3.0 / 26.0) * zdev * ef
    )
    s2 = zdev * (
        ef / 6.0
        + zdev * (-(9.0 / 22.0) * ec + zdev * (3.0 / 26.0) * ea)
    )
    return 3.0 * sigma + power4 * (1.0 + s1 + s2) / (
        mean * torch.sqrt(mean)
    )


def carlson_rj(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    p: torch.Tensor,
    *,
    iterations: int | None = None,
) -> torch.Tensor:
    """Return positive-real Carlson integral :math:`R_J(x,y,z,p)`."""

    x, y, z, p = _validate_real_floating(x, y, z, p)
    if iterations is None:
        iterations = 10 if x.dtype == torch.float32 else 12
    tiny = float(torch.finfo(x.dtype).tiny) ** 0.25
    xn, yn, zn, pn = (
        x.clamp_min(0.0),
        y.clamp_min(0.0),
        z.clamp_min(0.0),
        p.clamp_min(tiny),
    )
    invalid_pair = torch.minimum(
        torch.minimum(xn + yn, xn + zn),
        torch.minimum(yn + zn, pn),
    ) < tiny
    xn = torch.where(invalid_pair, xn + tiny, xn)
    yn = torch.where(invalid_pair, yn + tiny, yn)
    zn = torch.where(invalid_pair, zn + tiny, zn)
    sigma = torch.zeros_like(xn)
    power4 = torch.ones_like(xn)
    for _ in range(int(iterations)):
        sx, sy, sz = torch.sqrt(xn), torch.sqrt(yn), torch.sqrt(zn)
        lam = sx * (sy + sz) + sy * sz
        alpha = (pn * (sx + sy + sz) + sx * sy * sz).square()
        beta = pn * (pn + lam).square()
        sigma = sigma + power4 * carlson_rc(alpha, beta)
        power4 = 0.25 * power4
        xn, yn, zn, pn = (
            0.25 * (xn + lam),
            0.25 * (yn + lam),
            0.25 * (zn + lam),
            0.25 * (pn + lam),
        )
    mean = (xn + yn + zn + 2.0 * pn) / 5.0
    xdev = (mean - xn) / mean
    ydev = (mean - yn) / mean
    zdev = (mean - zn) / mean
    pdev = (mean - pn) / mean
    ea = xdev * (ydev + zdev) + ydev * zdev
    eb = xdev * ydev * zdev
    ec = pdev.square()
    e2 = ea - 3.0 * ec
    e3 = eb + 2.0 * pdev * (ea - ec)
    s1 = 1.0 + e2 * (
        -3.0 / 14.0
        + 0.75 * (3.0 / 22.0) * e2
        - 1.5 * (3.0 / 26.0) * e3
    )
    s2 = eb * (
        1.0 / 6.0
        + pdev * (-2.0 * (3.0 / 22.0) + pdev * (3.0 / 26.0))
    )
    s3 = pdev * ea * (1.0 / 3.0 - pdev * (3.0 / 22.0)) - pdev * ec / 3.0
    return 3.0 * sigma + power4 * (s1 + s2 + s3) / (
        mean * torch.sqrt(mean)
    )


def elliptic_k(parameter: torch.Tensor) -> torch.Tensor:
    """Return complete Legendre integral :math:`K(m)` for real ``m < 1``."""

    parameter = torch.as_tensor(parameter)
    return carlson_rf(
        torch.zeros_like(parameter),
        1.0 - parameter,
        torch.ones_like(parameter),
    )


def _elliptic_f_principal(
    amplitude: torch.Tensor,
    parameter: torch.Tensor,
) -> torch.Tensor:
    amplitude, parameter = torch.broadcast_tensors(amplitude, parameter)
    sine = torch.sin(amplitude)
    sine2 = sine.square()
    return sine * carlson_rf(
        (1.0 - sine2).clamp_min(0.0),
        (1.0 - parameter * sine2).clamp_min(0.0),
        torch.ones_like(sine2),
    )


def elliptic_f(amplitude: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    r"""Return incomplete Legendre integral :math:`F(\phi|m)`."""

    amplitude, parameter = torch.broadcast_tensors(amplitude, parameter)
    pi = torch.as_tensor(math.pi, dtype=amplitude.dtype, device=amplitude.device)
    absolute = amplitude.abs()
    periods = torch.floor((absolute + 0.5 * pi) / pi)
    reduced = absolute - periods * pi
    result = 2.0 * periods * elliptic_k(parameter)
    result = result + _elliptic_f_principal(reduced, parameter)
    return torch.copysign(result, amplitude)


def _elliptic_e_principal(
    amplitude: torch.Tensor,
    parameter: torch.Tensor,
) -> torch.Tensor:
    amplitude, parameter = torch.broadcast_tensors(amplitude, parameter)
    sine = torch.sin(amplitude)
    sine2 = sine.square()
    x = (1.0 - sine2).clamp_min(0.0)
    y = (1.0 - parameter * sine2).clamp_min(0.0)
    one = torch.ones_like(sine2)
    return sine * carlson_rf(x, y, one) - (
        parameter * sine.pow(3) / 3.0
    ) * carlson_rd(x, y, one)


def elliptic_e(amplitude: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    r"""Return incomplete Legendre integral :math:`E(\phi|m)`."""

    amplitude, parameter = torch.broadcast_tensors(amplitude, parameter)
    pi = torch.as_tensor(math.pi, dtype=amplitude.dtype, device=amplitude.device)
    absolute = amplitude.abs()
    periods = torch.floor((absolute + 0.5 * pi) / pi)
    reduced = absolute - periods * pi
    complete = _elliptic_e_principal(
        torch.full_like(reduced, 0.5 * math.pi),
        parameter,
    )
    result = 2.0 * periods * complete
    result = result + _elliptic_e_principal(reduced, parameter)
    return torch.copysign(result, amplitude)


def elliptic_pi_principal(
    characteristic: torch.Tensor,
    amplitude: torch.Tensor,
    parameter: torch.Tensor,
    *,
    iterations: int | None = None,
) -> torch.Tensor:
    r"""Return nonsingular principal-branch :math:`\Pi(n;\phi|m)`."""

    characteristic, amplitude, parameter = torch.broadcast_tensors(
        characteristic,
        amplitude,
        parameter,
    )
    sine = torch.sin(amplitude)
    sine2 = sine.square()
    x = (1.0 - sine2).clamp_min(0.0)
    y = (1.0 - parameter * sine2).clamp_min(0.0)
    one = torch.ones_like(sine2)
    p = 1.0 - characteristic * sine2
    return sine * carlson_rf(x, y, one) + (
        characteristic * sine.pow(3) / 3.0
    ) * carlson_rj(x, y, one, p, iterations=iterations)


def jacobi_sn_cn(
    argument: torch.Tensor,
    parameter: torch.Tensor,
    *,
    iterations: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Jacobi ``sn`` and ``cn`` for ``0 <= parameter <= 1``."""

    argument, parameter = _validate_real_floating(argument, parameter)
    m = parameter.clamp(0.0, 1.0)
    if iterations is None:
        iterations = 5 if m.dtype == torch.float32 else 10
    a = torch.ones_like(m)
    b = torch.sqrt((1.0 - m).clamp_min(0.0))
    ratios = []
    for _ in range(int(iterations)):
        next_a = 0.5 * (a + b)
        next_b = torch.sqrt((a * b).clamp_min(0.0))
        ratios.append(0.5 * (a - b) / next_a.clamp_min(torch.finfo(m.dtype).tiny))
        a, b = next_a, next_b
    phase = a * argument * float(2**int(iterations))
    for ratio in reversed(ratios):
        phase = 0.5 * (
            phase + torch.asin((ratio * torch.sin(phase)).clamp(-1.0, 1.0))
        )
    sn, cn = torch.sin(phase), torch.cos(phase)
    near_one = (1.0 - m) < 64.0 * torch.finfo(m.dtype).eps
    return (
        torch.where(near_one, torch.tanh(argument), sn),
        torch.where(near_one, torch.reciprocal(torch.cosh(argument)), cn),
    )
