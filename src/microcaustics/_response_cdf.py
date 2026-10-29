"""Shared CDF/PCHIP response primitives for transfer products and convolution.

All pixel weights and polynomial arithmetic remain on their input device.
Grid selection runs once per source. No reference response or per-map accuracy
comparison is computed during production. Exact source evolution is untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .compile import run_tensor_kernel


@dataclass(frozen=True)
class CDFGridChoice:
    """Resolution policy, chosen once per source, not separately per epoch."""

    step_days: float
    narrowest_width_days: float
    maximum_allowed_days: float
    native_bins: int


def choose_cdf_grid(
    widths,
    driver_cadence_days,
    max_delay_days,
    *,
    candidates=(0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    bins_across_central_width=16,
    max_native_bins=65536,
    pixel_delay_days=None,
):
    """Choose a source-wide coarse spacing for shifted CDF reconstruction.

    With a pixel delay scale, select its nearest permitted logarithmic bucket,
    limited by q84-q16/16 and the driver cadence. Four shifted grids then give
    polynomial intervals at one quarter of that spacing. Without a pixel scale,
    retain the coarsest bucket below q84-q16/16 and driver cadence/4.
    This is a resolution heuristic, not a proof of physical convergence.
    Unsupported grids raise a setup error. Ordinary deposition remains an
    explicit alternative, not an automatic per-map fallback.
    """
    widths = torch.as_tensor(widths)
    if widths.numel() == 0 or not bool(
        torch.isfinite(widths).all() & (widths > 0).all()
    ):
        raise ValueError("CDF selection requires positive, resolved response widths")
    if not math.isfinite(driver_cadence_days) or driver_cadence_days <= 0:
        raise ValueError("Driver cadence must be positive and finite")
    if not math.isfinite(max_delay_days) or max_delay_days <= 0:
        raise ValueError("Delay support must be positive and finite")
    if bins_across_central_width <= 0 or max_native_bins < 2:
        raise ValueError("Resolution and memory limits must be positive")
    if not candidates or any(not math.isfinite(s) or s <= 0 for s in candidates):
        raise ValueError("Native grid candidates must be positive and finite")
    width = float(widths.min())
    limit = min(
        width / bins_across_central_width,
        driver_cadence_days / (4 if pixel_delay_days is None else 1),
    )
    permitted = [s for s in candidates if s <= limit * (1 + 1e-6)]
    if not permitted:
        raise ValueError(
            "No CDF bucket resolves this response; use ordinary deposition"
        )
    step = (
        max(permitted)
        if pixel_delay_days is None
        else min(
            permitted,
            key=lambda value: abs(
                math.log(value / max(float(pixel_delay_days), 1e-12))
            ),
        )
    )
    count = math.ceil(max_delay_days / step) + 1
    if count > max_native_bins:
        raise ValueError("CDF grid exceeds its memory budget; use ordinary deposition")
    return CDFGridChoice(step, width, limit, count)


def central_response_widths(mass, bin_width):
    """Conservative widths from nonnegative diagnostic bin masses on-device.

    q84's left edge minus q16's right edge is a lower bound on the true width.
    Zero-response channels are excluded by the caller or rejected, never given
    arbitrary quantiles. Scaling each channel before summing avoids underflow.
    """
    scale = mass.amax(0).clamp_min(torch.finfo(mass.dtype).tiny)
    cumulative = (mass / scale).cumsum(0)
    total = cumulative[-1]
    q = cumulative.new_tensor([0.16, 0.84])[:, None] * total[None]
    indices = torch.searchsorted(cumulative.T.contiguous(), q.T.contiguous()).T
    width = (indices[1] - indices[0] - 1).clamp_min(0) * bin_width
    return torch.where(total > 0, width, torch.zeros_like(width))


def cdf_mean_delays(coefficients, edges):
    """Exact first moment of the reconstructed positive polynomial response."""
    mass, left, right = coefficients.chunk(3)
    h = edges[1] - edges[0]
    moment = (edges[:-1, None] * mass + h * (mass / 2 - left / 12 + right / 12)).sum(0)
    return moment / mass.sum(0).clamp_min(torch.finfo(mass.dtype).tiny)


def project_histograms(sampled, brightness, response, index, bins, bands, quadratic):
    """Project mean flux and nonnegative response histograms for one map."""
    mean = sampled @ brightness
    if quadratic:
        first, second = response[:, :bands], response[:, bands:]
        response = torch.cat((first, second.clamp_min(0), (-second).clamp_min(0)), 1)
    return mean, response_histogram(sampled, response, index, bins)


def response_histogram(sampled, response, index, bins):
    """Accumulate one spatial chunk without a batch-by-pixel temporary."""
    return response.new_zeros((bins, response.shape[1])).index_add(
        0, index, sampled[:, None] * response
    )


def pchip_coefficients(mass, widths=None):
    """Return [mass, h*left_slope, h*right_slope] for native bins.

    These coefficients multiply the derivatives of the same cubic Hermite basis
    used by CDFResponseRebinner. All inputs must be nonnegative interval masses.
    """
    if widths is not None:
        widths = widths[:, None]
        density = mass / widths
        if mass.shape[0] == 1:
            return torch.cat((mass, mass, mass), 0)
        a, b = density[:-1], density[1:]
        minimum = torch.minimum(a, b)
        tiny = torch.finfo(mass.dtype).tiny
        w1, w2 = 2 * widths[1:] + widths[:-1], widths[1:] + 2 * widths[:-1]
        denominator = w1 * minimum / a.clamp_min(tiny) + w2 * minimum / b.clamp_min(
            tiny
        )
        middle = minimum * (w1 + w2) / torch.where(minimum > 0, denominator, 1)
        first = ((2 * widths[0] + widths[1]) * density[0] - widths[0] * density[1]) / (
            widths[0] + widths[1]
        )
        last = (
            (2 * widths[-1] + widths[-2]) * density[-1] - widths[-1] * density[-2]
        ) / (widths[-1] + widths[-2])
        first = torch.minimum(first.clamp_min(0), 3 * density[0])
        last = torch.minimum(last.clamp_min(0), 3 * density[-1])
        slopes = torch.cat((first[None], middle, last[None]), 0)
        return torch.cat((mass, slopes[:-1] * widths, slopes[1:] * widths), 0)
    if mass.shape[0] == 1:
        slopes = torch.cat((mass, mass), 0)
    else:
        a, b = mass[:-1], mass[1:]
        minimum = torch.minimum(a, b)
        tiny = torch.finfo(mass.dtype).tiny
        denominator = minimum / a.clamp_min(tiny) + minimum / b.clamp_min(tiny)
        middle = 2 * minimum / torch.where(minimum > 0, denominator, 1)
        first = torch.minimum(((3 * mass[0] - mass[1]) / 2).clamp_min(0), 3 * mass[0])
        last = torch.minimum(((3 * mass[-1] - mass[-2]) / 2).clamp_min(0), 3 * mass[-1])
        slopes = torch.cat((first[None], middle, last[None]), 0)
    return torch.cat((mass, slopes[:-1], slopes[1:]), 0)


def deposit_response(sampled, response, index, fraction, bins):
    """Linear deposition onto fine bin centers, without a half-bin time shift."""
    weighted = sampled[:, None] * response
    mass = response.new_zeros((bins, response.shape[1]))
    mass = mass.index_add(0, index, weighted * (1 - fraction[:, None]))
    return mass.index_add(0, index + 1, weighted * fraction[:, None])


def project_deposited_response(
    sampled, brightness, response, index, fraction, bins, bands, quadratic, cubic=False
):
    """Project mean flux and fractionally deposited response weights through cubic order."""
    if cubic:
        first, second, third = response.split(bands, dim=1)
        response = torch.cat((first, second.clamp_min(0), (-second).clamp_min(0),
                              third.clamp_min(0), (-third).clamp_min(0)), 1)
    elif quadratic:
        first, second = response[:, :bands], response[:, bands:]
        response = torch.cat((first, second.clamp_min(0), (-second).clamp_min(0)), 1)
    return sampled @ brightness, deposit_response(
        sampled, response, index, fraction, bins
    )


def phase_cdf_coefficients(fine_mass, count):
    """Average shifted PCHIPs as exact polynomials on their common knot grid.

    Local sums avoid subtracting large cumulative totals in float32. Restricting
    each polynomial to the common grid is exact, not another interpolation.
    Thus display and convolution use the identical phase-averaged response.
    """
    # Interleave all four grids in one array of coarse windows. Neighbours in
    # the same phase are four rows apart. Widths in common-piece units suffice
    # because the physical step cancels in h*slope and the Hermite coordinates.
    start = torch.arange(-3, count, device=fine_mass.device)
    lo, hi = start.clamp_min(0), (start + 4).clamp_max(count)
    width = (hi - lo).to(fine_mass.dtype)[:, None]
    offset = torch.arange(16, device=fine_mass.device)
    gather = lo[:, None] * 4 + offset
    mask = offset[None] < width * 4
    mass = (fine_mass[gather.clamp_max(len(fine_mass) - 1)] * mask[..., None]).sum(1)
    density = mass / width
    index = torch.arange(count + 3, device=fine_mass.device)
    previous = (index - 4).clamp_min(0)
    following = (index + 4).clamp_max(count + 2)

    def interior(a, b, wa, wb):
        minimum = torch.minimum(a, b)
        tiny = torch.finfo(fine_mass.dtype).tiny
        w1, w2 = 2 * wb + wa, wb + 2 * wa
        denominator = w1 * minimum / a.clamp_min(tiny) + w2 * minimum / b.clamp_min(
            tiny
        )
        return minimum * (w1 + w2) / torch.where(minimum > 0, denominator, 1)

    left = interior(density[previous], density, width[previous], width)
    right = interior(density, density[following], width, width[following])
    first = ((2 * width + width[following]) * density - width * density[following]) / (
        width + width[following]
    )
    last = ((2 * width + width[previous]) * density - width * density[previous]) / (
        width + width[previous]
    )
    left = (
        torch.where(
            start[:, None] <= 0, torch.minimum(first.clamp_min(0), 3 * density), left
        )
        * width
    )
    right = (
        torch.where(
            (start + 4)[:, None] >= count,
            torch.minimum(last.clamp_min(0), 3 * density),
            right,
        )
        * width
    )
    # A minimal grid has one interval in phase zero, whose PCHIP is linear.
    left = torch.where(width == count, mass, left)
    right = torch.where(width == count, mass, right)
    piece = torch.arange(count, device=fine_mass.device)
    phase = torch.arange(4, device=fine_mass.device)
    selected = piece[None] - ((piece[None] - phase[:, None]) % 4) + 3
    mass, left, right = mass[selected], left[selected], right[selected]
    low = (piece[None, :, None] - lo[selected, None]) / width[selected]
    high = low + 1 / width[selected]
    delta, s1 = high - low, high + low
    s2 = high.square() + high * low + low.square()
    piece_mass = delta * (
        mass * (3 * s1 - 2 * s2) + left * (s2 - 2 * s1 + 1) + right * (s2 - s1)
    )

    def density_at(x):
        return (
            6 * mass * (x - x.square())
            + left * (1 - 4 * x + 3 * x.square())
            + right * (3 * x.square() - 2 * x)
        )

    result = torch.cat(
        (
            piece_mass.clamp_min(0),
            (delta * density_at(low)).clamp_min(0),
            (delta * density_at(high)).clamp_min(0),
        ),
        1,
    )
    return (result[0] + result[1] + result[2] + result[3]) / 4


def _piece_integrals(a, b, low, high, quadratic):
    """Integrate the three response basis functions times a+b*t or its square."""
    # Factored power differences remain accurate for short subintervals.
    delta = high - low
    i0 = delta
    i1 = delta * (high + low) / 2
    i2 = delta * (high.square() + high * low + low.square()) / 3
    i3 = delta * (high + low) * (high.square() + low.square()) / 4
    i4 = (
        delta
        * (
            high**4
            + high**3 * low
            + high.square() * low.square()
            + high * low**3
            + low**4
        )
        / 5
    )
    if quadratic:
        j0 = a.square() * i0 + 2 * a * b * i1 + b.square() * i2
        j1 = a.square() * i1 + 2 * a * b * i2 + b.square() * i3
        j2 = a.square() * i2 + 2 * a * b * i3 + b.square() * i4
    else:
        j0 = a * i0 + b * i1
        j1 = a * i1 + b * i2
        j2 = a * i2 + b * i3
    return torch.cat((6 * (j1 - j2), j0 - 4 * j1 + 3 * j2, 3 * j2 - 2 * j1), 1)


def _cubic_piece_integrals(a, b, low, high):
    """All three driver powers in one pass, sharing interval moments."""
    delta = high - low
    lo2, hi2 = low.square(), high.square()
    lo3, hi3 = lo2 * low, hi2 * high
    i0 = delta
    i1 = delta * (high + low) / 2
    i2 = delta * (hi2 + high * low + lo2) / 3
    i3 = delta * (high + low) * (hi2 + lo2) / 4
    i4 = delta * (hi2.square() + hi3 * low + hi2 * lo2 + high * lo3 + lo2.square()) / 5
    i5 = delta * (hi3 * hi2 + hi2.square() * low + hi3 * lo2
                  + hi2 * lo3 + high * lo2.square() + lo3 * lo2) / 6
    a2, b2 = a.square(), b.square()
    ab = 2 * a * b
    a3, aab, abb, b3 = a2 * a, 3 * a2 * b, 3 * a * b2, b2 * b

    def basis(j0, j1, j2):
        return torch.cat((6 * (j1 - j2), j0 - 4 * j1 + 3 * j2, 3 * j2 - 2 * j1), 1)

    return (
        basis(a * i0 + b * i1, a * i1 + b * i2, a * i2 + b * i3),
        basis(a2 * i0 + ab * i1 + b2 * i2, a2 * i1 + ab * i2 + b2 * i3,
              a2 * i2 + ab * i3 + b2 * i4),
        basis(a3 * i0 + aab * i1 + abb * i2 + b3 * i3,
              a3 * i1 + aab * i2 + abb * i3 + b3 * i4,
              a3 * i2 + aab * i3 + abb * i4 + b3 * i5),
    )


def driver_integrals(times, edges, sample_times, sample_values, baseline, quadratic, cubic=False):
    """Integrate exactly across the (at most one) driver knot in each bin."""
    step = edges[1] - edges[0]
    query = times[:, None] - edges[None, :-1]
    right = torch.searchsorted(sample_times, query.contiguous(), right=False).clamp(
        1, len(sample_times) - 1
    )
    left = right - 1
    slope = (sample_values[right] - sample_values[left]) / (
        sample_times[right] - sample_times[left]
    )
    start = sample_values[left] - baseline + (query - sample_times[left]) * slope
    cut = ((query - sample_times[left]) / step).clamp(0, 1)
    b1 = -step * slope
    lower = (left - 1).clamp_min(0)
    slope2 = (sample_values[left] - sample_values[lower]) / (
        sample_times[left] - sample_times[lower]
    ).clamp_min(torch.finfo(times.dtype).eps)
    b2 = -step * slope2
    a2 = sample_values[left] - baseline - b2 * cut
    zeros, ones = torch.zeros_like(cut), torch.ones_like(cut)
    if cubic:
        before = _cubic_piece_integrals(start, b1, zeros, cut)
        after = _cubic_piece_integrals(a2, b2, cut, ones)
        return tuple(a + b for a, b in zip(before, after, strict=True))
    first = _piece_integrals(start, b1, zeros, cut, False)
    first = first + _piece_integrals(a2, b2, cut, ones, False)
    if not quadratic:
        return first, None
    second = _piece_integrals(start, b1, zeros, cut, True)
    second = second + _piece_integrals(a2, b2, cut, ones, True)
    return first, second


def contract(mean, linear_driver, linear, quadratic_driver=None, quadratic=None):
    """Combine mean flux with integrated first- and optional second-order responses."""
    result = mean[None] + linear_driver @ linear
    if quadratic is not None:
        result = result + 0.5 * quadratic_driver @ quadratic
    return result


def callable_driver_integrals(signal, times, edges, baseline, quadratic, cubic=False):
    """Three-point Gaussian integration for a general scalar callable driver."""
    x = edges.new_tensor([0.1127016653792583, 0.5, 0.8872983346207417])
    w = edges.new_tensor([5 / 18, 4 / 9, 5 / 18])
    delay = edges[:-1, None] + (edges[1] - edges[0]) * x
    query = times[:, None, None] - delay[None]
    delta = (
        signal.amplitudes(
            query.flatten(), bands=1, device=times.device, dtype=times.dtype
        ).reshape(query.shape)
        - baseline
    )
    basis = torch.stack((6 * (x - x * x), 1 - 4 * x + 3 * x * x, 3 * x * x - 2 * x)) * w

    def integral(value):
        return torch.einsum("tkq,bq->tbk", value, basis).reshape(len(times), -1)

    if cubic:
        return integral(delta), integral(delta.square()), integral(delta.pow(3))
    return integral(delta), integral(delta.square()) if quadratic else None


def rebin_coefficients(coefficients, rebinner):
    """Integrate the *same* stored pieces for display, including signed kernels.

    Do not fit PCHIP again after subtracting the quadratic positive/negative
    components: that would produce a different kernel from the convolution.
    """
    mass, left, right = coefficients.chunk(3)
    index = rebinner.native_index
    weight = rebinner.coefficients
    widths = rebinner.widths[index, None]
    contribution = (
        mass[index] * weight[0, :, None]
        + left[index] * (weight[1, :, None] / widths)
        + right[index] * (weight[2, :, None] / widths)
    )
    result = coefficients.new_zeros(
        (len(rebinner.output_edges_days) - 1, coefficients.shape[1])
    )
    return result.index_add(0, rebinner.output_index, contribution)


def signal_cadence(signal, like):
    """Minimum tabulated spacing, or a one-day policy for callable drivers."""
    signal = getattr(signal, "_samples", signal)
    knots = getattr(signal, "times_days", None)
    return (
        float(knots.to(like).diff().min())
        if knots is not None and len(knots) > 1
        else 1.0
    )


def pixel_delay_scale(source):
    """Observer-frame light-crossing time of a source pixel (days)."""
    return (
        max(source.geometry.pixel_scale_m)
        * (1 + float(source.source_redshift))
        / (299792458.0 * 86400.0)
    )


class NativeCDFGrid:
    """Four shifted CDF grids shared by rendering and response convolution.

    Fine fractional deposition precedes reconstruction. The averaged response
    is stored as one set of polynomial pieces, not four separate convolutions.
    """

    def __init__(
        self,
        delay,
        linear_weights,
        *,
        step=None,
        cadence=1.0,
        runtime=None,
        pixel_delay_days=None,
    ):
        self.runtime = runtime
        if delay.numel() == 0 or not bool(torch.isfinite(delay).all()):
            raise ValueError("CDF response requires finite source delays")
        if bool((delay < 0).any()):
            raise ValueError("CDF response requires nonnegative delays")
        support = float(delay.max())
        if step is None:
            diagnostic_step = max(support, 1e-6) / 8190
            index = (delay / diagnostic_step).long().clamp(0, 8191)
            widths = []
            for weight in linear_weights:
                scale = weight.amax(0).clamp_min(torch.finfo(weight.dtype).tiny)
                mass = weight.new_zeros((8192, weight.shape[-1])).index_add(
                    0, index, weight / scale
                )
                measured = central_response_widths(mass, diagnostic_step)
                widths.append(measured)
            widths = torch.cat(widths)
            # A point response has no resolvable shape. Use the finest bucket,
            # rather than inventing a broad response or running a second solver.
            positive = widths[widths > 0]
            step = (
                choose_cdf_grid(
                    positive,
                    cadence,
                    max(support, 1e-6),
                    pixel_delay_days=pixel_delay_days,
                ).step_days
                if len(positive)
                else min(0.005, cadence / 4)
            )
        if not math.isfinite(step) or step <= 0:
            raise ValueError("response_delay_bin_days must be finite and positive")
        # Four shifted coarse CDF grids share knots spaced by step/4. Use
        # another four sub-bins for fractional pixel deposition. Neither grid
        # is tied to the requested transfer-function output spacing.
        self.phase_count = 4
        self.step = float(step)
        self.piece_step = self.step / self.phase_count
        self.deposition_step = self.piece_step / 4
        count = 4 * (math.ceil(support / step) + 1)
        if count > 65536:
            raise ValueError(
                "CDF grid exceeds 65536 bins; increase response_delay_bin_days"
            )
        self.count = count
        self.deposition_count = count * 4
        self.edges = (
            torch.arange(count + 1, device=delay.device, dtype=delay.dtype)
            * self.piece_step
        )
        coordinate = (delay / self.deposition_step - 0.5).clamp_min(0)
        self.index = coordinate.long().clamp(max=self.deposition_count - 2)
        self.fraction = (coordinate - self.index).clamp(0, 1)
        self._deposition_groups = None

    def run(self, name, function, *args):
        """Evaluate a tensor kernel eagerly or through the cached runtime backend."""
        if self.runtime is None:
            return function(*args)
        result, _ = run_tensor_kernel(
            self.runtime,
            name,
            function,
            *args,
            dynamic=not self.runtime.static_response_projection,
        )
        return result

    def project(self, sampled, brightness, response, bands, quadratic, cubic=False):
        """Return mean flux and phase-averaged CDF coefficients for one map."""
        from .config import Backend

        grouped = (
            self.runtime is not None and self.runtime.backend == Backend.TRITON
            and response.device.type == "cuda" and response.dtype == torch.float32
            and (quadratic or cubic) and sampled.is_contiguous()
            and self.fraction.is_contiguous()
            and response.shape[0] >= 4096 and response.stride(1) == 1
            and response.shape[0] <= self.deposition_count * 1024
            and not (torch.is_grad_enabled() and any(
                t.requires_grad for t in (sampled, brightness, response, self.fraction)))
        )
        if grouped:
            from .sources.triton_response import grouped_deposit, prepare_groups

            if self._deposition_groups is None:
                self._deposition_groups = prepare_groups(self.index, self.deposition_count)
            mean = sampled @ brightness
            mass = grouped_deposit(sampled, response, self.fraction,
                self._deposition_groups, self.deposition_count, bands,
                3 if cubic else 2 if quadratic else 1, self.runtime)
        else:
            mean, mass = self._deposit(sampled, brightness, response, bands, quadratic, cubic)
        coeff = self.run(
            "phase-averaged CDF polynomials", phase_cdf_coefficients, mass, self.count
        )
        return mean, coeff

    def _deposit(self, sampled, brightness, response, bands, quadratic, cubic):
        mean, mass = self.run(
            "CDF response projection",
            project_deposited_response,
            sampled,
            brightness,
            response,
            self.index,
            self.fraction,
            self.deposition_count,
            bands,
            quadratic,
            cubic,
        )
        return mean, mass

    def rebin(self, coefficients, output_edges):
        """Integrate response polynomials over the requested delay-bin edges."""
        from .response_rebin import CDFResponseRebinner

        plan = CDFResponseRebinner(self.edges, output_edges, runtime=self.runtime)
        return rebin_coefficients(coefficients, plan)

    def project_response(self, sampled, response, spatial_chunk_size):
        """Project a transfer product with bounded pixel temporaries."""
        mass = None
        for start in range(0, len(sampled), spatial_chunk_size):
            stop = min(start + spatial_chunk_size, len(sampled))
            chunk = self.run(
                "CDF transfer histogram",
                deposit_response,
                sampled[start:stop],
                response[start:stop],
                self.index[start:stop],
                self.fraction[start:stop],
                self.deposition_count,
            )
            mass = chunk if mass is None else mass + chunk
        return self.run(
            "phase-averaged CDF polynomials", phase_cdf_coefficients, mass, self.count
        )
