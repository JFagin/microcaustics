"""Thermal response approximations through cubic order.

The exact source remains the default. This plan keeps daily driver sampling but
contracts the wavelength-dependent mean image and response with each sparse
magnification map only once.
"""

from __future__ import annotations

import math

import torch

from ..compile import run_tensor_kernel
from ..config import Backend
from .reprocessing import ThermalReprocessingSource, _quadratic_response_weight_kernel
from .variability import TimeShiftedSource


def _project_response_chunk(
    sampled, brightness, response, left_index, right_index, right_fraction, bin_count
):
    """Contract one map with mean brightness and its delay-binned response."""

    mean = sampled @ brightness
    kernel = response.new_zeros((bin_count, response.shape[-1]))
    weighted = sampled[:, None] * response
    kernel.index_add_(0, left_index, weighted * (1.0 - right_fraction[:, None]))
    kernel.index_add_(0, right_index, weighted * right_fraction[:, None])
    return mean, kernel


class OrdinaryResponsePlan:
    """Cache one disk's mean brightness and unnormalized delayed response."""

    def __init__(
        self,
        source,
        source_chunks,
        times,
        *,
        delay_bin_days,
        response_order=1,
        runtime=None,
        _prepare_only=False,
    ):
        arrival_delay = 0.0
        if isinstance(source, TimeShiftedSource):
            arrival_delay = float(source.delay_days)
            source = source.source
        if not isinstance(source, ThermalReprocessingSource):
            raise TypeError("response evolution requires a ThermalReprocessingSource")
        if source.is_time_static:
            raise ValueError("linear_response requires an evolving thermal source")
        delay_bin_days = 0.1 if delay_bin_days is None else delay_bin_days
        if not math.isfinite(delay_bin_days) or delay_bin_days <= 0:
            raise ValueError("response_delay_bin_days must be finite and positive")
        if response_order not in (1, 2, 3):
            raise ValueError("response_order must be 1, 2, or 3")

        device, dtype = times.device, times.dtype
        valid = source.transfer.hit.to(device) & torch.isfinite(
            source.delay_days.to(device)
        )
        if not bool(valid.any()):
            raise ValueError("linear response requires finite delays on the disk")
        delay = source.delay_days.to(device=device, dtype=dtype).flatten()
        selected = torch.nonzero(valid.flatten()).flatten()
        valid_delay = delay[selected]
        minimum = float(valid_delay.min())
        maximum = float(valid_delay.max())
        count = math.ceil((maximum - minimum) / delay_bin_days) + 1
        centers = (
            minimum + torch.arange(count, device=device, dtype=dtype) * delay_bin_days
        ).clamp(max=maximum)
        # Linear deposition conserves both response mass and its first delay
        # moment. Nearest-bin histograms can spuriously shift continuum lags.
        left_index = (torch.searchsorted(centers, valid_delay, right=True) - 1).clamp(
            0, count - 1
        )
        right_index = (left_index + 1).clamp(max=count - 1)
        span = centers[right_index] - centers[left_index]
        right_fraction = torch.where(
            span > 0,
            (valid_delay - centers[left_index]) / span.clamp_min(1.0e-30),
            torch.zeros_like(valid_delay),
        ).clamp(0.0, 1.0)
        mean_value = torch.as_tensor(
            source.signal.metadata().get("mean_amplitude", 1.0), dtype=dtype
        )
        if mean_value.numel() != 1:
            raise ValueError("thermal response requires a scalar driver mean amplitude")
        mean_amplitude = float(mean_value.reshape(()))
        driving = centers.new_empty((0, count))
        if not _prepare_only:
            driving = torch.cat(
                [
                    source.signal.amplitudes(
                        (batch[:, None] - arrival_delay - centers[None]).reshape(-1),
                        bands=1,
                        device=device,
                        dtype=dtype,
                    ).reshape(len(batch), count)
                    for batch in times.split(64)
                ]
            )

        self.positions = selected
        self.left_index = left_index
        self.right_index = right_index
        self.right_fraction = right_fraction
        self.delay_centers_days = centers
        self.bin_count = count
        self.delay_bin_days = float(delay_bin_days)
        self.driver_delta = driving - mean_amplitude
        self.driver_delta_squared = (
            self.driver_delta.square() if response_order >= 2 else None
        )
        if response_order == 3 and not _prepare_only:
            self.driver_powers = torch.cat(
                (self.driver_delta, self.driver_delta_squared, self.driver_delta.pow(3)), 1
            )
        self.response_order = response_order
        self.runtime = runtime
        self.shape = source.geometry.shape
        self.parts = []
        total_bands = sum(valid_bands for _, valid_bands in source_chunks)
        fused_bytes = (
            selected.numel() * total_bands * times.element_size() * (1 + response_order)
        )
        budget = 0
        if runtime is not None and device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            budget = min(
                512 * 2**20,
                (runtime.available_memory_bytes or 512 * 2**20) // 16,
                free_bytes // 8,
            )
        fuse_chunks = (
            device.type == "cuda"
            and runtime is not None
            and runtime.backend in {Backend.TRITON, Backend.TORCH_COMPILE}
            and fused_bytes <= budget
        )
        if fuse_chunks:
            fused_brightness = torch.empty(
                (selected.numel(), total_bands), device=device, dtype=dtype
            )
            fused_response = torch.empty(
                (selected.numel(), total_bands * response_order),
                device=device,
                dtype=dtype,
            )
        column = 0
        # Heating and observer transfer do not depend on wavelength. Prepare
        # them once, without constructing a new constant-driver source for
        # every band chunk. The nonlinear exact-evolution path is unchanged.
        evaluation = source._evaluation_tensors(device=device, dtype=dtype)
        transfer, _, static_temperature4, response_temperature4, _ = evaluation
        mean_state = (
            (static_temperature4 + mean_amplitude * response_temperature4)[None],
            transfer,
            1,
        )
        redshift = torch.as_tensor(source.source_redshift, device=device, dtype=dtype)
        color = torch.as_tensor(source.color_correction, device=device, dtype=dtype)
        for chunk, valid_bands in source_chunks:
            if isinstance(chunk, TimeShiftedSource):
                chunk = chunk.source
            if response_order < 3:
                brightness = chunk._brightness_from_state(mean_state)[
                    0, ..., :valid_bands
                ].reshape(-1, valid_bands)[selected]
            wavelength = (
                torch.as_tensor(
                    chunk.geometry.wavelengths_angstrom, device=device, dtype=dtype
                )
                * 1.0e-10
                / (1.0 + redshift)
            )
            arguments = (
                static_temperature4,
                response_temperature4,
                transfer.gfactor,
                transfer.solid_angle_sr,
                transfer.hit,
                wavelength,
                redshift,
                color,
                chunk.geometry.pixel_scale_m[0] * chunk.geometry.pixel_scale_m[1],
                mean_amplitude,
                response_order >= 2,
            )
            if response_order == 3:
                arguments += (True,)
            if runtime is None:
                weights = _quadratic_response_weight_kernel(
                    *arguments
                )
            else:
                weights, _ = run_tensor_kernel(
                    runtime,
                    "thermal response weights",
                    _quadratic_response_weight_kernel,
                    *arguments,
                )
            if response_order == 3:
                brightness = weights[0][..., :valid_bands].reshape(-1, valid_bands)[selected]
                weights = weights[1:]
            response = torch.cat(tuple(
                w[..., :valid_bands].reshape(-1, valid_bands)[selected]
                for w in weights[:response_order]
            ), dim=1)
            if fuse_chunks:
                fused_brightness[:, column : column + valid_bands] = brightness
                for order in range(response_order):
                    fused_response[
                        :, order * total_bands + column : order * total_bands + column + valid_bands
                    ] = response[:, order * valid_bands : (order + 1) * valid_bands]
            else:
                self.parts.append((brightness, response, valid_bands))
            column += valid_bands
        if fuse_chunks:
            self.parts = [(fused_brightness, fused_response, total_bands)]

        if not _prepare_only:
            self.unlensed_projection = self._project_valid(
                torch.ones(selected.numel(), device=device, dtype=dtype)
            )

    def metadata(self):
        """Describe the ordinary deposition grid used for this response plan."""
        return {
            "response_reconstruction": "ordinary",
            "response_delay_bin_days": self.delay_bin_days,
        }

    def _project_valid(self, sampled):
        means, linear_kernels, quadratic_kernels, cubic_kernels = [], [], [], []
        for brightness, response, bands in self.parts:
            arguments = (
                sampled,
                brightness,
                response,
                self.left_index,
                self.right_index,
                self.right_fraction,
                self.bin_count,
            )
            if self.runtime is None:
                mean, kernel = _project_response_chunk(*arguments)
            else:
                # Reuse one compiled projection across map epochs; eager
                # per-map scatters otherwise dominate daily reconstruction.
                (mean, kernel), _ = run_tensor_kernel(
                    self.runtime,
                    "microlensed delay-response projection",
                    _project_response_chunk,
                    *arguments,
                    dynamic=not self.runtime.static_response_projection,
                )
            means.append(mean)
            linear_kernels.append(kernel[:, :bands])
            if self.response_order >= 2:
                quadratic_kernels.append(kernel[:, bands : 2 * bands])
            if self.response_order == 3:
                cubic_kernels.append(kernel[:, 2 * bands :])
        if self.response_order == 3:
            return torch.cat(means), torch.cat((
                torch.cat(linear_kernels, 1), torch.cat(quadratic_kernels, 1) / 2,
                torch.cat(cubic_kernels, 1) / 6,
            ), 0)
        return (
            torch.cat(means),
            torch.cat(linear_kernels, dim=1),
            torch.cat(quadratic_kernels, dim=1) if quadratic_kernels else None,
        )

    def project_map(self, sampled):
        """Return mean flux and delay kernel for one source-aligned map."""

        if sampled.shape[-2:] != self.shape:
            raise ValueError("sampled map must match the source shape")
        values = sampled.reshape(-1)[self.positions]
        return self._project_valid(values)

    def flux(self, projection, start, stop):
        """Evaluate the projected response over the selected driver epochs."""

        if self.response_order == 3:
            # Orders and factorials were packed once during map projection.
            # One GEMM (with the mean as its bias) replaces three separate GEMMs.
            return torch.addmm(projection[0][None], self.driver_powers[start:stop], projection[1])
        mean, linear, quadratic = projection
        result = mean[None] + self.driver_delta[start:stop] @ linear
        if quadratic is not None:
            result = result + 0.5 * self.driver_delta_squared[start:stop] @ quadratic
        return result

    def unlensed_flux(self, start, stop):
        """Evaluate the same response with the unlensed source projection."""
        return self.flux(self.unlensed_projection, start, stop)


class LinearResponsePlan(OrdinaryResponsePlan):
    """CDF/PCHIP thermal response, sharing export and convolution.

    Tabulated drivers are integrated analytically. Callable drivers use
    three-point Gauss quadrature on each native interval. No second response is
    evaluated for validation during production.
    """

    def __init__(
        self,
        source,
        source_chunks,
        times,
        *,
        delay_bin_days=None,
        response_order=1,
        runtime=None,
    ):
        from .._response_cdf import NativeCDFGrid, pixel_delay_scale, signal_cadence

        self.cdf = getattr(runtime, "response_reconstruction", "cdf") == "cdf"
        super().__init__(
            source,
            source_chunks,
            times,
            delay_bin_days=delay_bin_days,
            response_order=response_order,
            runtime=runtime,
            _prepare_only=self.cdf,
        )
        if not self.cdf:
            return
        arrival_delay = 0.0
        if isinstance(source, TimeShiftedSource):
            arrival_delay = float(source.delay_days)
            source = source.source
        delay = source.delay_days.to(times).flatten()[self.positions]
        self.grid = NativeCDFGrid(
            delay,
            [r[:, :b] for _, r, b in self.parts],
            step=delay_bin_days,
            cadence=signal_cadence(source.signal, times),
            runtime=runtime,
            pixel_delay_days=pixel_delay_scale(source),
        )
        self.delay_bin_days = self.grid.step
        self.edges = self.grid.edges
        # Scale before float32 accumulation to avoid subnormal physical weights.
        self.scales = [
            r.abs().sum(0).clamp_min(torch.finfo(r.dtype).tiny)
            for _, r, _ in self.parts
        ]
        self.parts = [
            (b, r / s, n) for (b, r, n), s in zip(self.parts, self.scales, strict=True)
        ]
        baseline = float(source.signal.metadata().get("mean_amplitude", 1.0))
        self._prepare_cdf_driver(source.signal, times - arrival_delay, baseline)
        self.unlensed_projection = self._project_valid(torch.ones_like(delay))

    def _prepare_cdf_driver(self, signal, times, baseline):
        from .._response_cdf import callable_driver_integrals, driver_integrals
        from .variability import TabulatedDrivingSignal

        table = getattr(signal, "_samples", signal)
        quadratic = self.response_order >= 2
        cubic = self.response_order == 3
        columns = 3 * self.grid.count
        needed = (
            times.numel() * columns * times.element_size() * self.response_order
        )
        budget = 512 * 2**20
        if times.device.type == "cuda":
            budget = min(budget, torch.cuda.mem_get_info(times.device)[0] // 4)
        if needed > budget:
            raise ValueError(
                "CDF driver arrays exceed the memory budget; increase response_delay_bin_days or shorten the requested duration"
            )
        analytic = (
            isinstance(table, TabulatedDrivingSignal) and len(table.times_days) > 1
        )
        if analytic:
            knots, values = table._table_for(times)
            if values.shape[1] != 1:
                raise ValueError("CDF response requires a scalar driving signal")
            if self.grid.piece_step > float(knots.diff().min()) * (1 + 1e-6):
                raise ValueError(
                    "response_delay_bin_days / 4 must not exceed the minimum driver cadence"
                )
            # Extend held endpoints to the entire polynomial support. Error-mode
            # drivers must cover that support, rather than silently extrapolate.
            low, high = float(times.min() - self.edges[-1]), float(times.max())
            if table.extrapolation == "error" and (
                low < float(knots[0]) or high > float(knots[-1])
            ):
                raise ValueError(
                    "driver history does not cover the complete CDF delay support"
                )
            if table.extrapolation == "hold":
                knots = torch.cat(
                    (
                        knots.new_tensor([min(low, float(knots[0])) - 1]),
                        knots,
                        knots.new_tensor([max(high, float(knots[-1])) + 1]),
                    )
                )
                values = torch.cat((values[:1], values, values[-1:]))
        if cubic:
            self.driver_powers = times.new_empty((len(times), columns * 3))
            first, second, third = self.driver_powers.split(columns, dim=1)
        else:
            first = times.new_empty((len(times), columns))
            second = torch.empty_like(first) if quadratic else None
        for start in range(0, len(times), 64):
            batch = times[start : start + 64]
            if analytic:
                # Fixed shape, including the final partial chunk.
                padded = torch.cat((batch, batch[-1:].expand(64 - len(batch))))
                integrated = self.grid.run(
                    "CDF driver integration",
                    driver_integrals,
                    padded,
                    self.edges,
                    knots,
                    values[:, 0],
                    baseline,
                    quadratic,
                    cubic,
                )
            else:
                integrated = callable_driver_integrals(
                    signal, batch, self.edges, baseline, quadratic, cubic
                )
            first[start : start + len(batch)] = integrated[0][:len(batch)]
            if quadratic:
                second[start : start + len(batch)] = integrated[1][:len(batch)]
            if cubic:
                third[start : start + len(batch)] = integrated[2][:len(batch)]
        self.driver_delta, self.driver_delta_squared = first, second

    def _project_valid(self, sampled):
        if not self.cdf:
            return super()._project_valid(sampled)
        means, linear, quadratic, cubic = [], [], [], []
        for (brightness, response, bands), scale in zip(
            self.parts, self.scales, strict=True
        ):
            mean, coeff = self.grid.project(
                sampled, brightness, response, bands, self.response_order >= 2,
                self.response_order == 3,
            )
            means.append(mean)
            linear.append(coeff[:, :bands] * scale[:bands])
            if self.response_order >= 2:
                quadratic.append(
                    (coeff[:, bands : 2 * bands] - coeff[:, 2 * bands : 3 * bands])
                    * scale[bands : 2 * bands]
                )
            if self.response_order == 3:
                cubic.append(
                    (coeff[:, 3 * bands : 4 * bands] - coeff[:, 4 * bands :])
                    * scale[2 * bands :]
                )
        if self.response_order == 3:
            return torch.cat(means), torch.cat((
                torch.cat(linear, 1), torch.cat(quadratic, 1) / 2,
                torch.cat(cubic, 1) / 6,
            ), 0)
        return (
            torch.cat(means),
            torch.cat(linear, 1),
            torch.cat(quadratic, 1) if quadratic else None,
        )

    def response_bin_masses(self, projection, output_edges, *, quadratic=False, order=None):
        """Integrate the exact same polynomials used by the light curve."""
        if not self.cdf:
            raise ValueError("response_bin_masses requires CDF reconstruction")
        order = (2 if quadratic else 1) if order is None else order
        if order not in (1, 2, 3) or order > self.response_order:
            raise ValueError("requested response order was not computed")
        if self.response_order == 3:
            coeff = projection[1].chunk(3, dim=0)[order - 1] * math.factorial(order)
        else:
            coeff = projection[order]
        if coeff is None:
            raise ValueError("no quadratic response was computed")
        return self.grid.rebin(coeff, output_edges)

    def metadata(self):
        """Describe the selected response reconstruction and its delay grid."""
        return {
            "response_order": self.response_order,
            "response_reconstruction": "cdf_pchip" if self.cdf else "ordinary",
            "response_delay_bin_days": self.delay_bin_days,
            "response_cdf_phases": self.grid.phase_count if self.cdf else None,
        }
