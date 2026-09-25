"""Experimental delay-binned linear response of a thermal disk.

The exact source remains the default. This plan keeps daily driver sampling but
contracts the wavelength-dependent mean image and response with each sparse
magnification map only once.
"""

from __future__ import annotations

import math

import torch

from ..compile import run_tensor_kernel
from ..config import Backend
from .reprocessing import ThermalReprocessingSource
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


class LinearResponsePlan:
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
    ):
        arrival_delay = 0.0
        if isinstance(source, TimeShiftedSource):
            arrival_delay = float(source.delay_days)
            source = source.source
        if not isinstance(source, ThermalReprocessingSource):
            raise TypeError("response evolution requires a ThermalReprocessingSource")
        if source.is_time_static:
            raise ValueError("linear_response requires an evolving thermal source")
        if not math.isfinite(delay_bin_days) or delay_bin_days <= 0:
            raise ValueError("response_delay_bin_days must be finite and positive")
        if response_order not in (1, 2):
            raise ValueError("response_order must be 1 or 2")

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
            source.signal.metadata().get("mean_amplitude", 1.0)
        )
        if mean_value.numel() != 1:
            raise ValueError("thermal response requires a scalar driver mean amplitude")
        mean_amplitude = float(mean_value.reshape(()))
        query = (times[:, None] - arrival_delay - centers[None]).reshape(-1)
        driving = source.signal.amplitudes(
            query, bands=1, device=device, dtype=dtype
        ).reshape(times.numel(), count)

        self.positions = selected
        self.left_index = left_index
        self.right_index = right_index
        self.right_fraction = right_fraction
        self.delay_centers_days = centers
        self.bin_count = count
        self.delay_bin_days = float(delay_bin_days)
        self.driver_delta = driving - mean_amplitude
        self.driver_delta_squared = (
            self.driver_delta.square() if response_order == 2 else None
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
        for chunk, valid_bands in source_chunks:
            if isinstance(chunk, TimeShiftedSource):
                chunk = chunk.source
            brightness = (
                chunk.at_driver_mean()
                .brightness(times[:1], device=device, dtype=dtype)[0, ..., :valid_bands]
                .reshape(-1, valid_bands)[selected]
            )
            if response_order == 2:
                linear_weights, quadratic_weights = (
                    chunk._quadratic_response_weight_pair(
                        driver_amplitude=mean_amplitude,
                        device=device,
                        dtype=dtype,
                        runtime=runtime,
                    )
                )
                response = linear_weights[..., :valid_bands].reshape(-1, valid_bands)[
                    selected
                ]
                quadratic = quadratic_weights[..., :valid_bands].reshape(
                    -1, valid_bands
                )[selected]
                response = torch.cat((response, quadratic), dim=1)
            else:
                response = chunk.linear_response_weights(
                    driver_amplitude=mean_amplitude, device=device, dtype=dtype
                )[..., :valid_bands].reshape(-1, valid_bands)[selected]
            if fuse_chunks:
                fused_brightness[:, column : column + valid_bands] = brightness
                fused_response[:, column : column + valid_bands] = response[
                    :, :valid_bands
                ]
                if response_order == 2:
                    fused_response[
                        :, total_bands + column : total_bands + column + valid_bands
                    ] = response[:, valid_bands:]
            else:
                self.parts.append((brightness, response, valid_bands))
            column += valid_bands
        if fuse_chunks:
            self.parts = [(fused_brightness, fused_response, total_bands)]

        self.unlensed_projection = self._project_valid(
            torch.ones(selected.numel(), device=device, dtype=dtype)
        )

    def _project_valid(self, sampled):
        means, linear_kernels, quadratic_kernels = [], [], []
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
            if self.response_order == 2:
                quadratic_kernels.append(kernel[:, bands:])
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

        mean, linear, quadratic = projection
        result = mean[None] + self.driver_delta[start:stop] @ linear
        if quadratic is not None:
            result = result + 0.5 * self.driver_delta_squared[start:stop] @ quadratic
        return result

    def unlensed_flux(self, start, stop):
        """Evaluate the same response with the unlensed source projection."""

        return self.flux(self.unlensed_projection, start, stop)
