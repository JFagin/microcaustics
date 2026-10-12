"""Conservative, device-resident CDF/PCHIP response reconstruction.

The integral on every native interval is fixed by its response mass. We
evaluate *local* Hermite differences instead of subtracting two global CDF
values near one. This is algebraically the same PCHIP reconstruction, but
retains weak tails in float32. No spatial interpolation is performed.
"""

from __future__ import annotations

import torch

from .compile import run_tensor_kernel
from .config import Backend, RuntimeConfig
from .runtime import resolve_runtime


def _pchip_rebin(mass, widths, native_index, output_index, coefficients, output_bins):
    secant = mass / widths[:, None]
    if widths.numel() == 1:
        slopes = torch.cat((secant, secant), dim=-2)
    else:
        left, right = secant[..., :-1, :], secant[..., 1:, :]
        w1 = (2 * widths[1:] + widths[:-1])[:, None]
        w2 = (widths[1:] + 2 * widths[:-1])[:, None]
        minimum = torch.minimum(left, right)
        tiny = torch.finfo(mass.dtype).tiny
        denominator = w1 * (minimum / left.clamp_min(tiny)) + w2 * (
            minimum / right.clamp_min(tiny)
        )
        denominator = torch.where(
            minimum > 0, denominator, torch.ones_like(denominator)
        )
        interior = minimum * ((w1 + w2) / denominator)
        first = (
            (2 * widths[0] + widths[1]) * secant[..., 0, :]
            - widths[0] * secant[..., 1, :]
        ) / (widths[0] + widths[1])
        last = (
            (2 * widths[-1] + widths[-2]) * secant[..., -1, :]
            - widths[-1] * secant[..., -2, :]
        ) / (widths[-1] + widths[-2])
        first = torch.minimum(first.clamp_min(0), 3 * secant[..., 0, :])
        last = torch.minimum(last.clamp_min(0), 3 * secant[..., -1, :])
        slopes = torch.cat((first[..., None, :], interior, last[..., None, :]), dim=-2)
    contribution = (
        mass[..., native_index, :] * coefficients[0, :, None]
        + slopes[..., native_index, :] * coefficients[1, :, None]
        + slopes[..., native_index + 1, :] * coefficients[2, :, None]
    ).clamp_min(0)  # Only roundoff can be negative for a monotone CDF.
    result = mass.new_zeros((*mass.shape[:-2], output_bins, mass.shape[-1]))
    return result.index_add(-2, output_index, contribution)


class CDFResponseRebinner:
    """Cache a conservative CDF/PCHIP mapping between two delay grids.

    Arrays have shape ``[..., native_delay_bin, band]`` and store bin *mass*,
    not density. Results retain device, dtype, and absolute response amplitude.
    Output outside the native support is zero. A restricted output range loses
    the excluded mass, without renormalization or extrapolation.

    Grid validation/sorting happens once. Repeated calls use only tensor
    operations on the input device. CUDA defaults to cached ``torch.compile``;
    pass a resolved runtime to select an eager or strict compiled backend.
    ``validate=False`` skips the value check for already validated internal
    responses and avoids a host synchronization on CUDA. Shape/device/dtype
    checks are always performed.

    This reconstruction preserves interval integrals, not exact pixelwise
    mean delays or unresolved sub-bin structure. Retain direct response moments
    for physical lags and test convergence of the native grid separately.
    """

    def __init__(self, native_edges_days, output_edges_days, *, runtime=None):
        edges = torch.as_tensor(native_edges_days)
        if not edges.is_floating_point():
            edges = edges.to(torch.get_default_dtype())
        if edges.dtype not in (torch.float32, torch.float64):
            raise ValueError("CDF reconstruction requires float32 or float64")
        output = torch.as_tensor(
            output_edges_days, device=edges.device, dtype=edges.dtype
        )
        for name, grid in (("native", edges), ("output", output)):
            if (
                grid.ndim != 1
                or grid.numel() < 2
                or not bool(torch.isfinite(grid).all() & (grid[1:] > grid[:-1]).all())
            ):
                raise ValueError(
                    f"{name} delay edges must be finite and strictly increasing"
                )
        self.native_edges_days = edges.detach().clone()
        self.output_edges_days = output.detach().clone()
        edges, output = self.native_edges_days, self.output_edges_days
        self.widths = edges.diff()
        # Split at every native or requested edge. Each subinterval belongs to
        # exactly one native polynomial and one output bin, including coarsening.
        knots = torch.cat((edges, output.clamp(edges[0], edges[-1]))).sort().values
        lo, hi = knots[:-1], knots[1:]
        keep = (hi > lo) & (lo >= output[0]) & (hi <= output[-1])
        lo, hi = lo[keep], hi[keep]
        self.native_index = (torch.searchsorted(edges, lo, right=True) - 1).clamp(
            0, edges.numel() - 2
        )
        self.output_index = (torch.searchsorted(output, lo, right=True) - 1).clamp(
            0, output.numel() - 2
        )
        width = self.widths[self.native_index]
        t0 = (lo - edges[self.native_index]) / width
        t1 = (hi - edges[self.native_index]) / width
        # Factored F(t1)-F(t0) avoids cancellation for narrowly separated edges.
        dt, s1 = t1 - t0, t0 + t1
        s2 = t0.square() + t0 * t1 + t1.square()
        self.coefficients = torch.stack(
            (
                dt * (3 * s1 - 2 * s2),
                dt * width * (s2 - 2 * s1 + 1),
                dt * width * (s2 - s1),
            )
        )
        if runtime is None:
            runtime = resolve_runtime(
                RuntimeConfig(
                    device=str(edges.device),
                    dtype=str(edges.dtype).removeprefix("torch."),
                    backend=Backend.TORCH_COMPILE
                    if edges.is_cuda
                    else Backend.TORCH_EAGER,
                )
            )
        runtime_device = runtime.device
        if runtime_device.type == "cuda" and runtime_device.index is None:
            runtime_device = torch.device("cuda", torch.cuda.current_device())
        if runtime_device != edges.device or runtime.dtype != edges.dtype:
            raise ValueError("rebin runtime must match the delay-grid device and dtype")
        self.runtime = runtime

    def __call__(self, mass: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
        """Reconstruct nonnegative bin masses, optionally validating values."""
        if (
            mass.ndim < 2
            or mass.shape[-2] != self.widths.numel()
            or mass.device != self.widths.device
            or mass.dtype != self.widths.dtype
        ):
            raise ValueError("response must match native bin count, device, and dtype")
        if validate and not bool((torch.isfinite(mass) & (mass >= 0)).all()):
            raise ValueError("CDF response masses must be finite and nonnegative")
        result, _ = run_tensor_kernel(
            self.runtime,
            "CDF/PCHIP response reconstruction",
            _pchip_rebin,
            mass,
            self.widths,
            self.native_index,
            self.output_index,
            self.coefficients,
            self.output_edges_days.numel() - 1,
            dynamic=False,
        )
        return result

    def experimental_signed(
        self, mass: torch.Tensor, *, validate: bool = True
    ) -> torch.Tensor:
        """Experimentally reconstruct a signed kernel without normalizing it.

        Positive and negative native-bin masses are reconstructed separately.
        This cannot recover cancellation already lost inside a native bin.
        It is deliberately NOT used by linear/quadratic light-curve evolution.
        """
        if validate and not bool(torch.isfinite(mass).all()):
            raise ValueError("signed response masses must be finite")
        pair = torch.stack((mass.clamp_min(0), (-mass).clamp_min(0)))
        result = self(pair, validate=False)
        return result[0] - result[1]
