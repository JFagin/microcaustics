"""Real-valued building blocks for the complex Taylor far field.

The point-mass deflection is analytic in ``conjugate(z)`` away from a lens.
These small functions contain only that mathematics. They cover spatial partitioning,
local-star packing, temporal batching, and backend-specific scheduling remain
separate concerns. Keeping real and imaginary components in ordinary floating
tensors gives eager, compiled PyTorch, and Triton the same contract.
"""

from __future__ import annotations

import math

import torch


@torch.no_grad()
def complex_taylor_coefficients(
    expansion_x,
    expansion_y,
    star_x,
    star_y,
    einstein_radius_squared,
    *,
    order: int = 4,
    star_chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate point-mass deflection coefficients at many centers.

    The returned real and imaginary tensors have shape
    ``broadcast(expansion_x, expansion_y).shape + (order + 1,)``. Coefficient
    ``n`` multiplies ``conjugate(z - z0)**n``. The supplied stars must all be
    outside the local-exact neighborhood of every paired expansion center.
    """

    if not 0 <= int(order) <= 16:
        raise ValueError("order must lie in [0, 16]")
    if int(star_chunk_size) < 1:
        raise ValueError("star_chunk_size must be positive")
    center_x = torch.as_tensor(expansion_x)
    center_y = torch.as_tensor(
        expansion_y,
        device=center_x.device,
        dtype=center_x.dtype,
    )
    center_x, center_y = torch.broadcast_tensors(center_x, center_y)
    if not center_x.is_floating_point():
        raise TypeError("expansion centers must use a floating dtype")
    stars_x = torch.as_tensor(star_x, device=center_x.device, dtype=center_x.dtype).reshape(-1)
    stars_y = torch.as_tensor(star_y, device=center_x.device, dtype=center_x.dtype).reshape(-1)
    mass = torch.as_tensor(
        einstein_radius_squared,
        device=center_x.device,
        dtype=center_x.dtype,
    ).reshape(-1)
    if stars_x.shape != stars_y.shape or stars_x.shape != mass.shape:
        raise ValueError("star coordinates and squared Einstein radii must match")
    if bool(torch.any(mass < 0)):
        raise ValueError("squared Einstein radii must be non-negative")
    flat_x = center_x.reshape(-1)
    flat_y = center_y.reshape(-1)
    real = torch.zeros(
        (flat_x.numel(), int(order) + 1),
        device=center_x.device,
        dtype=center_x.dtype,
    )
    imag = torch.zeros_like(real)
    minimum = 1.0e-30 if center_x.dtype == torch.float32 else 1.0e-300
    for start in range(0, stars_x.numel(), int(star_chunk_size)):
        stop = min(stars_x.numel(), start + int(star_chunk_size))
        dx = flat_x[:, None] - stars_x[None, start:stop]
        dy = flat_y[:, None] - stars_y[None, start:stop]
        inverse_r2 = (dx.square() + dy.square()).clamp_min(minimum).reciprocal()
        inverse_real = dx * inverse_r2
        inverse_imag = dy * inverse_r2
        power_real = inverse_real
        power_imag = inverse_imag
        chunk_mass = mass[None, start:stop]
        sign = 1.0
        for coefficient_order in range(int(order) + 1):
            real[:, coefficient_order].add_(
                sign * (chunk_mass * power_real).sum(dim=1)
            )
            imag[:, coefficient_order].add_(
                sign * (chunk_mass * power_imag).sum(dim=1)
            )
            if coefficient_order != int(order):
                next_real = power_real * inverse_real - power_imag * inverse_imag
                next_imag = power_real * inverse_imag + power_imag * inverse_real
                power_real, power_imag = next_real, next_imag
                sign = -sign
    shape = (*center_x.shape, int(order) + 1)
    return real.reshape(shape), imag.reshape(shape)


@torch.no_grad()
def translate_complex_taylor(
    coefficient_real,
    coefficient_imag,
    offset_x,
    offset_y,
    *,
    output_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate center expansions to one or more nearby node centers.

    Coefficients must have shape ``[center, order + 1]``. Offsets may have
    shape ``[node]`` (shared by all centers) or ``[center, node]``. Returned
    tensors have shape ``[center, node, output_order + 1]``.
    """

    real = torch.as_tensor(coefficient_real)
    imag = torch.as_tensor(coefficient_imag, device=real.device, dtype=real.dtype)
    if real.ndim != 2 or imag.shape != real.shape:
        raise ValueError("coefficient arrays must share shape [center, order + 1]")
    high_order = int(real.shape[1]) - 1
    if not 0 <= int(output_order) <= high_order:
        raise ValueError("output_order must lie within the input expansion order")
    shift_real = torch.as_tensor(offset_x, device=real.device, dtype=real.dtype)
    shift_imag = -torch.as_tensor(offset_y, device=real.device, dtype=real.dtype)
    shift_real, shift_imag = torch.broadcast_tensors(shift_real, shift_imag)
    if shift_real.ndim == 1:
        shift_real = shift_real[None].expand(real.shape[0], -1)
        shift_imag = shift_imag[None].expand(real.shape[0], -1)
    if shift_real.ndim != 2 or shift_real.shape[0] != real.shape[0]:
        raise ValueError("offsets must have shape [node] or [center, node]")
    translated_real = []
    translated_imag = []
    for target_order in range(int(output_order) + 1):
        value_real = torch.zeros_like(shift_real)
        value_imag = torch.zeros_like(shift_real)
        power_real = torch.ones_like(shift_real)
        power_imag = torch.zeros_like(shift_real)
        for source_order in range(target_order, high_order + 1):
            combination = float(math.comb(source_order, target_order))
            source_real = real[:, source_order, None]
            source_imag = imag[:, source_order, None]
            value_real.add_(
                combination * (source_real * power_real - source_imag * power_imag)
            )
            value_imag.add_(
                combination * (source_real * power_imag + source_imag * power_real)
            )
            next_real = power_real * shift_real - power_imag * shift_imag
            next_imag = power_real * shift_imag + power_imag * shift_real
            power_real, power_imag = next_real, next_imag
        translated_real.append(value_real)
        translated_imag.append(value_imag)
    return torch.stack(translated_real, dim=-1), torch.stack(translated_imag, dim=-1)


def evaluate_complex_taylor(
    coefficient_real,
    coefficient_imag,
    delta_x,
    delta_y,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate paired complex deflection expansions with Horner's rule."""

    real = torch.as_tensor(coefficient_real)
    imag = torch.as_tensor(coefficient_imag, device=real.device, dtype=real.dtype)
    if real.shape != imag.shape or real.ndim < 1:
        raise ValueError("coefficient arrays must share a non-scalar shape")
    dx = torch.as_tensor(delta_x, device=real.device, dtype=real.dtype)
    dy = -torch.as_tensor(delta_y, device=real.device, dtype=real.dtype)
    dx, dy = torch.broadcast_tensors(dx, dy)
    if tuple(real.shape[:-1]) != tuple(dx.shape):
        raise ValueError("coefficient leading dimensions must match delta coordinates")
    value_real = real[..., -1]
    value_imag = imag[..., -1]
    for index in range(int(real.shape[-1]) - 2, -1, -1):
        product_real = value_real * dx - value_imag * dy
        product_imag = value_real * dy + value_imag * dx
        value_real = product_real + real[..., index]
        value_imag = product_imag + imag[..., index]
    return value_real, value_imag
