"""Small physical-unit conversions used by public simulations."""

from __future__ import annotations

import torch

_GRAVITATIONAL_CONSTANT_SI = 6.67430e-11
_SOLAR_MASS_KG = 1.988409870698051e30
_SPEED_OF_LIGHT_SI = 299_792_458.0


def gravitational_radius_m(
    mass_solar: float | torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return :math:`GM/c^2` in metres for a mass in solar masses.

    The tensor-preserving interface avoids hand-written constants in source
    models and remains differentiable when ``mass_solar`` is a tensor.
    """

    resolved_dtype = (
        mass_solar.dtype
        if isinstance(mass_solar, torch.Tensor) and dtype is None
        else (torch.float64 if dtype is None else dtype)
    )
    mass = torch.as_tensor(mass_solar, dtype=resolved_dtype, device=device)
    factor = _GRAVITATIONAL_CONSTANT_SI * _SOLAR_MASS_KG / _SPEED_OF_LIGHT_SI**2
    return mass * factor
