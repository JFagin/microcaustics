"""Lightweight analytic surface-brightness profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .base import SourceGeometry, _as_times


def _band_values(
    value: float | Sequence[float],
    bands: int,
    *,
    name: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast a scalar or validate one value per source band."""

    tensor = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if tensor.numel() == 1:
        tensor = tensor.expand(bands)
    if tensor.numel() != bands:
        raise ValueError(f"{name} must be scalar or contain one value per band")
    return tensor


@dataclass(frozen=True)
class GaussianSource:
    """An elliptical multiband Gaussian with an optional smooth central hole.

    The major-axis Gaussian width may differ by band. ``axis_ratio`` is the
    minor-to-major width ratio and ``position_angle_rad`` is counterclockwise
    from the positive x-axis. If ``hole_radius_m`` is positive, the profile is
    multiplied by ``1 - exp(-(r / hole_radius_m)**hole_power)``. Each band is
    discretely normalized to its requested total observed flux density in Jy.
    """

    geometry: SourceGeometry
    sigma_m: float | tuple[float, ...]
    total_flux: float | tuple[float, ...] = 1.0
    axis_ratio: float = 1.0
    position_angle_rad: float = 0.0
    center_m: tuple[float, float] = (0.0, 0.0)
    hole_radius_m: float = 0.0
    hole_power: float = 4.0
    name: str = "gaussian"
    is_time_static: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.axis_ratio <= 1.0:
            raise ValueError("axis_ratio must lie in (0, 1]")
        if self.hole_radius_m < 0:
            raise ValueError("hole_radius_m must be non-negative")
        if self.hole_power <= 0:
            raise ValueError("hole_power must be positive")
        if len(self.center_m) != 2:
            raise ValueError("center_m must contain x and y")

    def _frame(self, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        bands = len(self.geometry.band_names)
        sigma = _band_values(
            self.sigma_m,
            bands,
            name="sigma_m",
            device=device,
            dtype=dtype,
        )
        if bool(torch.any(sigma <= 0)):
            raise ValueError("sigma_m must be positive")
        requested_flux = _band_values(
            self.total_flux,
            bands,
            name="total_flux",
            device=device,
            dtype=dtype,
        )
        if bool(torch.any(requested_flux < 0)):
            raise ValueError("total_flux must be non-negative")
        ny, nx = self.geometry.shape
        dy, dx = self.geometry.pixel_scale_m
        x = (torch.arange(nx, device=device, dtype=dtype) + 0.5 - 0.5 * nx) * dx
        y = (torch.arange(ny, device=device, dtype=dtype) + 0.5 - 0.5 * ny) * dy
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xx = xx - float(self.center_m[0])
        yy = yy - float(self.center_m[1])
        angle = torch.as_tensor(self.position_angle_rad, device=device, dtype=dtype)
        cosine, sine = torch.cos(angle), torch.sin(angle)
        major = cosine * xx + sine * yy
        minor = -sine * xx + cosine * yy
        elliptical_radius2 = major.square() + (minor / self.axis_ratio).square()
        profile = torch.exp(-0.5 * elliptical_radius2[..., None] / sigma.square())
        if self.hole_radius_m > 0:
            radius = torch.sqrt(xx.square() + yy.square())
            hole = 1.0 - torch.exp(
                -(radius / self.hole_radius_m).pow(self.hole_power)
            )
            profile = profile * hole[..., None]
        pixel_area = float(dx * dy)
        normalization = profile.sum(dim=(0, 1)) * pixel_area
        if bool(torch.any(normalization <= 0)):
            raise ValueError("profile has no positive flux on the requested grid")
        return profile * (requested_flux / normalization)

    def brightness(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the normalized profile for every requested time."""

        times = _as_times(times_days)
        device = times.device if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        frame = self._frame(device=device, dtype=dtype)
        return frame.unsqueeze(0).expand(int(times.numel()), -1, -1, -1)

    def metadata(self) -> Mapping[str, object]:
        """Return serializable analytic profile parameters."""

        return {
            "type": "gaussian",
            "name": self.name,
            "sigma_m": self.sigma_m,
            "axis_ratio": self.axis_ratio,
            "position_angle_rad": self.position_angle_rad,
            "center_m": self.center_m,
            "hole_radius_m": self.hole_radius_m,
            "hole_power": self.hole_power,
            "brightness_units": "Jy m^-2 projected source plane",
            "integrated_flux_units": "Jy",
            "is_time_static": True,
        }
