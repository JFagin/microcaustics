"""Backend-independent observer-transfer products."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ObserverScreen:
    """Coordinates and solid-angle weights for an observer image.

    ``x_rg`` and ``y_rg`` are impact parameters in gravitational radii and
    use package-standard ``[y, x]`` array order. ``solid_angle_sr`` contains
    the observer solid angle represented by each pixel. Keeping this geometry
    separate ensures analytic Kerr, external, and validation tracers use the
    same flux normalization.
    """

    x_rg: torch.Tensor
    y_rg: torch.Tensor
    solid_angle_sr: torch.Tensor
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = torch.as_tensor(self.x_rg)
        if x.ndim != 2 or not x.is_floating_point():
            raise ValueError("x_rg must be one floating [y, x] array")
        y = torch.as_tensor(self.y_rg, device=x.device, dtype=x.dtype)
        solid_angle = torch.as_tensor(
            self.solid_angle_sr,
            device=x.device,
            dtype=x.dtype,
        )
        if y.shape != x.shape or solid_angle.shape != x.shape:
            raise ValueError("observer-screen arrays must share one [y, x] shape")
        if bool(torch.any(~torch.isfinite(x))) or bool(torch.any(~torch.isfinite(y))):
            raise ValueError("observer-screen coordinates must be finite")
        if bool(torch.any(~torch.isfinite(solid_angle))) or bool(
            torch.any(solid_angle <= 0)
        ):
            raise ValueError("solid_angle_sr must be finite and positive")
        object.__setattr__(self, "x_rg", x)
        object.__setattr__(self, "y_rg", y)
        object.__setattr__(self, "solid_angle_sr", solid_angle)

    @classmethod
    def uniform(
        cls,
        shape: tuple[int, int],
        half_size_rg: float | tuple[float, float],
        *,
        gravitational_radius_m: float,
        observer_distance_m: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        metadata: Mapping[str, object] | None = None,
    ) -> ObserverScreen:
        """Construct a pixel-centered rectangular screen.

        ``half_size_rg`` is either one common half-width or ``(y, x)``. The
        small-angle solid angle is calculated from the physical gravitational
        radius and observer angular-diameter distance, both in meters.
        """

        if len(shape) != 2 or any(int(value) < 1 for value in shape):
            raise ValueError("shape must contain two positive integers")
        if isinstance(half_size_rg, tuple):
            if len(half_size_rg) != 2:
                raise ValueError("half_size_rg must be scalar or (y, x)")
            half_y, half_x = (float(value) for value in half_size_rg)
        else:
            half_y = half_x = float(half_size_rg)
        gravitational_radius = float(gravitational_radius_m)
        observer_distance = float(observer_distance_m)
        values = (half_y, half_x, gravitational_radius, observer_distance)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("screen sizes, radius, and distance must be positive")

        ny, nx = (int(value) for value in shape)
        dy_rg = 2.0 * half_y / ny
        dx_rg = 2.0 * half_x / nx
        y = (
            torch.arange(ny, device=device, dtype=dtype) + 0.5 - 0.5 * ny
        ) * dy_rg
        x = (
            torch.arange(nx, device=device, dtype=dtype) + 0.5 - 0.5 * nx
        ) * dx_rg
        screen_y, screen_x = torch.meshgrid(y, x, indexing="ij")
        pixel_solid_angle = (
            dx_rg
            * dy_rg
            * (gravitational_radius / observer_distance) ** 2
        )
        solid_angle = torch.full_like(screen_x, pixel_solid_angle)
        return cls(
            screen_x,
            screen_y,
            solid_angle,
            {
                "sampling": "pixel_centers",
                "half_size_y_rg": half_y,
                "half_size_x_rg": half_x,
                "observer_distance_kind": "angular_diameter",
                **dict(metadata or {}),
            },
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Observer-screen array shape in ``(y, x)`` order."""

        return tuple(int(value) for value in self.x_rg.shape)

    def to(self, *, device=None, dtype=None) -> ObserverScreen:
        """Return the screen on another device or floating dtype."""

        kwargs = {
            key: value
            for key, value in {"device": device, "dtype": dtype}.items()
            if value is not None
        }
        return ObserverScreen(
            self.x_rg.to(**kwargs),
            self.y_rg.to(**kwargs),
            self.solid_angle_sr.to(**kwargs),
            self.metadata,
        )

    def rotated(
        self,
        position_angle_deg: float | torch.Tensor,
    ) -> ObserverScreen:
        """Return a screen rotated counterclockwise in the observer plane.

        The operation preserves pixel solid angles and is differentiable with
        respect to a tensor-valued position angle. Positive angles rotate the
        Cartesian screen coordinates from ``(x, y)`` to
        ``(cos(a) x - sin(a) y, sin(a) x + cos(a) y)``.
        """

        angle = torch.deg2rad(
            torch.as_tensor(
                position_angle_deg,
                device=self.x_rg.device,
                dtype=self.x_rg.dtype,
            )
        )
        if angle.numel() != 1 or not bool(torch.isfinite(angle)):
            raise ValueError("position_angle_deg must be one finite scalar")
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        return ObserverScreen(
            cosine * self.x_rg - sine * self.y_rg,
            sine * self.x_rg + cosine * self.y_rg,
            self.solid_angle_sr,
            {
                **self.metadata,
                "position_angle_deg": float(angle.detach().cpu()) * 180.0 / math.pi,
            },
        )


@dataclass(frozen=True)
class ObserverTransfer:
    """One-to-one observer-screen transfer for an equatorial emitting disk.

    Every array uses package-standard ``[y, x]`` ordering. ``solid_angle_sr``
    is the observer solid angle represented by each screen pixel. Optional
    delay and azimuth maps let reverberation or non-axisymmetric source models
    reuse the same transfer without rerunning geodesics.
    """

    radius_rg: torch.Tensor
    gfactor: torch.Tensor
    solid_angle_sr: torch.Tensor
    hit: torch.Tensor
    relative_delay_days: torch.Tensor | None = None
    emission_azimuth_rad: torch.Tensor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_screen(
        cls,
        screen: ObserverScreen,
        radius_rg: torch.Tensor,
        gfactor: torch.Tensor,
        hit: torch.Tensor,
        *,
        relative_delay_days: torch.Tensor | None = None,
        emission_azimuth_rad: torch.Tensor | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ObserverTransfer:
        """Attach traced photon quantities to a validated observer screen."""

        return cls(
            radius_rg,
            gfactor,
            screen.solid_angle_sr,
            hit,
            relative_delay_days,
            emission_azimuth_rad,
            {**dict(screen.metadata), **dict(metadata or {})},
        )

    def __post_init__(self) -> None:
        radius = torch.as_tensor(self.radius_rg)
        if radius.ndim != 2 or not radius.is_floating_point():
            raise ValueError("radius_rg must be one floating [y, x] array")
        gfactor = torch.as_tensor(
            self.gfactor,
            device=radius.device,
            dtype=radius.dtype,
        )
        solid_angle = torch.as_tensor(
            self.solid_angle_sr,
            device=radius.device,
            dtype=radius.dtype,
        )
        hit = torch.as_tensor(self.hit, device=radius.device, dtype=torch.bool)
        if gfactor.shape != radius.shape or solid_angle.shape != radius.shape:
            raise ValueError("transfer arrays must share one [y, x] shape")
        if hit.shape != radius.shape:
            raise ValueError("hit must match the transfer array shape")
        if bool(torch.any(~torch.isfinite(solid_angle))) or bool(
            torch.any(solid_angle < 0)
        ):
            raise ValueError("solid_angle_sr must be finite and non-negative")
        if bool(torch.any(~torch.isfinite(radius[hit]))) or bool(
            torch.any(radius[hit] <= 0)
        ):
            raise ValueError("hit radii must be finite and positive")
        if bool(torch.any(~torch.isfinite(gfactor[hit]))) or bool(
            torch.any(gfactor[hit] <= 0)
        ):
            raise ValueError("hit frequency shifts must be finite and positive")

        optional = {}
        for name in ("relative_delay_days", "emission_azimuth_rad"):
            value = getattr(self, name)
            if value is None:
                optional[name] = None
                continue
            tensor = torch.as_tensor(value, device=radius.device, dtype=radius.dtype)
            if tensor.shape != radius.shape:
                raise ValueError(f"{name} must match the transfer shape")
            if bool(torch.any(~torch.isfinite(tensor[hit]))):
                raise ValueError(f"{name} must be finite on hit pixels")
            optional[name] = tensor
        object.__setattr__(self, "radius_rg", radius)
        object.__setattr__(self, "gfactor", gfactor)
        object.__setattr__(self, "solid_angle_sr", solid_angle)
        object.__setattr__(self, "hit", hit)
        for name, value in optional.items():
            object.__setattr__(self, name, value)

    @property
    def shape(self) -> tuple[int, int]:
        """Observer-screen array shape in ``(y, x)`` order."""

        return tuple(int(value) for value in self.radius_rg.shape)

    def to(self, *, device=None, dtype=None) -> ObserverTransfer:
        """Return the transfer on another device or floating dtype."""

        kwargs = {
            key: value
            for key, value in {"device": device, "dtype": dtype}.items()
            if value is not None
        }
        target_device = device if device is not None else self.radius_rg.device
        return ObserverTransfer(
            self.radius_rg.to(**kwargs),
            self.gfactor.to(**kwargs),
            self.solid_angle_sr.to(**kwargs),
            self.hit.to(device=target_device),
            None
            if self.relative_delay_days is None
            else self.relative_delay_days.to(**kwargs),
            None
            if self.emission_azimuth_rad is None
            else self.emission_azimuth_rad.to(**kwargs),
            self.metadata,
        )
