"""Source-plane trajectories independent of lens and source models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SourceTrajectory(Protocol):
    """A source-center path expressed in source-plane microarcseconds."""

    def position_uas(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return Cartesian ``(x, y)`` positions with shape ``[time, 2]``."""

        ...


@dataclass(frozen=True)
class LinearTrajectory:
    """A constant-velocity source path in the source plane.

    Positions use Cartesian ``(x, y)`` ordering. Velocities are angular
    source-plane velocities in microarcseconds per observer-frame day.
    """

    initial_position_uas: tuple[float, float] = (0.0, 0.0)
    velocity_uas_per_day: tuple[float, float] = (0.0, 0.0)
    reference_time_days: float = 0.0

    def __post_init__(self) -> None:
        if len(self.initial_position_uas) != 2:
            raise ValueError("initial_position_uas must contain x and y")
        if len(self.velocity_uas_per_day) != 2:
            raise ValueError("velocity_uas_per_day must contain x and y")

    def position_uas(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Evaluate the linear path at the requested observer-frame times."""

        resolved_dtype = torch.get_default_dtype() if dtype is None else dtype
        times = torch.as_tensor(times_days, device=device, dtype=resolved_dtype).reshape(-1)
        initial = torch.as_tensor(
            self.initial_position_uas,
            device=times.device,
            dtype=times.dtype,
        )
        velocity = torch.as_tensor(
            self.velocity_uas_per_day,
            device=times.device,
            dtype=times.dtype,
        )
        return initial + (times - self.reference_time_days)[:, None] * velocity


@dataclass(frozen=True)
class TabulatedTrajectory:
    """Explicit source positions paired one-to-one with requested times."""

    times_days: torch.Tensor
    positions_uas: torch.Tensor

    def __post_init__(self) -> None:
        times = torch.as_tensor(self.times_days)
        positions = torch.as_tensor(
            self.positions_uas,
            device=times.device,
            dtype=times.dtype if times.is_floating_point() else None,
        )
        if times.ndim != 1 or not times.is_floating_point():
            raise ValueError("times_days must be a one-dimensional floating tensor")
        if positions.shape != (times.numel(), 2):
            raise ValueError("positions_uas must have shape [time, 2]")
        if not positions.is_floating_point():
            raise TypeError("positions_uas must use a floating dtype")
        object.__setattr__(self, "times_days", times)
        object.__setattr__(self, "positions_uas", positions)

    def position_uas(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return positions when the requested time grid exactly matches."""

        requested = torch.as_tensor(times_days).reshape(-1).to(self.times_days)
        if requested.shape != self.times_days.shape or not torch.equal(
            requested,
            self.times_days,
        ):
            raise ValueError(
                "TabulatedTrajectory requires the same ordered time grid. "
                "interpolate explicitly before construction if desired"
            )
        return self.positions_uas.to(
            device=self.positions_uas.device if device is None else device,
            dtype=self.positions_uas.dtype if dtype is None else dtype,
        )
