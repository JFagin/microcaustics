"""Coordinate grids with explicit angular units."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlaneRegion:
    """A rectangular angular field without a required pixelization.

    ``field_of_view_uas`` and ``center_uas`` use ``(y, x)`` ordering so they
    align with array shapes. :attr:`bounds_uas` returns explicitly named
    Cartesian ordering ``(xmin, xmax, ymin, ymax)``.
    """

    field_of_view_uas: tuple[float, float]
    center_uas: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.field_of_view_uas) != 2 or any(
            float(value) <= 0 for value in self.field_of_view_uas
        ):
            raise ValueError("field_of_view_uas must contain two positive values")
        if len(self.center_uas) != 2:
            raise ValueError("center_uas must contain two values")
        object.__setattr__(
            self,
            "field_of_view_uas",
            tuple(float(value) for value in self.field_of_view_uas),
        )
        object.__setattr__(
            self,
            "center_uas",
            tuple(float(value) for value in self.center_uas),
        )

    @property
    def bounds_uas(self) -> tuple[float, float, float, float]:
        """Return ``(xmin, xmax, ymin, ymax)`` field boundaries."""

        center_y, center_x = self.center_uas
        fov_y, fov_x = self.field_of_view_uas
        return (
            center_x - 0.5 * fov_x,
            center_x + 0.5 * fov_x,
            center_y - 0.5 * fov_y,
            center_y + 0.5 * fov_y,
        )


@dataclass(frozen=True)
class PlaneGrid:
    """A uniformly sampled rectangular lens- or source-plane field.

    Fields of view and centers are expressed in microarcseconds. Pixel values
    represent cell centers, so the outermost coordinates lie half a pixel
    inside the requested field boundary. Shape, field of view, and center use
    array ordering ``(y, x)``.
    """

    shape: tuple[int, int]
    field_of_view_uas: tuple[float, float]
    center_uas: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(int(value) < 1 for value in self.shape):
            raise ValueError("shape must contain two positive integers")
        if len(self.field_of_view_uas) != 2 or any(
            float(value) <= 0 for value in self.field_of_view_uas
        ):
            raise ValueError("field_of_view_uas must contain two positive values")
        if len(self.center_uas) != 2:
            raise ValueError("center_uas must contain two values")
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        object.__setattr__(
            self,
            "field_of_view_uas",
            tuple(float(v) for v in self.field_of_view_uas),
        )
        object.__setattr__(self, "center_uas", tuple(float(v) for v in self.center_uas))

    @property
    def pixel_scale_uas(self) -> tuple[float, float]:
        """Return ``(dy, dx)`` in microarcseconds per pixel."""

        ny, nx = self.shape
        fov_y, fov_x = self.field_of_view_uas
        return fov_y / ny, fov_x / nx

    @property
    def bounds_uas(self) -> tuple[float, float, float, float]:
        """Return ``(xmin, xmax, ymin, ymax)`` field boundaries."""

        cy, cx = self.center_uas
        fov_y, fov_x = self.field_of_view_uas
        return (
            cx - 0.5 * fov_x,
            cx + 0.5 * fov_x,
            cy - 0.5 * fov_y,
            cy + 0.5 * fov_y,
        )

    @property
    def region(self) -> PlaneRegion:
        """Return the same field without its pixelization."""

        return PlaneRegion(self.field_of_view_uas, self.center_uas)

    def axes(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return source-plane ``(y, x)`` cell-center coordinate axes."""

        ymin, ymax = self.bounds_uas[2:]
        xmin, xmax = self.bounds_uas[:2]
        dy, dx = self.pixel_scale_uas
        y = torch.linspace(
            ymin + 0.5 * dy,
            ymax - 0.5 * dy,
            self.shape[0],
            device=device,
            dtype=dtype,
        )
        x = torch.linspace(
            xmin + 0.5 * dx,
            xmax - 0.5 * dx,
            self.shape[1],
            device=device,
            dtype=dtype,
        )
        return y, x

    def mesh(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, y)`` coordinate arrays with ``self.shape``."""

        y, x = self.axes(device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return xx, yy

    def covering_trajectory(
        self,
        trajectory,
        times_days,
        *,
        margin: float = 1.05,
    ) -> PlaneGrid:
        """Return a grid enclosing this field along a source trajectory.

        ``self`` describes the source support at one center. The returned
        field encloses that support at every requested trajectory position,
        while retaining the same pixel shape. ``margin`` expands the final
        bounding box about its center and is useful for interpolation safety.
        Trajectory positions follow the package's Cartesian ``(x, y)`` order.
        """

        if not math.isfinite(float(margin)) or float(margin) < 1.0:
            raise ValueError("margin must be finite and at least one")
        times = torch.as_tensor(times_days, dtype=torch.float64).reshape(-1)
        if times.numel() < 1 or not bool(torch.all(torch.isfinite(times))):
            raise ValueError("times_days must contain finite values")
        positions = trajectory.position_uas(
            times,
            device="cpu",
            dtype=torch.float64,
        )
        if positions.shape != (times.numel(), 2) or not bool(
            torch.all(torch.isfinite(positions))
        ):
            raise ValueError("trajectory positions must have finite shape [time, 2]")
        half_y = 0.5 * self.field_of_view_uas[0]
        half_x = 0.5 * self.field_of_view_uas[1]
        x_min = float(positions[:, 0].min()) - half_x
        x_max = float(positions[:, 0].max()) + half_x
        y_min = float(positions[:, 1].min()) - half_y
        y_max = float(positions[:, 1].max()) + half_y
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        return PlaneGrid(
            self.shape,
            (
                float(margin) * (y_max - y_min),
                float(margin) * (x_max - x_min),
            ),
            (center_y, center_x),
        )


@dataclass(frozen=True)
class ImagePlaneGrid:
    """A rectangular macro-image grid in angular arcseconds.

    Shape, field of view, and center use array ordering ``(y, x)``. Methods
    returning Cartesian coordinates explicitly return ``(x, y)``.
    """

    shape: tuple[int, int]
    field_of_view_arcsec: tuple[float, float]
    center_arcsec: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(int(value) < 1 for value in self.shape):
            raise ValueError("shape must contain two positive integers")
        if len(self.field_of_view_arcsec) != 2 or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in self.field_of_view_arcsec
        ):
            raise ValueError("field_of_view_arcsec must contain two positive values")
        if len(self.center_arcsec) != 2 or any(
            not math.isfinite(float(value))
            for value in self.center_arcsec
        ):
            raise ValueError("center_arcsec must contain two finite values")
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(
            self,
            "field_of_view_arcsec",
            tuple(float(value) for value in self.field_of_view_arcsec),
        )
        object.__setattr__(
            self,
            "center_arcsec",
            tuple(float(value) for value in self.center_arcsec),
        )

    @property
    def pixel_scale_arcsec(self) -> tuple[float, float]:
        """Return ``(dy, dx)`` in arcseconds per pixel."""

        ny, nx = self.shape
        fov_y, fov_x = self.field_of_view_arcsec
        return fov_y / ny, fov_x / nx

    @property
    def pixel_solid_angle_sr(self) -> float:
        """Return the small-angle solid angle of one rectangular pixel."""

        dy, dx = self.pixel_scale_arcsec
        arcsec_to_rad = math.pi / (180.0 * 3600.0)
        return float(dy * dx * arcsec_to_rad**2)

    @property
    def bounds_arcsec(self) -> tuple[float, float, float, float]:
        """Return ``(xmin, xmax, ymin, ymax)`` field boundaries."""

        center_y, center_x = self.center_arcsec
        fov_y, fov_x = self.field_of_view_arcsec
        return (
            center_x - 0.5 * fov_x,
            center_x + 0.5 * fov_x,
            center_y - 0.5 * fov_y,
            center_y + 0.5 * fov_y,
        )

    def axes(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image-plane ``(y, x)`` cell-center axes."""

        xmin, xmax, ymin, ymax = self.bounds_arcsec
        dy, dx = self.pixel_scale_arcsec
        y = torch.linspace(
            ymin + 0.5 * dy,
            ymax - 0.5 * dy,
            self.shape[0],
            device=device,
            dtype=dtype,
        )
        x = torch.linspace(
            xmin + 0.5 * dx,
            xmax - 0.5 * dx,
            self.shape[1],
            device=device,
            dtype=dtype,
        )
        return y, x

    def mesh(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image-plane ``(x, y)`` coordinate arrays."""

        y, x = self.axes(device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return xx, yy
