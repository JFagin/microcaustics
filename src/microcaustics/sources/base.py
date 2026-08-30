"""Source protocols that keep microlensing independent of source physics."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class SourceGeometry:
    """Spatial and spectral dimensions of a pixelated source."""

    shape: tuple[int, int]
    pixel_scale_m: tuple[float, float]
    wavelengths_angstrom: tuple[float, ...]
    band_names: tuple[str, ...]

    @classmethod
    def from_angular(
        cls,
        distances,
        *,
        shape: int | tuple[int, int],
        field_of_view_uas: float | tuple[float, float],
        bands: Mapping[str, float],
    ) -> SourceGeometry:
        """Construct physical source geometry from an angular field.

        Parameters
        ----------
        distances:
            A :class:`~microcaustics.LensingDistances` instance.
        shape:
            One square resolution or ``(ny, nx)``.
        field_of_view_uas:
            One square field size or ``(height, width)`` in microarcseconds.
        bands:
            Mapping from band name to observed wavelength in Angstrom.

        Notes
        -----
        Source brightness remains expressed per projected physical area. This
        constructor only removes the otherwise repetitive angular-to-metre
        conversion from custom-source workflows.
        """

        resolved_shape = (
            (int(shape), int(shape))
            if isinstance(shape, int)
            else tuple(int(value) for value in shape)
        )
        resolved_fov = (
            (float(field_of_view_uas), float(field_of_view_uas))
            if isinstance(field_of_view_uas, (int, float))
            else tuple(float(value) for value in field_of_view_uas)
        )
        if len(resolved_shape) != 2 or len(resolved_fov) != 2:
            raise ValueError("shape and field_of_view_uas must each contain two values")
        if not bands:
            raise ValueError("bands must contain at least one name and wavelength")
        field_m = distances.uas_to_source_length(
            torch.as_tensor(resolved_fov, dtype=torch.float32),
            dtype=torch.float32,
        )
        return cls(
            resolved_shape,
            tuple(
                float(length) / pixels
                for length, pixels in zip(field_m, resolved_shape, strict=True)
            ),
            tuple(float(wavelength) for wavelength in bands.values()),
            tuple(str(name) for name in bands),
        )

    def __post_init__(self) -> None:
        if (
            len(self.shape) != 2
            or any(not isinstance(value, int) for value in self.shape)
            or any(value < 1 for value in self.shape)
        ):
            raise ValueError("shape must contain two positive integers")
        if len(self.pixel_scale_m) != 2 or any(
            not math.isfinite(value) or value <= 0 for value in self.pixel_scale_m
        ):
            raise ValueError("pixel_scale_m must contain two positive values")
        if len(self.wavelengths_angstrom) != len(self.band_names):
            raise ValueError("wavelength and band-name counts must match")
        if not self.band_names:
            raise ValueError("at least one source band is required")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.wavelengths_angstrom
        ):
            raise ValueError("wavelengths_angstrom must be finite and positive")
        if any(not isinstance(name, str) or not name for name in self.band_names):
            raise ValueError("band names must be non-empty strings")
        if len(set(self.band_names)) != len(self.band_names):
            raise ValueError("band names must be unique")


@runtime_checkable
class PixelatedSource(Protocol):
    """Images produced by a static or time-dependent source model.

    ``brightness`` returns a tensor with shape ``[time, y, x, band]``. Values
    used for photometry are observed spectral flux density per projected
    source-plane area in ``Jy m^-2``. Integrating over source-pixel area then
    returns physical flux density in Jy. Dimensionless profiles may still be
    used for morphology or magnification-only calculations, but must be given
    a physical normalization before an absolute light curve is requested.
    """

    geometry: SourceGeometry
    is_time_static: bool

    def brightness(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return brightness with shape ``[time, y, x, band]``."""

        ...

    def metadata(self) -> Mapping[str, object]:
        """Return serializable source provenance and model settings."""

        ...


def _as_times(value: torch.Tensor | Sequence[float] | float) -> torch.Tensor:
    return torch.as_tensor(value).reshape(-1)


@dataclass(frozen=True)
class StaticSource:
    """A time-independent physical source image with arbitrary bands.

    The image must have shape ``[y, x, band]``. Rectangular images are
    supported. A later map operation decides whether padding is required by a
    particular algorithm. Values are in ``Jy m^-2`` on the projected source
    plane when this source is used for photometry.
    """

    image: torch.Tensor
    geometry: SourceGeometry
    name: str = "static"
    is_time_static: bool = True

    def __post_init__(self) -> None:
        image = torch.as_tensor(self.image)
        expected = (*self.geometry.shape, len(self.geometry.band_names))
        if tuple(image.shape) != expected:
            raise ValueError(f"image shape must be {expected}, got {tuple(image.shape)}")
        if not image.is_floating_point():
            raise TypeError("source brightness must use a floating dtype")
        object.__setattr__(self, "image", image)

    def brightness(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Repeat the static image for every requested source time."""

        times = _as_times(times_days)
        frame = self.image.to(
            dtype=self.image.dtype if dtype is None else dtype,
            device=self.image.device if device is None else device,
        )
        return frame.unsqueeze(0).expand(int(times.numel()), -1, -1, -1)

    def metadata(self) -> Mapping[str, object]:
        """Return serializable source provenance."""

        return {
            "type": "static",
            "name": self.name,
            "brightness_units": "Jy m^-2 projected source plane",
            "integrated_flux_units": "Jy",
            "is_time_static": True,
        }


@dataclass(frozen=True)
class CallableSource:
    """Adapt a callable returning physical ``[time, y, x, band]`` brightness.

    The callable returns observed spectral flux density per projected
    source-plane area in ``Jy m^-2`` when used for photometry.
    """

    function: Callable[[torch.Tensor], torch.Tensor]
    geometry: SourceGeometry
    name: str = "callable"
    is_time_static: bool = False
    user_metadata: Mapping[str, object] | None = None

    def brightness(
        self,
        times_days: torch.Tensor | Sequence[float] | float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Evaluate and validate the user-supplied batched source function."""

        times = _as_times(times_days)
        if device is not None:
            times = times.to(device=device)
        value = torch.as_tensor(self.function(times), device=times.device)
        expected_tail = (*self.geometry.shape, len(self.geometry.band_names))
        if value.ndim == 3 and int(times.numel()) == 1:
            value = value.unsqueeze(0)
        if tuple(value.shape) != (int(times.numel()), *expected_tail):
            raise ValueError(
                "source callable must return [time, y, x, band] with shape "
                f"{(int(times.numel()), *expected_tail)}, got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise TypeError("source brightness must use a floating dtype")
        return value.to(dtype=value.dtype if dtype is None else dtype)

    def metadata(self) -> Mapping[str, object]:
        """Return serializable source provenance supplied by the user."""

        return {
            "type": "callable",
            "name": self.name,
            "brightness_units": "Jy m^-2 projected source plane",
            "integrated_flux_units": "Jy",
            "is_time_static": self.is_time_static,
            **dict(self.user_metadata or {}),
        }
