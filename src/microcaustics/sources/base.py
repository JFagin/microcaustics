"""Source protocols that keep microlensing independent of source physics."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import torch

from ..geometry import PlaneGrid


@dataclass(frozen=True)
class SourceGeometry:
    """Spatial and spectral dimensions of a source.

    Supply ``field_of_view_uas`` for angular geometry or ``pixel_scale_m``
    for physical geometry, never both. Scalars describe square fields and
    tuples use ``(y, x)`` order. ``bands_angstrom`` maps band names to observed
    wavelengths. Angular geometry is resolved using the system distances
    during source setup. Physical pixel sizes are unavailable until then.
    """

    shape: int | tuple[int, int]
    pixel_scale_m: tuple[float, float] | None = None
    wavelengths_angstrom: tuple[float, ...] = ()
    band_names: tuple[str, ...] = ()
    field_of_view_uas: float | tuple[float, float] | None = field(
        default=None, kw_only=True
    )
    bands_angstrom: Mapping[str, float] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        shape = (
            (self.shape, self.shape)
            if isinstance(self.shape, int)
            else tuple(self.shape)
        )
        if (
            len(shape) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) for value in shape
            )
            or any(value < 1 for value in shape)
        ):
            raise ValueError("shape must contain two positive integers")
        object.__setattr__(self, "shape", shape)
        if (self.pixel_scale_m is None) == (self.field_of_view_uas is None):
            raise ValueError("supply exactly one of field_of_view_uas or pixel_scale_m")
        name = (
            "pixel_scale_m" if self.pixel_scale_m is not None else "field_of_view_uas"
        )
        values = getattr(self, name)
        values = (values, values) if isinstance(values, int | float) else tuple(values)
        if len(values) != 2 or any(
            not math.isfinite(value) or value <= 0 for value in values
        ):
            raise ValueError(f"{name} must contain two finite positive values")
        object.__setattr__(self, name, tuple(float(value) for value in values))
        if self.bands_angstrom is not None:
            if not self.bands_angstrom:
                raise ValueError(
                    "bands_angstrom must contain at least one name and wavelength"
                )
            names = tuple(self.bands_angstrom)
            wavelengths = tuple(float(value) for value in self.bands_angstrom.values())
            if (self.band_names or self.wavelengths_angstrom) and (
                tuple(self.band_names) != names
                or tuple(self.wavelengths_angstrom) != wavelengths
            ):
                raise ValueError(
                    "bands_angstrom conflicts with band_names/wavelengths_angstrom"
                )
            object.__setattr__(self, "band_names", names)
            object.__setattr__(self, "wavelengths_angstrom", wavelengths)
            object.__setattr__(
                self, "bands_angstrom", dict(zip(names, wavelengths, strict=True))
            )
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

    def resolve(
        self, distances, *, dtype: torch.dtype = torch.float32
    ) -> SourceGeometry:
        """Return physical geometry without mutating the angular specification.

        Systems call this automatically. It is also available for standalone
        source calculations that already have lensing distances.
        """
        if self.pixel_scale_m is not None:
            return self
        lengths = distances.uas_to_source_length(self.field_of_view_uas, dtype=dtype)
        return SourceGeometry(
            self.shape,
            tuple(
                float(length) / pixels
                for length, pixels in zip(lengths, self.shape, strict=True)
            ),
            self.wavelengths_angstrom,
            self.band_names,
        )

    @property
    def pixel_area_m2(self) -> float:
        """Physical pixel area, available after the source geometry is resolved."""
        if self.pixel_scale_m is None:
            raise RuntimeError(
                "angular source geometry is not resolved. Add the source to a "
                "MicrolensingSystem first, or pixelate it with distances for standalone use"
            )
        return self.pixel_scale_m[0] * self.pixel_scale_m[1]


def _geometry_grid(geometry, distances):
    """Convert resolved or angular geometry to a source grid without sampling it."""
    fov = geometry.field_of_view_uas
    if fov is None:
        fov = distances.source_length_to_uas(
            tuple(
                n * scale
                for n, scale in zip(geometry.shape, geometry.pixel_scale_m, strict=True)
            ),
            dtype=torch.float64,
        )
    return PlaneGrid(geometry.shape, tuple(float(value) for value in fov))


def _custom_geometry(geometry, shape, fov, bands):
    """Resolve the explicit-geometry and concise custom-source constructor forms."""
    if geometry is not None:
        if fov is not None or bands is not None:
            raise ValueError(
                "supply geometry or field_of_view_uas/bands_angstrom, not both"
            )
        return geometry
    if shape is None:
        raise ValueError("source_grid_shape is required for a callable source")
    return SourceGeometry(shape, field_of_view_uas=fov, bands_angstrom=bands)


class _SampledSource:
    """Resolve custom-source geometry without resampling physical brightness."""

    def recommended_grid(self, distances, policy=None) -> PlaneGrid:
        """Return the declared source extent and pixel shape without evaluating it."""
        del policy
        return _geometry_grid(self.geometry, distances)

    def pixelate(
        self,
        distances=None,
        *,
        source_redshift=None,
        H0=None,
        Om0=None,
        grid=None,
        policy=None,
        runtime=None,
    ):
        """Bind angular geometry once, preserving the declared source pixels.

        An explicit map grid may be larger or have a different resolution,
        but must enclose the source. No custom image is silently resampled.
        """
        from .physical import _resolve_source_distances

        distances = _resolve_source_distances(
            distances,
            source_redshift=source_redshift,
            H0=H0,
            Om0=Om0,
            runtime=runtime,
        )
        del policy
        native = self.recommended_grid(distances)
        if grid is not None and any(
            actual < expected * (1 - 1e-6)
            for actual, expected in zip(
                grid.field_of_view_uas, native.field_of_view_uas, strict=True
            )
        ):
            raise ValueError(
                "source_grid field of view must enclose the custom source geometry"
            )
        dtype = (
            torch.float64
            if getattr(runtime, "dtype", None) in (torch.float64, "float64")
            else torch.float32
        )
        geometry = self.geometry.resolve(distances, dtype=dtype)
        if geometry is self.geometry:
            return self
        return replace(self, geometry=geometry)


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
class StaticSource(_SampledSource):
    """A time-independent physical source image with arbitrary bands.

    The image must have shape ``[y, x, band]``. Rectangular images are
    supported. A later map operation decides whether padding is required by a
    particular algorithm. Values are in ``Jy m^-2`` on the projected source
    plane when this source is used for photometry. Supply ``field_of_view_uas``
    and ``bands_angstrom`` directly or an explicit ``geometry``. Image shape
    is inferred and physical geometry is resolved by the system.
    """

    image: torch.Tensor
    geometry: SourceGeometry | None = None
    name: str = "static"
    is_time_static: bool = True
    field_of_view_uas: float | tuple[float, float] | None = field(
        default=None, kw_only=True
    )
    bands_angstrom: Mapping[str, float] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        image = torch.as_tensor(self.image)
        if image.ndim != 3:
            raise ValueError("image must have shape [y, x, band]")
        geometry = _custom_geometry(
            self.geometry,
            tuple(image.shape[:2]),
            self.field_of_view_uas,
            self.bands_angstrom,
        )
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "field_of_view_uas", None)
        object.__setattr__(self, "bands_angstrom", None)
        expected = (*self.geometry.shape, len(self.geometry.band_names))
        if tuple(image.shape) != expected:
            raise ValueError(
                f"image shape must be {expected}, got {tuple(image.shape)}"
            )
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
class CallableSource(_SampledSource):
    """Adapt a callable returning physical ``[time, y, x, band]`` brightness.

    The callable returns observed spectral flux density per projected
    source-plane area in ``Jy m^-2`` when used for photometry. Supply the
    ``source_grid_shape``, ``field_of_view_uas``, and ``bands_angstrom`` directly
    or an explicit ``geometry``. A function accepting ``geometry`` as a keyword
    receives the resolved physical geometry. Times-only functions also work.
    No cosmology or distance object is required when used through a system.
    """

    function: Callable[[torch.Tensor], torch.Tensor]
    geometry: SourceGeometry | None = None
    name: str = "callable"
    is_time_static: bool = False
    user_metadata: Mapping[str, object] | None = None
    source_grid_shape: int | tuple[int, int] | None = field(default=None, kw_only=True)
    field_of_view_uas: float | tuple[float, float] | None = field(
        default=None, kw_only=True
    )
    bands_angstrom: Mapping[str, float] | None = field(default=None, kw_only=True)
    _accepts_geometry: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.geometry is not None and self.source_grid_shape is not None:
            raise ValueError("supply geometry or source_grid_shape, not both")
        geometry = _custom_geometry(
            self.geometry,
            self.source_grid_shape,
            self.field_of_view_uas,
            self.bands_angstrom,
        )
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "source_grid_shape", None)
        object.__setattr__(self, "field_of_view_uas", None)
        object.__setattr__(self, "bands_angstrom", None)
        if not callable(self.function):
            raise TypeError("function must be a callable source brightness model")
        try:
            parameters = inspect.signature(self.function).parameters
        except (TypeError, ValueError):
            parameters = {}
        parameter = parameters.get("geometry")
        accepts = parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        object.__setattr__(self, "_accepts_geometry", accepts)

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
        if dtype is not None:
            times = times.to(dtype=dtype)
        elif not times.is_floating_point():
            times = times.to(torch.get_default_dtype())
        kwargs = {"geometry": self.geometry} if self._accepts_geometry else {}
        value = torch.as_tensor(self.function(times, **kwargs), device=times.device)
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
