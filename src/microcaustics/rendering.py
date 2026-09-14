"""Thin multiband adapters around the caustics macro-image renderer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from .geometry import ImagePlaneGrid
from .results import RenderedMacroImage
from .sources import PixelatedSource

_ARCSEC_TO_RAD = math.pi / (180.0 * 3_600.0)


@runtime_checkable
class ImageObservationModel(Protocol):
    """Optional detector/observation layer applied after caustics rendering."""

    def __call__(
        self,
        noiseless: torch.Tensor,
        *,
        time_days: float,
        grid: ImagePlaneGrid,
        band_names: tuple[str, ...],
        wavelengths_angstrom: tuple[float, ...],
        generator: torch.Generator | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, Mapping[str, object]]:
        """Return an observed image or ``(image, variance, metadata)``."""

        ...


def gaussian_psf_kernel(
    fwhm_arcsec: float,
    pixel_scale_arcsec: float,
    *,
    size: int = 17,
    oversample_factor: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a normalized square Gaussian PSF kernel.

    ``pixel_scale_arcsec`` is the final image-pixel scale. The kernel is
    sampled on the renderer's oversampled grid when ``oversample_factor`` is
    greater than one. ``size`` is always the delivered odd kernel width.
    """

    fwhm = float(fwhm_arcsec)
    scale = float(pixel_scale_arcsec)
    if not math.isfinite(fwhm) or fwhm <= 0.0:
        raise ValueError("fwhm_arcsec must be finite and positive")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("pixel_scale_arcsec must be finite and positive")
    if not isinstance(size, int) or size < 3 or size % 2 == 0:
        raise ValueError("size must be an odd integer of at least three")
    if not isinstance(oversample_factor, int) or oversample_factor < 1:
        raise ValueError("oversample_factor must be a positive integer")
    sigma_pixels = fwhm / (scale / oversample_factor) / 2.354820045
    radius = size // 2
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma_pixels**2))
    return kernel / kernel.sum()


@dataclass(frozen=True)
class PeakScaledPoissonReadNoise:
    """High-S/N Poisson-plus-read approximation in arbitrary flux units.

    The brightest noiseless pixel is assigned ``peak_electrons`` expected
    counts. ``read_noise_fraction`` specifies the one-sigma read noise as a
    fraction of that peak in the original image units. This makes the model
    useful for normalized demonstration images while keeping every assumption
    explicit. Precision instrument work should provide its own calibrated
    :class:`ImageObservationModel`.
    """

    peak_electrons: float = 5000.0
    read_noise_fraction: float = 1.0 / 250.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.peak_electrons) or self.peak_electrons <= 0.0:
            raise ValueError("peak_electrons must be finite and positive")
        if (
            not math.isfinite(self.read_noise_fraction)
            or self.read_noise_fraction < 0.0
        ):
            raise ValueError("read_noise_fraction must be finite and non-negative")

    def __call__(self, noiseless: torch.Tensor, **context):
        """Draw one noisy image and return it with its variance."""

        peak = noiseless.amax().clamp_min(torch.finfo(noiseless.dtype).tiny)
        variance = (
            noiseless.clamp_min(0.0) * peak / self.peak_electrons
            + (peak * self.read_noise_fraction) ** 2
        )
        noise = torch.randn(
            noiseless.shape,
            generator=context.get("generator"),
            device=noiseless.device,
            dtype=noiseless.dtype,
        ) * torch.sqrt(variance)
        return noiseless + noise, variance, {
            "type": "peak-scaled Poisson plus read noise",
            "peak_electrons": float(self.peak_electrons),
            "read_noise_fraction": float(self.read_noise_fraction),
        }


def _require_caustics():
    try:
        import caustics
    except ImportError as error:
        raise ImportError(
            "macro-image rendering requires the optional 'macro' dependencies"
        ) from error
    return caustics


def caustics_pixelated_sources(
    source: PixelatedSource,
    time_days: float,
    source_angular_diameter_distance_m: float,
    image_grid: ImagePlaneGrid,
    *,
    source_center_arcsec: tuple[float, float] = (0.0, 0.0),
    brightness_scale: float | Sequence[float] = 1.0,
    convert_to_pixel_flux: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[object, ...]:
    """Convert one package source epoch into caustics ``Pixelated`` bands.

    The returned objects are ordinary caustics sources and can be combined
    with caustics ``LightStack`` or passed directly to
    :class:`CausticsMacroImageRenderer`. Source geometry is converted from
    meters to arcseconds with the supplied angular-diameter distance.

    When ``convert_to_pixel_flux`` is true, a source surface brightness in
    flux per projected square meter becomes flux per output image pixel using
    surface-brightness conservation. Set it false for arbitrary normalized
    images or when the source values already use the desired rendered units.
    """

    caustics = _require_caustics()
    if not isinstance(source, PixelatedSource):
        raise TypeError("source must implement PixelatedSource")
    distance = float(source_angular_diameter_distance_m)
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("source angular-diameter distance must be positive")
    if len(source_center_arcsec) != 2 or any(
        not math.isfinite(float(value)) for value in source_center_arcsec
    ):
        raise ValueError("source_center_arcsec must contain two finite values")
    resolved_device = torch.device("cpu" if device is None else device)
    resolved_dtype = torch.get_default_dtype() if dtype is None else dtype
    if not resolved_dtype.is_floating_point:
        raise TypeError("source dtype must be floating point")
    image = source.brightness(
        [float(time_days)],
        device=resolved_device,
        dtype=resolved_dtype,
    )[0]
    scales = torch.as_tensor(
        brightness_scale,
        device=resolved_device,
        dtype=resolved_dtype,
    ).reshape(-1)
    bands = len(source.geometry.band_names)
    if scales.numel() == 1:
        scales = scales.repeat(bands)
    if scales.shape != (bands,) or not torch.isfinite(scales).all():
        raise ValueError("brightness_scale must be finite and scalar or per-band")
    if convert_to_pixel_flux:
        scales = scales * distance**2 * image_grid.pixel_solid_angle_sr
    dy_m, dx_m = source.geometry.pixel_scale_m
    if not math.isclose(dy_m, dx_m, rel_tol=1.0e-8, abs_tol=0.0):
        raise ValueError("caustics Pixelated currently requires square source pixels")
    pixelscale_arcsec = dx_m / (distance * _ARCSEC_TO_RAD)
    center_y, center_x = source_center_arcsec
    return tuple(
        caustics.Pixelated(
            image=image[..., band],
            x0=float(center_x),
            y0=float(center_y),
            pixelscale=pixelscale_arcsec,
            scale=scales[band],
            name=f"{source.geometry.band_names[band]}_pixelated_source",
        )
        for band in range(bands)
    )


def _resolve_per_band(
    value,
    band_names: tuple[str, ...],
    *,
    name: str,
    allow_none: bool,
) -> tuple[object | None, ...]:
    if value is None:
        if allow_none:
            return (None,) * len(band_names)
        raise ValueError(f"{name} is required")
    if isinstance(value, Mapping):
        missing = set(band_names).difference(value)
        extra = set(value).difference(band_names)
        if missing or extra:
            raise ValueError(
                f"{name} mapping must match bands. Missing={missing}, extra={extra}"
            )
        return tuple(value[band] for band in band_names)
    if isinstance(value, tuple | list):
        if len(value) != len(band_names):
            raise ValueError(f"{name} sequence must contain one item per band")
        return tuple(value)
    return (value,) * len(band_names)


def _resolve_psfs(
    psf,
    band_names: tuple[str, ...],
) -> tuple[object, ...]:
    if psf is None:
        return ([[1.0]],) * len(band_names)
    if isinstance(psf, Mapping):
        return _resolve_per_band(
            psf,
            band_names,
            name="psf",
            allow_none=False,
        )
    try:
        tensor = torch.as_tensor(psf)
    except (TypeError, ValueError):
        tensor = None
    if tensor is not None and tensor.ndim == 2:
        return (tensor,) * len(band_names)
    if (
        tensor is not None
        and tensor.ndim == 3
        and tensor.shape[0] == len(band_names)
    ):
        return tuple(tensor[index] for index in range(len(band_names)))
    if isinstance(psf, tuple | list) and len(psf) == len(band_names):
        return tuple(psf)
    raise ValueError("psf must be 2D, [band,y,x], or a band mapping")


class CausticsMacroImageRenderer:
    """Render arbitrary caustics lens/source models in multiple bands.

    This class intentionally delegates ray tracing, source and lens-light
    evaluation, subpixel quadrature, upsampling, and PSF convolution to
    ``caustics.LensSource``. It adds only band coordination, typed results, and
    an optional observation callback for effects outside caustics' scope.

    ``sources`` and ``lens_light`` may be a shared caustics light model, a
    sequence in band order, or a mapping by band name. Multiple physical light
    components should be composed with ``caustics.LightStack``. Unresolved
    sources can use ``caustics.StarSource``. Sampled images can use
    ``caustics.Pixelated`` or :func:`caustics_pixelated_sources`.
    """

    def __init__(
        self,
        lens: object,
        grid: ImagePlaneGrid,
        *,
        band_names: Sequence[str],
        wavelengths_angstrom: Sequence[float],
        upsample_factor: int = 1,
        quadrature_level: int | None = None,
        psf_mode: str = "fft",
        chunk_size: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        units: str = "arbitrary pixel flux",
    ) -> None:
        caustics = _require_caustics()
        if not isinstance(lens, caustics.lenses.base.Lens):
            raise TypeError("lens must be a caustics Lens model")
        bands = tuple(str(value) for value in band_names)
        wavelengths = tuple(float(value) for value in wavelengths_angstrom)
        if not bands or len(bands) != len(wavelengths):
            raise ValueError("band names and wavelengths must be non-empty and match")
        if len(set(bands)) != len(bands) or any(not value for value in bands):
            raise ValueError("band names must be non-empty and unique")
        if any(not math.isfinite(value) or value <= 0 for value in wavelengths):
            raise ValueError("wavelengths must be finite and positive")
        dy, dx = grid.pixel_scale_arcsec
        if not math.isclose(dy, dx, rel_tol=1.0e-8, abs_tol=0.0):
            raise ValueError("caustics LensSource currently requires square pixels")
        if not isinstance(upsample_factor, int) or upsample_factor < 1:
            raise ValueError("upsample_factor must be a positive integer")
        if quadrature_level is not None and quadrature_level < 1:
            raise ValueError("quadrature_level must be positive or None")
        if psf_mode not in {"fft", "conv2d"}:
            raise ValueError("psf_mode must be 'fft' or 'conv2d'")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be positive or None")
        if not dtype.is_floating_point:
            raise TypeError("renderer dtype must be floating point")
        self.lens = lens
        self.grid = grid
        self.band_names = bands
        self.wavelengths_angstrom = wavelengths
        self.upsample_factor = upsample_factor
        self.quadrature_level = quadrature_level
        self.psf_mode = psf_mode
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.units = str(units)

    def _simulator(self, source, lens_light, psf, band_index):
        caustics = _require_caustics()
        if not isinstance(source, caustics.Source):
            raise TypeError(
                f"source for band {self.band_names[band_index]!r} must be a "
                "caustics Source"
            )
        if lens_light is not None and not isinstance(lens_light, caustics.Source):
            raise TypeError("lens_light must contain caustics Source models")
        center_y, center_x = self.grid.center_arcsec
        simulator = caustics.LensSource(
            lens=self.lens,
            source=source,
            lens_light=lens_light,
            pixelscale=self.grid.pixel_scale_arcsec[1],
            pixels_x=self.grid.shape[1],
            pixels_y=self.grid.shape[0],
            upsample_factor=self.upsample_factor,
            quad_level=self.quadrature_level,
            psf_mode=self.psf_mode,
            psf=psf,
            x0=center_x,
            y0=center_y,
            name=f"macro_image_{self.band_names[band_index]}",
        )
        return simulator.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _evaluate(simulator, parameters, **options):
        if parameters is None:
            return simulator(**options)
        return simulator(parameters, **options)

    def render(
        self,
        time_days: float,
        *,
        sources,
        lens_light=None,
        psf=None,
        parameters=None,
        parameters_by_band: Mapping[str, object] | None = None,
        observation_model: ImageObservationModel | None = None,
        generator: torch.Generator | None = None,
        retain_components: bool = False,
    ) -> RenderedMacroImage:
        """Render one observer epoch through caustics in every requested band.

        ``time_days`` is provenance for the returned product and is forwarded
        to the optional observation model. Native caustics source parameters
    are not evolved implicitly. Supply epoch-specific ``parameters`` (or
        ``parameters_by_band``), or convert an evolving package source with
        :func:`caustics_pixelated_sources` at this same epoch.
        """

        time_days = float(time_days)
        if not math.isfinite(time_days):
            raise ValueError("time_days must be finite")
        if parameters is not None and parameters_by_band is not None:
            raise ValueError("supply shared parameters or parameters_by_band, not both")
        if observation_model is not None and not callable(observation_model):
            raise TypeError("observation_model must be callable")
        band_sources = _resolve_per_band(
            sources,
            self.band_names,
            name="sources",
            allow_none=False,
        )
        band_lens_light = _resolve_per_band(
            lens_light,
            self.band_names,
            name="lens_light",
            allow_none=True,
        )
        band_psfs = _resolve_psfs(psf, self.band_names)
        if parameters_by_band is None:
            band_parameters = (parameters,) * len(self.band_names)
        else:
            band_parameters = _resolve_per_band(
                parameters_by_band,
                self.band_names,
                name="parameters_by_band",
                allow_none=True,
            )

        total_bands = []
        source_bands = []
        lens_bands = []
        for index, (source, foreground, kernel, parameters_for_band) in enumerate(
            zip(
                band_sources,
                band_lens_light,
                band_psfs,
                band_parameters,
                strict=True,
            )
        ):
            simulator = self._simulator(source, foreground, kernel, index)
            total = self._evaluate(
                simulator,
                parameters_for_band,
                chunk_size=self.chunk_size,
            )
            total_bands.append(total)
            if retain_components:
                source_bands.append(
                    self._evaluate(
                        simulator,
                        parameters_for_band,
                        source_light=True,
                        lens_light=False,
                        chunk_size=self.chunk_size,
                    )
                )
                lens_bands.append(
                    self._evaluate(
                        simulator,
                        parameters_for_band,
                        source_light=False,
                        lens_light=foreground is not None,
                        chunk_size=self.chunk_size,
                    )
                )
        noiseless = torch.stack(total_bands, dim=-1)
        components = {}
        if retain_components:
            components = {
                "lensed_source": torch.stack(source_bands, dim=-1),
                "lens_light": torch.stack(lens_bands, dim=-1),
            }
        variance = None
        observation_metadata: Mapping[str, object] = {"type": "none"}
        if observation_model is None:
            observed = noiseless
        else:
            output = observation_model(
                noiseless,
                time_days=time_days,
                grid=self.grid,
                band_names=self.band_names,
                wavelengths_angstrom=self.wavelengths_angstrom,
                generator=generator,
            )
            if isinstance(output, tuple):
                if len(output) != 3:
                    raise ValueError(
                        "observation model tuple must be (values, variance, metadata)"
                    )
                observed, variance, observation_metadata = output
            else:
                observed = output
            observed = torch.as_tensor(
                observed,
                device=noiseless.device,
                dtype=noiseless.dtype,
            )
            if variance is not None:
                variance = torch.as_tensor(
                    variance,
                    device=noiseless.device,
                    dtype=noiseless.dtype,
                )
        return RenderedMacroImage(
            values=observed,
            noiseless_values=noiseless,
            variance=variance,
            grid=self.grid,
            band_names=self.band_names,
            wavelengths_angstrom=self.wavelengths_angstrom,
            time_days=time_days,
            component_values=components,
            units=self.units,
            metadata={
                "renderer": "caustics.LensSource",
                "caustics_version": getattr(_require_caustics(), "__version__", None),
                "upsample_factor": self.upsample_factor,
                "quadrature_level": self.quadrature_level,
                "psf_mode": self.psf_mode,
                "chunk_size": self.chunk_size,
                "observation": dict(observation_metadata),
            },
        )
