"""IPM-cell microimages and flux-weighted centroids."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter

import torch

from .geometry import PlaneGrid
from .lens import LensingDistances
from .sources import PixelatedSource


@dataclass(frozen=True)
class MicroimageConfig:
    """Optional output selections. Omitted fields use all curve epochs/bands."""

    times_days: Sequence[float] | None = None
    bands: Sequence[str] | None = None

    def __post_init__(self):
        if self.times_days is not None:
            times = tuple(float(t) for t in self.times_days)
            if not times or any(not math.isfinite(t) for t in times):
                raise ValueError("microimage times must be nonempty and finite")
            if len(set(times)) != len(times):
                raise ValueError("microimage times must be unique")
            object.__setattr__(self, "times_days", times)
        if self.bands is not None:
            if isinstance(self.bands, str):
                raise TypeError("bands must be a sequence of band names, not a string")
            bands = tuple(self.bands)
            if not bands or len(set(bands)) != len(bands):
                raise ValueError("microimage bands must be nonempty and unique")
            object.__setattr__(self, "bands", bands)


@dataclass(frozen=True)
class MicroimageRequest:
    """Render a spatial source on its native angular grid during an IPM map.

    Brightness retains the source's own response delays. ``macro_delay_days``
    shifts only the source epoch, not the lens epoch. No microimage propagation
    delays are added. Trajectory/resampling support is deliberately not implicit.
    """

    source: PixelatedSource
    distances: LensingDistances
    macro_delay_days: float = 0.0
    band_indices: tuple[int, ...] | None = None
    times_days: frozenset[float] | None = None
    _capture_pixels: bool = False
    _prepared_pixels: dict = field(default_factory=dict, compare=False, repr=False)
    _static_pixels: dict = field(default_factory=dict, compare=False, repr=False)

    def includes(self, time):
        """Whether this evaluated epoch was selected for image output."""
        return self.times_days is None or float(time) in self.times_days

    def photometry(self, time, values):
        """Finish each frame before another OOM-split batch reuses the source buffers."""
        if not self._capture_pixels:
            return None
        pixels = self._prepared_pixels.pop(float(time))
        return (
            values.reshape(-1) @ pixels.reshape(-1, pixels.shape[-1]),
            pixels.sum((0, 1)),
        )

    @property
    def wavelengths_angstrom(self):
        """Wavelengths of the selected image bands, in output order."""
        values = tuple(self.source.geometry.wavelengths_angstrom)
        return (
            values
            if self.band_indices is None
            else tuple(values[i] for i in self.band_indices)
        )


@dataclass(frozen=True)
class MicroimageFrame:
    """Sparse Jy per IPM lens cell, not a galaxy-scale quad or point brightness."""

    cell_indices: torch.Tensor
    flux_jy: torch.Tensor
    grid: PlaneGrid
    time_days: float
    source_time_days: float
    wavelengths_angstrom: tuple[float, ...]
    normalization_correction_flux_jy: torch.Tensor

    @property
    def total_flux_jy(self):
        """Flux actually assigned to lens-plane cells, without scalar correction."""
        return self.flux_jy.sum(dim=0)

    @property
    def centroid_uas(self):
        """Flux-weighted lens-cell centers, [band, (x, y)], on the same device.

        Zero-flux bands return NaN. The unresolved normalization correction is
        excluded because it has no known position. Resolution convergence must
        be checked separately when quantitative astrometry is required.
        """
        _, nx = self.grid.shape
        dy, dx = self.grid.pixel_scale_uas
        xmin, _, ymin, _ = self.grid.bounds_uas
        x = xmin + (self.cell_indices.remainder(nx).to(self.flux_jy.dtype) + 0.5) * dx
        y = (
            ymin
            + (
                self.cell_indices.div(nx, rounding_mode="floor").to(self.flux_jy.dtype)
                + 0.5
            )
            * dy
        )
        moments = torch.stack((x @ self.flux_jy, y @ self.flux_jy), dim=-1)
        total = self.total_flux_jy
        return moments / torch.where(total > 0, total, torch.nan)[:, None]

    def rebin(self, factor: int):
        """Sum cells on device for display, preserving flux, not fine centroids.

        Partial edge cells are zero-padded. Compute astrometry on the original
        frame, before rebinning.
        """
        if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
            raise ValueError("factor must be a positive integer")
        if factor == 1:
            return self
        ny, nx = self.grid.shape
        new_ny, new_nx = math.ceil(ny / factor), math.ceil(nx / factor)
        index = (
            self.cell_indices // nx // factor
        ) * new_nx + self.cell_indices.remainder(nx) // factor
        unique, inverse = torch.unique(index, return_inverse=True)
        flux = self.flux_jy.new_zeros((unique.numel(), self.flux_jy.shape[-1]))
        flux.index_add_(0, inverse, self.flux_jy)
        dy, dx = self.grid.pixel_scale_uas
        fov = (new_ny * factor * dy, new_nx * factor * dx)
        xmin, _, ymin, _ = self.grid.bounds_uas
        grid = PlaneGrid((new_ny, new_nx), fov, (ymin + fov[0] / 2, xmin + fov[1] / 2))
        return replace(self, cell_indices=unique, flux_jy=flux, grid=grid)

    def crop(self, bounds_uas=None):
        """Keep intersecting native cells without binning or a dense allocation.

        Bounds are (xmin, xmax, ymin, ymax), rounded outward to cell edges.
        Omitted bounds tightly enclose nonzero cells. An empty image is left
        unchanged. Flux and centroids of the returned frame describe the crop;
        use the original frame for full-image photometry and astrometry.
        """
        ny, nx = self.grid.shape
        dy, dx = self.grid.pixel_scale_uas
        xmin, _, ymin, _ = self.grid.bounds_uas
        row = self.cell_indices // nx
        col = self.cell_indices % nx
        if bounds_uas is None:
            nonzero = (self.flux_jy != 0).any(dim=-1)
            if not bool(nonzero.any()):
                return self
            c0, c1 = int(col[nonzero].min()), int(col[nonzero].max()) + 1
            r0, r1 = int(row[nonzero].min()), int(row[nonzero].max()) + 1
        else:
            x0, x1, y0, y1 = map(float, bounds_uas)
            if not all(map(math.isfinite, (x0, x1, y0, y1))) or x1 <= x0 or y1 <= y0:
                raise ValueError("bounds_uas must be finite increasing x/y bounds")
            c0, c1 = (
                max(0, math.floor((x0 - xmin) / dx)),
                min(nx, math.ceil((x1 - xmin) / dx)),
            )
            r0, r1 = (
                max(0, math.floor((y0 - ymin) / dy)),
                min(ny, math.ceil((y1 - ymin) / dy)),
            )
            if c1 <= c0 or r1 <= r0:
                raise ValueError("bounds_uas must intersect the image grid")
        keep = (row >= r0) & (row < r1) & (col >= c0) & (col < c1)
        shape = (r1 - r0, c1 - c0)
        fov = (shape[0] * dy, shape[1] * dx)
        grid = PlaneGrid(
            shape, fov, (ymin + (r0 + r1) * dy / 2, xmin + (c0 + c1) * dx / 2)
        )
        return replace(
            self,
            cell_indices=(row[keep] - r0) * shape[1] + col[keep] - c0,
            flux_jy=self.flux_jy[keep],
            grid=grid,
            normalization_correction_flux_jy=torch.zeros_like(
                self.normalization_correction_flux_jy
            ),
        )

    def to_dense(self):
        """Allocate the full lens grid explicitly; unselected cells are zero."""
        out = self.flux_jy.new_zeros((*self.grid.shape, self.flux_jy.shape[-1]))
        out.reshape(-1, out.shape[-1])[self.cell_indices] = self.flux_jy
        return out


def prepare_microimages(request, source_grid, time_days, runtime):
    """Evaluate source once, returning Jy per source pixel and row prefix sums."""
    pixels, prefix = prepare_microimage_batch(
        request, source_grid, (time_days,), runtime
    )
    return (None, None) if pixels is None else (pixels[0], prefix[0])


def prepare_microimage_batch(request, source_grid, times_days, runtime):
    """Prepare each source epoch once; preserve broadcast static frames on device."""
    # Release failed or padded batches before allocating a retry's source cube.
    request._prepared_pixels.clear()
    geometry = request.source.geometry
    scale = geometry.pixel_scale_m
    if scale is None or tuple(geometry.shape) != source_grid.shape:
        raise ValueError("Microimages currently require the source's native pixel grid")
    uas_to_rad = math.pi / (180 * 3600 * 1e6)
    angular_scale = tuple(s / request.distances.source_m / uas_to_rad for s in scale)
    if not all(
        math.isclose(a, b, rel_tol=1e-6)
        for a, b in zip(angular_scale, source_grid.pixel_scale_uas, strict=True)
    ):
        raise ValueError("Source pixel scale does not match source_grid")
    if not math.isfinite(request.macro_delay_days):
        raise ValueError("macro_delay_days must be finite")
    times = torch.tensor(
        [time - request.macro_delay_days for time in times_days],
        device=runtime.device,
        dtype=runtime.dtype,
    )
    static = bool(getattr(request.source, "is_time_static", False))
    cache_key = (str(runtime.device), runtime.dtype)
    cached = request._static_pixels.get(cache_key) if static else None
    brightness = (
        cached.expand(len(times_days), -1, -1, -1)
        if cached is not None
        else request.source.brightness(
            times, device=runtime.device, dtype=runtime.dtype
        )
    )
    if static and cached is None:
        request._static_pixels[cache_key] = brightness[:1]
    if (
        brightness.ndim != 4
        or tuple(brightness.shape[1:3]) != source_grid.shape
        or brightness.shape[0] != len(times_days)
    ):
        raise ValueError("Source brightness must have shape [time, y, x, wavelength]")
    if brightness.shape[-1] < 1 or brightness.shape[-1] != len(
        geometry.wavelengths_angstrom
    ):
        raise ValueError("Brightness channels must match the source wavelengths")
    # A static source can broadcast one brightness frame. Avoid multiplying,
    # scanning and storing identical copies for every epoch in the batch.
    shared = len(times_days) > 1 and brightness.stride(0) == 0
    if shared:
        brightness = brightness[:1]
    pixels = (brightness * (scale[0] * scale[1])).contiguous()
    if request._capture_pixels:
        full_pixels = pixels.expand(len(times_days), -1, -1, -1) if shared else pixels
        request._prepared_pixels.update(zip(times_days, full_pixels, strict=True))
    if not any(request.includes(t) for t in times_days):
        return None, None
    if request.band_indices is not None:
        pixels = pixels[..., list(request.band_indices)].contiguous()
    prefix = torch.cat(
        (
            pixels.new_zeros((*pixels.shape[:2], 1, pixels.shape[-1])),
            pixels.cumsum(dim=2),
        ),
        dim=2,
    ).contiguous()
    if shared:
        pixels = pixels.expand(len(times_days), -1, -1, -1)
        prefix = prefix.expand(len(times_days), -1, -1, -1)
    return pixels, prefix


@torch.inference_mode()
def microimage_light_curve(
    simulation,
    lens_region,
    source_grid,
    times_days,
    source,
    distances,
    *,
    method,
    trajectory,
    schedule,
    map_observer,
    microimages,
    microimage_observer,
):
    """Native-grid, exact-source photometry sharing source pixels with IPM."""
    from .config import IPMConfig
    from .dynamic import dynamic_maps
    from .photometry import _increasing_times, _source_is_map_aligned
    from .results import LightCurve, TimingBreakdown

    if not callable(microimage_observer):
        raise ValueError("microimages requires a callable microimage_observer")
    if microimages is not None and not isinstance(microimages, MicroimageConfig):
        raise TypeError("microimages must be MicroimageConfig or None")
    config = MicroimageConfig() if microimages is None else microimages
    if not isinstance(method, IPMConfig):
        raise ValueError("Microimages require IPM")
    if not _source_is_map_aligned(source, distances, trajectory, source_grid):
        raise ValueError(
            "Microimages require a centered, stationary source on its native map grid"
        )
    times = _increasing_times(times_days, name="times_days").cpu()
    epochs = tuple(times.tolist())
    # Match at the precision of the evaluated time axis, not Python's default
    # float precision. This handles decimal epochs without snapping to neighbors.
    selected = (
        None
        if config.times_days is None
        else frozenset(torch.tensor(config.times_days, dtype=times.dtype).tolist())
    )
    if selected is not None and len(selected) != len(config.times_days):
        raise ValueError("microimage times must be distinct at the time axis precision")
    if selected is not None and not selected.issubset(epochs):
        raise ValueError(
            "microimage times must be evaluated map epochs at the time axis precision; "
            "no nearest-epoch selection or interpolation is performed"
        )
    names = tuple(source.geometry.band_names)
    if config.bands is not None and any(b not in names for b in config.bands):
        raise ValueError(f"Unknown microimage band; available bands are {names}")
    band_indices = (
        None if config.bands is None else tuple(names.index(b) for b in config.bands)
    )
    request = MicroimageRequest(
        source,
        distances,
        band_indices=band_indices,
        times_days=selected,
        _capture_pixels=True,
    )
    pending = request._prepared_pixels
    runtime = simulation.runtime
    runtime.synchronize(detailed=False)
    started = perf_counter()
    flux, unlensed = [], []
    try:
        for index, frame in enumerate(
            dynamic_maps(
                simulation,
                lens_region,
                source_grid,
                epochs,
                method=method,
                config=schedule,
                microimages=request,
            )
        ):
            lensed_flux, source_flux = frame._source_photometry
            flux.append(lensed_flux)
            unlensed.append(source_flux)
            frame = replace(frame, _source_photometry=None)
            if map_observer is not None:
                map_observer(index, frame)
            if frame.microimages is not None:
                microimage_observer(index, frame.microimages)
    finally:
        pending.clear()
        request._static_pixels.clear()
    runtime.synchronize(detailed=False)
    return LightCurve(
        times.to(device=runtime.device, dtype=runtime.dtype),
        torch.stack(flux),
        names,
        unlensed_flux=torch.stack(unlensed),
        metadata={
            "microimage_output": True,
            "source_evolution": "exact",
            "flux_units": "Jy",
        },
        timing=TimingBreakdown(
            steady_seconds=perf_counter() - started
            if runtime.profiling_enabled
            else 0.0,
            collected=runtime.profiling_enabled,
        ),
    )


def validate_render_tensors(
    source_pixels, source_prefix, cell_flux, shape, cells, device, dtype
):
    """Validate optional kernel buffers before a GPU pointer can be used."""
    if source_pixels is None:
        if source_prefix is not None or cell_flux is not None:
            raise ValueError("Microimage buffers require source_pixels")
        return
    rows, columns = shape
    if (
        source_pixels.ndim != 3
        or source_pixels.shape[:2] != (rows, columns)
        or source_pixels.shape[-1] < 1
    ):
        raise ValueError("source_pixels must have shape [source_y, source_x, bands]")
    bands = source_pixels.shape[-1]
    for name, value, expected in (
        ("source_pixels", source_pixels, (rows, columns, bands)),
        ("source_prefix", source_prefix, (rows, columns + 1, bands)),
        ("cell_flux", cell_flux, (cells, bands)),
    ):
        if value is None or tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        if value.device != device or value.dtype != dtype or not value.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous and match the raster device/dtype"
            )
