"""Plots for pixelated source models and rendered macroimages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..geometry import PlaneGrid
from ..results import RenderedMacroImage
from ..sources import PixelatedSource
from ._common import (
    add_scale_bar,
    as_numpy,
    axes_or_new,
    finish_axis,
    hide_image_axes,
    panel_colorbar,
    require_matplotlib,
    resolve_band,
)


@dataclass(frozen=True)
class TemporalSourceStandardization:
    """Streaming temporal statistics for a variable pixelated source.

    Arrays use package-standard ``[y, x, band]`` order.  ``standardize``
    applies the same physical support, temporal mean, standard deviation, and
    clipping to any compatible source frame.  Statistics are accumulated in
    float64 without retaining the full time cube.
    """

    mean: np.ndarray
    standard_deviation: np.ndarray
    representative: np.ndarray
    representative_index: int
    support: np.ndarray
    valid: np.ndarray
    clip: tuple[float, float]

    def standardize(self, frame) -> np.ndarray:
        """Return one standardized ``[y, x, band]`` brightness frame."""

        values = np.asarray(as_numpy(frame), dtype=np.float64)
        if values.shape != self.mean.shape:
            raise ValueError("frame must match the summarized source shape")
        standardized = np.divide(
            values - self.mean,
            self.standard_deviation,
            out=np.zeros_like(values, dtype=np.float64),
            where=self.valid,
        )
        standardized = np.where(self.support, standardized, 0.0)
        return np.nan_to_num(
            standardized,
            nan=0.0,
            posinf=self.clip[1],
            neginf=self.clip[0],
        ).clip(*self.clip)

    @property
    def representative_standardized(self) -> np.ndarray:
        """Return the selected representative frame after standardization."""

        return self.standardize(self.representative)


@dataclass(frozen=True)
class MeanSourceIsophotes:
    """Mean source images and one enclosed-flux isophote per band.

    Arrays use package-standard ``[y, x, band]`` image order. ``levels`` and
    the radius array have one entry per source band. The plotted contour
    retains the image's actual shape; ``area_equivalent_radii_uas`` only
    summarizes its enclosed pixel area as ``sqrt(A / pi)``.
    """

    mean_brightness: np.ndarray
    levels: np.ndarray
    enclosed_fractions: np.ndarray
    area_equivalent_radii_uas: np.ndarray
    x_uas: np.ndarray
    y_uas: np.ndarray
    requested_fraction: float

    @property
    def images(self) -> tuple[np.ndarray, ...]:
        """Return mean brightness planes in source-band order."""

        return tuple(
            self.mean_brightness[:, :, index]
            for index in range(self.mean_brightness.shape[-1])
        )


def _source_support(source, shape: tuple[int, int]) -> np.ndarray:
    """Return the source's wavelength-independent physical image support."""

    transfer = getattr(source, "transfer", None)
    if transfer is None:
        return np.ones(shape, dtype=bool)
    hit = np.asarray(as_numpy(transfer.hit), dtype=bool)
    if hit.shape != shape:
        raise ValueError("source transfer mask must match its pixel geometry")
    return hit


def standardize_source_over_time(
    source: PixelatedSource,
    times_days,
    *,
    batch_size: int = 4,
    dtype: torch.dtype = torch.float64,
    device=None,
    support=None,
    representative: str = "brightest",
    clip: tuple[float, float] = (-3.5, 3.5),
) -> TemporalSourceStandardization:
    """Summarize and standardize a time-dependent pixelated source.

    The source is evaluated in small batches and the temporal statistics are
    accumulated in float64.  This avoids materializing a potentially large
    ``[time, y, x, band]`` cube.  By default the returned representative epoch
    has the largest integrated brightness.  ``representative='most-variable'``
    instead chooses the epoch with the largest spatial RMS after
    standardization and performs one additional streaming pass.

    A single wavelength-independent support is used for every band.  Kerr
    sources obtain it from their observer-transfer hit mask.  Other source
    models default to their complete rectangular grid, or callers may provide
    an explicit two-dimensional boolean mask.
    """

    times = torch.as_tensor(times_days, device=device, dtype=dtype).reshape(-1)
    if times.numel() == 0:
        raise ValueError("times_days must contain at least one epoch")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if representative not in {"brightest", "most-variable"}:
        raise ValueError("representative must be 'brightest' or 'most-variable'")
    lower, upper = (float(clip[0]), float(clip[1]))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("clip must contain two increasing finite values")

    total = square_total = selected = None
    selected_index = 0
    selected_score = -np.inf
    count = 0
    for chunk in times.split(int(batch_size)):
        values = source.brightness(chunk, dtype=dtype, device=device)
        values = np.asarray(as_numpy(values), dtype=np.float64)
        if values.ndim != 4 or values.shape[-1] != len(source.geometry.band_names):
            raise ValueError("source brightness must have [time, y, x, band] shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("source brightness contains non-finite values")
        chunk_total = values.sum(axis=0, dtype=np.float64)
        chunk_square = np.square(values).sum(axis=0, dtype=np.float64)
        total = chunk_total if total is None else total + chunk_total
        square_total = chunk_square if square_total is None else square_total + chunk_square
        if representative == "brightest":
            scores = values.sum(axis=(1, 2, 3), dtype=np.float64)
            local = int(np.argmax(scores))
            if float(scores[local]) > selected_score:
                selected_score = float(scores[local])
                selected_index = count + local
                selected = values[local].copy()
        count += values.shape[0]

    mean = total / count
    variance = np.maximum(square_total / count - np.square(mean), 0.0)
    standard_deviation = np.sqrt(variance)
    spatial_shape = tuple(int(value) for value in mean.shape[:2])
    physical_support = (
        _source_support(source, spatial_shape)
        if support is None
        else np.asarray(as_numpy(support), dtype=bool)
    )
    if physical_support.shape != spatial_shape:
        raise ValueError("support must have the source's [y, x] shape")
    band_support = np.broadcast_to(physical_support[..., None], mean.shape)
    valid = band_support & (standard_deviation > np.finfo(np.float64).tiny)

    if representative == "most-variable":
        count = 0
        for chunk in times.split(int(batch_size)):
            values = np.asarray(
                as_numpy(source.brightness(chunk, dtype=dtype, device=device)),
                dtype=np.float64,
            )
            normalized = np.divide(
                values - mean,
                standard_deviation,
                out=np.zeros_like(values),
                where=valid[None],
            )
            scores = np.sqrt(np.mean(np.square(normalized), axis=(1, 2, 3)))
            local = int(np.argmax(scores))
            if float(scores[local]) > selected_score:
                selected_score = float(scores[local])
                selected_index = count + local
                selected = values[local].copy()
            count += values.shape[0]

    if selected is None:
        raise RuntimeError("failed to select a representative source epoch")
    return TemporalSourceStandardization(
        mean,
        standard_deviation,
        selected,
        selected_index,
        band_support,
        valid,
        (lower, upper),
    )


def mean_source_isophotes(
    source: PixelatedSource,
    times_days,
    grid: PlaneGrid,
    *,
    fraction: float = 0.95,
    batch_size: int = 4,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> MeanSourceIsophotes:
    """Measure enclosed-flux isophotes from a source's mean brightness.

    Brightness is accumulated in small temporal batches, so the full
    ``[time, y, x, band]`` cube is never retained. ``grid.bounds_uas`` defines
    the angular field occupied by the source images; its pixel resolution may
    differ from the source image resolution. One intensity level enclosing
    ``fraction`` of the mean positive finite flux is measured independently
    for every band.
    """

    times = torch.as_tensor(times_days, device=device, dtype=dtype).reshape(-1)
    if times.numel() == 0:
        raise ValueError("times_days must contain at least one epoch")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    requested_fraction = float(fraction)
    if (
        not np.isfinite(requested_fraction)
        or not 0.0 < requested_fraction < 1.0
    ):
        raise ValueError("fraction must be finite and lie within (0, 1)")

    total = None
    count = 0
    for chunk in times.split(int(batch_size)):
        values = np.asarray(
            as_numpy(source.brightness(chunk, dtype=dtype, device=device)),
            dtype=np.float64,
        )
        if values.ndim != 4 or values.shape[-1] != len(source.geometry.band_names):
            raise ValueError("source brightness must have [time, y, x, band] shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("source brightness contains non-finite values")
        chunk_total = values.sum(axis=0, dtype=np.float64)
        total = chunk_total if total is None else total + chunk_total
        count += values.shape[0]

    mean = total / count
    ny, nx, bands = mean.shape
    x0, x1, y0, y1 = grid.bounds_uas
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    x_uas = x0 + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y_uas = y0 + (np.arange(ny, dtype=np.float64) + 0.5) * dy
    levels = np.empty(bands, dtype=np.float64)
    enclosed = np.empty(bands, dtype=np.float64)
    radii = np.empty(bands, dtype=np.float64)
    for index in range(bands):
        image = mean[:, :, index]
        level = enclosed_flux_contour_levels(
            image, fractions=(requested_fraction,)
        )[0]
        positive = np.isfinite(image) & (image > 0.0)
        selected = positive & (image >= level)
        positive_flux = np.where(positive, image, 0.0)
        levels[index] = level
        enclosed[index] = positive_flux[selected].sum() / positive_flux.sum()
        radii[index] = np.sqrt(selected.sum() * abs(dx * dy) / np.pi)

    return MeanSourceIsophotes(
        mean,
        levels,
        enclosed,
        radii,
        x_uas,
        y_uas,
        requested_fraction,
    )


def plot_standardized_source_bands(
    summary: TemporalSourceStandardization,
    grid: PlaneGrid,
    band_names,
    *,
    scale_bar_uas: float = 1.0,
    cmap: str = "seismic",
    colorbar_label: str = (
        r"$(B_\lambda-\overline{B}_\lambda)/\sigma_{B_\lambda}$"
    ),
    figsize=None,
):
    """Plot every band of a standardized source with one shared color scale."""

    require_matplotlib()
    names = tuple(str(name) for name in band_names)
    values = summary.representative_standardized
    if values.shape[-1] != len(names):
        raise ValueError("band_names must match the summarized source")
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (2.6 * len(names), 2.7)
    figure, axes = plt.subplots(1, len(names), figsize=figsize, squeeze=False)
    axes = axes[0]
    artist = None
    for index, (axis, name) in enumerate(zip(axes, names, strict=True)):
        artist = axis.imshow(
            values[:, :, index],
            origin="lower",
            extent=grid.bounds_uas,
            cmap=cmap,
            vmin=summary.clip[0],
            vmax=summary.clip[1],
            aspect="equal",
        )
        axis.set_title(f"{name} band")
        add_scale_bar(
            axis,
            float(scale_bar_uas),
            label=rf"{float(scale_bar_uas):g} $\mu$as",
            color="black",
        )
        hide_image_axes(axis)
    panel_colorbar(figure, axes[-1], artist, label=colorbar_label)
    figure.tight_layout()
    return figure, axes


def enclosed_flux_contour_levels(
    image,
    fractions: tuple[float, ...] = (0.68, 0.95, 0.99),
) -> np.ndarray:
    """Return intensity thresholds enclosing requested fractions of flux.

    The returned thresholds are sorted in increasing intensity order, as
    required by :meth:`matplotlib.axes.Axes.contour`.  Consequently the first
    contour encloses the largest requested flux fraction.  Non-finite and
    non-positive pixels contribute no flux.  Any display-only smoothing must
    be applied to ``image`` before calling this function.

    Parameters
    ----------
    image:
        Two-dimensional source-brightness array.
    fractions:
        Strictly increasing enclosed-flux fractions between zero and one.
        The package tutorials use 68%, 95%, and 99% throughout.
    """

    values = np.asarray(as_numpy(image), dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("image must be two-dimensional")
    requested = np.asarray(fractions, dtype=np.float64)
    if (
        requested.ndim != 1
        or requested.size == 0
        or np.any(~np.isfinite(requested))
        or np.any((requested <= 0.0) | (requested >= 1.0))
        or np.any(np.diff(requested) <= 0.0)
    ):
        raise ValueError("fractions must increase strictly within (0, 1)")
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError("image contains no positive finite flux")
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered, dtype=np.float64)
    cumulative /= cumulative[-1]
    levels = [
        ordered[min(int(np.searchsorted(cumulative, fraction)), ordered.size - 1)]
        for fraction in requested
    ]
    return np.sort(np.asarray(levels, dtype=np.float64))


def plot_source_brightness(
    source: PixelatedSource,
    *,
    time_days: float = 0.0,
    band: int | str = 0,
    ax=None,
    log10: bool = False,
    cmap: str = "inferno",
    colorbar: bool = True,
    scale_bar_m: float | None = None,
    show_axes: bool = True,
    title: str | None = None,
):
    """Evaluate and plot one source brightness plane.

    This evaluates only the supplied source model. It never constructs or
    reruns a microlensing map.
    """

    require_matplotlib()
    index, name = resolve_band(band, source.geometry.band_names)
    frame = as_numpy(source.brightness(float(time_days)))[0, :, :, index]
    if log10:
        frame = np.ma.log10(np.ma.masked_less_equal(frame, 0.0))
    figure, ax = axes_or_new(ax, figsize=(5.0, 4.2))
    ny, nx = source.geometry.shape
    dy, dx = source.geometry.pixel_scale_m
    extent = (-0.5 * nx * dx, 0.5 * nx * dx, -0.5 * ny * dy, 0.5 * ny * dy)
    image = ax.imshow(frame, origin="lower", extent=extent, cmap=cmap, aspect="equal")
    ax.set_xlabel("Source $x$ [m]")
    ax.set_ylabel("Source $y$ [m]")
    if title is not None:
        ax.set_title(title)
    if scale_bar_m is not None:
        add_scale_bar(ax, float(scale_bar_m), label=f"{float(scale_bar_m):g} m")
    if not show_axes:
        hide_image_axes(ax)
    if colorbar:
        panel_colorbar(
            figure,
            ax,
            image,
            label=(r"$\log_{10}$ brightness" if log10 else "Brightness"),
        )
    if show_axes:
        finish_axis(ax)
    return figure, ax


def plot_rendered_macro_image(
    image: RenderedMacroImage,
    *,
    band: int | str = 0,
    ax=None,
    noiseless: bool = False,
    log10: bool = False,
    cmap: str = "magma",
    colorbar: bool = True,
    scale_bar_arcsec: float | None = None,
    show_axes: bool = True,
    title: str | None = None,
):
    """Plot one band of a rendered, optionally noisy macro-lensed image."""

    require_matplotlib()
    index, name = resolve_band(band, image.band_names)
    values = image.noiseless_values if noiseless else image.values
    frame = np.asarray(as_numpy(values)[:, :, index], dtype=np.float64)
    if log10:
        frame = np.ma.log10(np.ma.masked_less_equal(frame, 0.0))
    figure, ax = axes_or_new(ax, figsize=(5.0, 4.2))
    artist = ax.imshow(
        frame,
        origin="lower",
        extent=image.grid.bounds_arcsec,
        cmap=cmap,
        aspect="equal",
    )
    ax.set_xlabel(r"Image $x$ [arcsec]")
    ax.set_ylabel(r"Image $y$ [arcsec]")
    if title is not None:
        ax.set_title(title)
    if scale_bar_arcsec is not None:
        add_scale_bar(
            ax,
            float(scale_bar_arcsec),
            label=rf'{float(scale_bar_arcsec):g}"',
        )
    if not show_axes:
        hide_image_axes(ax)
    if colorbar:
        label = f"{image.units}" if not log10 else rf"$\log_{{10}}$({image.units})"
        panel_colorbar(figure, ax, artist, label=label)
    if show_axes:
        finish_axis(ax)
    return figure, ax
