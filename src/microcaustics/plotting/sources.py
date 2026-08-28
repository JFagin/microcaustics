"""Plots for pixelated source models and rendered macroimages."""

from __future__ import annotations

import numpy as np

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
