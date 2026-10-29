"""Plots of already calculated stellar microimages."""

import math

from ._common import (
    add_scale_bar,
    as_numpy,
    axes_or_new,
    hide_image_axes,
    panel_colorbar,
)


def plot_microimages(
    frame,
    *,
    band=0,
    max_pixels=None,
    bounds_uas=None,
    scale_bar_uas=None,
    ax=None,
    cmap="magma",
    log=True,
    vmin=None,
    vmax=None,
):
    """Display cell-averaged intensity in microJy/microarcsecond squared.

    Native cells are preserved by default. Bounds (xmin, xmax, ymin, ymax)
    crop before allocation and CPU transfer; omitted bounds enclose nonzero
    cells. Explicit max_pixels opts into display binning. The input is unchanged.
    ``band`` indexes the wavelengths delivered to the observer.
    """
    import numpy as np
    from matplotlib.colors import LogNorm

    if max_pixels is not None and (
        isinstance(max_pixels, bool)
        or not isinstance(max_pixels, int)
        or max_pixels < 1
    ):
        raise ValueError("max_pixels must be a positive integer")
    if not isinstance(band, int) or not 0 <= band < frame.flux_jy.shape[-1]:
        raise ValueError("band must index a wavelength in the frame")
    display = frame.crop(bounds_uas)
    if max_pixels is not None:
        display = display.rebin(max(1, math.ceil(max(display.grid.shape) / max_pixels)))
    # Do not allocate or transfer a dense cube for unused bands.
    image = display.flux_jy.new_zeros(display.grid.shape)
    image.reshape(-1)[display.cell_indices] = display.flux_jy[:, band]
    dy, dx = display.grid.pixel_scale_uas
    values = as_numpy(image * (1e6 / (dx * dy)))
    positive = values[values > 0]
    norm = None
    if log and positive.size:
        low = float(positive.min()) if vmin is None else vmin
        high = max(float(positive.max()), low * 1.01) if vmax is None else vmax
        norm = LogNorm(low, high)
        values = np.ma.masked_less_equal(values, 0)
    figure, ax = axes_or_new(ax, figsize=(5, 5))
    from matplotlib import colormaps

    ax.set_facecolor(colormaps.get_cmap(cmap)(0.0))
    artist = ax.imshow(
        values,
        origin="lower",
        extent=display.grid.bounds_uas,
        cmap=cmap,
        norm=norm,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        interpolation="nearest",
    )
    panel_colorbar(figure, ax, artist, label=r"$I_\nu$ [$\mu$Jy $\mu$as$^{-2}$]")
    hide_image_axes(ax)
    if scale_bar_uas is not None:
        add_scale_bar(ax, scale_bar_uas, label=rf"{scale_bar_uas:g} $\mu$as")
    return figure, ax
