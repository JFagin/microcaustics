"""Plots for steady and time-dependent transfer functions."""

from __future__ import annotations

import numpy as np

from ..results import TransferFunction, TransferFunctionSeries
from ._common import as_numpy, axes_or_new, band_colors, finish_axis, require_matplotlib


def transfer_response_density(
    transfer: TransferFunction | TransferFunctionSeries,
    *,
    epoch: int = 0,
    include_zero_boundaries: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return delay centers and normalized response density for plotting.

    No interpolation or smoothing is applied. When
    ``include_zero_boundaries`` is true, zero-valued samples are added at the
    first and last delay edges. This is useful for drawing the physically
    compact response without implying a non-zero value at :math:`t=0`.
    """

    product = transfer.at(epoch) if isinstance(transfer, TransferFunctionSeries) else transfer
    edges = np.asarray(as_numpy(product.delay_edges_days), dtype=np.float64)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = np.asarray(as_numpy(product.values), dtype=np.float64) / widths[:, None]
    if include_zero_boundaries:
        centers = np.concatenate(([edges[0]], centers, [edges[-1]]))
        density = np.pad(density, ((1, 1), (0, 0)), mode="constant")
    return centers, density


def plot_transfer_function(
    transfer: TransferFunction | TransferFunctionSeries,
    *,
    epoch: int = 0,
    ax=None,
    bands: tuple[str, ...] | None = None,
    title: str = "Transfer function",
    density: bool = True,
    drawstyle: str = "default",
):
    """Plot a steady response or one epoch of a response series.

    Transfer-function arrays store probability *mass* per delay bin. The
    default divides by each bin width and therefore plots response density,
    which is invariant to the chosen delay-grid resolution.
    """

    require_matplotlib()
    if isinstance(transfer, TransferFunctionSeries):
        product = transfer.at(epoch)
        title = f"{title} at {float(transfer.times_days[epoch]):g} d"
    else:
        product = transfer
    figure, ax = axes_or_new(ax, figsize=(6.4, 3.8))
    edges = as_numpy(product.delay_edges_days)
    if density:
        centers, values = transfer_response_density(
            product,
            include_zero_boundaries=True,
        )
    else:
        centers = 0.5 * (edges[:-1] + edges[1:])
        values = as_numpy(product.values)
    selected = product.band_names if bands is None else tuple(bands)
    colors = band_colors(product.band_names)
    for name in selected:
        if name not in product.band_names:
            raise KeyError(f"unknown band {name!r}")
        index = product.band_names.index(name)
        ax.plot(
            centers,
            values[:, index],
            color=colors[name],
            label=name,
            drawstyle=drawstyle,
        )
    ax.set_xlabel("Response delay [days]")
    ax.set_ylabel("Normalized response density" if density else "Normalized response")
    ax.set_title(title)
    ax.legend(ncol=min(3, len(selected)))
    finish_axis(ax, grid=True)
    return figure, ax


def plot_mean_delays(
    transfer: TransferFunctionSeries,
    *,
    ax=None,
    bands: tuple[str, ...] | None = None,
):
    """Plot the time evolution of microlensed mean response delays."""

    require_matplotlib()
    figure, ax = axes_or_new(ax, figsize=(6.4, 3.8))
    times = as_numpy(transfer.times_days)
    values = as_numpy(transfer.mean_delays_days)
    selected = transfer.band_names if bands is None else tuple(bands)
    colors = band_colors(transfer.band_names)
    for name in selected:
        if name not in transfer.band_names:
            raise KeyError(f"unknown band {name!r}")
        index = transfer.band_names.index(name)
        ax.plot(times, values[:, index], color=colors[name], label=name)
    ax.set_xlabel("Map epoch [days]")
    ax.set_ylabel("Mean response delay [days]")
    ax.set_title("Microlensed mean delays")
    ax.legend(ncol=min(3, len(selected)))
    finish_axis(ax, grid=True)
    return figure, ax
