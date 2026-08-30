"""Plots for magnification, caustic-label, and residual maps."""

from __future__ import annotations

import numpy as np

from ..results import CausticField, DistanceMap, LabelMap, MagnificationMap
from ._common import (
    add_scale_bar,
    as_numpy,
    axes_or_new,
    finish_axis,
    hide_image_axes,
    panel_colorbar,
    require_matplotlib,
)


def _overlay_segments(ax, segments, *, color="white", linewidth=0.7, alpha=0.9):
    from matplotlib.collections import LineCollection

    values = as_numpy(segments)
    if values.size:
        collection = LineCollection(
            values,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            zorder=4,
        )
        ax.add_collection(collection)
        return collection
    return None


def plot_magnification_map(
    magnification_map: MagnificationMap,
    *,
    ax=None,
    log10: bool = True,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    caustics: CausticField | None = None,
    caustic_color="white",
    caustic_linewidth: float = 0.7,
    title: str | None = None,
    scale_bar_uas: float | None = None,
    show_axes: bool = True,
):
    """Plot a :class:`~microcaustics.MagnificationMap` in source coordinates.

    Non-positive pixels are masked in logarithmic mode. The function never
    calls :func:`matplotlib.pyplot.show` and returns ``(figure, axes)`` so it
    works in notebooks and larger caller-owned layouts.
    """

    require_matplotlib()
    figure, ax = axes_or_new(ax, figsize=(5.0, 4.2))
    values = np.asarray(magnification_map.numpy(), dtype=np.float64)
    if log10:
        values = np.ma.masked_less_equal(values, 0.0)
        values = np.ma.log10(values)
        label = r"$\log_{10}\,\mu$"
    else:
        label = r"Magnification $\mu$"
    image = ax.imshow(
        values,
        origin="lower",
        extent=magnification_map.grid.bounds_uas,
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    if caustics is not None:
        _overlay_segments(
            ax,
            caustics.caustic_segments_uas,
            color=caustic_color,
            linewidth=caustic_linewidth,
        )
    ax.set_xlim(magnification_map.grid.bounds_uas[:2])
    ax.set_ylim(magnification_map.grid.bounds_uas[2:])
    ax.set_xlabel(r"Source-plane $x$ [$\mu$as]")
    ax.set_ylabel(r"Source-plane $y$ [$\mu$as]")
    if title is not None:
        ax.set_title(title)
    if scale_bar_uas is not None:
        add_scale_bar(
            ax,
            float(scale_bar_uas),
            label=rf"{float(scale_bar_uas):g} $\mu$as",
        )
    if not show_axes:
        hide_image_axes(ax)
    if colorbar:
        panel_colorbar(figure, ax, image, label=label)
    if show_axes:
        finish_axis(ax)
    return figure, ax


def plot_map_comparison(
    candidate: MagnificationMap,
    reference: MagnificationMap,
    *,
    residual: str = "magnitude",
    log10_maps: bool = True,
    cmap: str = "magma",
    residual_cmap: str = "coolwarm",
    residual_limit: float | None = None,
    scale_bar_uas: float | None = None,
    show_axes: bool = True,
):
    """Plot candidate, reference, and candidate-minus-reference residual.

    ``residual='magnitude'`` displays
    ``-2.5 log10(candidate/reference)`` in magnitudes. ``'fractional'`` and
    ``'linear'`` show fractional and linear residuals respectively.
    """

    require_matplotlib()
    from matplotlib import pyplot as plt

    if candidate.grid != reference.grid:
        raise ValueError("candidate and reference maps must share one PlaneGrid")
    cand = np.asarray(candidate.numpy(), dtype=np.float64)
    ref = np.asarray(reference.numpy(), dtype=np.float64)
    if residual == "magnitude":
        valid = (cand > 0.0) & (ref > 0.0)
        difference = np.full_like(cand, np.nan)
        difference[valid] = -2.5 * np.log10(cand[valid] / ref[valid])
        residual_label = "Magnitude residual [mag]"
    elif residual == "fractional":
        difference = np.divide(cand - ref, ref, out=np.full_like(cand, np.nan), where=ref != 0)
        residual_label = "Fractional residual"
    elif residual == "linear":
        difference = cand - ref
        residual_label = r"$\Delta\mu$"
    else:
        raise ValueError("residual must be 'magnitude', 'fractional', or 'linear'")
    finite = np.abs(difference[np.isfinite(difference)])
    limit = float(np.quantile(finite, 0.99)) if residual_limit is None and finite.size else residual_limit
    if not limit or not np.isfinite(limit):
        limit = 1.0

    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), sharex=True, sharey=True)
    for ax, product, title in zip(axes[:2], (candidate, reference), ("Candidate", "Reference"), strict=True):
        plot_magnification_map(
            product,
            ax=ax,
            log10=log10_maps,
            cmap=cmap,
            colorbar=False,
            title=title,
            scale_bar_uas=scale_bar_uas,
            show_axes=show_axes,
        )
    residual_image = axes[2].imshow(
        difference,
        origin="lower",
        extent=candidate.grid.bounds_uas,
        interpolation="nearest",
        aspect="equal",
        cmap=residual_cmap,
        vmin=-float(limit),
        vmax=float(limit),
    )
    axes[2].set_title("Residual")
    axes[2].set_xlabel(r"Source-plane $x$ [$\mu$as]")
    axes[2].set_ylabel("")
    if scale_bar_uas is not None:
        add_scale_bar(
            axes[2],
            float(scale_bar_uas),
            label=rf"{float(scale_bar_uas):g} $\mu$as",
        )
    if not show_axes:
        hide_image_axes(axes[2])
    panel_colorbar(figure, axes[2], residual_image, label=residual_label)
    for ax in axes:
        if show_axes:
            finish_axis(ax)
    figure.tight_layout()
    return figure, axes


def plot_label_map(
    label_map: LabelMap,
    *,
    ax=None,
    caustics: CausticField | None = None,
    cmap: str = "tab20",
    colorbar: bool = True,
    title: str = "Caustic labels",
    scale_bar_uas: float | None = None,
    show_axes: bool = True,
):
    """Plot an integer label or winding-number map."""

    require_matplotlib()
    figure, ax = axes_or_new(ax, figsize=(5.0, 4.2))
    values = as_numpy(label_map.values)
    from matplotlib import colormaps
    from matplotlib.colors import BoundaryNorm

    minimum = int(np.nanmin(values)) if values.size else 0
    maximum = int(np.nanmax(values)) if values.size else 0
    ticks = np.arange(minimum, maximum + 1, dtype=int)
    boundaries = np.arange(minimum - 0.5, maximum + 1.5, 1.0)
    discrete_cmap = colormaps.get_cmap(cmap).resampled(max(1, len(ticks)))
    image = ax.imshow(
        values,
        origin="lower",
        extent=label_map.grid.bounds_uas,
        interpolation="nearest",
        aspect="equal",
        cmap=discrete_cmap,
        norm=BoundaryNorm(boundaries, discrete_cmap.N),
    )
    if caustics is not None:
        _overlay_segments(ax, caustics.caustic_segments_uas)
    ax.set_xlim(label_map.grid.bounds_uas[:2])
    ax.set_ylim(label_map.grid.bounds_uas[2:])
    ax.set_xlabel(r"Source-plane $x$ [$\mu$as]")
    ax.set_ylabel(r"Source-plane $y$ [$\mu$as]")
    ax.set_title(title)
    if scale_bar_uas is not None:
        add_scale_bar(
            ax,
            float(scale_bar_uas),
            label=rf"{float(scale_bar_uas):g} $\mu$as",
        )
    if not show_axes:
        hide_image_axes(ax)
    if colorbar:
        panel_colorbar(figure, ax, image, label="Label", ticks=ticks)
    if show_axes:
        finish_axis(ax)
    return figure, ax


def plot_distance_map(
    distance_map: DistanceMap,
    *,
    ax=None,
    caustics: CausticField | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    scale_bar_uas: float | None = None,
    show_axes: bool = True,
):
    """Plot distance to the nearest caustic in microarcseconds."""

    require_matplotlib()
    figure, ax = axes_or_new(ax, figsize=(5.0, 4.2))
    image = ax.imshow(
        as_numpy(distance_map.values_uas),
        origin="lower",
        extent=distance_map.grid.bounds_uas,
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
    )
    if caustics is not None:
        _overlay_segments(ax, caustics.caustic_segments_uas)
    ax.set_xlim(distance_map.grid.bounds_uas[:2])
    ax.set_ylim(distance_map.grid.bounds_uas[2:])
    ax.set_xlabel(r"Source-plane $x$ [$\mu$as]")
    ax.set_ylabel(r"Source-plane $y$ [$\mu$as]")
    ax.set_title("Distance to caustic")
    if scale_bar_uas is not None:
        add_scale_bar(
            ax,
            float(scale_bar_uas),
            label=rf"{float(scale_bar_uas):g} $\mu$as",
        )
    if not show_axes:
        hide_image_axes(ax)
    if colorbar:
        panel_colorbar(figure, ax, image, label=r"Distance [$\mu$as]")
    if show_axes:
        finish_axis(ax)
    return figure, ax


def plot_caustic_diagnostics(
    magnification_map: MagnificationMap,
    label_map: LabelMap,
    distance_map: DistanceMap,
    winding_map: LabelMap,
    *,
    caustics: CausticField | None = None,
    scale_bar_uas: float | None = None,
):
    """Plot registered magnification, parity, distance, and winding maps.

    All four products must use the same source-plane grid. Integer products
    receive discrete color maps and integer-only colorbar ticks. This helper
    only renders already calculated public result objects; it performs no
    lensing calculation and does not retain additional accelerator tensors.
    """

    require_matplotlib()
    from matplotlib import colormaps, pyplot as plt
    from matplotlib.colors import BoundaryNorm

    products = (label_map, distance_map, winding_map)
    if any(product.grid != magnification_map.grid for product in products):
        raise ValueError("all diagnostic maps must share one source-plane grid")

    figure, axes = plt.subplots(1, 4, figsize=(15.8, 3.7))
    bounds = magnification_map.grid.bounds_uas
    common = dict(origin="lower", extent=bounds, aspect="equal")
    magnification = np.asarray(magnification_map.numpy(), dtype=np.float64)
    mag_artist = axes[0].imshow(
        np.log10(np.clip(magnification, 1.0e-12, None)), cmap="magma", **common
    )

    parity = np.asarray(as_numpy(label_map.values), dtype=int)
    parity_ticks = np.arange(int(parity.min()), int(parity.max()) + 1)
    parity_cmap = colormaps["Blues"].resampled(max(1, parity_ticks.size))
    parity_norm = BoundaryNorm(
        np.arange(parity_ticks[0] - 0.5, parity_ticks[-1] + 1.5), parity_cmap.N
    )
    parity_artist = axes[1].imshow(
        parity,
        cmap=parity_cmap,
        norm=parity_norm,
        interpolation="nearest",
        **common,
    )
    distance_artist = axes[2].imshow(
        as_numpy(distance_map.values_uas), cmap="viridis", **common
    )

    winding = np.asarray(as_numpy(winding_map.values), dtype=int)
    winding_ticks = np.arange(int(winding.min()), int(winding.max()) + 1)
    winding_cmap = colormaps["Reds"].resampled(max(1, winding_ticks.size))
    winding_norm = BoundaryNorm(
        np.arange(winding_ticks[0] - 0.5, winding_ticks[-1] + 1.5),
        winding_cmap.N,
    )
    winding_artist = axes[3].imshow(
        winding,
        cmap=winding_cmap,
        norm=winding_norm,
        interpolation="nearest",
        **common,
    )

    if caustics is not None:
        for axis in axes:
            _overlay_segments(
                axis,
                caustics.caustic_segments_uas,
                color="white",
                linewidth=0.35,
                alpha=0.85,
            )
    for axis, title in zip(
        axes,
        ("Magnification", "Binary parity", "Distance", "Winding number"),
        strict=True,
    ):
        axis.set_xlim(bounds[:2])
        axis.set_ylim(bounds[2:])
        axis.set_title(title)
        if scale_bar_uas is not None:
            add_scale_bar(
                axis,
                float(scale_bar_uas),
                label=rf"{float(scale_bar_uas):g} $\mu$as",
                color="white",
            )
        hide_image_axes(axis)
    panel_colorbar(figure, axes[0], mag_artist, label=r"$\log_{10}\mu$")
    panel_colorbar(
        figure, axes[1], parity_artist, label="Parity", ticks=parity_ticks
    )
    panel_colorbar(
        figure, axes[2], distance_artist, label=r"Distance [$\mu$as]"
    )
    panel_colorbar(
        figure,
        axes[3],
        winding_artist,
        label="Winding number",
        ticks=winding_ticks,
    )
    figure.tight_layout()
    return figure, axes
