"""Internal helpers shared by the optional plotting interface."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


def as_numpy(value) -> np.ndarray:
    """Return an array-like value as a detached CPU NumPy array."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def axes_or_new(ax, *, figsize: tuple[float, float]):
    """Return ``(figure, axes)`` while honoring a caller-owned axes."""

    from matplotlib import pyplot as plt

    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def resolve_band(
    band: int | str,
    band_names: Sequence[str],
) -> tuple[int, str]:
    """Resolve a numeric or named band selector."""

    names = tuple(str(name) for name in band_names)
    if isinstance(band, str):
        try:
            index = names.index(band)
        except ValueError as error:
            raise KeyError(f"unknown band {band!r}. Available bands are {names}") from error
    else:
        index = int(band)
        if not -len(names) <= index < len(names):
            raise IndexError(f"band index {index} is outside {len(names)} bands")
        index %= len(names)
    return index, names[index]


def require_matplotlib() -> None:
    """Raise an actionable error when the optional plotting extra is absent."""

    try:
        import matplotlib  # noqa: F401
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "Plotting requires Matplotlib. Install `microcaustics[plot]`."
        ) from error


def band_colors(band_names: Sequence[str]) -> dict[str, object]:
    """Return stable colors, using familiar optical-band colors when possible."""

    from matplotlib import pyplot as plt

    optical = {
        "u": "#6f4ca0",
        "g": "#2ca25f",
        "r": "#d7301f",
        "i": "#e67e22",
        "z": "#8c510a",
        "y": "#4d4d4d",
    }
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return {
        str(name): optical.get(str(name).lower(), cycle[index % len(cycle)])
        for index, name in enumerate(band_names)
    }


def publication_style(*, font_size: float = 13.0) -> dict[str, object]:
    """Return a compact Matplotlib style for publication-quality figures.

    Use this mapping with ``plt.style.context(mcp.publication_style())`` or update
    ``plt.rcParams`` explicitly. Returning a mapping avoids silently changing
    a user's process-wide Matplotlib configuration on import.
    """

    return {
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
        "axes.linewidth": 1.1,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }


paper_style = publication_style
"""Backward-compatible alias for :func:`publication_style`."""


def finish_axis(ax, *, grid: bool = False) -> None:
    """Apply inward major/minor ticks and optional restrained grid lines."""

    from matplotlib import ticker

    ax.minorticks_on()
    if ax.get_xscale() == "log":
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    if ax.get_yscale() == "log":
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(
        which="major", direction="in", top=True, right=True, length=6, width=1.1
    )
    ax.tick_params(
        which="minor", direction="in", top=True, right=True, length=3, width=1.1
    )
    if grid:
        ax.grid(alpha=0.18, linewidth=0.7)


def add_scale_bar(
    ax,
    length: float,
    *,
    label: str | None = None,
    location: str = "lower left",
    color: str = "white",
    font_size: float = 11.0,
    thickness_fraction: float = 0.006,
    pad: float = 0.35,
    borderpad: float = 0.5,
    separation: float = 4.0,
):
    """Add a physical scale bar to an image axes.

    Parameters
    ----------
    ax
        Matplotlib axes whose data coordinates use the same physical units as
        ``length``. Image extents are preferred. Ordinary collection/patch
        diagrams use the current axes limits.
    length
        Physical length of the bar in the axes data coordinates.
    label
        Text shown below the bar. If omitted, ``length`` is formatted without
        an assumed unit.
    location, color, font_size
        Standard visual controls for the anchored scale bar.
    thickness_fraction
        Bar thickness as a fraction of the displayed image height.

    Returns
    -------
    matplotlib.offsetbox.AnchoredOffsetbox
        The artist added to ``ax``.

    Notes
    -----
    The function intentionally does not hide axes or choose physical units;
    callers can use it for source-plane microarcseconds, Einstein radii, or
    any other coordinate system represented by the image extent.
    """

    if not np.isfinite(length) or float(length) <= 0.0:
        raise ValueError("scale-bar length must be finite and positive")
    if not np.isfinite(thickness_fraction) or float(thickness_fraction) <= 0.0:
        raise ValueError("thickness_fraction must be finite and positive")

    from matplotlib import font_manager
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    if ax.images:
        extent = ax.images[0].get_extent()
        height = abs(float(extent[3]) - float(extent[2]))
    else:
        lower, upper = ax.get_ylim()
        height = abs(float(upper) - float(lower))
        if not np.isfinite(height) or height <= 0.0:
            raise ValueError("axes must have finite non-zero data limits")
    artist = AnchoredSizeBar(
        ax.transData,
        float(length),
        f"{float(length):g}" if label is None else str(label),
        location,
        pad=float(pad),
        borderpad=float(borderpad),
        sep=float(separation),
        color=color,
        frameon=False,
        size_vertical=height * float(thickness_fraction),
        fontproperties=font_manager.FontProperties(size=float(font_size)),
    )
    artist.set_zorder(20)
    artist.set_clip_on(False)
    ax.add_artist(artist)
    return artist


def hide_image_axes(ax, *, keep_frame: bool = True) -> None:
    """Hide image ticks while preserving overlays and, by default, the frame.

    This is the package convention for source-, lens-, and image-plane maps
    that carry a physical scale bar instead of redundant coordinate ticks.
    Keeping the panel frame makes multi-panel image boundaries unambiguous;
    pass ``keep_frame=False`` only when a frameless image is intentional.
    """

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(bool(keep_frame))
        if keep_frame:
            spine.set_linewidth(0.8)


def panel_colorbar(
    figure,
    ax,
    mappable,
    *,
    label: str | None = None,
    ticks=None,
    size: str = "4%",
    pad: float = 0.06,
):
    """Attach a colorbar whose height exactly matches one image panel."""

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    colorbar_axis = divider.append_axes("right", size=size, pad=float(pad))
    colorbar = figure.colorbar(mappable, cax=colorbar_axis, ticks=ticks)
    if label:
        colorbar.set_label(label)
    colorbar.ax.tick_params(direction="out", length=3)
    return colorbar
