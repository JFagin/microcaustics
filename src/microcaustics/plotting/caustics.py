"""Plots for critical curves and anchor/gauge diagnostics."""

from __future__ import annotations

from ..results import AnchorGaugeLabels, CausticField, LabeledCausticFrame
from ._common import (
    add_scale_bar,
    as_numpy,
    axes_or_new,
    finish_axis,
    hide_image_axes,
    require_matplotlib,
)
from .maps import _overlay_segments


def plot_caustics(
    field: CausticField,
    *,
    plane: str = "source",
    ax=None,
    color="C3",
    linewidth: float = 0.8,
    title: str | None = None,
):
    """Plot source-plane caustics or lens-plane critical curves."""

    require_matplotlib()
    figure, ax = axes_or_new(ax, figsize=(5.0, 4.2))
    if plane == "source":
        segments = field.caustic_segments_uas
        prefix = "Source-plane"
        default_title = "Caustics"
    elif plane in {"lens", "image"}:
        segments = field.critical_segments_uas
        prefix = "Lens-plane"
        default_title = "Critical curves"
    else:
        raise ValueError("plane must be 'source' or 'lens'")
    _overlay_segments(ax, segments, color=color, linewidth=linewidth, alpha=1.0)
    values = as_numpy(segments)
    if values.size:
        minimum = values.reshape(-1, 2).min(axis=0)
        maximum = values.reshape(-1, 2).max(axis=0)
        padding = 0.04 * max(float((maximum - minimum).max()), 1.0)
        ax.set_xlim(minimum[0] - padding, maximum[0] + padding)
        ax.set_ylim(minimum[1] - padding, maximum[1] + padding)
    ax.set_aspect("equal")
    ax.set_xlabel(fr"{prefix} $x$ [$\mu$as]")
    ax.set_ylabel(fr"{prefix} $y$ [$\mu$as]")
    ax.set_title(title or default_title)
    finish_axis(ax)
    return figure, ax


def plot_anchor_gauge(
    labels: AnchorGaugeLabels | LabeledCausticFrame,
    *,
    ax=None,
    show_caustics: bool = True,
    annotate: bool = True,
    show_paths: bool = True,
    scale_bar_uas: float | None = None,
    show_axes: bool = True,
    title: str | None = "Anchor/gauge caustic labels",
):
    """Visualize anchor/gauge labels, paths, and diagnostic binary regions.

    When ``labels`` is a :class:`~microcaustics.LabeledCausticFrame` with a
    diagnostic label map, the two crossing-parity regions are drawn beneath
    the caustics. Marker colors and the centered ``0``/``1`` annotations show
    the actual inferred class rather than an arbitrary point index.
    """

    require_matplotlib()
    if isinstance(labels, LabeledCausticFrame):
        frame = labels
        diagnostics = frame.labels
    else:
        frame = None
        diagnostics = labels
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    figure, ax = axes_or_new(ax, figsize=(4.7, 4.4))
    class_colors = ("#3b75af", "#d09a56")
    if frame is not None and frame.label_map is not None:
        values = as_numpy(frame.label_map.values)
        ax.imshow(
            values,
            origin="lower",
            extent=frame.label_map.grid.bounds_uas,
            interpolation="nearest",
            aspect="equal",
            cmap=ListedColormap(class_colors),
            vmin=-0.5,
            vmax=1.5,
            alpha=0.78,
            zorder=0,
        )
    if show_caustics and frame is not None:
        _overlay_segments(
            ax,
            frame.caustics.caustic_segments_uas,
            color="black",
            linewidth=0.85,
            alpha=0.95,
        )
    anchors = as_numpy(diagnostics.anchor_points_uas)
    gauges = as_numpy(diagnostics.gauge_points_uas)
    anchor_classes = as_numpy(diagnostics.anchor_offsets).astype(int)
    gauge_classes = as_numpy(diagnostics.gauge_labels).astype(int)
    center = diagnostics.metadata.get("source_center_uas")
    if center is None:
        center = tuple(gauges.mean(axis=0)) if gauges.size else (0.0, 0.0)
    center_x, center_y = (float(value) for value in center)
    if show_paths:
        for (x, y), value in zip(anchors, anchor_classes, strict=True):
            ax.plot(
                (x, center_x),
                (y, center_y),
                color=class_colors[int(value) & 1],
                linewidth=0.75,
                alpha=0.62,
                zorder=2,
            )
    for value in (0, 1):
        selected = anchor_classes == value
        if selected.any():
            ax.scatter(
                anchors[selected, 0],
                anchors[selected, 1],
                marker="s",
                s=82,
                facecolor=class_colors[value],
                edgecolor="white",
                linewidth=1.1,
                zorder=5,
            )
        selected = gauge_classes == value
        if selected.any():
            ax.scatter(
                gauges[selected, 0],
                gauges[selected, 1],
                marker="D",
                s=76,
                facecolor=class_colors[value],
                edgecolor="white",
                linewidth=1.0,
                zorder=6,
            )
    center_class = int(diagnostics.center_label) & 1
    ax.scatter(
        [center_x],
        [center_y],
        marker="*",
        s=150,
        facecolor=class_colors[center_class],
        edgecolor="white",
        linewidth=1.0,
        zorder=7,
    )
    if annotate:
        for (x, y), value in zip(anchors, anchor_classes, strict=True):
            ax.annotate(
                str(int(value) & 1),
                (x, y),
                ha="center",
                va="center",
                color="white",
                weight="bold",
                fontsize=7.5,
                zorder=8,
            )
        for (x, y), value in zip(gauges, gauge_classes, strict=True):
            ax.annotate(
                str(int(value) & 1),
                (x, y),
                ha="center",
                va="center",
                color="white",
                weight="bold",
                fontsize=7.5,
                zorder=8,
            )
        ax.annotate(
            str(center_class),
            (center_x, center_y),
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=7.5,
            zorder=8,
        )
    all_x = list(anchors[:, 0]) + list(gauges[:, 0]) + [center_x]
    all_y = list(anchors[:, 1]) + list(gauges[:, 1]) + [center_y]
    if frame is not None and frame.label_map is not None:
        x0, x1, y0, y1 = frame.label_map.grid.bounds_uas
        all_x.extend((x0, x1))
        all_y.extend((y0, y1))
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1.0)
    ax.set_xlim(min(all_x) - 0.04 * span, max(all_x) + 0.04 * span)
    ax.set_ylim(min(all_y) - 0.04 * span, max(all_y) + 0.04 * span)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Source-plane $x$ [$\mu$as]")
    ax.set_ylabel(r"Source-plane $y$ [$\mu$as]")
    if title is not None:
        ax.set_title(title)
    handles = [
        Line2D([], [], marker="s", linestyle="none", markersize=7,
               markerfacecolor="0.5", markeredgecolor="white", label="Anchor"),
        Line2D([], [], marker="D", linestyle="none", markersize=6.5,
               markerfacecolor="0.5", markeredgecolor="white", label="Gauge"),
        Line2D([], [], marker="*", linestyle="none", markersize=10,
               markerfacecolor="0.5", markeredgecolor="white", label="Source center"),
        Line2D([], [], color="black", linewidth=0.9, label="Caustic"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, borderaxespad=0.35)
    if scale_bar_uas is not None:
        add_scale_bar(
            ax,
            float(scale_bar_uas),
            label=rf"{float(scale_bar_uas):g} $\mu$as",
        )
    if show_axes:
        finish_axis(ax)
    else:
        hide_image_axes(ax)
    return figure, ax
