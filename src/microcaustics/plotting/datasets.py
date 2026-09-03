"""Dataset galleries drawn from in-memory public simulation results."""

from __future__ import annotations

import numpy as np

from ..photometry import flux_to_magnitude
from ._common import as_numpy, finish_axis, require_matplotlib
from .light_curves import _curve_values
from .maps import plot_magnification_map


def plot_light_curve_dataset(
    curves,
    *,
    center_magnifications,
    microlensing_fluxes=None,
    band="i",
    annotations=None,
):
    """Plot AB light curves beside three source-center diagnostics.

    ``curves`` contains labeled ``LightCurve`` results. Each corresponding
    ``center_magnifications`` array samples the map center at the label epochs,
    not the potentially finer photometry epochs. Optional ``microlensing_fluxes``
    are Jy arrays with the same shape as each curve, calculated by the caller
    without intrinsic driving. No source model, simulation, or file is loaded.
    Returns ``(figure, axes)`` with axes shaped ``(len(curves), 4)``.
    """
    require_matplotlib()
    from matplotlib import pyplot as plt

    curves = tuple(curves)
    count = len(curves)
    centers = tuple(center_magnifications)
    baselines = (
        (None,) * count if microlensing_fluxes is None else tuple(microlensing_fluxes)
    )
    annotations = ("",) * count if annotations is None else tuple(annotations)
    if not count or any(
        len(items) != count for items in (centers, baselines, annotations)
    ):
        raise ValueError("supply one center series, baseline, and annotation per curve")
    for curve, center, baseline in zip(curves, centers, baselines, strict=True):
        if curve.labels is None:
            raise ValueError("dataset diagnostics require labeled light curves")
        if as_numpy(center).shape != tuple(curve.labels.times_days.shape):
            raise ValueError("center magnification must match the label epoch axis")
        if baseline is not None and as_numpy(baseline).shape != tuple(curve.flux.shape):
            raise ValueError("microlensing flux must match the photometry shape")
        if band not in curve.band_names:
            raise KeyError(f"unknown band {band!r}")

    figure, axes = plt.subplots(
        count,
        4,
        squeeze=False,
        figsize=(14.2, 2.28 * count),
        gridspec_kw={
            "width_ratios": (5.4, 1.45, 1.15, 1.55),
            "hspace": 0.10,
            "wspace": 0.32,
        },
    )
    for index, (curve, center, baseline, annotation) in enumerate(
        zip(curves, centers, baselines, annotations, strict=True)
    ):
        time = as_numpy(curve.times_days)
        label_time = as_numpy(curve.labels.times_days)
        band_index = curve.band_names.index(band)
        axis = axes[index, 0]
        if baseline is not None:
            magnitude = as_numpy(flux_to_magnitude(baseline))[:, band_index]
            axis.plot(
                time, magnitude, color="black", lw=1.35, label="Microlensing only"
            )
        axis.plot(
            time,
            _curve_values(curve, magnitude=True, normalize=False)[:, band_index],
            color="darkorange",
            lw=1.15,
            label="Microlensing + intrinsic variability"
            if baseline is not None
            else "Light curve",
        )
        axis.invert_yaxis()
        axis.set_ylabel(f"LC {index + 1}\nbrightness [mag]")
        if annotation:
            axis.text(0.012, 0.08, annotation, transform=axis.transAxes, fontsize=8.5)
        if index == 0:
            axis.legend(loc="upper right", frameon=True, fontsize=9)
        diagnostics = (
            (np.log10(np.clip(as_numpy(center), 1e-12, None)), r"$\log_{10}\mu$"),
            (as_numpy(curve.labels.crossing_events), "crossing"),
            (
                as_numpy(curve.labels.center_distances_uas),
                r"$d_{\rm caustic}$ [$\mu$as]",
            ),
        )
        for column, (values, label) in enumerate(diagnostics, start=1):
            axis = axes[index, column]
            if column == 2:
                axis.step(label_time, values, where="pre", color="black", lw=1.05)
                axis.set(ylim=(-0.05, 1.05), yticks=(0, 1))
                if index == 0:
                    axis.set_title("Labels at source center", fontsize=10)
            else:
                axis.plot(label_time, values, color="black", lw=1.05)
            axis.set_xlim(float(time[0]), float(time[-1]))
            axis.set_ylabel(label, fontsize=9, labelpad=2)
        for axis in axes[index]:
            if index < count - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("time [days]")
            finish_axis(axis)
    figure.align_ylabels()
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.065, top=0.98)
    return figure, axes


def plot_labeled_map_gallery(frames):
    """Plot live labeled map frames with caustics and one shared color scale.

    Accepts ``LabeledMapFrame`` results, each of which supplies its own angular
    grid and epoch. Scale bars replace coordinate ticks. Returns ``(figure,
    axes)`` with a one-dimensional axes array, excluding the colorbar axis.
    The function neither generates maps nor writes files.
    """
    require_matplotlib()
    from matplotlib import pyplot as plt

    frames = tuple(frames)
    if not frames:
        raise ValueError("at least one labeled map frame is required")
    logs = [
        np.log10(np.clip(frame.magnification_map.numpy(), 1e-12, None))
        for frame in frames
    ]
    finite = np.concatenate([values[np.isfinite(values)] for values in logs])
    if not finite.size:
        raise ValueError("the maps contain no finite magnifications")
    vmin, vmax = np.percentile(finite, (0.25, 99.75))
    figure = plt.figure(figsize=(2.96 * len(frames), 3.15))
    layout = figure.add_gridspec(
        1, len(frames) + 1, width_ratios=([1] * len(frames) + [0.045]), wspace=0.06
    )
    axes = np.asarray(
        [figure.add_subplot(layout[0, index]) for index in range(len(frames))]
    )
    colorbar_axis = figure.add_subplot(layout[0, -1])
    for index, (axis, frame) in enumerate(zip(axes, frames, strict=True)):
        width = min(frame.magnification_map.grid.field_of_view_uas)
        plot_magnification_map(
            frame.magnification_map,
            ax=axis,
            caustics=frame.caustics.caustics,
            log10=True,
            vmin=float(vmin),
            vmax=float(vmax),
            colorbar=False,
            scale_bar_uas=2.0 if width >= 4.0 else min(0.5, width / 4),
            show_axes=False,
            title=rf"LC {index + 1}, $t={frame.time_days:g}$ days",
        )
    colorbar = figure.colorbar(axes[-1].images[-1], cax=colorbar_axis)
    colorbar.set_label(r"$\log_{10}\mu$")
    colorbar.ax.tick_params(direction="in")
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.88)
    return figure, axes
