"""Plots for single- and multi-image light curves and survey observations."""

from __future__ import annotations

import numpy as np

from ..observations import PhotometricObservations
from ..results import LightCurve, MultiImageLightCurves
from ._common import as_numpy, axes_or_new, band_colors, finish_axis, require_matplotlib


def _curve_values(
    curve: LightCurve,
    *,
    magnitude: bool,
    normalize: bool,
    zero_point_flux=None,
):
    values = np.asarray(as_numpy(curve.flux), dtype=np.float64)
    if zero_point_flux is not None and normalize:
        raise ValueError("zero_point_flux and normalize=True are mutually exclusive")
    if normalize:
        scale = np.nanmedian(values, axis=0, keepdims=True)
        values = np.divide(values, scale, out=np.full_like(values, np.nan), where=scale > 0)
    if magnitude:
        if zero_point_flux is not None:
            if isinstance(zero_point_flux, dict):
                scale = np.asarray(
                    [float(zero_point_flux[name]) for name in curve.band_names],
                    dtype=np.float64,
                )[None, :]
            else:
                scale = np.asarray(zero_point_flux, dtype=np.float64)
                if scale.ndim == 0:
                    scale = np.full((1, len(curve.band_names)), float(scale))
                else:
                    scale = scale.reshape(1, -1)
                if scale.shape[1] != len(curve.band_names):
                    raise ValueError("zero_point_flux must contain one value per band")
            if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError("zero_point_flux values must be finite and positive")
            values = values / scale
        # ``np.where`` evaluates both branches, so taking ``log10`` inside it
        # emits a warning for physically dark epochs even though those values
        # are subsequently replaced.  Mask first to keep zero-flux source
        # models (for example, a pre-explosion supernova) quiet and explicit.
        magnitudes = np.full_like(values, np.nan, dtype=float)
        positive = values > 0
        magnitudes[positive] = -2.5 * np.log10(values[positive])
        values = magnitudes
    return values


def plot_light_curve(
    curve: LightCurve,
    *,
    ax=None,
    magnitude: bool = False,
    normalize: bool = False,
    show_unlensed: bool = False,
    bands: tuple[str, ...] | None = None,
    invert_magnitude_axis: bool = True,
    title: str = "Light curve",
    zero_point_flux=None,
):
    """Plot selected bands of a public :class:`~microcaustics.LightCurve`."""

    require_matplotlib()
    figure, ax = axes_or_new(ax, figsize=(6.4, 3.8))
    selected = curve.band_names if bands is None else tuple(bands)
    colors = band_colors(curve.band_names)
    times = as_numpy(curve.times_days)
    values = _curve_values(
        curve,
        magnitude=magnitude,
        normalize=normalize,
        zero_point_flux=zero_point_flux,
    )
    for name in selected:
        if name not in curve.band_names:
            raise KeyError(f"unknown band {name!r}")
        index = curve.band_names.index(name)
        ax.plot(times, values[:, index], color=colors[name], label=name)
        if show_unlensed and curve.unlensed_flux is not None:
            baseline = LightCurve(
                times_days=curve.times_days,
                flux=curve.unlensed_flux,
                band_names=curve.band_names,
            )
            unlensed = _curve_values(
                baseline,
                magnitude=magnitude,
                normalize=normalize,
                zero_point_flux=zero_point_flux,
            )
            ax.plot(times, unlensed[:, index], color=colors[name], linestyle="--", alpha=0.65)
    ax.set_xlabel("Time [days]")
    if magnitude:
        ax.set_ylabel(
            "Relative magnitude"
            if normalize
            else ("Brightness [mag]" if zero_point_flux is not None else "Magnitude + constant")
        )
        if invert_magnitude_axis:
            ax.invert_yaxis()
    else:
        ax.set_ylabel("Normalized flux" if normalize else "Flux")
    ax.set_title(title)
    ax.legend(ncol=min(3, len(selected)))
    finish_axis(ax, grid=True)
    return figure, ax


def plot_multi_image_light_curves(
    curves: MultiImageLightCurves,
    *,
    band: str,
    axes=None,
    magnitude: bool = True,
    normalize: bool = True,
    sharex: bool = True,
    zero_point_flux=None,
):
    """Plot one band for each macroimage in vertically aligned panels."""

    require_matplotlib()
    from matplotlib import pyplot as plt

    if axes is None:
        figure, axes = plt.subplots(
            len(curves.images), 1, figsize=(7.0, 2.15 * len(curves.images)), sharex=sharex,
        )
    else:
        axes = np.atleast_1d(axes)
        if len(axes) != len(curves.images):
            raise ValueError("axes must contain one panel per macroimage")
        figure = axes[0].figure
    axes = np.atleast_1d(axes)
    for ax, image in zip(axes, curves.images, strict=True):
        plot_light_curve(
            image.light_curve,
            ax=ax,
            magnitude=magnitude,
            normalize=normalize,
            bands=(band,),
            zero_point_flux=zero_point_flux,
            title=rf"Image {image.image_name}  ($\Delta t={image.arrival_time_delay_days:g}$ d)",
        )
        ax.legend().remove()
    figure.tight_layout()
    return figure, axes


def plot_photometric_observations(
    observations: PhotometricObservations,
    *,
    image: int | str = 0,
    ax=None,
    marker_size: float = 4.0,
    capsize: float = 2.0,
):
    """Plot noisy multiband observations for one resolved macroimage."""

    require_matplotlib()
    figure, ax = axes_or_new(ax, figsize=(6.4, 3.8))
    if isinstance(image, str):
        try:
            image_index = observations.image_names.index(image)
        except ValueError as error:
            raise KeyError(f"unknown macroimage {image!r}") from error
    else:
        image_index = int(image)
    colors = band_colors(tuple(dict.fromkeys(observations.band_names)))
    times = as_numpy(observations.time_days)
    values = as_numpy(observations.magnitude)[:, image_index]
    errors = as_numpy(observations.magnitude_error)[:, image_index]
    bands = np.asarray(observations.band_names)
    for name in dict.fromkeys(observations.band_names):
        selected = bands == name
        ax.errorbar(
            times[selected], values[selected], yerr=errors[selected], fmt="o",
            markersize=marker_size, capsize=capsize, linewidth=0.8,
            color=colors[name], label=name,
        )
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Brightness [mag]")
    ax.invert_yaxis()
    ax.set_title(f"Observed image {observations.image_names[image_index]}")
    ax.legend(ncol=3)
    finish_axis(ax, grid=True)
    return figure, ax
