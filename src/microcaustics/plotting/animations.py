"""Small helpers for reproducible, fixed-normalization animations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ._common import (
    add_scale_bar,
    band_colors,
    finish_axis,
    hide_image_axes,
    panel_colorbar,
)


def save_fixed_palette_gif(
    frames: Iterable[np.ndarray],
    path: str | Path,
    *,
    fps: float = 8.0,
    colors: int = 256,
    dither: bool = True,
    palette_rgb: np.ndarray | None = None,
    freeze_static: bool = True,
) -> Path:
    """Save RGB frames with one shared GIF palette.

    Pillow's default animated-GIF path may independently quantize each frame.
    A scientifically fixed Matplotlib normalization can then *appear* to
    change because the indexed GIF palette changes.  This helper derives one
    palette from representative pixels across the complete animation and
    quantizes every frame against that same palette. Dithering is fixed to
    pixel coordinates so changing map pixels cannot alter a static colorbar.

    Parameters
    ----------
    frames
        Iterable of ``[height, width, 3]`` RGB arrays. Floating arrays are
        interpreted on ``[0, 1]``. Integer arrays are interpreted on
        ``[0, 255]``.
    path
        Destination GIF path.
    fps
        Playback rate in frames per second.
    colors
        Number of entries in the shared palette, between 2 and 256.
    dither
        Apply a fixed ordered dither to reduce banding. Unlike error diffusion,
        this never propagates changing quantization errors into nearby pixels.
    palette_rgb
        Optional ``[2..256, 3]`` palette of integer RGB colors on ``[0, 255]``.
        Overrides ``colors``. Supplying the plotted colormap and annotation
        colors preserves the full gradient and thin overlays that automatic
        palette sampling might otherwise miss.
    freeze_static
        Encode pixels that are identical in every RGB frame once, using
        error-diffusion dithering on frame zero. Reuse their exact palette
        indices throughout the GIF. This smooths stationary colorbars without
        adding noise to moving images or allowing the bar to flicker. Only
        unchanged input pixels are frozen. Set to False to apply ``dither``
        uniformly, including static elements.
    """

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - optional notebook extra
        raise ImportError("GIF export requires Pillow.") from error

    arrays = []
    for frame in frames:
        values = np.asarray(frame)
        if values.ndim != 3 or values.shape[-1] not in (3, 4):
            raise ValueError("GIF frames must have shape [height, width, 3 or 4]")
        values = values[..., :3]
        if np.issubdtype(values.dtype, np.floating):
            values = np.rint(np.clip(values, 0.0, 1.0) * 255.0)
        arrays.append(np.asarray(np.clip(values, 0, 255), dtype=np.uint8))
    if not arrays:
        raise ValueError("at least one GIF frame is required")
    if not 2 <= int(colors) <= 256:
        raise ValueError("colors must be between 2 and 256")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("all GIF frames must have the same shape")

    if palette_rgb is None:
        # Sample across the animation without favoring its first frame.
        sample_frames = arrays[:: max(1, len(arrays) // 16)]
        sample = np.concatenate(
            [frame[::4, ::4].reshape(-1, 3) for frame in sample_frames], axis=0
        )
        palette = Image.fromarray(sample.reshape(-1, 1, 3)).quantize(
            colors=int(colors),
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
    else:
        rgb = np.asarray(palette_rgb)
        if (
            rgb.ndim != 2
            or rgb.shape[1] != 3
            or not 2 <= len(rgb) <= 256
            or not np.issubdtype(rgb.dtype, np.integer)
            or np.any(rgb < 0)
            or np.any(rgb > 255)
        ):
            raise ValueError(
                "palette_rgb must be [2..256, 3] integer RGB values on [0, 255]"
            )
        palette = Image.new("P", (1, 1))
        padded = np.concatenate((rgb, np.repeat(rgb[-1:], 256 - len(rgb), axis=0)))
        palette.putpalette(padded.astype(np.uint8).ravel().tolist())

    # A small, zero-mean Bayer pattern is spatially fixed, not frame-dependent.
    bayer = np.array(
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32
    )
    offsets = ((bayer + 0.5) / 16.0 - 0.5) * 6.0
    offsets = np.tile(offsets, ((shape[0] + 3) // 4, (shape[1] + 3) // 4))
    offsets = offsets[: shape[0], : shape[1], None]
    static_mask = None
    if freeze_static and len(arrays) > 1:
        static_mask = np.ones(shape[:2], dtype=bool)
        for frame in arrays[1:]:
            static_mask &= np.all(frame == arrays[0], axis=-1)
        # Quantize once. Repeating error diffusion on later frames would let
        # changing map pixels influence the otherwise stationary colorbar.
        static_indices = Image.fromarray(arrays[0]).quantize(
            palette=palette, dither=Image.Dither.FLOYDSTEINBERG
        )
    indexed = []
    for frame in arrays:
        if dither:
            frame = np.rint(np.clip(frame.astype(np.float32) + offsets, 0, 255)).astype(
                np.uint8
            )
        encoded = Image.fromarray(frame).quantize(
            palette=palette, dither=Image.Dither.NONE
        )
        if static_mask is not None:
            encoded.paste(static_indices, mask=Image.fromarray(static_mask))
        indexed.append(encoded)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        destination,
        save_all=True,
        append_images=indexed[1:],
        duration=max(1, round(1000.0 / float(fps))),
        loop=0,
        optimize=False,
        disposal=2,
    )
    return destination


def plot_bandpass_background(
    axis, bandpasses="lsst", *, alpha: float = 0.10, show_axis: bool = True
):
    """Draw faint observed-frame response curves behind a spectrum axis.

    The right axis shows dimensionless throughput, independent of the flux
    scale on the left. Return that axis so callers can adjust or inspect it.
    """

    from ..bandpasses import resolve_bandpasses

    resolved = resolve_bandpasses(bandpasses)
    colors = band_colors(resolved.names)
    response_axis = axis.twinx()
    for bandpass in resolved.bandpasses:
        wavelength = np.asarray(bandpass.wavelength_angstrom)
        response = np.asarray(bandpass.response)
        color = colors[bandpass.name]
        response_axis.fill_between(
            wavelength, response, color=color, alpha=alpha, linewidth=0
        )
        response_axis.plot(wavelength, response, color=color, alpha=0.35, lw=0.7)
    response_axis.set(ylim=(0.0, 1.05), ylabel="Filter throughput")
    response_axis.set_yticks((0.0, 0.5, 1.0))
    response_axis.tick_params(axis="y", colors="0.55", labelsize=8)
    response_axis.yaxis.label.set_color("0.55")
    response_axis.spines["right"].set_color("0.75")
    response_axis.set_zorder(axis.get_zorder() - 1)
    axis.patch.set_alpha(0.0)
    if not show_axis:
        response_axis.set_yticks(())
        response_axis.set_ylabel("")
        response_axis.spines["right"].set_visible(False)
    return response_axis


def animate_spectrum_and_photometry(
    light_curve,
    path: str | Path,
    *,
    max_frames: int = 48,
    fps: float = 8.0,
    wavelength_limits: tuple[float, float] | None = (3000.0, 11000.0),
    figsize: tuple[float, float] = (8.4, 6.2),
    bandpasses=None,
) -> Path:
    """Animate one retained spectrum above its synchronized light curves.

    The input must be a :class:`~microcaustics.LightCurve` produced with
    ``return_spectrum=True``. At most ``max_frames`` evenly spaced epochs are
    rendered, while the lower panel always shows the full photometric cadence.
    Axis limits and the GIF palette remain fixed across the animation.
    ``bandpasses`` optionally adds faint response curves behind the spectrum.
    """

    import matplotlib.pyplot as plt

    spectrum = light_curve.spectrum
    if spectrum is None:
        raise ValueError("light_curve must contain a retained spectrum")
    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    times = spectrum.times_days.detach().cpu().numpy()
    curve_times = light_curve.times_days.detach().cpu().numpy()
    wavelengths = spectrum.wavelengths_angstrom.detach().cpu().numpy()
    total = spectrum.total_flux.detach().cpu().numpy()
    continuum = spectrum.continuum_flux.detach().cpu().numpy()
    microlensing_only = (
        None
        if spectrum.microlensing_only_continuum_flux is None
        else spectrum.microlensing_only_continuum_flux.detach().cpu().numpy()
    )
    magnitudes = light_curve.magnitude.detach().cpu().numpy()
    microlensing_magnitudes = (
        None
        if light_curve.microlensing_only_magnitude is None
        else light_curve.microlensing_only_magnitude.detach().cpu().numpy()
    )
    frame_count = min(max_frames, len(times))
    frame_indices = np.unique(
        np.linspace(0, len(times) - 1, frame_count).round().astype(int)
    )
    selected = total[:, (
        np.ones_like(wavelengths, dtype=bool)
        if wavelength_limits is None
        else (wavelengths >= wavelength_limits[0])
        & (wavelengths <= wavelength_limits[1])
    )]
    if selected.size == 0:
        raise ValueError("wavelength_limits do not overlap the retained spectrum")
    spectrum_max = 1.08e3 * float(np.nanpercentile(selected, 99.9))
    magnitude_values = (
        magnitudes.ravel()
        if microlensing_magnitudes is None
        else np.concatenate((magnitudes.ravel(), microlensing_magnitudes.ravel()))
    )
    magnitude_pad = max(0.05, 0.04 * float(np.ptp(magnitude_values)))
    magnitude_limits = (
        float(np.nanmin(magnitude_values) - magnitude_pad),
        float(np.nanmax(magnitude_values) + magnitude_pad),
    )
    colors = band_colors(light_curve.band_names)

    figure, (spectrum_axis, curve_axis) = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": (1.0, 0.82)},
    )
    total_line, = spectrum_axis.plot(
        wavelengths, 1.0e3 * total[0], color="0.12", lw=1.7, label="Total"
    )
    continuum_line, = spectrum_axis.plot(
        wavelengths,
        1.0e3 * continuum[0],
        color="#3569a8",
        ls="--",
        lw=1.35,
        label="Continuum",
    )
    microlensing_line = None
    if microlensing_only is not None:
        microlensing_line, = spectrum_axis.plot(
            wavelengths,
            1.0e3 * microlensing_only[0],
            color="0.60",
            lw=1.0,
            label="Mean-driver continuum",
        )
    spectrum_axis.set(
        xlim=wavelength_limits,
        ylim=(0.0, spectrum_max),
        xlabel=r"Observed wavelength [$\AA$]",
        ylabel=r"Observed $F_\nu$ [mJy]",
    )
    if bandpasses is not None:
        plot_bandpass_background(spectrum_axis, bandpasses)
    spectrum_axis.legend(loc="upper right", frameon=True)
    finish_axis(spectrum_axis)

    for band_index, band in enumerate(light_curve.band_names):
        curve_axis.plot(
            curve_times,
            magnitudes[:, band_index],
            color=colors[band],
            lw=1.1,
            label=band,
        )
        if microlensing_magnitudes is not None:
            curve_axis.plot(
                curve_times,
                microlensing_magnitudes[:, band_index],
                color=colors[band],
                ls="--",
                lw=0.9,
                alpha=0.8,
            )
    marker = curve_axis.axvline(times[0], color="0.15", ls=":", lw=1.2)
    curve_axis.set(
        xlim=(float(curve_times[0]), float(curve_times[-1])),
        ylim=magnitude_limits[::-1],
        xlabel="Observer time [days]",
        ylabel="brightness [mag]",
    )
    curve_axis.legend(loc="upper right", ncol=3, frameon=True, fontsize=8)
    finish_axis(curve_axis)
    figure.tight_layout()

    frames = []
    for index in frame_indices:
        total_line.set_ydata(1.0e3 * total[index])
        continuum_line.set_ydata(1.0e3 * continuum[index])
        if microlensing_line is not None:
            microlensing_line.set_ydata(1.0e3 * microlensing_only[index])
        marker.set_xdata([times[index], times[index]])
        spectrum_axis.set_title(f"Evolving observed spectrum: t = {times[index]:.0f} days")
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    destination = save_fixed_palette_gif(
        frames,
        path,
        fps=fps,
        dither=False,
    )
    plt.close(figure)
    return destination


def animate_standardized_source_bands(
    source,
    times_days,
    standardization,
    grid,
    path: str | Path,
    *,
    bands: Iterable[str] | str | None = None,
    batch_size: int = 4,
    fps: float = 8.0,
    scale_bar_uas: float = 1.0,
    cmap: str = "seismic",
    figsize: tuple[float, float] | None = None,
    dtype=None,
    device=None,
) -> Path:
    """Animate standardized multiband source images with fixed colors.

    ``standardization`` is returned by
    :func:`microcaustics.plotting.standardize_source_over_time`. The same
    temporal mean, support mask, clipping interval, Matplotlib normalization,
    and GIF palette are applied to every frame. This avoids both scientific
    renormalization and palette flicker. ``bands`` selects displayed band names
    without changing the source or its normalization. For example,
    ``bands="i"`` makes a compact single-band animation. Omit it for all bands.
    The colormap and neutral annotation colors are explicitly preserved in
    the GIF palette. Moving disk pixels remain undithered. Static elements
    such as the colorbar are encoded once and reused in every frame.
    """

    import matplotlib.pyplot as plt
    import torch

    times = torch.as_tensor(times_days)
    if times.ndim != 1 or times.numel() < 1:
        raise ValueError("times_days must be a non-empty one-dimensional sequence")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    available = tuple(source.geometry.band_names)
    band_names = (
        available
        if bands is None
        else (bands,)
        if isinstance(bands, str)
        else tuple(bands)
    )
    if not band_names or len(set(band_names)) != len(band_names):
        raise ValueError("bands must contain at least one unique band name")
    if any(name not in available for name in band_names):
        raise ValueError(f"bands must be selected from {available}")
    band_indices = tuple(available.index(name) for name in band_names)
    figure, axes = plt.subplots(
        1,
        len(band_names),
        figsize=figsize
        or ((5.6, 4.8) if len(band_names) == 1 else (2.65 * len(band_names), 2.7)),
        squeeze=False,
    )
    axes = axes[0]
    artists = []
    for axis, band in zip(axes, band_names, strict=True):
        artist = axis.imshow(
            np.zeros(grid.shape),
            origin="lower",
            extent=grid.bounds_uas,
            cmap=cmap,
            vmin=standardization.clip[0],
            vmax=standardization.clip[1],
            aspect="equal",
            interpolation="bilinear",
        )
        artists.append(artist)
        axis.set_title(f"{band} band")
        add_scale_bar(
            axis,
            scale_bar_uas,
            label=rf"{scale_bar_uas:g} $\mu$as",
            color="black",
            font_size=10,
        )
        hide_image_axes(axis)
    panel_colorbar(
        figure,
        axes[-1],
        artists[-1],
        label=r"$(B_\lambda-\overline{B}_\lambda)/\sigma_{B_\lambda}$",
    )
    time_text = axes[0].text(
        0.03,
        0.97,
        "",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    figure.subplots_adjust(
        left=0.04 if len(band_names) == 1 else 0.015,
        right=0.80 if len(band_names) == 1 else 0.955,
        bottom=0.04,
        top=0.88,
        wspace=0.13,
    )
    frames = []
    for chunk in times.split(int(batch_size)):
        values = source.brightness(chunk, dtype=dtype, device=device)
        values = values.detach().cpu().numpy().astype(np.float64, copy=False)
        for time_day, image in zip(chunk, values, strict=True):
            standardized = standardization.standardize(image)
            for band_index, artist in zip(band_indices, artists, strict=True):
                artist.set_data(standardized[:, :, band_index])
            time_text.set_text(f"t = {float(time_day) / 365.0:.1f} yr")
            figure.canvas.draw()
            frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    palette_rgb = np.vstack(
        (
            (plt.get_cmap(cmap)(np.linspace(0, 1, 240))[:, :3] * 255).astype(np.uint8),
            np.repeat(np.linspace(0, 255, 16, dtype=np.uint8)[:, None], 3, axis=1),
        )
    )
    destination = save_fixed_palette_gif(
        frames,
        path,
        fps=fps,
        dither=False,
        palette_rgb=palette_rgb,
    )
    plt.close(figure)
    return destination
