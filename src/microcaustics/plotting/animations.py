"""Small helpers for reproducible, fixed-normalization animations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ._common import add_scale_bar, hide_image_axes, panel_colorbar


def save_fixed_palette_gif(
    frames: Iterable[np.ndarray],
    path: str | Path,
    *,
    fps: float = 8.0,
    colors: int = 256,
    dither: bool = True,
) -> Path:
    """Save RGB frames with one shared GIF palette.

    Pillow's default animated-GIF path may independently quantize each frame.
    A scientifically fixed Matplotlib normalization can then *appear* to
    change because the indexed GIF palette changes.  This helper derives one
    palette from representative pixels across the complete animation and
    quantizes every frame against that same palette.

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
        Apply deterministic Floyd--Steinberg dithering against the shared
        palette. This substantially reduces visible color banding while the
        palette and scientific normalization remain fixed across all frames.
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

    # A deterministic spatial/temporal subsample limits palette construction
    # memory without favoring the first animation frame.
    sample_frames = arrays[:: max(1, len(arrays) // 16)]
    sample = np.concatenate(
        [frame[::4, ::4].reshape(-1, 3) for frame in sample_frames], axis=0
    )
    palette_source = Image.fromarray(sample.reshape(-1, 1, 3), mode="RGB")
    palette = palette_source.quantize(
        colors=int(colors), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    dither_mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    indexed = [
        Image.fromarray(frame, mode="RGB").quantize(
            palette=palette, dither=dither_mode
        )
        for frame in arrays
    ]
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


def animate_standardized_source_bands(
    source,
    times_days,
    standardization,
    grid,
    path: str | Path,
    *,
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
    renormalization and palette flicker.
    """

    import matplotlib.pyplot as plt
    import torch

    times = torch.as_tensor(times_days)
    if times.ndim != 1 or times.numel() < 1:
        raise ValueError("times_days must be a non-empty one-dimensional sequence")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    band_names = tuple(source.geometry.band_names)
    figure, axes = plt.subplots(
        1,
        len(band_names),
        figsize=figsize or (2.65 * len(band_names), 2.7),
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
        label=r"$(B_\lambda-\overline{B}_\lambda)/\sigma_{\overline{B}_\lambda}$",
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
        left=0.015, right=0.955, bottom=0.04, top=0.88, wspace=0.13
    )
    frames = []
    for chunk in times.split(int(batch_size)):
        values = source.brightness(chunk, dtype=dtype, device=device)
        values = values.detach().cpu().numpy().astype(np.float64, copy=False)
        for time_day, image in zip(chunk, values, strict=True):
            standardized = standardization.standardize(image)
            for band_index, artist in enumerate(artists):
                artist.set_data(standardized[:, :, band_index])
            time_text.set_text(f"t = {float(time_day) / 365.0:.1f} yr")
            figure.canvas.draw()
            frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    destination = save_fixed_palette_gif(frames, path, fps=fps, dither=True)
    plt.close(figure)
    return destination
