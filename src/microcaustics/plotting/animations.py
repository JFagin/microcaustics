"""Small helpers for reproducible, fixed-normalization animations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np


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
