"""Self-contained animation previews for notebooks."""

from __future__ import annotations

import base64
import io
import operator
from pathlib import Path


def gif_preview(
    path: str | Path,
    *,
    start_frame: int = 0,
    stop_frame: int | None = None,
    max_bytes: int = 7_000_000,
):
    """Return an embedded HTML GIF for ``IPython.display.display``.

    Select consecutive frames from ``start_frame`` to ``stop_frame`` (exclusive).
    If needed, shorten the end of this interval until the encoded GIF fits
    ``max_bytes``. Never resize, skip frames, change colors, or change playback
    timing. The full GIF on disk is not modified. The default leaves room below
    the approximately 10 MB HTML-output cutoff observed in GitHub's viewer.

    Pillow and IPython are needed only when this notebook helper is called.
    """
    from IPython.display import HTML
    from PIL import Image

    start_frame = operator.index(start_frame)
    max_bytes = operator.index(max_bytes)
    if start_frame < 0 or max_bytes <= 0:
        raise ValueError("start_frame must be nonnegative and max_bytes positive")
    frames, durations = [], []
    with Image.open(path) as original:
        if original.format != "GIF":
            raise ValueError("gif_preview requires a GIF file")
        end = original.n_frames if stop_frame is None else operator.index(stop_frame)
        if not start_frame < end <= original.n_frames:
            raise ValueError("frame interval must be nonempty and within the GIF")
        loop = original.info.get("loop")
        for index in range(start_frame, end):
            original.seek(index)
            rgb = original.convert("RGB")
            indexed = rgb.quantize(colors=256, dither=Image.Dither.NONE)
            if indexed.convert("RGB").tobytes() != rgb.tobytes():
                raise ValueError("Cannot preserve this frame's colors exactly")
            frames.append(indexed)
            durations.append(original.info.get("duration", 0))

    count = len(frames)
    while True:
        buffer = io.BytesIO()
        options = {} if loop is None else {"loop": loop}
        frames[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=frames[1:count],
            duration=durations[:count],
            disposal=1,
            optimize=True,
            **options,
        )
        raw = buffer.getvalue()
        if len(raw) <= max_bytes:
            break
        if count <= min(2, len(frames)):
            raise ValueError("The full-resolution frames exceed the preview size limit")
        count = max(2, min(count - 1, int(count * max_bytes / len(raw) * 0.98)))

    # Fail rather than silently alter the pixels, frame count, or timing.
    with Image.open(io.BytesIO(raw)) as check:
        if check.n_frames != count:
            raise ValueError("GIF encoding merged frames unexpectedly")
        for index in range(count):
            check.seek(index)
            if (
                check.convert("RGB").tobytes() != frames[index].convert("RGB").tobytes()
                or check.info.get("duration", 0) != durations[index]
            ):
                raise ValueError("GIF encoding changed the preview frames or timing")

    encoded = base64.b64encode(raw).decode("ascii")
    return HTML(
        f'<img src="data:image/gif;base64,{encoded}" '
        f'alt="Animation frames {start_frame} through {start_frame + count - 1}" '
        'style="max-width:100%;height:auto">'
    )
