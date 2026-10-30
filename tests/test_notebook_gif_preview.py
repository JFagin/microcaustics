"""Notebook previews preserve contiguous frames, pixels, and playback timing."""

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from microcaustics.plotting import gif_preview

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("IPython")


def make_gif(path):
    rng = np.random.default_rng(42)
    frames = []
    palette = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    for _ in range(12):
        frame = Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8)).convert(
            "P"
        )
        frame.putpalette(palette.ravel())
        frames.append(frame)
    durations = [80 + 10 * index for index in range(12)]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False,
    )


@pytest.mark.parametrize("limit", [7_000_000, 25_000])
def test_preview_preserves_contiguous_frames_and_original_file(tmp_path, limit):
    path = tmp_path / "original.gif"
    make_gif(path)
    before = hashlib.sha256(path.read_bytes()).digest()
    html = gif_preview(path, start_frame=2, stop_frame=10, max_bytes=limit).data
    raw = base64.b64decode(html.split("base64,", 1)[1].split('"', 1)[0])
    assert len(raw) <= limit
    with Image.open(path) as original, Image.open(io.BytesIO(raw)) as preview:
        assert preview.size == original.size
        assert preview.info["loop"] == 0
        if limit == 7_000_000:
            assert preview.n_frames == 8
        else:
            assert 2 <= preview.n_frames < 8
        for index in range(preview.n_frames):
            preview.seek(index)
            original.seek(index + 2)
            assert preview.convert("RGB").tobytes() == original.convert("RGB").tobytes()
            assert preview.info["duration"] == original.info["duration"]
    assert hashlib.sha256(path.read_bytes()).digest() == before


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(start_frame=-1),
        dict(stop_frame=13),
        dict(start_frame=3, stop_frame=3),
        dict(max_bytes=0),
        dict(max_bytes=1),
    ],
)
def test_invalid_preview_request_fails_without_altering_gif(tmp_path, kwargs):
    path = tmp_path / "original.gif"
    make_gif(path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        gif_preview(path, **kwargs)
    assert path.read_bytes() == before


def test_single_frame_interval(tmp_path):
    path = tmp_path / "original.gif"
    make_gif(path)
    html = gif_preview(path, start_frame=4, stop_frame=5).data
    raw = base64.b64decode(html.split("base64,", 1)[1].split('"', 1)[0])
    with Image.open(io.BytesIO(raw)) as image:
        assert image.n_frames == 1
        assert image.info["duration"] == 120


def test_notebook_file_and_animation_output_sizes():
    root = Path(__file__).resolve().parents[1] / "examples/notebooks"
    count = 0
    for path in root.rglob("*.ipynb"):
        assert path.stat().st_size < 25_000_000, path
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                assert "image/gif" not in data, path
                html = data.get("text/html", "")
                if isinstance(html, list):
                    html = "".join(html)
                if "data:image/gif;base64," in html:
                    count += 1
                    assert len(html.encode("utf-8")) < 9_400_000, path
    assert count == 9
