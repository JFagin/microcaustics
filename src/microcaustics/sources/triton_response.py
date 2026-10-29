"""Optional grouped delay deposition, with unchanged fractional pixel weights."""

from __future__ import annotations

import torch

from ..runtime import warn_compilation

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

_PREPARED = set()


if triton is not None:

    @triton.jit
    def _grouped_deposit(
        sampled,
        response,
        fraction,
        permutation,
        offsets,
        output,
        right_output,
        BANDS: tl.constexpr,
        CHANNELS: tl.constexpr,
        STRIDE: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
    ):
        bin_id = tl.program_id(0)
        tiles = tl.cdiv(BANDS, COLS)
        derivative = tl.program_id(1) // tiles
        col = (tl.program_id(1) % tiles) * COLS + tl.arange(0, COLS)
        row = tl.arange(0, ROWS)
        source_col = derivative * BANDS + col
        positive_col = tl.maximum(2 * derivative - 1, 0) * BANDS + col
        pos_left = tl.full((ROWS, COLS), 0, tl.float32)
        pos_right = tl.full((ROWS, COLS), 0, tl.float32)
        neg_left = tl.full((ROWS, COLS), 0, tl.float32)
        neg_right = tl.full((ROWS, COLS), 0, tl.float32)
        begin = tl.load(offsets + bin_id)
        end = tl.load(offsets + bin_id + 1)
        for start in range(begin, end, ROWS):
            valid = start + row < end
            pixel = tl.load(permutation + start + row, valid, other=0)
            mu = tl.load(sampled + pixel, valid, other=0)
            f = tl.load(fraction + pixel, valid, other=0)
            value = tl.load(
                response + pixel[:, None] * STRIDE + source_col[None, :],
                valid[:, None] & (col[None, :] < BANDS),
                other=0,
            )
            pos = mu[:, None] * tl.where(derivative == 0, value, tl.maximum(value, 0))
            neg = mu[:, None] * tl.maximum(-value, 0)
            pos_left += pos * (1 - f[:, None])
            pos_right += pos * f[:, None]
            neg_left += neg * (1 - f[:, None])
            neg_right += neg * f[:, None]
        tl.store(
            output + bin_id * CHANNELS + positive_col, tl.sum(pos_left, 0), col < BANDS
        )
        tl.store(
            right_output + bin_id * CHANNELS + positive_col,
            tl.sum(pos_right, 0),
            col < BANDS,
        )
        if derivative > 0:
            tl.store(
                output + bin_id * CHANNELS + positive_col + BANDS,
                tl.sum(neg_left, 0),
                col < BANDS,
            )
            tl.store(
                right_output + bin_id * CHANNELS + positive_col + BANDS,
                tl.sum(neg_right, 0),
                col < BANDS,
            )


def prepare_groups(index, bins):
    """Sort fixed pixel memberships once. All subsequent work stays on device."""
    permutation = torch.argsort(index, stable=True)
    counts = torch.bincount(index, minlength=bins)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return permutation, offsets


def grouped_deposit(
    sampled, response, fraction, groups, bins, bands, order, runtime, rows=16, cols=64
):
    """Deposit grouped signed response weights into adjacent delay bins on CUDA."""
    channels = bands * (2 * order - 1)
    output = response.new_empty((bins, channels))
    right = torch.empty_like(output)
    signature = (bands, order, response.stride(0))
    if signature not in _PREPARED:
        warn_compilation(
            "grouped delay-response deposition",
            backend="Triton",
            device=response.device,
            dtype=response.dtype,
            enabled=runtime.warn_on_compile,
        )
        _PREPARED.add(signature)
    _grouped_deposit[(bins, order * triton.cdiv(bands, cols))](
        sampled,
        response,
        fraction,
        *groups,
        output,
        right,
        bands,
        channels,
        response.stride(0),
        rows,
        cols,
        num_warps=4,
        enable_fp_fusion=False,
    )
    output[1:].add_(right[:-1])
    return output
