"""Optional wavelength-vectorized photometry for CUDA thermal disks."""

from __future__ import annotations

import weakref
from collections import OrderedDict

import torch

from ..runtime import warn_compilation
from .thin_disk import _C, _H, _K_B

try:  # Triton is optional for CPU, Apple, and non-Triton runtimes.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on optional runtime
    triton = None
    tl = None

_PREPARED_SPECIALIZATIONS: set[tuple[object, ...]] = set()
_ACTIVE_TILE_CACHE: OrderedDict[tuple[int, int], tuple] = OrderedDict()
_ACTIVE_TILE_CACHE_LIMIT = 8
_TILE_HEIGHT = 16
_TILE_WIDTH = 32
_TILE_PIXELS = _TILE_HEIGHT * _TILE_WIDTH


def _active_source_tiles(hit: torch.Tensor, block_pixels: int = _TILE_PIXELS):
    """Cache an exact 2D tile list for a sufficiently sparse observer mask.

    No faint flux is omitted: a tile is inactive only if every hit bit is
    false. The cache is weakly keyed by the mask and checked against its
    version, so replacing or editing a transfer cannot reuse stale indices.
    """

    height, width = hit.shape
    tile_height = block_pixels // _TILE_WIDTH
    if (
        height < 128
        or width < 128
        or height % tile_height
        or width % _TILE_WIDTH
    ):
        return None
    try:
        version = hit._version
    except RuntimeError:
        # Inference tensors without version counters cannot be cached safely.
        return None
    key = (id(hit), block_pixels)
    entry = _ACTIVE_TILE_CACHE.get(key)
    if entry is not None:
        owner, cached_version, tiles = entry
        if owner() is hit and cached_version == version:
            _ACTIVE_TILE_CACHE.move_to_end(key)
            if tiles is not None:
                stream = torch.cuda.current_stream(hit.device)
                if stream.cuda_stream != tiles[3]:
                    stream.wait_event(tiles[2])
                    tiles[0].record_stream(stream)
                    tiles[1].record_stream(stream)
            return tiles
        del _ACTIVE_TILE_CACHE[key]

    index = (
        torch.arange(height * width, device=hit.device, dtype=torch.int32)
        .reshape(
            height // tile_height,
            tile_height,
            width // _TILE_WIDTH,
            _TILE_WIDTH,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .flatten()
    )
    blocks = height * width // block_pixels
    active = torch.nonzero(
        hit.flatten()[index].reshape(blocks, block_pixels).any(dim=1)
    ).flatten().to(torch.int32).contiguous()
    # Full and nearly full masks are better handled by the linear kernel.
    tiles = None
    if active.numel() < 0.9 * blocks:
        stream = torch.cuda.current_stream(hit.device)
        ready = torch.cuda.Event()
        ready.record(stream)
        tiles = (index, active, ready, stream.cuda_stream)
    _ACTIVE_TILE_CACHE[key] = (weakref.ref(hit), version, tiles)
    if len(_ACTIVE_TILE_CACHE) > _ACTIVE_TILE_CACHE_LIMIT:
        _ACTIVE_TILE_CACHE.popitem(last=False)
    return tiles


if triton is not None:

    @triton.jit
    def _thermal_flux_partials(
        temperature_ptr,
        gfactor_ptr,
        solid_angle_ptr,
        hit_ptr,
        wavelength_ptr,
        color_ptr,
        dimming_ptr,
        area_ptr,
        left_ptr,
        right_ptr,
        fraction_ptr,
        lensed_ptr,
        intrinsic_ptr,
        pixel_index_ptr,
        active_block_ptr,
        PIXELS: tl.constexpr,
        BANDS: tl.constexpr,
        BLOCKS: tl.constexpr,
        TEMPERATURE_TIMES: tl.constexpr,
        LEFT_TIMES: tl.constexpr,
        RIGHT_TIMES: tl.constexpr,
        FRACTION_TIMES: tl.constexpr,
        LEFT_PIXELS: tl.constexpr,
        RIGHT_PIXELS: tl.constexpr,
        LEFT_STRIDE: tl.constexpr,
        RIGHT_STRIDE: tl.constexpr,
        HC: tl.constexpr,
        KB: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_B: tl.constexpr,
        USE_COMPACT: tl.constexpr,
    ):
        block = tl.program_id(0)
        epoch = tl.program_id(1)
        physical_block = tl.load(active_block_ptr + block) if USE_COMPACT else block
        raw_pixel = physical_block * BLOCK_P + tl.arange(0, BLOCK_P)
        valid_pixel = raw_pixel < PIXELS
        pixel = (
            tl.load(pixel_index_ptr + raw_pixel, valid_pixel, other=0)
            if USE_COMPACT
            else raw_pixel
        )
        band = tl.arange(0, BLOCK_B)
        valid_band = band < BANDS
        source_epoch = 0 if TEMPERATURE_TIMES == 1 else epoch
        left_epoch = 0 if LEFT_TIMES == 1 else epoch
        right_epoch = 0 if RIGHT_TIMES == 1 else epoch
        fraction_epoch = 0 if FRACTION_TIMES == 1 else epoch
        temperature4 = tl.load(
            temperature_ptr + source_epoch * PIXELS + pixel,
            valid_pixel,
            other=0.0,
        )
        gfactor = tl.load(gfactor_ptr + pixel, valid_pixel, other=1.0)
        solid_angle = tl.load(solid_angle_ptr + pixel, valid_pixel, other=0.0)
        hit = tl.load(hit_ptr + pixel, valid_pixel, other=0)
        left_pixel = tl.full((BLOCK_P,), 0, tl.int32) if LEFT_PIXELS == 1 else pixel
        right_pixel = tl.full((BLOCK_P,), 0, tl.int32) if RIGHT_PIXELS == 1 else pixel
        left = tl.load(
            left_ptr + left_epoch * LEFT_STRIDE + left_pixel,
            valid_pixel,
            other=0.0,
        )
        right = tl.load(
            right_ptr + right_epoch * RIGHT_STRIDE + right_pixel,
            valid_pixel,
            other=0.0,
        )
        fraction = tl.load(fraction_ptr + fraction_epoch)
        magnification = left + fraction * (right - left)
        wavelength = tl.load(wavelength_ptr + band, valid_band, other=1.0)
        color = tl.load(color_ptr)
        dimming = tl.load(dimming_ptr)
        area = tl.load(area_ptr)
        temperature = tl.sqrt(tl.sqrt(tl.maximum(temperature4, 0.0)))
        exponent = HC / (
            wavelength[:, None]
            * gfactor[None, :]
            * KB
            * color
            * tl.maximum(temperature[None, :], 1.0e-12)
        )
        brightness = (
            2.0
            * HC
            / (wavelength[:, None] * wavelength[:, None] * wavelength[:, None])
            / tl.extra.cuda.libdevice.expm1(tl.minimum(exponent, 85.0))
            / (color * color * color * color)
            * solid_angle[None, :]
            * dimming
            * 1.0e26
            * area
        )
        brightness = tl.where(
            valid_band[:, None] & valid_pixel[None, :] & hit[None, :],
            brightness,
            0.0,
        )
        lensed = tl.sum(brightness * magnification[None, :], axis=1)
        intrinsic = tl.sum(brightness, axis=1)
        offset = (epoch * BANDS + band) * BLOCKS + block
        tl.store(lensed_ptr + offset, lensed, valid_band)
        tl.store(intrinsic_ptr + offset, intrinsic, valid_band)


def triton_thermal_flux(left, right, fraction, arguments, runtime):
    """Return fused fluxes when the CUDA float32 fast path applies."""

    # libdevice.expm1 is CUDA-specific; HIP retains the portable Torch path.
    if triton is None or torch.version.hip is not None or torch.is_grad_enabled():
        return None
    temperature, gfactor, solid_angle, hit, wavelength, color, dimming, area = arguments
    if temperature.device.type != "cuda" or temperature.dtype != torch.float32:
        return None
    tensors = (
        temperature,
        gfactor,
        solid_angle,
        wavelength,
        color,
        dimming,
        area,
        left,
        right,
        fraction,
    )
    if not all(
        value.device == temperature.device
        and value.dtype == temperature.dtype
        and value.is_contiguous()
        for value in tensors
    ):
        return None
    if hit.device != temperature.device or not hit.is_contiguous():
        return None
    if temperature.ndim != 3 or gfactor.ndim != 2 or wavelength.ndim != 1:
        return None
    if left.ndim != 3 or right.ndim != 3 or fraction.ndim != 1:
        return None
    if (
        temperature.shape[-2:] != gfactor.shape
        or solid_angle.shape != gfactor.shape
        or hit.shape != gfactor.shape
        or any(value.numel() != 1 for value in (color, dimming, area))
    ):
        return None
    pixels = gfactor.numel()
    bands = wavelength.numel()
    if (
        bands > 16
        or pixels < 256
        or any(
            value.shape[-2:] not in (gfactor.shape, (1, 1)) for value in (left, right)
        )
    ):
        return None
    epochs = max(temperature.shape[0], left.shape[0], right.shape[0], fraction.numel())
    if any(
        value not in (1, epochs)
        for value in (
            temperature.shape[0],
            left.shape[0],
            right.shape[0],
            fraction.numel(),
        )
    ):
        return None
    # The full 16-band tile benefits from fewer spatial blocks; narrower
    # chunks retain the lower-register-pressure launch shape.
    block_p = runtime.thermal_flux_block_pixels
    if block_p is None:
        block_p = 512 if bands == 16 else 256
    blocks = triton.cdiv(pixels, block_p)
    # Each compact tile indexes exactly one user-selected pixel block.
    # A nearly full hit mask still uses the complete linear grid.
    compact = _active_source_tiles(hit, block_p) if bands == 16 else None
    if compact is not None and compact[1].numel() == 0:
        empty = torch.zeros(
            (epochs, bands), device=temperature.device, dtype=temperature.dtype
        )
        return empty, empty.clone()
    specialization = (
        temperature.device,
        temperature.dtype,
        pixels,
        bands,
        epochs,
        temperature.shape[0],
        left.shape,
        right.shape,
        fraction.numel(),
        compact is not None,
        block_p,
        runtime.warn_on_compile,
    )
    if specialization not in _PREPARED_SPECIALIZATIONS:
        warn_compilation(
            "thermal spectral photometry",
            backend="Triton",
            device=temperature.device,
            dtype=temperature.dtype,
            enabled=runtime.warn_on_compile,
        )
        _PREPARED_SPECIALIZATIONS.add(specialization)
    allocate = torch.zeros if compact is not None else torch.empty
    partial_lensed = allocate(
        (epochs, bands, blocks), device=temperature.device, dtype=temperature.dtype
    )
    partial_intrinsic = (
        torch.zeros_like(partial_lensed)
        if compact is not None
        else torch.empty_like(partial_lensed)
    )
    tensor_args = (
        temperature,
        gfactor,
        solid_angle,
        hit,
        wavelength,
        color,
        dimming,
        area,
        left,
        right,
        fraction,
        partial_lensed,
        partial_intrinsic,
    )
    shape_args = (
        pixels,
        bands,
        blocks,
        temperature.shape[0],
        left.shape[0],
        right.shape[0],
        fraction.numel(),
        left.shape[1] * left.shape[2],
        right.shape[1] * right.shape[2],
        left.stride(0),
        right.stride(0),
        _H * _C,
        _K_B,
        block_p,
        triton.next_power_of_2(bands),
    )
    if compact is None:
        index = active = hit  # Unused in the linear specialization.
        launch_blocks = blocks
    else:
        index, active = compact[:2]
        launch_blocks = active.numel()
    # The output stride does not depend on how many hit-bearing tiles exist.
    _thermal_flux_partials[(launch_blocks, epochs)](
        *tensor_args,
        index,
        active,
        *shape_args,
        compact is not None,
        num_warps=4,
    )
    return partial_lensed.sum(-1), partial_intrinsic.sum(-1)
