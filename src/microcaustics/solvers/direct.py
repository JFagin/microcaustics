"""Exact chunked point-mass lens equation implemented in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..compile import run_tensor_kernel

if TYPE_CHECKING:
    from ..simulation import MicrolensingSimulation


@dataclass(frozen=True)
class DirectRaytraceDiagnostics:
    """Working-set information from an exact point-mass ray trace."""

    rays: int
    point_masses: int
    star_chunk_size: int
    ray_chunk_size: int
    ray_chunks: int
    star_chunks_per_ray_chunk: int
    requested_backend: str
    effective_backend: str


def _deflection_block(
    ray_x: torch.Tensor,
    ray_y: torch.Tensor,
    star_x: torch.Tensor,
    star_y: torch.Tensor,
    radius_squared: torch.Tensor,
    minimum_radius_squared: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one exact ray-by-star deflection reduction."""

    dx = ray_x[:, None] - star_x[None]
    dy = ray_y[:, None] - star_y[None]
    separation_squared = (dx.square() + dy.square()).clamp_min(minimum_radius_squared)
    weight = radius_squared[None] / separation_squared
    return (dx * weight).sum(dim=1), (dy * weight).sum(dim=1)


def _jacobian_block(
    ray_x: torch.Tensor,
    ray_y: torch.Tensor,
    star_x: torch.Tensor,
    star_y: torch.Tensor,
    radius_squared: torch.Tensor,
    minimum_radius_squared: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one exact ray-by-star Jacobian reduction."""

    dx = ray_x[:, None] - star_x[None]
    dy = ray_y[:, None] - star_y[None]
    separation_squared = (dx.square() + dy.square()).clamp_min(minimum_radius_squared)
    weight = radius_squared[None] / separation_squared.square()
    return (
        ((dy.square() - dx.square()) * weight).sum(dim=1),
        (-2.0 * dx * dy * weight).sum(dim=1),
    )


def _automatic_ray_chunk_size(
    *,
    rays: int,
    stars_per_chunk: int,
    element_size: int,
    max_pair_bytes: int,
) -> int:
    """Choose a conservative chunk for temporary ray-by-star tensors."""

    # dx, dy, squared radius, weight, and two weighted components may coexist
    # in eager execution. The estimate is deliberately conservative and does
    # not rely on allocator-specific tensor reuse.
    bytes_per_pair = 6 * int(element_size)
    affordable = int(max_pair_bytes) // max(1, bytes_per_pair * stars_per_chunk)
    return max(1, min(int(rays), affordable))


@torch.no_grad()
def raytrace_direct(
    simulation: MicrolensingSimulation,
    x_uas,
    y_uas,
    *,
    time_days: float = 0.0,
    star_chunk_size: int = 4096,
    ray_chunk_size: int | None = None,
    max_pair_bytes: int = 256 * 1024**2,
) -> tuple[torch.Tensor, torch.Tensor, DirectRaytraceDiagnostics]:
    """Evaluate the exact single-plane point-mass lens equation.

    Parameters
    ----------
    simulation:
        Macro lens, point-mass field, and resolved runtime.
    x_uas, y_uas:
        Broadcast-compatible lens-plane angular coordinates in
        microarcseconds. The returned source coordinates have their broadcast
        shape.
    time_days:
        Time at which linearly moving point lenses are evaluated.
    star_chunk_size:
        Maximum point lenses included in one pairwise working block.
    ray_chunk_size:
        Maximum rays in one pairwise working block. If omitted, a conservative
        value is derived from ``max_pair_bytes``.
    max_pair_bytes:
        Approximate temporary-memory budget used only when selecting an
        automatic ray chunk.

    Returns
    -------
    source_x_uas, source_y_uas, diagnostics
        Source-plane coordinates and the actual chunk schedule. Compile and
        warmup timing remain the caller's responsibility. Diagnostics report
        the backend that actually evaluated the point-mass blocks.

    Notes
    -----
    The smooth convergence and external shear are applied analytically. The
    compact convergence is represented only by the supplied point masses. It
    is not added a second time as a sheet.
    """

    if int(star_chunk_size) < 1:
        raise ValueError("star_chunk_size must be positive")
    if int(max_pair_bytes) < 1:
        raise ValueError("max_pair_bytes must be positive")
    runtime = simulation.runtime
    x = torch.as_tensor(x_uas, dtype=runtime.dtype, device=runtime.device)
    y = torch.as_tensor(y_uas, dtype=runtime.dtype, device=runtime.device)
    x, y = torch.broadcast_tensors(x, y)
    shape = x.shape
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    n_rays = int(flat_x.numel())
    field = simulation.lens_state(time_days)
    n_stars = len(field)
    star_chunk = min(max(1, int(star_chunk_size)), max(1, n_stars))
    if ray_chunk_size is None:
        ray_chunk = _automatic_ray_chunk_size(
            rays=n_rays,
            stars_per_chunk=star_chunk,
            element_size=flat_x.element_size(),
            max_pair_bytes=int(max_pair_bytes),
        )
    else:
        if int(ray_chunk_size) < 1:
            raise ValueError("ray_chunk_size must be positive")
        ray_chunk = min(n_rays, int(ray_chunk_size)) if n_rays else 1

    source_x = torch.empty_like(flat_x)
    source_y = torch.empty_like(flat_y)
    radius_squared = field.einstein_radius_uas.square()
    macro = simulation.macro_lens
    angle = torch.as_tensor(
        2.0 * macro.shear_angle_rad,
        dtype=runtime.dtype,
        device=runtime.device,
    )
    shear = torch.as_tensor(
        macro.shear,
        dtype=runtime.dtype,
        device=runtime.device,
    )
    gamma1 = shear * torch.cos(angle)
    gamma2 = shear * torch.sin(angle)
    kappa_sheet = torch.as_tensor(
        macro.smooth_convergence,
        dtype=runtime.dtype,
        device=runtime.device,
    )
    bulk_offset_x, bulk_offset_y = simulation.bulk_source_offset_uas(time_days)
    minimum_radius_squared = torch.as_tensor(
        1.0e-30 if runtime.dtype == torch.float32 else 1.0e-300,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    compiled_used = False
    compiled_complete = True

    for ray_start in range(0, n_rays, ray_chunk):
        ray_stop = min(n_rays, ray_start + ray_chunk)
        ray_x = flat_x[ray_start:ray_stop]
        ray_y = flat_y[ray_start:ray_stop]
        alpha_x = torch.zeros_like(ray_x)
        alpha_y = torch.zeros_like(ray_y)
        for star_start in range(0, n_stars, star_chunk):
            star_stop = min(n_stars, star_start + star_chunk)
            (chunk_alpha_x, chunk_alpha_y), used = run_tensor_kernel(
                runtime,
                "direct deflection block",
                _deflection_block,
                ray_x,
                ray_y,
                field.x_uas[star_start:star_stop],
                field.y_uas[star_start:star_stop],
                radius_squared[star_start:star_stop],
                minimum_radius_squared,
            )
            alpha_x.add_(chunk_alpha_x)
            alpha_y.add_(chunk_alpha_y)
            compiled_used = compiled_used or used
            compiled_complete = compiled_complete and used
        alpha_x.add_(kappa_sheet * ray_x + gamma1 * ray_x + gamma2 * ray_y)
        alpha_y.add_(kappa_sheet * ray_y + gamma2 * ray_x - gamma1 * ray_y)
        source_x[ray_start:ray_stop] = ray_x - alpha_x + bulk_offset_x
        source_y[ray_start:ray_stop] = ray_y - alpha_y + bulk_offset_y

    ray_chunks = (n_rays + ray_chunk - 1) // ray_chunk if n_rays else 0
    star_chunks = (n_stars + star_chunk - 1) // star_chunk if n_stars else 0
    diagnostics = DirectRaytraceDiagnostics(
        rays=n_rays,
        point_masses=n_stars,
        star_chunk_size=star_chunk,
        ray_chunk_size=ray_chunk,
        ray_chunks=ray_chunks,
        star_chunks_per_ray_chunk=star_chunks,
        requested_backend=runtime.backend.value,
        effective_backend=(
            "torch-compile" if compiled_used and compiled_complete else "torch-eager"
        ),
    )
    return source_x.reshape(shape), source_y.reshape(shape), diagnostics


@torch.no_grad()
def jacobian_determinant_direct(
    simulation: MicrolensingSimulation,
    x_uas,
    y_uas,
    *,
    time_days: float = 0.0,
    star_chunk_size: int = 4096,
    ray_chunk_size: int | None = None,
    max_pair_bytes: int = 256 * 1024**2,
) -> tuple[torch.Tensor, DirectRaytraceDiagnostics]:
    """Evaluate the exact determinant of the lens-equation Jacobian.

    Inputs and chunk settings follow :func:`raytrace_direct`. The returned
    determinant has the broadcast shape of ``x_uas`` and ``y_uas``. Critical
    curves are the zero contours of this scalar field. Mapping those curves
    through :func:`raytrace_direct` produces source-plane caustics.
    """

    if int(star_chunk_size) < 1:
        raise ValueError("star_chunk_size must be positive")
    if int(max_pair_bytes) < 1:
        raise ValueError("max_pair_bytes must be positive")
    runtime = simulation.runtime
    x = torch.as_tensor(x_uas, dtype=runtime.dtype, device=runtime.device)
    y = torch.as_tensor(y_uas, dtype=runtime.dtype, device=runtime.device)
    x, y = torch.broadcast_tensors(x, y)
    shape = x.shape
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    n_rays = int(flat_x.numel())
    field = simulation.lens_state(time_days)
    n_stars = len(field)
    star_chunk = min(max(1, int(star_chunk_size)), max(1, n_stars))
    if ray_chunk_size is None:
        ray_chunk = _automatic_ray_chunk_size(
            rays=n_rays,
            stars_per_chunk=star_chunk,
            element_size=flat_x.element_size(),
            max_pair_bytes=int(max_pair_bytes),
        )
    else:
        if int(ray_chunk_size) < 1:
            raise ValueError("ray_chunk_size must be positive")
        ray_chunk = min(n_rays, int(ray_chunk_size)) if n_rays else 1

    determinant = torch.empty_like(flat_x)
    radius_squared = field.einstein_radius_uas.square()
    macro = simulation.macro_lens
    angle = torch.as_tensor(
        2.0 * macro.shear_angle_rad,
        dtype=runtime.dtype,
        device=runtime.device,
    )
    shear = torch.as_tensor(macro.shear, dtype=runtime.dtype, device=runtime.device)
    gamma1 = shear * torch.cos(angle)
    gamma2 = shear * torch.sin(angle)
    kappa_sheet = torch.as_tensor(
        macro.smooth_convergence,
        dtype=runtime.dtype,
        device=runtime.device,
    )
    minimum_radius_squared = torch.as_tensor(
        1.0e-30 if runtime.dtype == torch.float32 else 1.0e-300,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    compiled_used = False
    compiled_complete = True

    for ray_start in range(0, n_rays, ray_chunk):
        ray_stop = min(n_rays, ray_start + ray_chunk)
        ray_x = flat_x[ray_start:ray_stop]
        ray_y = flat_y[ray_start:ray_stop]
        point_xx = torch.zeros_like(ray_x)
        point_xy = torch.zeros_like(ray_x)
        for star_start in range(0, n_stars, star_chunk):
            star_stop = min(n_stars, star_start + star_chunk)
            (chunk_xx, chunk_xy), used = run_tensor_kernel(
                runtime,
                "direct Jacobian block",
                _jacobian_block,
                ray_x,
                ray_y,
                field.x_uas[star_start:star_stop],
                field.y_uas[star_start:star_stop],
                radius_squared[star_start:star_stop],
                minimum_radius_squared,
            )
            point_xx.add_(chunk_xx)
            point_xy.add_(chunk_xy)
            compiled_used = compiled_used or used
            compiled_complete = compiled_complete and used
        alpha_xx = kappa_sheet + gamma1 + point_xx
        alpha_yy = kappa_sheet - gamma1 - point_xx
        alpha_xy = gamma2 + point_xy
        determinant[ray_start:ray_stop] = (1.0 - alpha_xx) * (
            1.0 - alpha_yy
        ) - alpha_xy.square()

    ray_chunks = (n_rays + ray_chunk - 1) // ray_chunk if n_rays else 0
    star_chunks = (n_stars + star_chunk - 1) // star_chunk if n_stars else 0
    diagnostics = DirectRaytraceDiagnostics(
        rays=n_rays,
        point_masses=n_stars,
        star_chunk_size=star_chunk,
        ray_chunk_size=ray_chunk,
        ray_chunks=ray_chunks,
        star_chunks_per_ray_chunk=star_chunks,
        requested_backend=runtime.backend.value,
        effective_backend=(
            "torch-compile" if compiled_used and compiled_complete else "torch-eager"
        ),
    )
    return determinant.reshape(shape), diagnostics
