"""Primary equatorial Kerr photon transfer.

This module is kept out of the top-level public API until the complete static
transfer, pinhole repair, and SIM5 regression suite have migrated. The pure
Torch calculation is already isolated here so it can be tested independently.
"""

from __future__ import annotations

import math
import threading
import warnings
from dataclasses import dataclass

import torch

from ..runtime import RuntimeCapabilities, _torch_compile_supported
from .geodesics import (
    invert_radial_motion,
    kerr_radial_root_parts,
    polar_mino_time,
)
from .kerr import circular_disk_gfactor, kerr_isco_radius
from .transfer import ObserverScreen, ObserverTransfer

_COMPILED_PRIMARY: dict[tuple[object, ...], object] = {}
_COMPILED_PRIMARY_LOCK = threading.Lock()


@dataclass(frozen=True)
class PrimaryKerrTrace:
    """Primary image plus constants needed by observer-coordinate tracing."""

    transfer: ObserverTransfer
    photon_lambda: torch.Tensor
    carter_eta: torch.Tensor
    mino_time: torch.Tensor
    radial_root_count: torch.Tensor
    radial_root_real: torch.Tensor
    radial_root_imag: torch.Tensor
    repaired_pixels: int = 0
    interpolated_mask: torch.Tensor | None = None


def _trace_primary_tensors(
    screen_x: torch.Tensor,
    screen_y: torch.Tensor,
    spin: torch.Tensor,
    inclination_rad: torch.Tensor,
    disk_inner_rg: torch.Tensor,
    disk_outer_rg: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Pure tensor kernel shared by eager and compiled execution."""

    alpha = -screen_x
    beta = -screen_y
    photon_lambda = -alpha * torch.sin(inclination_rad)
    carter_eta = (
        (alpha.square() - spin.square()) * torch.cos(inclination_rad).square()
        + beta.square()
    )
    mino_time, polar_valid = polar_mino_time(
        spin,
        carter_eta,
        photon_lambda,
        beta,
        inclination_rad,
    )
    root_real, root_imag = kerr_radial_root_parts(
        spin,
        carter_eta,
        photon_lambda,
    )
    radius, radial_valid, root_count = invert_radial_motion(
        spin,
        root_real,
        root_imag,
        mino_time,
    )
    gfactor = circular_disk_gfactor(radius, spin, photon_lambda)
    hit = (
        polar_valid
        & radial_valid
        & torch.isfinite(radius)
        & torch.isfinite(gfactor)
        & (radius >= disk_inner_rg)
        & (radius <= disk_outer_rg)
        & (gfactor > 0.0)
    )
    return (
        hit,
        radius,
        gfactor,
        photon_lambda,
        carter_eta,
        mino_time,
        polar_valid,
        root_count,
        root_real,
        root_imag,
    )


def clear_compiled_primary_cache() -> None:
    """Forget cached compiled primary kernels without touching Torch caches."""

    with _COMPILED_PRIMARY_LOCK:
        _COMPILED_PRIMARY.clear()


def _primary_calculation(
    screen: ObserverScreen,
    *,
    compile_solver: bool,
    compile_mode: str,
    fallback_to_eager: bool,
):
    if not compile_solver:
        return _trace_primary_tensors, "torch eager"
    if not _torch_compile_supported(
        screen.x_rg.device,
        RuntimeCapabilities.detect(),
    ):
        if not fallback_to_eager:
            raise RuntimeError(
                "torch.compile is unavailable or its required native "
                "compiler toolchain could not be found"
            )
        return (
            _trace_primary_tensors,
            "torch eager (torch.compile toolchain unavailable)",
        )
    key = (
        str(screen.x_rg.device),
        screen.x_rg.dtype,
        screen.shape,
        str(compile_mode),
    )
    with _COMPILED_PRIMARY_LOCK:
        calculation = _COMPILED_PRIMARY.get(key)
        cache_hit = calculation is not None
        if calculation is None:
            calculation = torch.compile(
                _trace_primary_tensors,
                mode=compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            _COMPILED_PRIMARY[key] = calculation
    return calculation, (
        f"torch.compile({compile_mode}, cache={'hit' if cache_hit else 'miss'})"
    )


def trace_primary_equatorial(
    screen: ObserverScreen,
    *,
    spin: float | torch.Tensor,
    inclination_deg: float | torch.Tensor,
    disk_inner_rg: float | torch.Tensor | None = None,
    disk_outer_rg: float | torch.Tensor = 50.0,
    repair_isolated_misses: bool = True,
    repair_minimum_neighbors: int = 5,
    repair_max_passes: int = 1,
    compile_solver: bool = False,
    compile_mode: str = "reduce-overhead",
    fallback_to_eager: bool = True,
) -> PrimaryKerrTrace:
    """Trace first equatorial intersections from a distant observer.

    The screen uses the same project-camera to Bardeen convention as the
    validated paper and SIM5 bridge. This stage calculates light bending,
    capture, the primary disk image, and circular-orbit frequency shifts. It
    does not yet calculate coordinate arrival time or emission azimuth.
    """

    device, dtype = screen.x_rg.device, screen.x_rg.dtype
    spin_tensor = torch.as_tensor(spin, device=device, dtype=dtype).reshape(())
    inclination = torch.as_tensor(
        inclination_deg,
        device=device,
        dtype=dtype,
    ).reshape(())
    if bool(~torch.isfinite(inclination)) or not 0.0 < float(
        inclination.detach().cpu()
    ) < 90.0:
        raise ValueError("inclination_deg must lie strictly between 0 and 90")
    if bool(~torch.isfinite(spin_tensor)) or not -0.998 <= float(
        spin_tensor.detach().cpu()
    ) <= 0.998:
        raise ValueError("spin must lie in [-0.998, 0.998]")
    inclination_rad = inclination * (math.pi / 180.0)
    inner = (
        kerr_isco_radius(spin_tensor)
        if disk_inner_rg is None
        else torch.as_tensor(disk_inner_rg, device=device, dtype=dtype).reshape(())
    )
    outer = torch.as_tensor(
        disk_outer_rg,
        device=device,
        dtype=dtype,
    ).reshape(())
    if not 0.0 < float(inner.detach().cpu()) < float(outer.detach().cpu()):
        raise ValueError("require 0 < disk_inner_rg < disk_outer_rg")
    if not 1 <= int(repair_minimum_neighbors) <= 8:
        raise ValueError("repair_minimum_neighbors must lie in [1, 8]")
    if int(repair_max_passes) < 1:
        raise ValueError("repair_max_passes must be positive")

    calculation, execution = _primary_calculation(
        screen,
        compile_solver=bool(compile_solver),
        compile_mode=compile_mode,
        fallback_to_eager=bool(fallback_to_eager),
    )
    try:
        outputs = calculation(
            screen.x_rg,
            screen.y_rg,
            spin_tensor,
            inclination_rad,
            inner,
            outer,
        )
    except Exception as error:
        if not compile_solver or not fallback_to_eager:
            raise
        clear_compiled_primary_cache()
        warnings.warn(
            "compiled primary Kerr calculation failed. Using eager Torch: "
            f"{type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        calculation = _trace_primary_tensors
        execution = f"torch eager fallback after {type(error).__name__}"
        outputs = calculation(
            screen.x_rg,
            screen.y_rg,
            spin_tensor,
            inclination_rad,
            inner,
            outer,
        )
    (
        hit,
        radius,
        gfactor,
        photon_lambda,
        carter_eta,
        mino_time,
        polar_valid,
        root_count,
        root_real,
        root_imag,
    ) = outputs
    repaired = torch.zeros((), device=device, dtype=torch.long)
    interpolated_mask = torch.zeros_like(hit)
    if repair_isolated_misses:
        for _ in range(int(repair_max_passes)):
            hit_float = hit.to(dtype)
            weighted_radius = torch.where(hit, radius, torch.zeros_like(radius))
            pooled = (
                torch.nn.functional.avg_pool2d(
                    torch.stack((hit_float, weighted_radius), dim=0).unsqueeze(0),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    count_include_pad=True,
                )[0]
                * 9.0
            )
            neighbor_count = pooled[0] - hit_float
            neighbor_radius_sum = pooled[1] - weighted_radius
            candidate = (~hit) & (
                neighbor_count >= float(repair_minimum_neighbors)
            )
            repaired_radius = neighbor_radius_sum / neighbor_count.clamp_min(1.0)
            repaired_gfactor = circular_disk_gfactor(
                repaired_radius,
                spin_tensor,
                photon_lambda,
            )
            candidate = (
                candidate
                & torch.isfinite(repaired_radius)
                & torch.isfinite(repaired_gfactor)
                & (repaired_radius >= inner)
                & (repaired_radius <= outer)
                & (repaired_gfactor > 0.0)
            )
            if not bool(torch.any(candidate)):
                break
            hit = hit | candidate
            radius = torch.where(candidate, repaired_radius, radius)
            gfactor = torch.where(candidate, repaired_gfactor, gfactor)
            repaired = repaired + candidate.sum()
            interpolated_mask = interpolated_mask | candidate
    nan = torch.full_like(radius, float("nan"))
    transfer = ObserverTransfer.from_screen(
        screen,
        torch.where(hit, radius, nan),
        torch.where(hit, gfactor, nan),
        hit,
        metadata={
            "backend": "analytic_separated_kerr",
            "image_order": "primary",
            "spin": float(spin_tensor.detach().cpu()),
            "inclination_deg": float(inclination.detach().cpu()),
            "disk_inner_rg": float(inner.detach().cpu()),
            "disk_outer_rg": float(outer.detach().cpu()),
            "observer_coordinates": False,
            "execution": execution,
            "pinhole_repair": bool(repair_isolated_misses),
            "pinhole_repair_max_passes": int(repair_max_passes),
            "repaired_pixels": int(repaired.detach().cpu()),
        },
    )
    return PrimaryKerrTrace(
        transfer,
        photon_lambda,
        carter_eta,
        torch.where(polar_valid, mino_time, nan),
        root_count,
        root_real,
        root_imag,
        int(repaired.detach().cpu()),
        interpolated_mask,
    )
