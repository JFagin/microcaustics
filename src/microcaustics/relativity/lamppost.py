"""Primary axial-lamppost transfer to an equatorial Kerr disk."""

from __future__ import annotations

import math
import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from ..runtime import RuntimeCapabilities, _torch_compile_supported
from .coordinates import _polar_integrals
from .geodesics import (
    incoming_radial_mino_time,
    inner_four_root_phase,
    kerr_radial_root_parts,
    lamppost_radial_radius,
    polar_mino_time,
    radial_root_real_count,
)
from .kerr import (
    circular_disk_gfactor,
    circular_disk_zamo_lorentz_factor,
    kerr_isco_radius,
)

_QUADRATURE_CACHE: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}
_COMPILED_LAMPPOST: dict[tuple[object, ...], object] = {}
_COMPILED_LAMPPOST_LOCK = threading.Lock()


@dataclass(frozen=True)
class AxisLamppostRayTransfer:
    """One-dimensional launch-angle rays from an axial point source."""

    launch_angle_rad: torch.Tensor
    initial_radial_sign: torch.Tensor
    hit: torch.Tensor
    radius_rg: torch.Tensor
    travel_time_rg: torch.Tensor
    gfactor_lamp_to_disk: torch.Tensor
    carter_eta: torch.Tensor
    source_mino_time: torch.Tensor
    disk_mino_time: torch.Tensor
    spin: float
    source_height_rg: float
    disk_inner_rg: float
    disk_outer_rg: float
    runtime_s: float
    chunk_size: int
    compile_warmup_s: float = 0.0
    execution: str = "torch eager"


@dataclass(frozen=True)
class AxisLamppostProfile:
    """Conservative annular lamppost illumination and delay profile."""

    radius_rg: torch.Tensor
    lamp_delay_rg: torch.Tensor
    illumination: torch.Tensor
    gfactor_lamp_to_disk: torch.Tensor
    hit_fraction: torch.Tensor
    rays: AxisLamppostRayTransfer
    lamp_g_power: float

    def interpolate(
        self,
        radius_rg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpolate delay, illumination, and frequency shift onto a disk."""

        radius = torch.as_tensor(radius_rg)
        coordinates = self.radius_rg.to(device=radius.device, dtype=radius.dtype)
        return tuple(
            _interp1d_clamped(
                radius,
                coordinates,
                value.to(device=radius.device, dtype=radius.dtype),
            )
            for value in (
                self.lamp_delay_rg,
                self.illumination,
                self.gfactor_lamp_to_disk,
            )
        )


def _gauss_legendre(
    order: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(order) < 8:
        raise ValueError("quadrature_order must be at least 8")
    key = (int(order), str(device), dtype)
    cached = _QUADRATURE_CACHE.get(key)
    if cached is None:
        nodes, weights = np.polynomial.legendre.leggauss(int(order))
        cached = (
            torch.as_tensor(0.5 * (nodes + 1.0), device=device, dtype=dtype),
            torch.as_tensor(0.5 * weights, device=device, dtype=dtype),
        )
        _QUADRATURE_CACHE[key] = cached
    return cached


def _safe_denominator(value: torch.Tensor) -> torch.Tensor:
    tiny = math.sqrt(torch.finfo(value.dtype).tiny)
    return torch.where(
        value >= 0.0,
        value.clamp_min(tiny),
        value.clamp_max(-tiny),
    )


def _radial_interval_time(
    spin: torch.Tensor,
    root_real: torch.Tensor,
    root_imag: torch.Tensor,
    phase_start: torch.Tensor,
    phase_end: torch.Tensor,
    inner_four_real: torch.Tensor,
    *,
    quadrature_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = _gauss_legendre(
        quadrature_order,
        device=phase_start.device,
        dtype=phase_start.dtype,
    )
    span = (phase_end - phase_start).clamp_min(0.0)
    phase = phase_start[:, None] + span[:, None] * nodes[None]
    measure = span[:, None] * weights[None]
    count, order = phase.shape
    expanded_spin = spin[:, None].expand(count, order).reshape(-1)
    expanded_real = root_real[:, None, :].expand(count, order, 4).reshape(-1, 4)
    expanded_imag = root_imag[:, None, :].expand(count, order, 4).reshape(-1, 4)
    expanded_inner = inner_four_real[:, None].expand(count, order).reshape(-1)
    radius, valid = lamppost_radial_radius(
        expanded_spin,
        expanded_real,
        expanded_imag,
        phase.reshape(-1),
        expanded_inner,
    )
    radius = radius.reshape(count, order)
    valid = valid.reshape(count, order)
    local_spin = spin[:, None]
    delta = _safe_denominator(
        radius.square() - 2.0 * radius + local_spin.square()
    )
    radial_momentum = radius.square() + local_spin.square()
    time_integrand = radius.square() + 2.0 * radius * radial_momentum / delta
    radial_time = (measure * time_integrand).sum(-1)
    return radial_time, valid.all(-1) & torch.isfinite(radial_time)


def _trace_chunk(
    launch_angle: torch.Tensor,
    initial_radial_sign: torch.Tensor,
    spin: torch.Tensor,
    source_radius: torch.Tensor,
    theta_source: torch.Tensor,
    disk_inner: torch.Tensor,
    disk_outer: torch.Tensor,
    *,
    quadrature_order: int,
) -> tuple[torch.Tensor, ...]:
    spin_ray = spin.expand_as(launch_angle)
    photon_lambda = torch.zeros_like(launch_angle)
    sigma_source = (
        source_radius.square()
        + spin.square() * torch.cos(theta_source).square()
    )
    minus_gtt_source = 1.0 - 2.0 * source_radius / sigma_source
    carter_eta = (
        sigma_source * torch.sin(launch_angle).square() / minus_gtt_source
        - spin.square() * torch.cos(theta_source).square()
    )
    beta_sign = -torch.ones_like(launch_angle)
    polar_time, polar_valid = polar_mino_time(
        spin_ray,
        carter_eta,
        photon_lambda,
        beta_sign,
        theta_source,
    )
    root_real, root_imag = kerr_radial_root_parts(
        spin_ray,
        carter_eta,
        photon_lambda,
    )
    source_radius_ray = source_radius.expand_as(launch_angle)
    source_time_outer, source_valid_outer = incoming_radial_mino_time(
        spin_ray,
        root_real,
        root_imag,
        source_radius_ray,
    )
    root_counts = radial_root_real_count(root_real, root_imag)
    _, q2, q3, _ = root_real.unbind(dim=-1)
    root_scale = torch.maximum(
        torch.ones_like(source_radius_ray),
        torch.hypot(root_real, root_imag).amax(-1),
    )
    real_tolerance = math.sqrt(torch.finfo(launch_angle.dtype).eps) * root_scale
    inner_four_real = (
        (root_counts == 4)
        & (source_radius_ray >= q2 - real_tolerance)
        & (source_radius_ray <= q3 + real_tolerance)
    )
    source_time_inner, source_valid_inner = inner_four_root_phase(
        root_real,
        source_radius_ray,
    )
    source_time = torch.where(
        inner_four_real,
        source_time_inner,
        source_time_outer,
    )
    source_valid = torch.where(
        inner_four_real,
        source_valid_inner,
        source_valid_outer,
    )
    source_phase = -initial_radial_sign * source_time
    disk_phase = source_phase + polar_time
    radius, radius_valid = lamppost_radial_radius(
        spin_ray,
        root_real,
        root_imag,
        disk_phase,
        inner_four_real,
    )
    radial_time, finite_path = _radial_interval_time(
        spin_ray,
        root_real,
        root_imag,
        source_phase,
        disk_phase,
        inner_four_real,
        quadrature_order=quadrature_order,
    )
    gtheta, _, polar_coordinate_time = _polar_integrals(
        spin_ray,
        carter_eta,
        photon_lambda,
        beta_sign,
        theta_source,
        order=quadrature_order,
    )
    polar_tolerance = 5.0e-4 if launch_angle.dtype == torch.float32 else 1.0e-8
    polar_endpoint_valid = (
        torch.isfinite(gtheta)
        & torch.isfinite(polar_coordinate_time)
        & (polar_coordinate_time >= 0.0)
        & (
            (gtheta - polar_time).abs()
            <= polar_tolerance
            * torch.maximum(torch.ones_like(polar_time), polar_time.abs())
        )
    )
    coordinate_time = radial_time + spin_ray.square() * polar_coordinate_time
    disk_g_to_infinity = circular_disk_gfactor(
        radius,
        spin_ray,
        photon_lambda,
    )
    lamp_to_disk_gfactor = torch.sqrt(minus_gtt_source) / disk_g_to_infinity
    hit = (
        polar_valid
        & polar_endpoint_valid
        & source_valid
        & radius_valid
        & finite_path
        & torch.isfinite(radius)
        & torch.isfinite(coordinate_time)
        & torch.isfinite(lamp_to_disk_gfactor)
        & (radius >= disk_inner)
        & (radius <= disk_outer)
        & (lamp_to_disk_gfactor > 0.0)
        & ((initial_radial_sign < 0.0) | (disk_phase <= 0.0))
    )
    return (
        radius,
        coordinate_time,
        lamp_to_disk_gfactor,
        carter_eta,
        source_phase,
        disk_phase,
        hit,
    )


def clear_compiled_lamppost_cache() -> None:
    """Forget package-cached compiled lamppost callables."""

    with _COMPILED_LAMPPOST_LOCK:
        _COMPILED_LAMPPOST.clear()


def trace_axis_lamppost(
    *,
    spin: float,
    source_height_rg: float,
    disk_outer_rg: float,
    nalpha: int = 4096,
    theta0: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    quadrature_order: int = 24,
    chunk_size: int = 8192,
    compile_solver: bool = False,
    compile_mode: str = "reduce-overhead",
    fallback_to_eager: bool = True,
) -> AxisLamppostRayTransfer:
    """Trace isotropic rays from a static axial source to the first disk hit."""

    device = torch.device(device)
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("lamppost tracing requires float32 or float64")
    if int(nalpha) < 16:
        raise ValueError("nalpha must be at least 16")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if not 0.0 <= float(theta0) < 0.5 * math.pi:
        raise ValueError("theta0 must lie in [0, pi/2)")
    spin_tensor = torch.as_tensor(spin, device=device, dtype=dtype)
    source_radius = torch.as_tensor(source_height_rg, device=device, dtype=dtype)
    theta_source = torch.as_tensor(theta0, device=device, dtype=dtype)
    disk_inner = kerr_isco_radius(spin_tensor)
    disk_outer = torch.as_tensor(disk_outer_rg, device=device, dtype=dtype)
    horizon = 1.0 + torch.sqrt((1.0 - spin_tensor.square()).clamp_min(0.0))
    if float(source_radius.detach().cpu()) <= float(horizon.detach().cpu()):
        raise ValueError("source_height_rg must lie outside the Kerr horizon")
    if float(disk_outer.detach().cpu()) <= float(disk_inner.detach().cpu()):
        raise ValueError("disk_outer_rg must exceed the ISCO")
    sigma_source = (
        source_radius.square()
        + spin_tensor.square() * torch.cos(theta_source).square()
    )
    if float((1.0 - 2.0 * source_radius / sigma_source).detach().cpu()) <= 0.0:
        raise ValueError("a static lamppost requires -g_tt > 0 at its source")

    launch_angle = torch.linspace(
        0.0,
        math.pi - 2.0e-5,
        2 * int(nalpha) - 1,
        device=device,
        dtype=dtype,
    )
    radial_sign = torch.where(
        launch_angle <= 0.5 * math.pi,
        -torch.ones_like(launch_angle),
        torch.ones_like(launch_angle),
    )
    outputs = [
        torch.full_like(launch_angle, float("nan")) for _ in range(6)
    ]
    hit = torch.zeros_like(launch_angle, dtype=torch.bool)
    solver = _trace_chunk
    effective_chunk_size = int(chunk_size)
    compile_warmup = 0.0
    execution = "torch eager"
    if compile_solver:
        compile_supported = _torch_compile_supported(
            device,
            RuntimeCapabilities.detect(),
        )
        if not compile_supported:
            if not fallback_to_eager:
                raise RuntimeError(
                    "torch.compile is unavailable or its required native "
                    "compiler toolchain could not be found"
                )
            # An unavailable host compiler is an expected portability case,
            # not a failed numerical kernel.  Record it in provenance without
            # emitting a warning on every notebook execution.
            execution = "torch eager (torch.compile toolchain unavailable)"
        else:
            # The complete axial launch grid is small. One static call avoids
            # a distinct compiled tail shape and makes compile cost explicit.
            effective_chunk_size = int(launch_angle.numel())
            effective_compile_mode = str(compile_mode)
            if device.type == "cuda":
                # This calculation intentionally reuses cached quadrature
                # tensors inside one invocation. CUDA Graph replay treats
                # those internal outputs as overwritten. Retain Inductor
                # fusion while selecting the corresponding non-graph mode.
                if effective_compile_mode == "reduce-overhead":
                    effective_compile_mode = "default"
                elif effective_compile_mode == "max-autotune":
                    effective_compile_mode = "max-autotune-no-cudagraphs"
            key = (
                str(device),
                dtype,
                effective_chunk_size,
                int(quadrature_order),
                effective_compile_mode,
            )
            with _COMPILED_LAMPPOST_LOCK:
                candidate = _COMPILED_LAMPPOST.get(key)
            cache_hit = candidate is not None
            if candidate is None:
                candidate = torch.compile(
                    _trace_chunk,
                    mode=effective_compile_mode,
                    fullgraph=False,
                    dynamic=False,
                )
                warm_started = time.perf_counter()
                try:
                    if device.type == "cuda":
                        mark_step = getattr(
                            getattr(torch, "compiler", None),
                            "cudagraph_mark_step_begin",
                            None,
                        )
                        if callable(mark_step):
                            mark_step()
                    candidate(
                        launch_angle,
                        radial_sign,
                        spin_tensor,
                        source_radius,
                        theta_source,
                        disk_inner,
                        disk_outer,
                        quadrature_order=quadrature_order,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    compile_warmup = time.perf_counter() - warm_started
                    with _COMPILED_LAMPPOST_LOCK:
                        _COMPILED_LAMPPOST[key] = candidate
                except Exception as error:
                    if not fallback_to_eager:
                        raise
                    warnings.warn(
                        "compiled lamppost calculation failed. Using eager "
                        f"Torch: {type(error).__name__}: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    candidate = None
                    effective_chunk_size = int(chunk_size)
            if candidate is not None:
                solver = candidate
                execution = (
                    f"torch.compile({effective_compile_mode}, "
                    f"cache={'hit' if cache_hit else 'miss'})"
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for start in range(0, launch_angle.numel(), effective_chunk_size):
        stop = min(start + effective_chunk_size, launch_angle.numel())
        if compile_solver and device.type == "cuda":
            mark_step = getattr(
                getattr(torch, "compiler", None),
                "cudagraph_mark_step_begin",
                None,
            )
            if callable(mark_step):
                mark_step()
        chunk = solver(
            launch_angle[start:stop],
            radial_sign[start:stop],
            spin_tensor,
            source_radius,
            theta_source,
            disk_inner,
            disk_outer,
            quadrature_order=quadrature_order,
        )
        for destination, value in zip(outputs, chunk[:-1], strict=True):
            destination[start:stop] = value
        hit[start:stop] = chunk[-1]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    radius, travel, gfactor, eta, source_time, disk_time = outputs
    if bool(torch.any(hit)):
        travel = torch.where(hit, travel - travel[hit].min(), travel)
    nan = torch.full_like(radius, float("nan"))
    return AxisLamppostRayTransfer(
        launch_angle,
        radial_sign,
        hit,
        torch.where(hit, radius, nan),
        torch.where(hit, travel, nan),
        torch.where(hit, gfactor, nan),
        eta,
        torch.where(torch.isfinite(source_time), source_time, nan),
        torch.where(hit, disk_time, nan),
        float(spin),
        float(source_height_rg),
        float(disk_inner.detach().cpu()),
        float(disk_outer_rg),
        float(runtime),
        int(effective_chunk_size),
        float(compile_warmup),
        execution,
    )


def _interp1d_clamped(
    query: torch.Tensor,
    coordinates: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    flat = query.reshape(-1)
    right = torch.searchsorted(coordinates, flat, right=True).clamp(
        1,
        coordinates.numel() - 1,
    )
    left = right - 1
    fraction = (flat - coordinates[left]) / (
        coordinates[right] - coordinates[left]
    ).clamp_min(torch.finfo(values.dtype).tiny)
    result = values[left] + fraction * (values[right] - values[left])
    result = torch.where(flat <= coordinates[0], values[0], result)
    result = torch.where(flat >= coordinates[-1], values[-1], result)
    return result.reshape(query.shape)


def _additive_continuation(
    values: torch.Tensor,
    reached: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    valid = reached & torch.isfinite(values)
    indices = torch.where(valid)[0]
    if indices.numel() == 0:
        return baseline
    first, last = indices[0], indices[-1]
    result = torch.where(valid, values, baseline)
    grid = torch.arange(values.numel(), device=values.device)
    result = torch.where(
        grid < first,
        baseline + (values[first] - baseline[first]),
        result,
    )
    return torch.where(grid > last, baseline + (values[last] - baseline[last]), result)


def _multiplicative_continuation(
    values: torch.Tensor,
    reached: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    valid = reached & torch.isfinite(values) & (values > 0.0)
    indices = torch.where(valid)[0]
    if indices.numel() == 0:
        return baseline
    first, last = indices[0], indices[-1]
    tiny = torch.finfo(values.dtype).tiny
    low = baseline * (values[first] / baseline[first].clamp_min(tiny))
    high = baseline * (values[last] / baseline[last].clamp_min(tiny))
    result = torch.where(valid, values, baseline)
    grid = torch.arange(values.numel(), device=values.device)
    result = torch.where(grid < first, low, result)
    return torch.where(grid > last, high, result)


def axis_lamppost_profile(
    *,
    spin: float,
    source_height_rg: float | None = None,
    height_above_isco_rg: float | None = None,
    disk_outer_rg: float,
    nalpha: int = 4096,
    radial_bins: int = 512,
    theta0: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    lamp_g_power: float = 2.0,
    quadrature_order: int = 24,
    chunk_size: int = 8192,
    compile_solver: bool = False,
    compile_mode: str = "reduce-overhead",
    fallback_to_eager: bool = True,
) -> AxisLamppostProfile:
    """Build a conservative, no-splat annular GR lamppost profile.

    Supply either the absolute Boyer--Lindquist ``source_height_rg`` or the
    more physical ``height_above_isco_rg``. The latter is converted with the
    package's differentiable Kerr ISCO function.
    """

    if int(radial_bins) < 8:
        raise ValueError("radial_bins must be at least 8")
    if not math.isfinite(lamp_g_power):
        raise ValueError("lamp_g_power must be finite")
    if (source_height_rg is None) == (height_above_isco_rg is None):
        raise ValueError(
            "supply exactly one of source_height_rg or height_above_isco_rg"
        )
    if source_height_rg is None:
        from .kerr import lamppost_source_height_rg

        source_height_rg = float(
            lamppost_source_height_rg(spin, height_above_isco_rg).detach().cpu()
        )
    else:
        source_height_rg = float(source_height_rg)
    rays = trace_axis_lamppost(
        spin=spin,
        source_height_rg=source_height_rg,
        disk_outer_rg=disk_outer_rg,
        nalpha=nalpha,
        theta0=theta0,
        device=device,
        dtype=dtype,
        quadrature_order=quadrature_order,
        chunk_size=chunk_size,
        compile_solver=compile_solver,
        compile_mode=compile_mode,
        fallback_to_eager=fallback_to_eager,
    )
    edges = torch.linspace(
        rays.disk_inner_rg,
        disk_outer_rg,
        int(radial_bins) + 1,
        device=rays.radius_rg.device,
        dtype=rays.radius_rg.dtype,
    )
    centers = 0.5 * (edges[1:] + edges[:-1])
    widths = edges[1:] - edges[:-1]
    valid = rays.hit & torch.isfinite(rays.radius_rg)
    if int(valid.sum()) < 4:
        raise RuntimeError("axis lamppost produced too few disk hits")
    order = torch.argsort(rays.radius_rg[valid])
    radius = rays.radius_rg[valid][order]
    angle = rays.launch_angle_rad[valid][order]
    delay = rays.travel_time_rg[valid][order]
    gfactor = rays.gfactor_lamp_to_disk[valid][order]
    distinct = torch.ones_like(radius, dtype=torch.bool)
    distinct[1:] = (radius[1:] - radius[:-1]) > (
        16.0
        * torch.finfo(dtype).eps
        * torch.maximum(torch.ones_like(radius[1:]), radius[1:].abs())
    )
    radius, angle, delay, gfactor = (
        value[distinct] for value in (radius, angle, delay, gfactor)
    )
    reached_centers = (centers >= radius[0]) & (centers <= radius[-1])
    reached_edges = (edges >= radius[0]) & (edges <= radius[-1])
    edge_angle = _interp1d_clamped(edges, radius, angle)
    center_delay = _interp1d_clamped(centers, radius, delay)
    center_gfactor = _interp1d_clamped(centers, radius, gfactor)
    solid_angle = 2.0 * math.pi * torch.abs(
        torch.cos(edge_angle[1:]) - torch.cos(edge_angle[:-1])
    )
    reached = reached_edges[1:] & reached_edges[:-1] & reached_centers
    local_spin = torch.as_tensor(spin, device=centers.device, dtype=centers.dtype)
    safe_radius = centers.clamp_min(torch.finfo(dtype).eps)
    delta = (
        safe_radius.square() - 2.0 * safe_radius + local_spin.square()
    ).clamp_min(torch.finfo(dtype).eps)
    metric_rr = safe_radius.square() / delta
    metric_phiphi = (
        safe_radius.square()
        + local_spin.square()
        + 2.0 * local_spin.square() / safe_radius
    )
    stationary_area = 2.0 * math.pi * torch.sqrt(
        (metric_rr * metric_phiphi).clamp_min(torch.finfo(dtype).tiny)
    ) * widths
    proper_area = (
        circular_disk_zamo_lorentz_factor(safe_radius, local_spin)
        * stationary_area
    )
    illumination = (
        solid_angle
        * center_gfactor.clamp_min(1.0e-12).pow(float(lamp_g_power))
        / proper_area
    )
    flat_delay = torch.sqrt(centers.square() + float(source_height_rg) ** 2)
    flat_illumination = (4.0 / 3.0) * float(source_height_rg) / (
        centers.square() + float(source_height_rg) ** 2
    ).clamp_min(torch.finfo(dtype).tiny).pow(1.5)
    center_delay = _additive_continuation(
        center_delay,
        reached,
        flat_delay,
    )
    center_gfactor = _additive_continuation(
        center_gfactor,
        reached & (center_gfactor > 0.0),
        torch.ones_like(center_gfactor),
    )
    illumination = _multiplicative_continuation(
        illumination,
        reached,
        flat_illumination,
    )
    center_delay = center_delay - center_delay.min()
    flat_power = (flat_illumination * proper_area).sum().clamp_min(
        torch.finfo(dtype).tiny
    )
    gr_power = (illumination * proper_area).sum().clamp_min(
        torch.finfo(dtype).tiny
    )
    illumination = illumination * (flat_power / gr_power)
    angle_spacing = rays.launch_angle_rad[1] - rays.launch_angle_rad[0]
    ray_weight = 2.0 * math.pi * torch.sin(rays.launch_angle_rad) * angle_spacing
    hit_fraction = ray_weight[rays.hit].sum() / ray_weight.sum().clamp_min(
        torch.finfo(dtype).tiny
    )
    return AxisLamppostProfile(
        centers,
        center_delay,
        illumination,
        center_gfactor,
        hit_fraction,
        rays,
        float(lamp_g_power),
    )
