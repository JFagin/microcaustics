"""Readable eager reference for local-exact Taylor far-field approximation tracing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import torch

from ..compile import run_tensor_kernel
from ..config import FarFieldApproxConfig
from ..geometry import PlaneRegion
from ..runtime import warn_backend_fallback
from .taylor import (
    complex_taylor_coefficients,
    evaluate_complex_taylor,
    translate_complex_taylor,
)


def _evaluate_far_field_deflection(
    coefficient_real: torch.Tensor,
    coefficient_imag: torch.Tensor,
    delta_x: torch.Tensor,
    delta_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    local_mass: torch.Tensor,
    minimum_radius_squared: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized portable far-field approximation deflection query."""

    value_real = coefficient_real[..., -1]
    value_imag = coefficient_imag[..., -1]
    conjugate_y = -delta_y
    for index in range(coefficient_real.shape[-1] - 2, -1, -1):
        product_real = value_real * delta_x - value_imag * conjugate_y
        product_imag = value_real * conjugate_y + value_imag * delta_x
        value_real = product_real + coefficient_real[..., index]
        value_imag = product_imag + coefficient_imag[..., index]
    dx = query_x[:, None] - local_x
    dy = query_y[:, None] - local_y
    weight = local_mass / (dx.square() + dy.square()).clamp_min(
        minimum_radius_squared
    )
    return (
        value_real + (dx * weight).sum(dim=1),
        value_imag + (dy * weight).sum(dim=1),
    )


def _evaluate_far_field_jacobian(
    coefficient_real: torch.Tensor,
    coefficient_imag: torch.Tensor,
    delta_x: torch.Tensor,
    delta_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    local_mass: torch.Tensor,
    minimum_radius_squared: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized portable far-field approximation point-mass Jacobian query."""

    value_real = coefficient_real[..., -1]
    value_imag = coefficient_imag[..., -1]
    conjugate_y = -delta_y
    for index in range(coefficient_real.shape[-1] - 2, -1, -1):
        product_real = value_real * delta_x - value_imag * conjugate_y
        product_imag = value_real * conjugate_y + value_imag * delta_x
        value_real = product_real + coefficient_real[..., index]
        value_imag = product_imag + coefficient_imag[..., index]
    dx = query_x[:, None] - local_x
    dy = query_y[:, None] - local_y
    inverse_r4 = (dx.square() + dy.square()).clamp_min(
        minimum_radius_squared
    ).square().reciprocal()
    weight = local_mass * inverse_r4
    return (
        value_real + ((dy.square() - dx.square()) * weight).sum(dim=1),
        value_imag + (-2.0 * dx * dy * weight).sum(dim=1),
    )

if TYPE_CHECKING:
    from ..simulation import MicrolensingSimulation


@dataclass(frozen=True)
class FarFieldDiagnostics:
    """Spatial decomposition and build cost of a far-field approximation."""

    cells: tuple[int, int]
    nodes_per_cell: tuple[int, int]
    maximum_local_stars: int
    mean_local_stars: float
    build_seconds: float


class TaylorFarFieldApproximation:
    """One-frame local-exact, complex-Taylor point-mass approximation.

    This eager implementation is the correctness reference for later compiled
    and Triton implementations. Stars within a rounded margin of each spatial
    cell are evaluated exactly. All other stars contribute through an order-p
    expansion translated from a higher-order cell-center series to regular
    subcell nodes. Construct a separate object for each independent lens
    state. Dynamic temporal reuse belongs to the scheduler rather than this
    mathematical object.
    """

    def __init__(
        self,
        simulation: MicrolensingSimulation,
        region: PlaneRegion,
        config: FarFieldApproxConfig,
        *,
        time_days: float = 0.0,
        star_chunk_size: int = 4096,
        _coefficient_override=None,
    ) -> None:
        """Build local-star packs and far-field coefficients."""

        if not config.enabled:
            raise ValueError("far-field approximation requires FarFieldApproxConfig(enabled=True)")
        if int(star_chunk_size) < 1:
            raise ValueError("star_chunk_size must be positive")
        self.simulation = simulation
        self.region = region
        self.config = config
        self.time_days = float(time_days)
        self.star_chunk_size = int(star_chunk_size)
        self._coefficient_override = _coefficient_override
        self.last_query_backend = "not-evaluated"
        self._build()

    def _build(self) -> None:
        started = perf_counter()
        runtime = self.simulation.runtime
        field = self.simulation.lens_state(self.time_days)
        fov_y, fov_x = self.region.field_of_view_uas
        self.nx = max(1, int(self.config.cells_per_axis * math.sqrt(fov_x / fov_y)))
        self.ny = max(1, int(self.config.cells_per_axis * math.sqrt(fov_y / fov_x)))
        xmin, xmax, ymin, ymax = self.region.bounds_uas
        self.x_edges = torch.linspace(
            xmin,
            xmax,
            self.nx + 1,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        self.y_edges = torch.linspace(
            ymin,
            ymax,
            self.ny + 1,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        self.cell_dx = float(fov_x / self.nx)
        self.cell_dy = float(fov_y / self.ny)
        radius = self.config.exact_radius_cells * max(self.cell_dx, self.cell_dy)
        # A star exactly on the rounded boundary must be assigned locally in
        # every backend. Expand by a few ulps so fused multiply/add choices do
        # not create a gap between the local and far-field classifications.
        membership_radius = radius * (1.0 + 16.0 * torch.finfo(runtime.dtype).eps)
        nodes_x = int(self.config.nodes_per_cell_axis)
        nodes_y = int(self.config.nodes_per_cell_axis)
        offset_x = (
            (torch.arange(nodes_x, device=runtime.device, dtype=runtime.dtype) + 0.5)
            / nodes_x
            - 0.5
        ) * self.cell_dx
        offset_y = (
            (torch.arange(nodes_y, device=runtime.device, dtype=runtime.dtype) + 0.5)
            / nodes_y
            - 0.5
        ) * self.cell_dy
        mesh_y, mesh_x = torch.meshgrid(offset_y, offset_x, indexing="ij")
        flat_offset_x = mesh_x.reshape(-1)
        flat_offset_y = mesh_y.reshape(-1)
        self._triton_disabled_reason: str | None = None
        zero = torch.zeros((), device=runtime.device, dtype=runtime.dtype)
        radius2 = membership_radius * membership_radius
        mass = field.einstein_radius_uas.square()
        n_cells = self.nx * self.ny
        cell_i = torch.arange(self.nx, device=runtime.device).repeat_interleave(self.ny)
        cell_j = torch.arange(self.ny, device=runtime.device).repeat(self.nx)
        n_stars = len(field)
        mask_grid = torch.empty(
            (n_cells, n_stars),
            device=runtime.device,
            dtype=torch.bool,
        )
        elements_per_cell = max(1, n_stars)
        element_size = torch.empty((), dtype=runtime.dtype).element_size()
        membership_cell_batch = max(
            1,
            min(n_cells, (32 * 1024**2) // max(2 * element_size * elements_per_cell, 1)),
        )
        star_x = field.x_uas[None]
        star_y = field.y_uas[None]
        for start in range(0, n_cells, membership_cell_batch):
            stop = min(n_cells, start + membership_cell_batch)
            ii = cell_i[start:stop]
            jj = cell_j[start:stop]
            dx = torch.maximum(
                torch.maximum(self.x_edges[ii, None] - star_x, star_x - self.x_edges[ii + 1, None]),
                zero,
            )
            dy = torch.maximum(
                torch.maximum(self.y_edges[jj, None] - star_y, star_y - self.y_edges[jj + 1, None]),
                zero,
            )
            mask_grid[start:stop] = dx.square() + dy.square() <= radius2
        local_counts_tensor = mask_grid.sum(dim=1)
        max_local_count = (
            int(local_counts_tensor.max().detach().cpu()) if n_cells else 0
        )
        coefficient_shape = (
            self.nx,
            self.ny,
            nodes_y,
            nodes_x,
            self.config.taylor_order + 1,
        )
        if self._coefficient_override is None:
            center_real_all = torch.empty(
                (n_cells, self.config.center_translation_order + 1),
                device=runtime.device,
                dtype=runtime.dtype,
            )
            center_imag_all = torch.empty_like(center_real_all)
            triton_coefficients = False
            if self._use_triton():
                from .triton_taylor import center_coefficients_rounded_local_triton

                try:
                    center_real_all, center_imag_all = (
                        center_coefficients_rounded_local_triton(
                            field.x_uas,
                            field.y_uas,
                            mass,
                            nx=self.nx,
                            ny=self.ny,
                            xmin=xmin,
                            ymin=ymin,
                            cell_dx=self.cell_dx,
                            cell_dy=self.cell_dy,
                            exact_radius=membership_radius,
                            order=self.config.center_translation_order,
                        )
                    )
                    triton_coefficients = True
                except Exception as error:
                    if runtime.strict_backend:
                        raise
                    warn_backend_fallback(
                        "Triton far-field approximation coefficient construction",
                        error,
                    )
                    self._triton_disabled_reason = f"{type(error).__name__}: {error}"
            if not triton_coefficients:
                for cell in range(n_cells):
                    ix = int(cell_i[cell])
                    iy = int(cell_j[cell])
                    far = ~mask_grid[cell]
                    center_x = 0.5 * (self.x_edges[ix] + self.x_edges[ix + 1])
                    center_y = 0.5 * (self.y_edges[iy] + self.y_edges[iy + 1])
                    center_real, center_imag = complex_taylor_coefficients(
                        center_x.reshape(1),
                        center_y.reshape(1),
                        field.x_uas[far],
                        field.y_uas[far],
                        mass[far],
                        order=self.config.center_translation_order,
                        star_chunk_size=self.star_chunk_size,
                    )
                    center_real_all[cell] = center_real[0]
                    center_imag_all[cell] = center_imag[0]
            node_real, node_imag = translate_complex_taylor(
                center_real_all,
                center_imag_all,
                flat_offset_x,
                flat_offset_y,
                output_order=self.config.taylor_order,
            )
            self.coefficient_real = node_real.reshape(coefficient_shape).contiguous()
            self.coefficient_imag = node_imag.reshape(coefficient_shape).contiguous()
            self.coefficient_build_backend = (
                "triton" if triton_coefficients else "torch-eager"
            )
        else:
            override_real, override_imag = self._coefficient_override
            override_real = torch.as_tensor(
                override_real,
                device=runtime.device,
                dtype=runtime.dtype,
            )
            override_imag = torch.as_tensor(
                override_imag,
                device=runtime.device,
                dtype=runtime.dtype,
            )
            if tuple(override_real.shape) != coefficient_shape or (
                override_imag.shape != override_real.shape
            ):
                raise ValueError("temporal coefficient override has the wrong shape")
            self.coefficient_real = override_real.contiguous()
            self.coefficient_imag = override_imag.contiguous()
            self.coefficient_build_backend = "precomputed"
        max_local = max(1, max_local_count)
        local_shape = (n_cells, max_local)
        self.local_x = torch.zeros(local_shape, device=runtime.device, dtype=runtime.dtype)
        self.local_y = torch.zeros_like(self.local_x)
        self.local_mass = torch.zeros_like(self.local_x)
        rank_elements_per_cell = max(1, n_stars)
        pack_cell_batch = max(
            1,
            min(n_cells, (64 * 1024**2) // max(8 * rank_elements_per_cell, 1)),
        )
        for start in range(0, n_cells, pack_cell_batch):
            stop = min(n_cells, start + pack_cell_batch)
            mask = mask_grid[start:stop]
            rank = torch.cumsum(mask, dim=1, dtype=torch.long) - 1
            chunk_cell, star = torch.nonzero(mask, as_tuple=True)
            if star.numel():
                cell = chunk_cell + start
                local = rank[chunk_cell, star]
                self.local_x[cell, local] = field.x_uas[star]
                self.local_y[cell, local] = field.y_uas[star]
                self.local_mass[cell, local] = mass[star]
        del mask_grid
        runtime.synchronize()
        self.diagnostics = FarFieldDiagnostics(
            cells=(self.ny, self.nx),
            nodes_per_cell=(nodes_y, nodes_x),
            maximum_local_stars=max_local_count,
            mean_local_stars=float(local_counts_tensor.to(torch.float64).mean().detach().cpu()),
            build_seconds=perf_counter() - started,
        )

    def _use_triton(self) -> bool:
        """Whether this object can use the fused order-four CUDA evaluator."""

        runtime = self.simulation.runtime
        return bool(
            self._triton_disabled_reason is None
            and runtime.backend.value == "triton"
            and runtime.device.type == "cuda"
            and runtime.dtype == torch.float32
            and self.config.taylor_order == 4
        )

    def _indices(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve spatial cells, expansion nodes, and node offsets."""

        xmin, xmax, ymin, ymax = self.region.bounds_uas
        scale = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax), 1.0)
        tolerance = 32.0 * torch.finfo(x.dtype).eps * scale
        if bool(torch.any(
            (x < xmin - tolerance) | (x > xmax + tolerance)
            | (y < ymin - tolerance) | (y > ymax + tolerance)
        )):
            raise ValueError("far-field approximation queries must lie inside its lens-plane region")
        ix = torch.floor((x - xmin) / self.cell_dx).long().clamp(0, self.nx - 1)
        iy = torch.floor((y - ymin) / self.cell_dy).long().clamp(0, self.ny - 1)
        local_x = ((x - self.x_edges[ix]) / self.cell_dx).clamp(0.0, 1.0)
        local_y = ((y - self.y_edges[iy]) / self.cell_dy).clamp(0.0, 1.0)
        nodes = int(self.config.nodes_per_cell_axis)
        node_x = torch.floor(local_x * nodes).long().clamp(0, nodes - 1)
        node_y = torch.floor(local_y * nodes).long().clamp(0, nodes - 1)
        expansion_x = self.x_edges[ix] + (node_x.to(x.dtype) + 0.5) * (self.cell_dx / nodes)
        expansion_y = self.y_edges[iy] + (node_y.to(y.dtype) + 0.5) * (self.cell_dy / nodes)
        return ix, iy, node_x, node_y, expansion_x, expansion_y

    def deflection(self, x_uas, y_uas) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate point-mass deflection with exact local-star corrections."""

        runtime = self.simulation.runtime
        x = torch.as_tensor(x_uas, device=runtime.device, dtype=runtime.dtype)
        y = torch.as_tensor(y_uas, device=runtime.device, dtype=runtime.dtype)
        x, y = torch.broadcast_tensors(x, y)
        shape = x.shape
        flat_x, flat_y = x.reshape(-1), y.reshape(-1)
        ix, iy, node_x, node_y, expansion_x, expansion_y = self._indices(flat_x, flat_y)
        coefficients_real = self.coefficient_real[ix, iy, node_y, node_x]
        coefficients_imag = self.coefficient_imag[ix, iy, node_y, node_x]
        flat_cell = ix * self.ny + iy
        minimum = torch.as_tensor(
            1.0e-30 if runtime.dtype == torch.float32 else 1.0e-300,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        if runtime.backend.value == "torch-compile":
            (alpha_x, alpha_y), compiled = run_tensor_kernel(
                runtime,
                "far-field approximation deflection query",
                _evaluate_far_field_deflection,
                coefficients_real,
                coefficients_imag,
                flat_x - expansion_x,
                flat_y - expansion_y,
                flat_x,
                flat_y,
                self.local_x[flat_cell],
                self.local_y[flat_cell],
                self.local_mass[flat_cell],
                minimum,
            )
            self.last_query_backend = (
                "torch-compile" if compiled else "torch-eager"
            )
        else:
            alpha_x, alpha_y = evaluate_complex_taylor(
                coefficients_real,
                coefficients_imag,
                flat_x - expansion_x,
                flat_y - expansion_y,
            )
            minimum_value = float(minimum)
            for cell in torch.unique(flat_cell).detach().cpu().tolist():
                query = flat_cell == int(cell)
                dx = flat_x[query, None] - self.local_x[int(cell)][None]
                dy = flat_y[query, None] - self.local_y[int(cell)][None]
                weight = self.local_mass[int(cell)][None] / (
                    dx.square() + dy.square()
                ).clamp_min(minimum_value)
                alpha_x[query] += (dx * weight).sum(dim=1)
                alpha_y[query] += (dy * weight).sum(dim=1)
            self.last_query_backend = "torch-eager"
        return alpha_x.reshape(shape), alpha_y.reshape(shape)

    def raytrace(self, x_uas, y_uas) -> tuple[torch.Tensor, torch.Tensor]:
        """Map lens-plane coordinates through the approximate lens equation."""

        runtime = self.simulation.runtime
        x = torch.as_tensor(x_uas, device=runtime.device, dtype=runtime.dtype)
        y = torch.as_tensor(y_uas, device=runtime.device, dtype=runtime.dtype)
        x, y = torch.broadcast_tensors(x, y)
        shape = x.shape
        if self._use_triton():
            from .triton_taylor import evaluate_far_field_p4_triton

            try:
                result_x, result_y = evaluate_far_field_p4_triton(self, x, y)
                self.last_query_backend = "triton"
                return result_x.reshape(shape), result_y.reshape(shape)
            except Exception as error:
                if runtime.strict_backend:
                    raise
                warn_backend_fallback("Triton far-field approximation ray tracing", error)
                self._triton_disabled_reason = f"{type(error).__name__}: {error}"
        alpha_x, alpha_y = self.deflection(x, y)
        macro = self.simulation.macro_lens
        angle = torch.as_tensor(2.0 * macro.shear_angle_rad, device=runtime.device, dtype=runtime.dtype)
        gamma1 = macro.shear * torch.cos(angle)
        gamma2 = macro.shear * torch.sin(angle)
        sheet = macro.smooth_convergence
        source_x = (1.0 - sheet - gamma1) * x - gamma2 * y - alpha_x
        source_y = -gamma2 * x + (1.0 - sheet + gamma1) * y - alpha_y
        return source_x, source_y

    def jacobian_determinant(self, x_uas, y_uas) -> torch.Tensor:
        """Evaluate ``det(d beta / d theta)`` analytically.

        The derivative of the complex Taylor polynomial supplies the far
        field. Local stars use the exact point-mass derivatives, so critical
        curves do not require finite differencing or a separate interpolation
        table.
        """

        runtime = self.simulation.runtime
        x = torch.as_tensor(x_uas, device=runtime.device, dtype=runtime.dtype)
        y = torch.as_tensor(y_uas, device=runtime.device, dtype=runtime.dtype)
        x, y = torch.broadcast_tensors(x, y)
        shape = x.shape
        if self._use_triton():
            from .triton_taylor import evaluate_far_field_p4_triton

            try:
                result = evaluate_far_field_p4_triton(
                    self,
                    x,
                    y,
                    jacobian=True,
                ).reshape(shape)
                self.last_query_backend = "triton"
                return result
            except Exception as error:
                if runtime.strict_backend:
                    raise
                warn_backend_fallback("Triton far-field approximation determinant", error)
                self._triton_disabled_reason = f"{type(error).__name__}: {error}"
        flat_x, flat_y = x.reshape(-1), y.reshape(-1)
        ix, iy, node_x, node_y, expansion_x, expansion_y = self._indices(flat_x, flat_y)
        coefficients_real = self.coefficient_real[ix, iy, node_y, node_x]
        coefficients_imag = self.coefficient_imag[ix, iy, node_y, node_x]
        powers = torch.arange(
            1,
            coefficients_real.shape[-1],
            device=runtime.device,
            dtype=runtime.dtype,
        )
        derivative_real = coefficients_real[..., 1:] * powers
        derivative_imag = coefficients_imag[..., 1:] * powers
        flat_cell = ix * self.ny + iy
        minimum = torch.as_tensor(
            1.0e-30 if runtime.dtype == torch.float32 else 1.0e-300,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        if runtime.backend.value == "torch-compile":
            (point_xx, point_xy), compiled = run_tensor_kernel(
                runtime,
                "far-field approximation Jacobian query",
                _evaluate_far_field_jacobian,
                derivative_real,
                derivative_imag,
                flat_x - expansion_x,
                flat_y - expansion_y,
                flat_x,
                flat_y,
                self.local_x[flat_cell],
                self.local_y[flat_cell],
                self.local_mass[flat_cell],
                minimum,
            )
            self.last_query_backend = (
                "torch-compile" if compiled else "torch-eager"
            )
        else:
            point_xx, point_xy = evaluate_complex_taylor(
                derivative_real,
                derivative_imag,
                flat_x - expansion_x,
                flat_y - expansion_y,
            )
            minimum_value = float(minimum)
            for cell in torch.unique(flat_cell).detach().cpu().tolist():
                query = flat_cell == int(cell)
                dx = flat_x[query, None] - self.local_x[int(cell)][None]
                dy = flat_y[query, None] - self.local_y[int(cell)][None]
                inverse_r4 = (
                    (dx.square() + dy.square())
                    .clamp_min(minimum_value)
                    .square()
                    .reciprocal()
                )
                weight = self.local_mass[int(cell)][None] * inverse_r4
                point_xx[query] += (
                    (dy.square() - dx.square()) * weight
                ).sum(dim=1)
                point_xy[query] += (-2.0 * dx * dy * weight).sum(dim=1)
            self.last_query_backend = "torch-eager"
        macro = self.simulation.macro_lens
        angle = torch.as_tensor(2.0 * macro.shear_angle_rad, device=runtime.device, dtype=runtime.dtype)
        gamma1 = macro.shear * torch.cos(angle)
        gamma2 = macro.shear * torch.sin(angle)
        alpha_xx = macro.smooth_convergence + gamma1 + point_xx
        alpha_yy = macro.smooth_convergence - gamma1 - point_xx
        alpha_xy = gamma2 + point_xy
        determinant = (1.0 - alpha_xx) * (1.0 - alpha_yy) - alpha_xy.square()
        return determinant.reshape(shape)


def temporal_taylor_far_fields(
    simulation: MicrolensingSimulation,
    region: PlaneRegion,
    config: FarFieldApproxConfig,
    times_days,
    *,
    star_chunk_size: int = 4096,
) -> tuple[tuple[TaylorFarFieldApproximation, ...], dict[str, object]]:
    """Build an exact far-field approximation state for every requested epoch.

    CUDA/Triton runtimes accumulate the far-field Taylor coefficients for all
    epochs in one batched call. Each returned state nevertheless uses the star
    positions, local membership, and coefficients of its own epoch. No
    temporal interpolation is performed.
    """

    times = tuple(float(value) for value in times_days)
    if not times:
        return (), {
            "far_field_frame_count": 0,
            "far_field_exact_each_frame": True,
            "far_field_batched_accumulator": False,
        }
    frame_indices = list(range(len(times)))
    anchor_overrides: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    batched_accumulator = False
    runtime = simulation.runtime
    if (
        len(frame_indices) > 1
        and runtime.backend.value == "triton"
        and runtime.device.type == "cuda"
        and runtime.dtype == torch.float32
    ):
        from .triton_taylor import (
            center_coefficients_rounded_local_batch_triton,
        )

        try:
            fov_y, fov_x = region.field_of_view_uas
            nx = max(1, int(config.cells_per_axis * math.sqrt(fov_x / fov_y)))
            ny = max(1, int(config.cells_per_axis * math.sqrt(fov_y / fov_x)))
            cell_dx = float(fov_x / nx)
            cell_dy = float(fov_y / ny)
            membership_radius = (
                float(config.exact_radius_cells)
                * max(cell_dx, cell_dy)
                * (1.0 + 16.0 * torch.finfo(runtime.dtype).eps)
            )
            states = [simulation.lens_state(times[index]) for index in frame_indices]
            center_real, center_imag = (
                center_coefficients_rounded_local_batch_triton(
                    torch.stack([state.x_uas for state in states]),
                    torch.stack([state.y_uas for state in states]),
                    states[0].einstein_radius_uas.square(),
                    nx=nx,
                    ny=ny,
                    xmin=region.bounds_uas[0],
                    ymin=region.bounds_uas[2],
                    cell_dx=cell_dx,
                    cell_dy=cell_dy,
                    exact_radius=membership_radius,
                    order=config.center_translation_order,
                )
            )
            nodes = int(config.nodes_per_cell_axis)
            offset_x = (
                (torch.arange(nodes, device=runtime.device, dtype=runtime.dtype) + 0.5)
                / nodes
                - 0.5
            ) * cell_dx
            offset_y = (
                (torch.arange(nodes, device=runtime.device, dtype=runtime.dtype) + 0.5)
                / nodes
                - 0.5
            ) * cell_dy
            mesh_y, mesh_x = torch.meshgrid(offset_y, offset_x, indexing="ij")
            node_real, node_imag = translate_complex_taylor(
                center_real.reshape(-1, config.center_translation_order + 1),
                center_imag.reshape(-1, config.center_translation_order + 1),
                mesh_x.reshape(-1),
                mesh_y.reshape(-1),
                output_order=config.taylor_order,
            )
            coefficient_shape = (
                len(frame_indices),
                nx,
                ny,
                nodes,
                nodes,
                config.taylor_order + 1,
            )
            node_real = node_real.reshape(coefficient_shape)
            node_imag = node_imag.reshape(coefficient_shape)
            anchor_overrides = {
                index: (node_real[position], node_imag[position])
                for position, index in enumerate(frame_indices)
            }
            batched_accumulator = True
        except Exception as error:
            if runtime.strict_backend:
                raise
            warn_backend_fallback(
                "batched Triton far-field approximation coefficient construction",
                error,
            )
    output = tuple(
        TaylorFarFieldApproximation(
            simulation,
            region,
            config,
            time_days=times[index],
            star_chunk_size=star_chunk_size,
            _coefficient_override=anchor_overrides.get(index),
        )
        for index in frame_indices
    )
    return output, {
        "far_field_frame_count": len(output),
        "far_field_exact_each_frame": True,
        "far_field_batched_accumulator": batched_accumulator,
    }


def temporal_taylor_far_field_window(
    simulation: MicrolensingSimulation,
    region: PlaneRegion,
    config: FarFieldApproxConfig,
    all_times_days,
    frame_indices,
    *,
    star_chunk_size: int = 4096,
) -> tuple[tuple[TaylorFarFieldApproximation, ...], dict[str, object]]:
    """Build exact far-field approximation states for selected positions on a global axis."""

    all_times = tuple(float(value) for value in all_times_days)
    requested = tuple(int(value) for value in frame_indices)
    if not requested:
        return (), {
            "far_field_frame_count": 0,
            "far_field_exact_each_frame": True,
            "far_field_batched_accumulator": False,
            "far_field_requested_frames": [],
        }
    if any(index < 0 or index >= len(all_times) for index in requested):
        raise ValueError("temporal frame index is outside the complete time axis")
    selected, metadata = temporal_taylor_far_fields(
        simulation,
        region,
        config,
        [all_times[index] for index in requested],
        star_chunk_size=star_chunk_size,
    )
    return selected, {
        **metadata,
        "far_field_requested_frames": list(requested),
    }


class BatchedTaylorFarFieldApproximation:
    """A unified ray-query queue for several temporal far-field approximation states.

    Individual frame construction remains independently testable through
    :class:`TaylorFarFieldApproximation`. This container pads only the local-star axis,
    stacks the already validated coefficient tables, and evaluates all frames
    in one Triton query launch. The eager fallback simply stacks the scalar
    frame results, preserving behavior on CPU, Apple, and float64 runtimes.
    """

    def __init__(self, far_fields) -> None:
        """Pack compatible one-frame far-field approximations into a temporal batch."""

        started = perf_counter()
        self.far_fields = tuple(far_fields)
        if not self.far_fields:
            raise ValueError("at least one far-field approximation is required")
        first = self.far_fields[0]
        self.simulation = first.simulation
        self.region = first.region
        self.config = first.config
        self.nx = first.nx
        self.ny = first.ny
        self.cell_dx = first.cell_dx
        self.cell_dy = first.cell_dy
        for item in self.far_fields[1:]:
            if (
                item.region != self.region
                or item.config != self.config
                or item.nx != self.nx
                or item.ny != self.ny
                or item.simulation.macro_lens != self.simulation.macro_lens
            ):
                raise ValueError("temporal far-field approximations must share geometry and lens model")
        self.frame_count = len(self.far_fields)
        maximum_local = max(int(item.local_x.shape[-1]) for item in self.far_fields)
        cells = self.nx * self.ny
        runtime = self.simulation.runtime
        local_shape = (self.frame_count, cells, maximum_local)
        self.local_x = torch.zeros(local_shape, device=runtime.device, dtype=runtime.dtype)
        self.local_y = torch.zeros_like(self.local_x)
        self.local_mass = torch.zeros_like(self.local_x)
        for frame, item in enumerate(self.far_fields):
            width = int(item.local_x.shape[-1])
            self.local_x[frame, :, :width] = item.local_x
            self.local_y[frame, :, :width] = item.local_y
            self.local_mass[frame, :, :width] = item.local_mass
        self.coefficient_real = torch.stack(
            [item.coefficient_real for item in self.far_fields]
        ).contiguous()
        self.coefficient_imag = torch.stack(
            [item.coefficient_imag for item in self.far_fields]
        ).contiguous()
        runtime.synchronize()
        self.pack_seconds = perf_counter() - started

    def _use_triton(self) -> bool:
        runtime = self.simulation.runtime
        return bool(
            runtime.backend.value == "triton"
            and runtime.device.type == "cuda"
            and runtime.dtype == torch.float32
            and self.config.taylor_order == 4
        )

    def _validate_points(self, x: torch.Tensor, y: torch.Tensor) -> None:
        xmin, xmax, ymin, ymax = self.region.bounds_uas
        scale = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax), 1.0)
        tolerance = 32.0 * torch.finfo(x.dtype).eps * scale
        if bool(torch.any(
            (x < xmin - tolerance) | (x > xmax + tolerance)
            | (y < ymin - tolerance) | (y > ymax + tolerance)
        )):
            raise ValueError("far-field approximation queries must lie inside its lens-plane region")

    def raytrace(self, x_uas, y_uas) -> tuple[torch.Tensor, torch.Tensor]:
        """Map one shared ray-coordinate array through every temporal frame."""

        runtime = self.simulation.runtime
        x = torch.as_tensor(x_uas, device=runtime.device, dtype=runtime.dtype)
        y = torch.as_tensor(y_uas, device=runtime.device, dtype=runtime.dtype)
        x, y = torch.broadcast_tensors(x, y)
        self._validate_points(x, y)
        if self._use_triton():
            from .triton_taylor import evaluate_far_field_p4_batch_triton

            try:
                return evaluate_far_field_p4_batch_triton(self, x, y)
            except Exception as error:
                if runtime.strict_backend:
                    raise
                warn_backend_fallback("batched Triton far-field approximation ray tracing", error)
        mapped = [item.raytrace(x, y) for item in self.far_fields]
        return (
            torch.stack([value[0] for value in mapped]),
            torch.stack([value[1] for value in mapped]),
        )

    def raytrace_ragged(
        self,
        x_by_frame,
        y_by_frame,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Map variable-length per-frame ray queues in one fused launch.

        The method is intended for temporal products such as marching-squares
        endpoints, whose count differs between frames.  No padding is used:
        coordinates are concatenated once, tagged with their physical frame,
        and returned as inexpensive views into the compact output buffers.
        """

        runtime = self.simulation.runtime
        x_rows = tuple(
            torch.as_tensor(value, device=runtime.device, dtype=runtime.dtype).reshape(-1)
            for value in x_by_frame
        )
        y_rows = tuple(
            torch.as_tensor(value, device=runtime.device, dtype=runtime.dtype).reshape(-1)
            for value in y_by_frame
        )
        if len(x_rows) != self.frame_count or len(y_rows) != self.frame_count:
            raise ValueError("ragged queues must match the temporal frame count")
        lengths = tuple(int(value.numel()) for value in x_rows)
        if any(x.shape != y.shape for x, y in zip(x_rows, y_rows, strict=True)):
            raise ValueError("each ragged x/y queue must have matching shapes")
        total = sum(lengths)
        if total == 0:
            empty = tuple(value.clone() for value in x_rows)
            return empty, tuple(value.clone() for value in y_rows)
        flat_x = torch.cat(x_rows)
        flat_y = torch.cat(y_rows)
        self._validate_points(flat_x, flat_y)
        if self._use_triton():
            from .triton_taylor import evaluate_far_field_p4_indexed_triton

            frame_index = torch.repeat_interleave(
                torch.arange(
                    self.frame_count,
                    device=runtime.device,
                    dtype=torch.int32,
                ),
                torch.tensor(lengths, device=runtime.device, dtype=torch.int64),
                output_size=total,
            )
            try:
                mapped_x, mapped_y = evaluate_far_field_p4_indexed_triton(
                    self,
                    flat_x,
                    flat_y,
                    frame_index,
                )
                return tuple(mapped_x.split(lengths)), tuple(mapped_y.split(lengths))
            except Exception as error:
                if runtime.strict_backend:
                    raise
                warn_backend_fallback("ragged Triton far-field approximation ray tracing", error)
        mapped = tuple(
            item.raytrace(x, y)
            for item, x, y in zip(self.far_fields, x_rows, y_rows, strict=True)
        )
        return (
            tuple(value[0] for value in mapped),
            tuple(value[1] for value in mapped),
        )

    def raytrace_indexed_flat(
        self,
        x_uas: torch.Tensor,
        y_uas: torch.Tensor,
        frame_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Trace a compact queue carrying one temporal frame index per point.

        The production caustic pipeline uses this handoff directly after
        sparse marching. It avoids reconstructing and concatenating ragged
        per-frame endpoint buffers before evaluating the far field.
        """

        runtime = self.simulation.runtime
        x = torch.as_tensor(
            x_uas,
            device=runtime.device,
            dtype=runtime.dtype,
        ).reshape(-1)
        y = torch.as_tensor(
            y_uas,
            device=runtime.device,
            dtype=runtime.dtype,
        ).reshape(-1)
        frames = torch.as_tensor(
            frame_index,
            device=runtime.device,
            dtype=torch.int32,
        ).reshape(-1)
        if x.shape != y.shape or x.shape != frames.shape:
            raise ValueError("flat x, y, and frame-index queues must match")
        if x.numel() == 0:
            return x.clone(), y.clone()
        self._validate_points(x, y)
        if bool(torch.any((frames < 0) | (frames >= self.frame_count))):
            raise ValueError("flat queue frame indices are out of range")
        if self._use_triton():
            from .triton_taylor import evaluate_far_field_p4_indexed_triton

            try:
                return evaluate_far_field_p4_indexed_triton(
                    self,
                    x,
                    y,
                    frames,
                )
            except Exception as error:
                if runtime.strict_backend:
                    raise
                warn_backend_fallback(
                    "indexed Triton far-field approximation ray tracing",
                    error,
                )

        mapped_x = torch.empty_like(x)
        mapped_y = torch.empty_like(y)
        for frame, far_field in enumerate(self.far_fields):
            selected = frames == frame
            if not bool(torch.any(selected)):
                continue
            frame_x, frame_y = far_field.raytrace(x[selected], y[selected])
            mapped_x[selected] = frame_x
            mapped_y[selected] = frame_y
        return mapped_x, mapped_y

    def jacobian_determinant(self, x_uas, y_uas) -> torch.Tensor:
        """Evaluate analytic Jacobian determinants for every temporal frame."""

        runtime = self.simulation.runtime
        x = torch.as_tensor(x_uas, device=runtime.device, dtype=runtime.dtype)
        y = torch.as_tensor(y_uas, device=runtime.device, dtype=runtime.dtype)
        x, y = torch.broadcast_tensors(x, y)
        self._validate_points(x, y)
        if self._use_triton():
            from .triton_taylor import evaluate_far_field_p4_batch_triton

            try:
                return evaluate_far_field_p4_batch_triton(
                    self,
                    x,
                    y,
                    jacobian=True,
                )
            except Exception as error:
                if runtime.strict_backend:
                    raise
                warn_backend_fallback("batched Triton far-field approximation determinant", error)
        return torch.stack(
            [item.jacobian_determinant(x, y) for item in self.far_fields]
        )
