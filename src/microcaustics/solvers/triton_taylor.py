"""Optional fused Triton evaluation for order-four Taylor far-field approximation."""

from __future__ import annotations

import math

import torch

try:  # Triton is intentionally optional on CPU and Apple platforms.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on optional runtime
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _center_coefficients_rounded_local_kernel(
        star_x_ptr,
        star_y_ptr,
        mass_ptr,
        output_real_ptr,
        output_imag_ptr,
        n_stars,
        cells_per_frame,
        xmin,
        ymin,
        cell_dx,
        cell_dy,
        exact_radius2,
        NY: tl.constexpr,
        ORDER_COUNT: tl.constexpr,
        BLOCK_ORDERS: tl.constexpr,
        BLOCK_STARS: tl.constexpr,
    ):
        packed_cell = tl.program_id(0)
        frame = packed_cell // cells_per_frame
        cell = packed_cell - frame * cells_per_frame
        cell_x = cell // NY
        cell_y = cell - cell_x * NY
        left_x = xmin + cell_x * cell_dx
        left_y = ymin + cell_y * cell_dy
        right_x = left_x + cell_dx
        right_y = left_y + cell_dy
        center_x = left_x + 0.5 * cell_dx
        center_y = left_y + 0.5 * cell_dy
        orders = tl.arange(0, BLOCK_ORDERS)
        active_orders = orders < ORDER_COUNT
        accumulated_real = tl.zeros((BLOCK_ORDERS,), tl.float32)
        accumulated_imag = tl.zeros((BLOCK_ORDERS,), tl.float32)
        for start in tl.range(0, n_stars, BLOCK_STARS):
            stars = start + tl.arange(0, BLOCK_STARS)
            active = stars < n_stars
            star_offset = frame * n_stars + stars
            star_x = tl.load(star_x_ptr + star_offset, mask=active, other=0.0)
            star_y = tl.load(star_y_ptr + star_offset, mask=active, other=0.0)
            mass = tl.load(mass_ptr + star_offset, mask=active, other=0.0)
            distance_to_cell_x = tl.maximum(
                tl.maximum(left_x - star_x, star_x - right_x),
                0.0,
            )
            distance_to_cell_y = tl.maximum(
                tl.maximum(left_y - star_y, star_y - right_y),
                0.0,
            )
            local = (
                distance_to_cell_x * distance_to_cell_x
                + distance_to_cell_y * distance_to_cell_y
                <= exact_radius2
            )
            effective_mass = tl.where(active & ~local, mass, 0.0)
            dx = center_x - star_x
            dy = center_y - star_y
            inverse_r2 = 1.0 / tl.maximum(dx * dx + dy * dy, 1.0e-30)
            inverse_real = dx * inverse_r2
            inverse_imag = dy * inverse_r2
            power_real = inverse_real
            power_imag = inverse_imag
            sign = 1.0
            for order in tl.static_range(0, ORDER_COUNT):
                value_real = sign * tl.sum(effective_mass * power_real, axis=0)
                value_imag = sign * tl.sum(effective_mass * power_imag, axis=0)
                accumulated_real += tl.where(orders == order, value_real, 0.0)
                accumulated_imag += tl.where(orders == order, value_imag, 0.0)
                next_real = power_real * inverse_real - power_imag * inverse_imag
                next_imag = power_real * inverse_imag + power_imag * inverse_real
                power_real = next_real
                power_imag = next_imag
                sign = -sign
        output = packed_cell * ORDER_COUNT + orders
        tl.store(output_real_ptr + output, accumulated_real, mask=active_orders)
        tl.store(output_imag_ptr + output, accumulated_imag, mask=active_orders)

    @triton.jit
    def _far_field_p4_query_kernel(
        x_ptr,
        y_ptr,
        frame_index_ptr,
        local_x_ptr,
        local_y_ptr,
        local_mass_ptr,
        coefficient_real_ptr,
        coefficient_imag_ptr,
        output_x_ptr,
        output_y_ptr,
        n_rays,
        rays_per_frame,
        cells_per_frame,
        max_local,
        xmin,
        ymin,
        cell_dx,
        cell_dy,
        beta_xx,
        beta_xy,
        beta_yy,
        NX: tl.constexpr,
        NY: tl.constexpr,
        NODES: tl.constexpr,
        RAY_BLOCK: tl.constexpr,
        STAR_BLOCK: tl.constexpr,
        DO_JACOBIAN: tl.constexpr,
        FRAME_INDEXED: tl.constexpr,
    ):
        ray = tl.program_id(0) * RAY_BLOCK + tl.arange(0, RAY_BLOCK)
        valid = ray < n_rays
        if FRAME_INDEXED:
            frame = tl.load(frame_index_ptr + ray, mask=valid, other=0).to(tl.int64)
            local_ray = ray
        else:
            frame = ray // rays_per_frame
            local_ray = ray - frame * rays_per_frame
        x = tl.load(x_ptr + local_ray, mask=valid, other=0.0)
        y = tl.load(y_ptr + local_ray, mask=valid, other=0.0)
        cell_x = tl.maximum(
            0,
            tl.minimum(NX - 1, tl.floor((x - xmin) / cell_dx)),
        ).to(tl.int64)
        cell_y = tl.maximum(
            0,
            tl.minimum(NY - 1, tl.floor((y - ymin) / cell_dy)),
        ).to(tl.int64)
        cell = cell_x * NY + cell_y
        packed_cell = frame * cells_per_frame + cell
        point_xx = tl.zeros((RAY_BLOCK,), tl.float32)
        point_xy = tl.zeros((RAY_BLOCK,), tl.float32)
        alpha_x = tl.zeros((RAY_BLOCK,), tl.float32)
        alpha_y = tl.zeros((RAY_BLOCK,), tl.float32)
        for start in tl.range(0, max_local, STAR_BLOCK):
            stars = start + tl.arange(0, STAR_BLOCK)
            star_valid = stars < max_local
            offset = packed_cell[:, None] * max_local + stars[None, :]
            mask = valid[:, None] & star_valid[None, :]
            mass = tl.load(local_mass_ptr + offset, mask=mask, other=0.0)
            star_x = tl.load(local_x_ptr + offset, mask=mask, other=0.0)
            star_y = tl.load(local_y_ptr + offset, mask=mask, other=0.0)
            dx = x[:, None] - star_x
            dy = y[:, None] - star_y
            inverse_r2 = 1.0 / tl.maximum(dx * dx + dy * dy, 1.0e-30)
            if DO_JACOBIAN:
                mass_over_r4 = mass * inverse_r2 * inverse_r2
                point_xx += tl.sum(
                    (dy * dy - dx * dx) * mass_over_r4,
                    axis=1,
                )
                point_xy += tl.sum(-2.0 * dx * dy * mass_over_r4, axis=1)
            else:
                weight = mass * inverse_r2
                alpha_x += tl.sum(dx * weight, axis=1)
                alpha_y += tl.sum(dy * weight, axis=1)

        cell_left_x = xmin + cell_x * cell_dx
        cell_left_y = ymin + cell_y * cell_dy
        fraction_x = tl.maximum(
            0.0,
            tl.minimum(1.0, (x - cell_left_x) / cell_dx),
        )
        fraction_y = tl.maximum(
            0.0,
            tl.minimum(1.0, (y - cell_left_y) / cell_dy),
        )
        node_x = tl.maximum(
            0,
            tl.minimum(NODES - 1, tl.floor(fraction_x * NODES)),
        ).to(tl.int64)
        node_y = tl.maximum(
            0,
            tl.minimum(NODES - 1, tl.floor(fraction_y * NODES)),
        ).to(tl.int64)
        expansion_x = cell_left_x + (node_x + 0.5) * (cell_dx / NODES)
        expansion_y = cell_left_y + (node_y + 0.5) * (cell_dy / NODES)
        delta_real = x - expansion_x
        delta_imag = -(y - expansion_y)
        coefficient_base = (
            ((packed_cell * NODES + node_y) * NODES + node_x) * 5
        )
        if DO_JACOBIAN:
            value_real = 4.0 * tl.load(
                coefficient_real_ptr + coefficient_base + 4,
                mask=valid,
                other=0.0,
            )
            value_imag = 4.0 * tl.load(
                coefficient_imag_ptr + coefficient_base + 4,
                mask=valid,
                other=0.0,
            )
            product_real = value_real * delta_real - value_imag * delta_imag
            product_imag = value_real * delta_imag + value_imag * delta_real
            value_real = product_real + 3.0 * tl.load(
                coefficient_real_ptr + coefficient_base + 3,
                mask=valid,
                other=0.0,
            )
            value_imag = product_imag + 3.0 * tl.load(
                coefficient_imag_ptr + coefficient_base + 3,
                mask=valid,
                other=0.0,
            )
            product_real = value_real * delta_real - value_imag * delta_imag
            product_imag = value_real * delta_imag + value_imag * delta_real
            value_real = product_real + 2.0 * tl.load(
                coefficient_real_ptr + coefficient_base + 2,
                mask=valid,
                other=0.0,
            )
            value_imag = product_imag + 2.0 * tl.load(
                coefficient_imag_ptr + coefficient_base + 2,
                mask=valid,
                other=0.0,
            )
            product_real = value_real * delta_real - value_imag * delta_imag
            product_imag = value_real * delta_imag + value_imag * delta_real
            point_xx += product_real + tl.load(
                coefficient_real_ptr + coefficient_base + 1,
                mask=valid,
                other=0.0,
            )
            point_xy += product_imag + tl.load(
                coefficient_imag_ptr + coefficient_base + 1,
                mask=valid,
                other=0.0,
            )
            matrix_xx = beta_xx - point_xx
            matrix_xy = beta_xy - point_xy
            matrix_yy = beta_yy + point_xx
            tl.store(
                output_x_ptr + ray,
                matrix_xx * matrix_yy - matrix_xy * matrix_xy,
                mask=valid,
            )
        else:
            value_real = tl.load(
                coefficient_real_ptr + coefficient_base + 4,
                mask=valid,
                other=0.0,
            )
            value_imag = tl.load(
                coefficient_imag_ptr + coefficient_base + 4,
                mask=valid,
                other=0.0,
            )
            for coefficient in tl.static_range(3, -1, -1):
                product_real = value_real * delta_real - value_imag * delta_imag
                product_imag = value_real * delta_imag + value_imag * delta_real
                value_real = product_real + tl.load(
                    coefficient_real_ptr + coefficient_base + coefficient,
                    mask=valid,
                    other=0.0,
                )
                value_imag = product_imag + tl.load(
                    coefficient_imag_ptr + coefficient_base + coefficient,
                    mask=valid,
                    other=0.0,
                )
            alpha_x += value_real
            alpha_y += value_imag
            tl.store(
                output_x_ptr + ray,
                beta_xx * x + beta_xy * y - alpha_x,
                mask=valid,
            )
            tl.store(
                output_y_ptr + ray,
                beta_xy * x + beta_yy * y - alpha_y,
                mask=valid,
            )


def triton_taylor_available() -> bool:
    """Whether CUDA float32 Triton kernels can be launched."""

    return bool(triton is not None and torch.cuda.is_available())


def center_coefficients_rounded_local_triton(
    star_x: torch.Tensor,
    star_y: torch.Tensor,
    mass: torch.Tensor,
    *,
    nx: int,
    ny: int,
    xmin: float,
    ymin: float,
    cell_dx: float,
    cell_dy: float,
    exact_radius: float,
    order: int,
    star_block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cell-center coefficients while excluding rounded local regions."""

    if not triton_taylor_available():
        raise RuntimeError("Triton Taylor coefficient construction is unavailable")
    if star_x.device.type != "cuda" or star_x.dtype != torch.float32:
        raise ValueError("Triton Taylor coefficient construction requires CUDA float32")
    star_x = star_x.contiguous().reshape(-1)
    star_y = star_y.contiguous().reshape(-1)
    mass = mass.contiguous().reshape(-1)
    if star_x.shape != star_y.shape or star_x.shape != mass.shape:
        raise ValueError("star coordinates and masses must have matching shapes")
    order_count = int(order) + 1
    if not 1 <= order_count <= 16:
        raise ValueError("order must lie in [0, 15]")
    cells = int(nx) * int(ny)
    real = torch.empty((cells, order_count), device=star_x.device, dtype=star_x.dtype)
    imag = torch.empty_like(real)
    block_orders = triton.next_power_of_2(order_count)
    _center_coefficients_rounded_local_kernel[(cells,)](
        star_x,
        star_y,
        mass,
        real,
        imag,
        star_x.numel(),
        cells,
        float(xmin),
        float(ymin),
        float(cell_dx),
        float(cell_dy),
        float(exact_radius) ** 2,
        NY=int(ny),
        ORDER_COUNT=order_count,
        BLOCK_ORDERS=block_orders,
        BLOCK_STARS=int(star_block),
        num_warps=4,
    )
    return real, imag


def center_coefficients_rounded_local_batch_triton(
    star_x: torch.Tensor,
    star_y: torch.Tensor,
    mass: torch.Tensor,
    *,
    nx: int,
    ny: int,
    xmin: float,
    ymin: float,
    cell_dx: float,
    cell_dy: float,
    exact_radius: float,
    order: int,
    star_block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build rounded-local center coefficients for many temporal frames."""

    if not triton_taylor_available():
        raise RuntimeError("Triton Taylor coefficient construction is unavailable")
    if star_x.device.type != "cuda" or star_x.dtype != torch.float32:
        raise ValueError("Triton Taylor coefficient construction requires CUDA float32")
    if star_x.ndim != 2 or star_y.shape != star_x.shape:
        raise ValueError("batched star coordinates must share shape [frame, star]")
    if mass.ndim == 1:
        mass = mass[None].expand(star_x.shape[0], -1)
    if mass.shape != star_x.shape:
        raise ValueError("batched mass must have shape [star] or [frame, star]")
    star_x = star_x.contiguous()
    star_y = star_y.contiguous()
    mass = mass.contiguous()
    frames, stars = (int(value) for value in star_x.shape)
    order_count = int(order) + 1
    if not 1 <= order_count <= 16:
        raise ValueError("order must lie in [0, 15]")
    cells = int(nx) * int(ny)
    real = torch.empty(
        (frames, cells, order_count),
        device=star_x.device,
        dtype=star_x.dtype,
    )
    imag = torch.empty_like(real)
    block_orders = triton.next_power_of_2(order_count)
    _center_coefficients_rounded_local_kernel[(frames * cells,)](
        star_x,
        star_y,
        mass,
        real,
        imag,
        stars,
        cells,
        float(xmin),
        float(ymin),
        float(cell_dx),
        float(cell_dy),
        float(exact_radius) ** 2,
        NY=int(ny),
        ORDER_COUNT=order_count,
        BLOCK_ORDERS=block_orders,
        BLOCK_STARS=int(star_block),
        num_warps=4,
    )
    return real, imag


def evaluate_far_field_p4_triton(
    far_field,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    jacobian: bool = False,
    ray_block: int = 64,
    star_block: int = 32,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Launch the fused order-four local-exact/far-field query kernel."""

    if not triton_taylor_available():
        raise RuntimeError("Triton Taylor evaluation is unavailable")
    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise ValueError("Triton Taylor evaluation requires CUDA float32")
    if far_field.config.taylor_order != 4:
        raise ValueError("the fused Triton evaluator supports Taylor order four")
    x = x.contiguous().reshape(-1)
    y = y.contiguous().reshape(-1)
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes")
    output_x = torch.empty_like(x)
    output_y = torch.empty_like(x)
    macro = far_field.simulation.macro_lens
    angle = 2.0 * float(macro.shear_angle_rad)
    gamma1 = float(macro.shear) * math.cos(angle)
    gamma2 = float(macro.shear) * math.sin(angle)
    beta_xx = 1.0 - macro.smooth_convergence - gamma1
    beta_xy = -gamma2
    beta_yy = 1.0 - macro.smooth_convergence + gamma1
    grid = (triton.cdiv(x.numel(), int(ray_block)),)
    _far_field_p4_query_kernel[grid](
        x,
        y,
        x,
        far_field.local_x,
        far_field.local_y,
        far_field.local_mass,
        far_field.coefficient_real,
        far_field.coefficient_imag,
        output_x,
        output_y,
        x.numel(),
        x.numel(),
        far_field.nx * far_field.ny,
        far_field.local_x.shape[-1],
        far_field.region.bounds_uas[0],
        far_field.region.bounds_uas[2],
        far_field.cell_dx,
        far_field.cell_dy,
        beta_xx,
        beta_xy,
        beta_yy,
        NX=far_field.nx,
        NY=far_field.ny,
        NODES=far_field.config.nodes_per_cell_axis,
        RAY_BLOCK=int(ray_block),
        STAR_BLOCK=int(star_block),
        DO_JACOBIAN=bool(jacobian),
        FRAME_INDEXED=False,
        num_warps=4,
    )
    return output_x if jacobian else (output_x, output_y)


def evaluate_far_field_p4_batch_triton(
    far_field_batch,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    jacobian: bool = False,
    ray_block: int = 64,
    star_block: int = 32,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one shared ray queue for a stack of temporal far-field approximations."""

    if not triton_taylor_available():
        raise RuntimeError("Triton Taylor evaluation is unavailable")
    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise ValueError("Triton Taylor evaluation requires CUDA float32")
    if far_field_batch.config.taylor_order != 4:
        raise ValueError("the fused Triton evaluator supports Taylor order four")
    x, y = torch.broadcast_tensors(x, y)
    output_shape = (far_field_batch.frame_count, *x.shape)
    rays_per_frame = int(x.numel())
    flat_x = x.contiguous().reshape(-1)
    flat_y = y.contiguous().reshape(-1)
    output_x = torch.empty(
        far_field_batch.frame_count * rays_per_frame,
        device=x.device,
        dtype=x.dtype,
    )
    output_y = torch.empty_like(output_x)
    macro = far_field_batch.simulation.macro_lens
    angle = 2.0 * float(macro.shear_angle_rad)
    gamma1 = float(macro.shear) * math.cos(angle)
    gamma2 = float(macro.shear) * math.sin(angle)
    beta_xx = 1.0 - macro.smooth_convergence - gamma1
    beta_xy = -gamma2
    beta_yy = 1.0 - macro.smooth_convergence + gamma1
    total_rays = far_field_batch.frame_count * rays_per_frame
    grid = (triton.cdiv(total_rays, int(ray_block)),)
    _far_field_p4_query_kernel[grid](
        flat_x,
        flat_y,
        flat_x,
        far_field_batch.local_x,
        far_field_batch.local_y,
        far_field_batch.local_mass,
        far_field_batch.coefficient_real,
        far_field_batch.coefficient_imag,
        output_x,
        output_y,
        total_rays,
        rays_per_frame,
        far_field_batch.nx * far_field_batch.ny,
        far_field_batch.local_x.shape[-1],
        far_field_batch.region.bounds_uas[0],
        far_field_batch.region.bounds_uas[2],
        far_field_batch.cell_dx,
        far_field_batch.cell_dy,
        beta_xx,
        beta_xy,
        beta_yy,
        NX=far_field_batch.nx,
        NY=far_field_batch.ny,
        NODES=far_field_batch.config.nodes_per_cell_axis,
        RAY_BLOCK=int(ray_block),
        STAR_BLOCK=int(star_block),
        DO_JACOBIAN=bool(jacobian),
        FRAME_INDEXED=False,
        num_warps=4,
    )
    return (
        output_x.reshape(output_shape)
        if jacobian
        else (
            output_x.reshape(output_shape),
            output_y.reshape(output_shape),
        )
    )


def evaluate_far_field_p4_indexed_triton(
    far_field_batch,
    x: torch.Tensor,
    y: torch.Tensor,
    frame_index: torch.Tensor,
    *,
    ray_block: int = 64,
    star_block: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a ragged ray queue whose entries select temporal frames.

    Unlike :func:`evaluate_far_field_p4_batch_triton`, coordinates need not
    have the same length in every frame.  ``frame_index`` associates each ray
    with one packed temporal far-field approximation, allowing caustic endpoint queues to be
    traced in one launch without padding to the largest frame.
    """

    if not triton_taylor_available():
        raise RuntimeError("Triton Taylor evaluation is unavailable")
    if x.device.type != "cuda" or x.dtype != torch.float32:
        raise ValueError("Triton Taylor evaluation requires CUDA float32")
    if far_field_batch.config.taylor_order != 4:
        raise ValueError("the fused Triton evaluator supports Taylor order four")
    x = x.contiguous().reshape(-1)
    y = y.contiguous().reshape(-1)
    frame_index = frame_index.to(device=x.device, dtype=torch.int32).contiguous().reshape(-1)
    if x.shape != y.shape or x.shape != frame_index.shape:
        raise ValueError("x, y, and frame_index must have matching shapes")
    output_x = torch.empty_like(x)
    output_y = torch.empty_like(x)
    macro = far_field_batch.simulation.macro_lens
    angle = 2.0 * float(macro.shear_angle_rad)
    gamma1 = float(macro.shear) * math.cos(angle)
    gamma2 = float(macro.shear) * math.sin(angle)
    beta_xx = 1.0 - macro.smooth_convergence - gamma1
    beta_xy = -gamma2
    beta_yy = 1.0 - macro.smooth_convergence + gamma1
    grid = (triton.cdiv(x.numel(), int(ray_block)),)
    _far_field_p4_query_kernel[grid](
        x,
        y,
        frame_index,
        far_field_batch.local_x,
        far_field_batch.local_y,
        far_field_batch.local_mass,
        far_field_batch.coefficient_real,
        far_field_batch.coefficient_imag,
        output_x,
        output_y,
        x.numel(),
        1,
        far_field_batch.nx * far_field_batch.ny,
        far_field_batch.local_x.shape[-1],
        far_field_batch.region.bounds_uas[0],
        far_field_batch.region.bounds_uas[2],
        far_field_batch.cell_dx,
        far_field_batch.cell_dy,
        beta_xx,
        beta_xy,
        beta_yy,
        NX=far_field_batch.nx,
        NY=far_field_batch.ny,
        NODES=far_field_batch.config.nodes_per_cell_axis,
        RAY_BLOCK=int(ray_block),
        STAR_BLOCK=int(star_block),
        DO_JACOBIAN=False,
        FRAME_INDEXED=True,
        num_warps=4,
    )
    return output_x, output_y
