"""Publication-style diagrams of the accelerated numerical methods."""

from __future__ import annotations

import math

import numpy as np
import torch

from ..config import FarFieldApproxConfig, IPMConfig
from ..geometry import PlaneGrid, PlaneRegion
from ._common import add_scale_bar, as_numpy, hide_image_axes, require_matplotlib


def plot_far_field_method(
    simulation,
    lens_region: PlaneRegion,
    config: FarFieldApproxConfig,
    *,
    time_days: float = 0.0,
    selected_cell: tuple[int, int] | None = None,
    scale_bar_uas: float | None = None,
):
    """Show the production local-exact/complex-Taylor calculation.

    The three panels use the same regular-cell partition, rounded exact-star
    region, Taylor coefficients, and nearest expansion-node convention as
    :class:`~microcaustics.solvers.TaylorFarFieldApproximation`.  A ray uses one Taylor
    expansion centered on the node in its uniform nodal subcell. It does not
    blend four neighboring nodes. ``selected_cell`` is ``(x_index, y_index)``.
    """

    require_matplotlib()
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, ConnectionPatch, Patch, Polygon, Rectangle

    from ..solvers.far_field import TaylorFarFieldApproximation

    far_field = TaylorFarFieldApproximation(
        simulation, lens_region, config, time_days=time_days
    )
    field = simulation.lens_state(time_days)
    star_x, star_y = as_numpy(field.x_uas), as_numpy(field.y_uas)
    mass = (
        as_numpy(field.mass_solar)
        if field.mass_solar is not None
        else as_numpy(field.einstein_radius_uas) ** 2
    )
    if selected_cell is None:
        star_ix = np.floor(
            (star_x - float(far_field.x_edges[0])) / far_field.cell_dx
        ).astype(int)
        star_iy = np.floor(
            (star_y - float(far_field.y_edges[0])) / far_field.cell_dy
        ).astype(int)
        inside = (
            (star_ix >= 0) & (star_ix < far_field.nx)
            & (star_iy >= 0) & (star_iy < far_field.ny)
        )
        occupied = np.unique(np.column_stack((star_ix[inside], star_iy[inside])), axis=0)
        if occupied.size:
            center = np.asarray(
                ((far_field.nx - 1) / 2, (far_field.ny - 1) / 2)
            )
            selected_cell = tuple(
                int(value)
                for value in occupied[np.argmin(np.sum((occupied - center) ** 2, axis=1))]
            )
        else:
            selected_cell = (far_field.nx // 2, far_field.ny // 2)
    ix, iy = (int(value) for value in selected_cell)
    if not (0 <= ix < far_field.nx and 0 <= iy < far_field.ny):
        raise ValueError("selected_cell lies outside the far-field partition")

    x0, x1 = float(far_field.x_edges[ix]), float(far_field.x_edges[ix + 1])
    y0, y1 = float(far_field.y_edges[iy]), float(far_field.y_edges[iy + 1])
    cell_width, cell_height = x1 - x0, y1 - y0
    radius = float(config.exact_radius_cells) * max(cell_width, cell_height)
    distance_x = np.maximum.reduce((x0 - star_x, star_x - x1, np.zeros_like(star_x)))
    distance_y = np.maximum.reduce((y0 - star_y, star_y - y1, np.zeros_like(star_y)))
    local = distance_x * distance_x + distance_y * distance_y <= radius * radius
    query = np.asarray((x0 + 0.36 * cell_width, y0 + 0.38 * cell_height))

    positive_mass = mass[np.isfinite(mass) & (mass > 0)]
    mass_scale = float(np.median(positive_mass)) if positive_mass.size else 1.0
    marker_size = np.clip(4.6 * np.sqrt(np.maximum(mass, 0) / mass_scale), 2.2, 24.0)
    field_size = np.clip(0.55 * np.sqrt(np.maximum(mass, 0) / mass_scale), 0.18, 3.0)

    figure, axes = plt.subplots(1, 3, figsize=(7.5, 3.10))
    figure.subplots_adjust(left=0.015, right=0.990, bottom=0.075, top=0.865, wspace=0.08)
    ax_field, ax_local, ax_far = axes
    xmin, xmax, ymin, ymax = lens_region.bounds_uas
    ax_field.scatter(
        star_x, star_y, s=field_size, color="red", edgecolor="none",
        alpha=0.65, rasterized=True, zorder=2,
    )
    for edge in as_numpy(far_field.x_edges):
        ax_field.axvline(edge, color="0.82", linewidth=0.36, zorder=0)
    for edge in as_numpy(far_field.y_edges):
        ax_field.axhline(edge, color="0.82", linewidth=0.36, zorder=0)
    aperture_radius = float(np.nanmax(np.hypot(star_x, star_y))) if star_x.size else 0.0
    if aperture_radius > 0:
        ax_field.add_patch(Circle(
            (0.0, 0.0), aperture_radius, fill=False, edgecolor="0.30",
            linewidth=1.05, linestyle=(0, (5, 3)), zorder=4,
        ))
    ax_field.add_patch(Rectangle(
        (x0, y0), cell_width, cell_height,
        facecolor=(0.15, 0.15, 0.15, 0.10), edgecolor="0.10",
        linewidth=0.90, zorder=5,
    ))
    ax_field.annotate(
        "one ray cell", xy=(0.5 * (x0 + x1), y1), xytext=(0.49, 0.59),
        textcoords="axes fraction", ha="center", va="center", fontsize=7.4,
        color="0.18", arrowprops=dict(arrowstyle="->", linewidth=0.75, color="0.10"),
    )
    ax_field.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax_field.set_title(
        f"Partition lens plane into {far_field.nx} × {far_field.ny} cells",
        pad=4,
        fontsize=9.2,
    )

    rounded_vertices = []
    for cx, cy, a0, a1 in (
        (x1, y0, -0.5 * np.pi, 0.0), (x1, y1, 0.0, 0.5 * np.pi),
        (x0, y1, 0.5 * np.pi, np.pi), (x0, y0, np.pi, 1.5 * np.pi),
    ):
        angles = np.linspace(a0, a1, 18)
        rounded_vertices.extend(np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles))))
    ax_local.add_patch(Polygon(
        np.asarray(rounded_vertices), closed=True, facecolor="#C9E3F1",
        edgecolor="#0072B2", linewidth=1.35, linestyle=(0, (4, 2)), alpha=0.55,
    ))
    ax_local.add_patch(Rectangle(
        (x0, y0), cell_width, cell_height, facecolor="none",
        edgecolor="0.10", linewidth=1.45, zorder=5,
    ))
    view_pad = max(0.30 * max(cell_width, cell_height), 1.15 * radius)
    local_view = (x0 - view_pad, x1 + view_pad, y0 - view_pad, y1 + view_pad)
    in_view = (
        (star_x >= local_view[0]) & (star_x <= local_view[1])
        & (star_y >= local_view[2]) & (star_y <= local_view[3])
    )
    far_visible = in_view & ~local
    ax_local.scatter(
        star_x[far_visible], star_y[far_visible], s=marker_size[far_visible],
        color="0.72", edgecolor="0.42", linewidth=0.25, zorder=2,
    )
    ax_local.scatter(
        star_x[in_view & local], star_y[in_view & local],
        s=1.18 * marker_size[in_view & local], color="#0072B2",
        edgecolor="black", linewidth=0.35, zorder=4,
    )
    ax_local.scatter(
        [query[0]], [query[1]], s=66, marker="X", facecolor="white",
        edgecolor="black", linewidth=1.0, zorder=8,
    )
    ax_local.annotate(
        "ray position", xy=query,
        xytext=(x0 + 0.52 * cell_width, y0 + 0.80 * cell_height),
        ha="center", va="center", fontsize=6.9,
        arrowprops=dict(arrowstyle="->", linewidth=0.8, color="0.20", shrinkB=4.0),
    )
    ax_local.set(xlim=local_view[:2], ylim=local_view[2:])
    ax_local.set_title("Sum nearby stars exactly", pad=4, fontsize=8.2)

    nodes = int(config.nodes_per_cell_axis)
    node_x = x0 + (np.arange(nodes) + 0.5) * cell_width / nodes
    node_y = y0 + (np.arange(nodes) + 0.5) * cell_height / nodes
    mesh_x, mesh_y = np.meshgrid(node_x, node_y, indexing="xy")
    coefficients_x = as_numpy(far_field.coefficient_real[ix, iy, :, :, 0])
    coefficients_y = as_numpy(far_field.coefficient_imag[ix, iy, :, :, 0])
    magnitude = np.hypot(coefficients_x, coefficients_y)
    vector_norm = max(float(np.nanmax(magnitude)), np.finfo(float).tiny)
    vector_scale = 0.070 * min(cell_width, cell_height) / vector_norm
    vector_x, vector_y = vector_scale * coefficients_x, vector_scale * coefficients_y
    selected_x = int(np.clip(np.floor((query[0] - x0) / cell_width * nodes), 0, nodes - 1))
    selected_y = int(np.clip(np.floor((query[1] - y0) / cell_height * nodes), 0, nodes - 1))
    regular = np.ones((nodes, nodes), dtype=bool)
    regular[selected_y, selected_x] = False
    ax_far.add_patch(Rectangle(
        (x0, y0), cell_width, cell_height, facecolor="#F7F3F8",
        edgecolor="0.10", linewidth=1.45, zorder=0,
    ))
    ax_far.quiver(
        mesh_x[regular], mesh_y[regular], vector_x[regular], vector_y[regular],
        color="#CC79A7", angles="xy", scale_units="xy", scale=1,
        width=0.0030, headwidth=3.4, headlength=4.0, zorder=3,
    )
    ax_far.scatter(mesh_x[regular], mesh_y[regular], s=1.8, color="#CC79A7", zorder=4)
    chosen = np.asarray((node_x[selected_x], node_y[selected_y]))
    ax_far.plot(
        (chosen[0], query[0]), (chosen[1], query[1]), color="#D97706",
        linewidth=0.8, linestyle=(0, (2, 2)), zorder=2,
    )
    ax_far.quiver(
        [chosen[0]], [chosen[1]], [vector_x[selected_y, selected_x]],
        [vector_y[selected_y, selected_x]], color="#D97706", angles="xy",
        scale_units="xy", scale=1, width=0.0034, headwidth=3.4, headlength=4.0,
        zorder=6,
    )
    ax_far.scatter([chosen[0]], [chosen[1]], s=7.0, color="#D97706", zorder=6)
    ax_far.scatter(
        [query[0]], [query[1]], s=38, marker="X", facecolor="white",
        edgecolor="black", linewidth=0.9, zorder=7,
    )
    far_pad = 0.16 * max(cell_width, cell_height)
    far_view = (x0 - far_pad, x1 + far_pad, y0 - far_pad, y1 + far_pad)
    ax_far.set(xlim=far_view[:2], ylim=far_view[2:])
    ax_far.set_title(
        f"Evaluate order-{config.taylor_order} Taylor far-field deflection",
        pad=4, fontsize=8.2,
    )

    connector_style = dict(
        color="0.55", linewidth=0.55, linestyle=(0, (3, 2)),
        alpha=0.76, clip_on=False, zorder=10,
    )
    for cell_y in (y1, y0):
        figure.add_artist(ConnectionPatch(
            xyA=(x1, cell_y), coordsA=ax_field.transData,
            xyB=(x0, cell_y), coordsB=ax_local.transData, **connector_style,
        ))
        figure.add_artist(ConnectionPatch(
            xyA=(x1, cell_y), coordsA=ax_local.transData,
            xyB=(x0, cell_y), coordsB=ax_far.transData, **connector_style,
        ))
    ax_far.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="None", markerfacecolor="red", markeredgecolor="none", markersize=5.4, label="Microlens"),
            Line2D([], [], marker="o", linestyle="None", markerfacecolor="#0072B2", markeredgecolor="black", markersize=7.2, label="Exact local star"),
            Line2D([], [], marker="o", linestyle="None", markerfacecolor="0.72", markeredgecolor="0.42", markersize=7.0, label="Far star"),
            Line2D([], [], marker="o", linestyle="None", markerfacecolor="#CC79A7", markeredgecolor="none", markersize=5.0, label="Taylor expansion node"),
            Line2D([], [], marker="o", linestyle="None", markerfacecolor="#D97706", markeredgecolor="none", markersize=5.0, label="Selected expansion node"),
            Patch(facecolor="none", edgecolor="0.10", linewidth=1.45, label="Ray-containing cell"),
            Patch(facecolor="#C9E3F1", edgecolor="#0072B2", linewidth=1.35, linestyle=(0, (4, 2)), label="Exact-star region"),
        ],
        loc="upper right", bbox_to_anchor=(0.985, 0.985), fontsize=5.8,
        handletextpad=0.42, borderpad=0.42, labelspacing=0.62,
        frameon=True, facecolor="white", edgecolor="0.35", framealpha=0.96,
    )

    def nice_scale(width):
        target = 0.20 * float(width)
        base = 10.0 ** np.floor(np.log10(max(target, 1e-12)))
        return max((value * base for value in (1.0, 2.0, 5.0) if value * base <= target), default=base)

    for index, (axis, view) in enumerate(zip(axes, ((xmin, xmax, ymin, ymax), local_view, far_view), strict=True)):
        axis.set_aspect("equal", adjustable="box")
        hide_image_axes(axis)
        bar = float(scale_bar_uas) if index == 0 and scale_bar_uas is not None else nice_scale(view[1] - view[0])
        add_scale_bar(
            axis,
            bar,
            label=rf"{bar:g} $\mu$as",
            color="black",
            font_size=8.3,
            pad=0.15,
            borderpad=0.14,
            separation=3.0,
        )
    return figure, axes


def plot_lens_field_strategies(
    simulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    config: IPMConfig,
    rectangle_region: PlaneRegion,
    *,
    rectangle_rotation_deg: float = 0.0,
    time_days: float = 0.0,
    scale_bar_uas: float | None = None,
    maximum_display_stars: int = 8_000,
):
    """Compare tiled-scout, full-field, and rectangular lens apertures.

    This is a lens-plane diagnostic. All three panels share one stellar
    realization and the full-field coordinate system.  The first panel runs
    the real source scout and displays its retained integration cells. The
    second evaluates the complete supplied lens field. The third shows the
    explicitly truncated rectangular aperture.  It does not regenerate any
    magnification map.
    """

    require_matplotlib()
    if not config.tiled:
        raise ValueError("config must enable tiled source scouting")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle

    from ..solvers.far_field import TaylorFarFieldApproximation
    from ..solvers.ipm import _source_scout_cells

    far_field = None
    if config.far_field_approx.enabled:
        far_field = TaylorFarFieldApproximation(
            simulation, lens_region, config.far_field_approx, time_days=time_days
        )
    selected, fine_ny, fine_nx, metadata = _source_scout_cells(
        simulation,
        far_field,
        lens_region,
        source_grid,
        config,
        time_days=time_days,
    )
    selection = np.zeros(fine_ny * fine_nx, dtype=np.uint8)
    selection[as_numpy(selected).astype(np.int64, copy=False)] = 1
    selection = selection.reshape(fine_ny, fine_nx)

    field = simulation.lens_state(time_days)
    star_x = as_numpy(field.x_uas)
    star_y = as_numpy(field.y_uas)
    if len(star_x) > int(maximum_display_stars):
        sample = np.linspace(
            0, len(star_x) - 1, int(maximum_display_stars), dtype=np.int64
        )
        star_x, star_y = star_x[sample], star_y[sample]

    xmin, xmax, ymin, ymax = lens_region.bounds_uas
    rxmin, rxmax, rymin, rymax = rectangle_region.bounds_uas
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharex=True, sharey=True)
    axes[0].imshow(
        np.ma.masked_where(selection == 0, selection),
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
        alpha=0.88,
        rasterized=True,
    )
    axes[1].add_patch(
        Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor="#dbeaf5", edgecolor="#1f77b4", linewidth=1.2,
        )
    )
    rectangle_corners = np.asarray(
        (
            (rxmin, rymin),
            (rxmax, rymin),
            (rxmax, rymax),
            (rxmin, rymax),
        ),
        dtype=float,
    )
    if not np.isclose(float(rectangle_rotation_deg), 0.0):
        angle = np.deg2rad(float(rectangle_rotation_deg))
        cosine, sine = np.cos(angle), np.sin(angle)
        local_x, local_y = rectangle_corners.T
        rectangle_corners = np.column_stack(
            (cosine * local_x - sine * local_y,
             sine * local_x + cosine * local_y)
        )
    axes[2].add_patch(
        Polygon(
            rectangle_corners,
            closed=True,
            facecolor="#dbeaf5",
            edgecolor="#1f77b4",
            linewidth=1.4,
        )
    )
    radius = 0.5 * min(xmax - xmin, ymax - ymin)
    center = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
    for axis in axes:
        axis.scatter(
            star_x, star_y, s=0.45, color="#d62728", alpha=0.36,
            linewidths=0, rasterized=True, zorder=3,
        )
        axis.add_patch(
            Circle(
                center, radius, fill=False, edgecolor="0.25",
                linewidth=0.9, linestyle=(0, (5, 3)), zorder=4,
            )
        )
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        axis.set_aspect("equal", adjustable="box")
        hide_image_axes(axis)
    axes[0].set_title("Source-scouted tiles")
    axes[1].set_title("Full lens field")
    axes[2].set_title("Rectangular lens field")
    axes[0].text(
        0.04,
        0.95,
        f"{100.0 * float(metadata['selected_fine_fraction']):.2g}% cells retained",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        zorder=5,
    )
    if scale_bar_uas is not None:
        for axis in axes:
            add_scale_bar(
                axis,
                float(scale_bar_uas),
                label=rf"{float(scale_bar_uas):g} $\mu$as",
                color="black",
            )
    figure.tight_layout()
    return figure, axes


def plot_ipm_scout_method(
    simulation,
    lens_region: PlaneRegion,
    source_grid: PlaneGrid,
    config: IPMConfig,
    *,
    time_days: float = 0.0,
    selected_lens_position_uas: tuple[float, float] | None = None,
    lens_scale_bar_uas: float | None = None,
    source_scale_bar_uas: float | None = None,
):
    """Show the production source scout and curved IPM construction.

    The two panels deliberately mirror the method schematic used in the paper:
    the left panel shows the retained coarse scout tiles and a magnified
    ``k x k`` fine-grid tile, while the right panel follows one fine cell into
    the source plane and magnifies its exact triangle--pixel overlap.  All
    geometry is evaluated from ``simulation`` and ``config``. The diagram is
    therefore valid for any supported scout ratio ``k``, traced refinement
    ``r``, and virtual refinement ``v`` rather than being a hard-coded cartoon.
    """

    require_matplotlib()
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch, Patch, Polygon, Rectangle
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    from ..solvers.far_field import TaylorFarFieldApproximation
    from ..solvers.ipm import (
        _source_scout_cells,
        interpolated_nodes,
        triangles_from_node_lattices,
    )

    far_field = None
    if config.far_field_approx.enabled:
        far_field = TaylorFarFieldApproximation(
            simulation,
            lens_region,
            config.far_field_approx,
            time_days=time_days,
        )
    selected, fine_ny, fine_nx, metadata, scout_corners = _source_scout_cells(
        simulation,
        far_field,
        lens_region,
        source_grid,
        config,
        time_days=time_days,
        _return_corners=True,
    )
    if selected.numel() == 0:
        raise RuntimeError("the illustrative scout selected no lens cells")
    xmin, xmax, ymin, ymax = lens_region.bounds_uas
    dx, dy = (xmax - xmin) / fine_nx, (ymax - ymin) / fine_ny
    rows = torch.div(selected, fine_nx, rounding_mode="floor")
    columns = selected - rows * fine_nx
    center_x = xmin + (columns.to(simulation.runtime.dtype) + 0.5) * dx
    center_y = ymin + (rows.to(simulation.runtime.dtype) + 0.5) * dy
    if far_field is None:
        mapped_x, mapped_y, _ = simulation.raytrace_direct(
            center_x, center_y, time_days=time_days
        )
    else:
        mapped_x, mapped_y = far_field.raytrace(center_x, center_y)
    target_x = float(source_grid.center_uas[1])
    target_y = float(source_grid.center_uas[0])
    source_bounds = source_grid.bounds_uas
    if selected_lens_position_uas is None:
        # A center-only choice can land on an extremely compressed critical
        # cell, which is scientifically valid but makes a poor explanatory
        # diagram.  Rank a small set of central retained cells by the shape of
        # their genuinely ray-traced quadrilateral and prefer a well-resolved,
        # non-degenerate example.  This affects visualization only.
        source_distance = (mapped_x - target_x).square() + (mapped_y - target_y).square()
        candidate_count = min(512, int(selected.numel()))
        candidates = torch.topk(
            source_distance,
            k=candidate_count,
            largest=False,
        ).indices
        candidate_rows, candidate_columns = rows[candidates], columns[candidates]
        candidate_x0 = xmin + candidate_columns.to(simulation.runtime.dtype) * dx
        candidate_y0 = ymin + candidate_rows.to(simulation.runtime.dtype) * dy
        corner_lens_x = torch.stack(
            (candidate_x0, candidate_x0 + dx, candidate_x0 + dx, candidate_x0), dim=1
        )
        corner_lens_y = torch.stack(
            (candidate_y0, candidate_y0, candidate_y0 + dy, candidate_y0 + dy), dim=1
        )
        if far_field is None:
            corner_source_x, corner_source_y, _ = simulation.raytrace_direct(
                corner_lens_x, corner_lens_y, time_days=time_days
            )
        else:
            corner_source_x, corner_source_y = far_field.raytrace(
                corner_lens_x, corner_lens_y
            )
        edge_u_x = corner_source_x[:, 1] - corner_source_x[:, 0]
        edge_u_y = corner_source_y[:, 1] - corner_source_y[:, 0]
        edge_v_x = corner_source_x[:, 3] - corner_source_x[:, 0]
        edge_v_y = corner_source_y[:, 3] - corner_source_y[:, 0]
        length_u = torch.sqrt(edge_u_x.square() + edge_u_y.square())
        length_v = torch.sqrt(edge_v_x.square() + edge_v_y.square())
        safe_product = torch.clamp(length_u * length_v, min=torch.finfo(length_u.dtype).tiny)
        cross = torch.abs(edge_u_x * edge_v_y - edge_u_y * edge_v_x)
        aspect = torch.minimum(length_u, length_v) / torch.clamp(
            torch.maximum(length_u, length_v), min=torch.finfo(length_u.dtype).tiny
        )
        orthogonality = cross / safe_product
        pixel_area = float(source_grid.pixel_scale_uas[0] * source_grid.pixel_scale_uas[1])
        resolved_area = torch.sqrt(torch.clamp(cross / pixel_area, min=0.0))
        # About 24 source pixels across gives enough grid cells to explain the
        # exact overlap without turning the inset into an unreadable moire.
        resolution_ratio = torch.clamp(resolved_area / 24.0, min=1.0e-6)
        resolution_score = torch.exp(-0.5 * (torch.log(resolution_ratio) / 0.6).square())
        source_half_width = max(
            0.5 * (source_bounds[1] - source_bounds[0]),
            0.5 * (source_bounds[3] - source_bounds[2]),
        )
        central_penalty = 1.0 + 2.0 * torch.sqrt(source_distance[candidates]) / source_half_width
        shape_score = aspect * orthogonality * resolution_score / central_penalty
        corners_inside = (
            (corner_source_x >= source_bounds[0])
            & (corner_source_x <= source_bounds[1])
            & (corner_source_y >= source_bounds[2])
            & (corner_source_y <= source_bounds[3])
        ).all(dim=1)
        if bool(corners_inside.any()):
            shape_score = torch.where(
                corners_inside, shape_score, torch.full_like(shape_score, -torch.inf)
            )
        finite = torch.isfinite(shape_score)
        if bool(finite.any()):
            shape_score = torch.where(
                finite, shape_score, torch.full_like(shape_score, -torch.inf)
            )
            best = candidates[torch.argmax(shape_score)]
        else:
            best = candidates[0]
    else:
        requested_x, requested_y = (float(value) for value in selected_lens_position_uas)
        best = torch.argmin(
            (center_x - requested_x).square() + (center_y - requested_y).square()
        )
    row, column = int(rows[best]), int(columns[best])
    cell_x0, cell_y0 = xmin + column * dx, ymin + row * dy
    refinement = int(config.refinement)
    lens_nodes_x = torch.linspace(
        cell_x0,
        cell_x0 + dx,
        refinement + 1,
        device=simulation.runtime.device,
        dtype=simulation.runtime.dtype,
    )
    lens_nodes_y = torch.linspace(
        cell_y0,
        cell_y0 + dy,
        refinement + 1,
        device=simulation.runtime.device,
        dtype=simulation.runtime.dtype,
    )
    lens_mesh_y, lens_mesh_x = torch.meshgrid(lens_nodes_y, lens_nodes_x, indexing="ij")
    if far_field is None:
        true_x, true_y, _ = simulation.raytrace_direct(
            lens_mesh_x, lens_mesh_y, time_days=time_days
        )
    else:
        true_x, true_y = far_field.raytrace(lens_mesh_x, lens_mesh_y)
    virtual_x, virtual_y = interpolated_nodes(
        true_x[None], true_y[None], virtual_refinement=config.virtual_refinement
    )
    triangles = as_numpy(triangles_from_node_lattices(virtual_x, virtual_y))

    def _finish_axis(axis):
        axis.set_aspect("equal", adjustable="box")
        hide_image_axes(axis)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("0.25")

    def _nice_scale(width):
        """Choose a 1/2/5 scale bar close to one quarter of a view width."""

        target = max(float(width) / 4.0, np.finfo(float).tiny)
        decade = 10.0 ** math.floor(math.log10(target))
        return max(value for value in (decade, 2 * decade, 5 * decade) if value <= target)

    def _clip_half_plane(polygon, axis, boundary, keep_greater):
        if len(polygon) == 0:
            return polygon
        result = []
        previous = polygon[-1]
        previous_inside = (
            previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
        )
        for current in polygon:
            current_inside = (
                current[axis] >= boundary if keep_greater else current[axis] <= boundary
            )
            if current_inside != previous_inside:
                denominator = current[axis] - previous[axis]
                fraction = 0.0 if denominator == 0 else (boundary - previous[axis]) / denominator
                crossing = previous + fraction * (current - previous)
                result.append(crossing)
            if current_inside:
                result.append(current)
            previous, previous_inside = current, current_inside
        return np.asarray(result, dtype=float).reshape(-1, 2)

    def _clip_to_pixel(triangle, x0, x1, y0, y1):
        polygon = np.asarray(triangle, dtype=float)
        for axis, boundary, keep_greater in (
            (0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)
        ):
            polygon = _clip_half_plane(polygon, axis, boundary, keep_greater)
            if len(polygon) == 0:
                break
        return polygon

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 5.2))
    field = simulation.lens_state(time_days)
    star_x, star_y = as_numpy(field.x_uas), as_numpy(field.y_uas)
    mass = (
        as_numpy(field.mass_solar)
        if field.mass_solar is not None
        else as_numpy(field.einstein_radius_uas) ** 2
    )
    positive_mass = mass[np.isfinite(mass) & (mass > 0)]
    mass_scale = float(np.median(positive_mass)) if positive_mass.size else 1.0
    marker_size = np.clip(1.8 * np.sqrt(np.maximum(mass, 0) / mass_scale), 0.35, 10.0)
    axes[0].scatter(
        star_x,
        star_y,
        s=marker_size,
        color="#ff3030",
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    ratio = int(config.scout_ratio)
    selected_mask = np.zeros((fine_ny, fine_nx), dtype=bool)
    selected_mask[as_numpy(rows).astype(int), as_numpy(columns).astype(int)] = True
    coarse_mask = selected_mask.reshape(
        fine_ny // ratio, ratio, fine_nx // ratio, ratio
    ).any(axis=(1, 3))
    masked = np.ma.masked_where(~coarse_mask, coarse_mask.astype(float))
    axes[0].imshow(
        masked,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
        alpha=0.82,
        zorder=1,
    )
    lens_width, lens_height = xmax - xmin, ymax - ymin
    axes[0].add_patch(
        plt.Circle(
            ((xmin + xmax) / 2, (ymin + ymax) / 2),
            0.5 * min(lens_width, lens_height),
            fill=False,
            edgecolor="0.3",
            linewidth=1.4,
            linestyle=(0, (6, 4)),
            zorder=3,
        )
    )
    coarse_row, coarse_column = row // ratio, column // ratio
    coarse_x0 = xmin + coarse_column * ratio * dx
    coarse_y0 = ymin + coarse_row * ratio * dy
    axes[0].add_patch(
        Rectangle(
            (coarse_x0, coarse_y0),
            ratio * dx,
            ratio * dy,
            fill=False,
            edgecolor="#1f77b4",
            linewidth=1.6,
            zorder=5,
        )
    )
    axes[0].set(
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        title="Lens-plane tile selection",
    )
    axes[0].text(
        0.04,
        0.94,
        f"{100.0 * float(metadata['selected_fine_fraction']):.2g}% scout cells\nretained",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.5),
        zorder=8,
    )

    # Magnify the selected coarse scout tile and its genuine k x k fine cells.
    lens_inset = inset_axes(
        axes[0],
        width="47%",
        height="44%",
        loc="lower right",
        borderpad=0.0,
    )
    lens_inset.set_facecolor("#dbeaf5")
    for index in range(ratio + 1):
        line_x = coarse_x0 + index * dx
        line_y = coarse_y0 + index * dy
        lens_inset.axvline(line_x, color="#1f77b4", linewidth=0.9, zorder=1)
        lens_inset.axhline(line_y, color="#1f77b4", linewidth=0.9, zorder=1)
    lens_inset.add_patch(
        Rectangle(
            (cell_x0, cell_y0),
            dx,
            dy,
            fill=False,
            edgecolor="#165a88",
            linewidth=1.5,
            zorder=4,
        )
    )
    lens_virtual_axis_x = np.linspace(cell_x0, cell_x0 + dx, config.virtual_refinement + 1)
    lens_virtual_axis_y = np.linspace(cell_y0, cell_y0 + dy, config.virtual_refinement + 1)
    lens_virtual_y, lens_virtual_x = np.meshgrid(
        lens_virtual_axis_y, lens_virtual_axis_x, indexing="ij"
    )
    true_x_np, true_y_np = as_numpy(true_x), as_numpy(true_y)
    inside_true = (
        (true_x_np >= source_bounds[0])
        & (true_x_np <= source_bounds[1])
        & (true_y_np >= source_bounds[2])
        & (true_y_np <= source_bounds[3])
    )
    lens_inset.scatter(
        lens_virtual_x,
        lens_virtual_y,
        s=10,
        facecolor="white",
        edgecolor="#1f77b4",
        linewidth=0.45,
        zorder=5,
    )
    lens_inset.scatter(
        as_numpy(lens_mesh_x)[inside_true],
        as_numpy(lens_mesh_y)[inside_true],
        s=34,
        facecolor="#2ca02c",
        edgecolor="black",
        linewidth=0.65,
        zorder=6,
    )
    lens_inset.scatter(
        as_numpy(lens_mesh_x)[~inside_true],
        as_numpy(lens_mesh_y)[~inside_true],
        s=32,
        marker="x",
        color="#d62728",
        linewidth=1.1,
        zorder=6,
    )
    corner_positions = (
        (cell_x0, cell_y0 + dy, "1", -1, 1),
        (cell_x0 + dx, cell_y0 + dy, "2", 1, 1),
        (cell_x0 + dx, cell_y0, "3", 1, -1),
        (cell_x0, cell_y0, "4", -1, -1),
    )
    for x_value, y_value, label, x_sign, y_sign in corner_positions:
        lens_inset.annotate(
            label,
            (x_value, y_value),
            xytext=(5 * x_sign, 5 * y_sign),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            zorder=8,
        )
    lens_pad = 0.32 * ratio * max(dx, dy)
    lens_inset.set(
        xlim=(coarse_x0 - lens_pad, coarse_x0 + ratio * dx + lens_pad),
        ylim=(coarse_y0 - lens_pad, coarse_y0 + ratio * dy + lens_pad),
    )
    lens_inset.text(
        0.5,
        0.98,
        "scout tile and fine grid",
        transform=lens_inset.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.7),
    )
    _finish_axis(lens_inset)
    add_scale_bar(
        lens_inset,
        _nice_scale(np.diff(lens_inset.get_xlim())[0]),
        label=rf"{_nice_scale(np.diff(lens_inset.get_xlim())[0]):g} $\mu$as",
        color="black",
        font_size=7.2,
        pad=0.12,
        borderpad=0.14,
        separation=2.5,
    )
    axes[0].add_artist(
        ConnectionPatch(
            xyA=(coarse_x0 + ratio * dx, coarse_y0 + ratio * dy),
            coordsA=axes[0].transData,
            xyB=(0.02, 0.98),
            coordsB=lens_inset.transAxes,
            color="0.35",
            linewidth=0.8,
            linestyle="--",
        )
    )

    # Show the selected cell at its true source-plane location, then magnify it.
    axes[1].add_collection(
        LineCollection(triangles, colors="#1f77b4", linewidths=0.55, alpha=0.9)
    )
    true_xy = np.column_stack((as_numpy(true_x).ravel(), as_numpy(true_y).ravel()))
    virtual_xy = np.column_stack((as_numpy(virtual_x).ravel(), as_numpy(virtual_y).ravel()))
    axes[1].scatter(
        virtual_xy[:, 0],
        virtual_xy[:, 1],
        s=8,
        facecolor="white",
        edgecolor="#1f77b4",
        linewidth=0.45,
        zorder=4,
    )
    axes[1].scatter(
        true_xy[inside_true.ravel(), 0],
        true_xy[inside_true.ravel(), 1],
        s=18,
        facecolor="#2ca02c",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )
    if np.any(~inside_true):
        axes[1].scatter(
            true_xy[(~inside_true).ravel(), 0],
            true_xy[(~inside_true).ravel(), 1],
            s=18,
            marker="x",
            color="#d62728",
            linewidth=0.9,
            zorder=5,
        )
    px = float(source_grid.pixel_scale_uas[1])
    py = float(source_grid.pixel_scale_uas[0])
    local_x0, local_x1 = triangles[..., 0].min(), triangles[..., 0].max()
    local_y0, local_y1 = triangles[..., 1].min(), triangles[..., 1].max()
    axes[1].add_patch(
        Rectangle(
            (source_bounds[0], source_bounds[2]),
            source_bounds[1] - source_bounds[0],
            source_bounds[3] - source_bounds[2],
            fill=False,
            edgecolor="0.45",
            linewidth=0.8,
            linestyle="--",
        )
    )
    axes[1].set(
        xlim=(source_bounds[0], source_bounds[1]),
        ylim=(source_bounds[2], source_bounds[3]),
        title="Exact IPM in the source plane",
    )

    source_inset = inset_axes(
        axes[1],
        width="48%",
        height="44%",
        loc="lower right",
        borderpad=0.0,
    )
    triangle_array = np.asarray(triangles)
    source_inset.add_collection(
        LineCollection(triangle_array, colors="#1f77b4", linewidths=0.8, alpha=0.95)
    )
    grid_x0 = source_bounds[0] + math.floor((local_x0 - source_bounds[0]) / px) * px
    grid_y0 = source_bounds[2] + math.floor((local_y0 - source_bounds[2]) / py) * py
    for value in np.arange(grid_x0, local_x1 + px, px):
        source_inset.axvline(value, color="#9ecae1", linewidth=0.34, zorder=0)
    for value in np.arange(grid_y0, local_y1 + py, py):
        source_inset.axhline(value, color="#9ecae1", linewidth=0.34, zorder=0)
    source_inset.scatter(
        virtual_xy[:, 0], virtual_xy[:, 1], s=12, facecolor="white",
        edgecolor="#1f77b4", linewidth=0.55, zorder=4
    )
    source_inset.scatter(
        true_xy[inside_true.ravel(), 0], true_xy[inside_true.ravel(), 1],
        s=26, facecolor="#2ca02c", edgecolor="black", linewidth=0.55, zorder=5
    )
    if np.any(~inside_true):
        source_inset.scatter(
            true_xy[(~inside_true).ravel(), 0], true_xy[(~inside_true).ravel(), 1],
            s=26, marker="x", color="#d62728", linewidth=1.0, zorder=5
        )

    # Highlight one genuine triangle--source-pixel intersection.
    target_triangle = triangle_array[
        np.argmin(np.sum((triangle_array.mean(axis=1) - np.asarray((target_x, target_y))) ** 2, axis=1))
    ]
    centroid = target_triangle.mean(axis=0)
    pixel_column = math.floor((centroid[0] - source_bounds[0]) / px)
    pixel_row = math.floor((centroid[1] - source_bounds[2]) / py)
    pixel_x0 = source_bounds[0] + pixel_column * px
    pixel_y0 = source_bounds[2] + pixel_row * py
    overlap = _clip_to_pixel(
        target_triangle, pixel_x0, pixel_x0 + px, pixel_y0, pixel_y0 + py
    )
    if len(overlap) >= 3:
        source_inset.add_patch(
            Polygon(
                overlap,
                closed=True,
                facecolor="#f2b632",
                edgecolor="#d99000",
                linewidth=0.9,
                alpha=0.82,
                zorder=3,
            )
        )
    virtual_x_np, virtual_y_np = as_numpy(virtual_x[0]), as_numpy(virtual_y[0])
    for x_value, y_value, label, x_sign, y_sign in (
        (virtual_x_np[-1, 0], virtual_y_np[-1, 0], "1", -1, 1),
        (virtual_x_np[-1, -1], virtual_y_np[-1, -1], "2", 1, 1),
        (virtual_x_np[0, -1], virtual_y_np[0, -1], "3", 1, -1),
        (virtual_x_np[0, 0], virtual_y_np[0, 0], "4", -1, -1),
    ):
        source_inset.annotate(
            label,
            (x_value, y_value),
            xytext=(6 * x_sign, 6 * y_sign),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            zorder=7,
        )
    pad_x = max(2 * px, 0.18 * (local_x1 - local_x0))
    pad_y = max(2 * py, 0.18 * (local_y1 - local_y0))
    source_inset.set(
        xlim=(local_x0 - pad_x, local_x1 + pad_x),
        ylim=(local_y0 - pad_y, local_y1 + pad_y),
    )
    source_inset.text(
        0.5,
        0.98,
        "triangle-pixel overlap",
        transform=source_inset.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.7),
    )
    _finish_axis(source_inset)
    add_scale_bar(
        source_inset,
        _nice_scale(np.diff(source_inset.get_xlim())[0]),
        label=rf"{_nice_scale(np.diff(source_inset.get_xlim())[0]):g} $\mu$as",
        color="black",
        font_size=7.2,
        pad=0.12,
        borderpad=0.14,
        separation=2.5,
    )
    axes[1].add_artist(
        ConnectionPatch(
            xyA=(float(centroid[0]), float(centroid[1])),
            coordsA=axes[1].transData,
            xyB=(0.03, 0.98),
            coordsB=source_inset.transAxes,
            color="0.35",
            linewidth=0.8,
            linestyle="--",
        )
    )
    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", color="#ff3030", markersize=5, label="Microlens"),
        Patch(facecolor="#c6dbef", edgecolor="#1f77b4", label="Retained scout tile"),
        Line2D([], [], color="#1f77b4", linewidth=0.9, label=rf"Fine integration grid ($k={ratio}$)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#2ca02c", markeredgecolor="black", markersize=6, label="Ray-traced node inside source plane"),
        Line2D([], [], marker="x", linestyle="none", color="#d62728", markersize=6, label="Ray-traced node outside source plane"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#1f77b4", markersize=6, label=("Biquadratic virtual node" if refinement == 2 else rf"Degree-{refinement} virtual node")),
        Line2D([], [], color="#1f77b4", linewidth=1.2, label=rf"Mapped triangles ($r={refinement}$, $v={config.virtual_refinement}$)"),
        Line2D([], [], color="#9ecae1", linewidth=0.8, label="Source-pixel grid"),
        Patch(facecolor="#f2b632", edgecolor="#d99000", label="Exact overlap area"),
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=7.4,
        borderpad=0.45,
        handlelength=1.4,
        labelspacing=0.35,
    )
    for axis, scale in zip(axes, (lens_scale_bar_uas, source_scale_bar_uas), strict=True):
        _finish_axis(axis)
        if scale is None:
            width = np.diff(axis.get_xlim())[0]
            scale = 10 ** math.floor(math.log10(max(width / 4.0, np.finfo(float).tiny)))
        add_scale_bar(
            axis,
            scale,
            label=rf"{scale:g} $\mu$as",
            color="black",
            font_size=8.5,
        )
    figure.subplots_adjust(left=0.025, right=0.995, bottom=0.035, top=0.91, wspace=0.09)
    figure._microcaustics_metadata = {
        **metadata,
        "displayed_fine_cell_column": column,
        "displayed_fine_cell_row": row,
        "displayed_lens_cell_center_uas": (
            cell_x0 + 0.5 * dx,
            cell_y0 + 0.5 * dy,
        ),
    }
    return figure, axes
